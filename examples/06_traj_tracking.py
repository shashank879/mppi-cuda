"""Closed-loop EE trajectory tracking in MuJoCo.

The target moves along a parametric trajectory (circle by default, optional
figure-eight or waypoint loop). At each control tick we re-sample the
trajectory at the upcoming H+1 timesteps and hand the buffer to the cost;
both PyTorch and CUDA backends consume the same `target_traj`.

Tracking metrics reported at the end:
  - per-tick ||ee(t) - target(t)||           (visualised over time)
  - accumulated RMS in mm
  - per-tick action jitter ||u_t - u_{t-1}||

Usage:
    PYTHONPATH=. python examples/06_traj_tracking.py
    PYTHONPATH=. python examples/06_traj_tracking.py --traj figure_eight
    PYTHONPATH=. python examples/06_traj_tracking.py --backend cuda_kernel
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    MujocoFrankaEnv,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
    forward_kinematics,
    trajectories as traj,
)


def accel_to_position_target(u_accel, q, qdot, dt, lookahead: int = 5):
    qdot_target = qdot + dt * u_accel
    return q + lookahead * dt * qdot_target


def make_traj_fn(name: str):
    if name == "circle":
        return traj.Circle(center=(0.45, 0.10, 0.50), radius=0.10, period=4.0, plane="xy")
    if name == "figure_eight":
        return traj.FigureEight(center=(0.45, 0.10, 0.50), radius=0.10, period=6.0, plane="xy")
    if name == "waypoints":
        return traj.Waypoints(
            points=[(0.45, 0.00, 0.50), (0.50, 0.15, 0.55),
                    (0.45, 0.20, 0.50), (0.40, 0.10, 0.45)],
            segment_time=1.5,
        )
    if name == "square":
        return traj.Waypoints(
            points=[(0.50, 0.00, 0.50), (0.50, 0.50, 0.50),
                    (0.00, 0.50, 0.50), (0.00, 0.00, 0.50)],
            segment_time=1.5,
        )
    raise ValueError(f"unknown trajectory: {name}")


def run(
    device: str = "cpu",
    backend: str = "pytorch",
    traj_name: str = "circle",
    horizon: int = 10,
    num_samples: int = 4096,
    n_ticks: int = 300,
    savepath=None,
):
    dtype = torch.float32

    traj_fn = make_traj_fn(traj_name)
    env = MujocoFrankaEnv(
        control_dt=0.02,
        target_marker=traj_fn(0.0),     # initial position; updated each tick
    )

    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)

    # Cost starts with a placeholder (H+1, 3) buffer; we overwrite it each tick.
    placeholder = traj.build_target_traj(
        traj_fn, t_now=0.0, dt=env.control_dt, horizon=horizon,
    )
    cost = ReachingCost(
        target_pos=placeholder,
        w_pos=500.0,
        w_u=0.005,
        w_qdot=0.05,
        terminal_scale=20.0,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX,
        device=device, dtype=dtype,
    )

    if backend == "cuda_kernel":
        from mppi_cuda import CudaMPPIController
        if CudaMPPIController is None:
            raise RuntimeError(
                "CUDA kernel extension is not built. Run `pip install -e .` in a CUDA env."
            )
        controller = CudaMPPIController(
            dynamics=predictive, cost=cost,
            action_dim=7, horizon=horizon, num_samples=num_samples,
            sigma=2.5, temperature=1.0,
            u_min=-20.0, u_max=20.0,
            device="cuda", dtype=dtype, seed=0,
        )
    else:
        controller = MPPIController(
            dynamics=predictive.step,
            running_cost=cost.running_cost,
            terminal_cost=cost.terminal_cost,
            action_dim=7, horizon=horizon, num_samples=num_samples,
            sigma=2.5, temperature=1.0,
            u_min=-20.0, u_max=20.0,
            device=device, dtype=dtype, seed=0,
        )

    s = env.reset()
    ee_log = [env.ee_position.copy()]
    target_log = [traj_fn(0.0).copy()]
    action_log = []
    tick_times = []

    u_prev = np.zeros(7)
    for tick in range(n_ticks):
        t_now = tick * env.control_dt

        # Slide the lookahead window: sample target at t_now, t_now+dt, ...
        target_buf = traj.build_target_traj(
            traj_fn, t_now=t_now, dt=env.control_dt, horizon=horizon,
        )
        cost.set_target_traj(torch.from_numpy(target_buf).to(device=device, dtype=dtype))

        # Update the marker so it tracks the visual sphere.
        env.data.mocap_pos[0] = target_buf[0]

        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        t0 = time.perf_counter()
        u = controller.step(x)
        tick_times.append(time.perf_counter() - t0)
        u_np = u.cpu().numpy().astype(np.float64)
        action_log.append(u_np.copy())

        q = s[:7].astype(np.float64)
        qdot = s[7:].astype(np.float64)
        q_target = accel_to_position_target(u_np, q, qdot, env.control_dt)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)
        s = env.step(q_target)
        ee_log.append(env.ee_position.copy())
        target_log.append(target_buf[0].copy())
        u_prev = u_np

    ee_log = np.array(ee_log)
    target_log = np.array(target_log)
    action_log = np.array(action_log)

    # ---- metrics ----
    tracking_err_mm = np.linalg.norm(ee_log - target_log, axis=1) * 1000.0
    # Skip a brief settling window so the steady-state metric reflects tracking,
    # not the initial home → trajectory approach.
    settle_ticks = 40
    rms_mm = float(np.sqrt(np.mean(tracking_err_mm[settle_ticks:] ** 2)))
    mean_mm = float(np.mean(tracking_err_mm[settle_ticks:]))
    p99_mm = float(np.percentile(tracking_err_mm[settle_ticks:], 99))

    # Action jitter = mean ||u_t - u_{t-1}||
    if len(action_log) > 1:
        action_diff = np.linalg.norm(np.diff(action_log, axis=0), axis=1)
        jitter = float(np.mean(action_diff))
    else:
        jitter = float("nan")

    mean_tick_ms = float(np.mean(tick_times) * 1000.0)

    print(f"Backend:                  {backend}")
    print(f"Trajectory:               {traj_name}")
    print(f"Horizon, samples:         H={horizon}, K={num_samples}")
    print(f"Sim duration:             {n_ticks * env.control_dt:.2f} s ({n_ticks} ticks)")
    print(f"Tracking RMS (post {settle_ticks*env.control_dt:.1f}s settle):  {rms_mm:.2f} mm")
    print(f"  mean: {mean_mm:.2f} mm | p99: {p99_mm:.2f} mm")
    print(f"Action jitter (mean ||Δu||):  {jitter:.3f}")
    print(f"Mean per-tick latency:    {mean_tick_ms:.2f} ms")

    if savepath:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(savepath, exist_ok=True)

        fig = plt.figure(figsize=(13, 4.5))

        ax = fig.add_subplot(131, projection="3d")
        ax.plot(target_log[:, 0], target_log[:, 1], target_log[:, 2],
                color="red", ls="--", lw=1, label="target", alpha=0.7)
        ax.plot(ee_log[:, 0], ee_log[:, 1], ee_log[:, 2],
                color="tab:blue", lw=2, label="EE")
        ax.scatter(*ee_log[0], color="green", s=60)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"EE vs target ({traj_name})")
        ax.legend(fontsize=8)

        ax2 = fig.add_subplot(132)
        t_axis = np.arange(len(ee_log)) * env.control_dt
        ax2.plot(t_axis, tracking_err_mm, lw=1.5)
        ax2.axvline(settle_ticks * env.control_dt, color="gray", ls=":",
                    alpha=0.6, label="settle line")
        ax2.set_xlabel("sim time (s)"); ax2.set_ylabel("EE tracking error (mm)")
        ax2.set_title(f"Tracking error  (RMS post-settle: {rms_mm:.1f} mm)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(133)
        if len(action_log) > 1:
            # action_diff[k] = || u_{k+1} - u_k ||; plot vs time of u_{k+1}
            action_times = env.control_dt * np.arange(1, len(action_log))
            ax3.plot(action_times, action_diff, lw=1.5)
            ax3.set_xlabel("sim time (s)"); ax3.set_ylabel(r"$\|u_t - u_{t-1}\|$")
            ax3.set_title(f"Action jitter (mean {jitter:.3f})")
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(savepath, f"traj_tracking_{traj_name}_{backend}.png")
        plt.savefig(out, dpi=120)
        print(f"Saved plot: {out}")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", default="cuda_kernel",
                   choices=["pytorch", "cuda_kernel"],
                   help="MPPI backend. pytorch works without the CUDA extension.")
    p.add_argument("--traj", default="circle",
                   choices=["circle", "figure_eight", "waypoints", "square"])
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--samples", type=int, default=16384)
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--savepath", default="docs")
    args = p.parse_args()

    run(
        device=args.device,
        backend=args.backend,
        traj_name=args.traj,
        horizon=args.horizon,
        num_samples=args.samples,
        n_ticks=args.ticks,
        savepath=args.savepath,
    )
