"""Day 2b demo: closed-loop MPPI controlling a MuJoCo Franka.

Plant:  MuJoCo Franka with PD-controlled actuators (real physics).
Model:  Double integrator in joint space (controller's predictive model).

The mismatch between the two is exactly what MPC is designed to absorb
via per-tick re-planning. Watch the final EE error and the convergence
curve — they tell you how well the closed loop is closing.
"""

from __future__ import annotations
import os, time

import numpy as np
import torch

from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    MujocoFrankaEnv,
    forward_kinematics,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)


def accel_to_position_target(u_accel, q, qdot, dt, lookahead: int = 5):
    """Bridge MPPI's acceleration output to a position target for the PD actuator.

    Naive integration (lookahead=1) commands q_next = q + dt*qdot_target.
    But MuJoCo's overdamped PD reaches that tiny target in milliseconds and
    settles — the arm never sustains the velocity the controller is planning
    for. Commanding `lookahead` ticks ahead keeps the actuator in the
    chasing regime with continuous velocity, at the cost of a small phase
    lead in the closed loop (which MPC handles via re-planning).
    """
    qdot_target = qdot + dt * u_accel
    q_target = q + lookahead * dt * qdot_target
    return q_target


def run(device: str = "cpu", savepath=None):
    dtype = torch.float32

    # --- Environment (plant) ---
    env = MujocoFrankaEnv(control_dt=0.02)

    # --- Controller's predictive model ---
    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)

    # --- Cost ---
    target_pos = torch.tensor([0.5, 0.3, 0.5], device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=target_pos,
        # Heavy reach cost dominates exploration noise AND lets the controller
        # apply enough corrective effort to compensate for the gravity bias that
        # the double-integrator predictive model doesn't see.
        w_pos=500.0,
        w_u=0.005,
        w_qdot=0.05,
        terminal_scale=20.0,
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        device=device,
        dtype=dtype,
    )

    # --- Controller (retuned for MuJoCo) ---
    #
    # Iteration log:
    #   v1 (u_max=8,  sigma=2): saturated, converged at 20 mm/s, final 287 mm. Too slow.
    #   v2 (u_max=40, sigma=5): no saturation but no convergence improvement.
    #                           PD tracking lag was the binding constraint, not u_max.
    #   v3: added lookahead=5 in bridge; final 146 mm but oscillatory at bottom.
    #   v4 (this):  lookahead=5 + smaller sigma=2.5, smaller u_max=20,
    #               heavier reach-cost weighting (in `cost` above) to overpower noise.
    controller = MPPIController(
        dynamics=predictive.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7,
        horizon=40,
        num_samples=1024,
        sigma=2.5,
        temperature=1.0,
        u_min=-20.0,
        u_max=20.0,
        device=device,
        dtype=dtype,
        seed=0,
    )

    # --- Reset ---
    s = env.reset()  # defaults to home

    # --- Logging ---
    state_log = [s.copy()]
    ee_log = [env.ee_position.copy()]
    cmd_log = []
    tick_times = []

    n_ticks = 200
    for t in range(n_ticks):
        # 1. State to controller as a torch tensor
        x = torch.from_numpy(s).to(device=device, dtype=dtype)

        # 2. MPPI tick (this is what the CUDA kernel will replace)
        t0 = time.perf_counter()
        u = controller.step(x)
        tick_times.append(time.perf_counter() - t0)
        u_np = u.cpu().numpy().astype(np.float64)

        # 3. Bridge: accel command -> position target for MuJoCo
        q = s[:7].astype(np.float64)
        qdot = s[7:].astype(np.float64)
        q_target = accel_to_position_target(u_np, q, qdot, env.control_dt)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)

        # 4. Step the simulator
        s = env.step(q_target)

        state_log.append(s.copy())
        ee_log.append(env.ee_position.copy())
        cmd_log.append(u_np)

    # --- Report ---
    ee_log = np.array(ee_log)
    state_log = np.array(state_log)
    cmd_log = np.array(cmd_log)
    tick_times_ms = np.array(tick_times) * 1000

    err_final = np.linalg.norm(ee_log[-1] - target_pos.cpu().numpy())

    print(f"Device:               {device}")
    print(f"Plant:                MuJoCo Franka ({env.n_sim_steps} sim steps per tick)")
    print(f"K, H:                 {controller.K}, {controller.H}")
    print(f"Ran {n_ticks} control ticks ({n_ticks * env.control_dt:.1f} s simulated)")
    print(f"Per-tick latency:     mean {tick_times_ms.mean():.2f} ms, "
          f"median {np.median(tick_times_ms):.2f} ms, "
          f"p99 {np.percentile(tick_times_ms, 99):.2f} ms")
    print(f"Target EE:            {target_pos.cpu().numpy()}")
    print(f"Final  EE:            {ee_log[-1]}")
    print(f"Final  position err:  {err_final * 1000:.2f} mm")

    if savepath:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(13, 4.5))

        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(ee_log[:, 0], ee_log[:, 1], ee_log[:, 2], lw=2)
        ax1.scatter(*ee_log[0], color="green", s=80, label="start")
        ax1.scatter(*target_pos.cpu().numpy(), color="red", s=80, label="target")
        ax1.scatter(*ee_log[-1], color="blue", s=80, label="final")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        ax1.set_title("EE trajectory (MuJoCo)")
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(132)
        t_axis = np.arange(len(ee_log)) * env.control_dt
        err_traj = np.linalg.norm(ee_log - target_pos.cpu().numpy(), axis=1) * 1000
        ax2.plot(t_axis, err_traj)
        ax2.set_xlabel("sim time (s)")
        ax2.set_ylabel("EE error (mm)")
        ax2.set_title("Convergence (MuJoCo closed loop)")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(133)
        for j in range(7):
            ax3.plot(t_axis[1:], cmd_log[:, j], label=f"j{j+1}", lw=0.8)
        ax3.set_xlabel("sim time (s)")
        ax3.set_ylabel("commanded accel (rad/s²)")
        ax3.set_title("Controller output")
        ax3.legend(fontsize=7, ncol=2, loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(savepath, "mujoco_reach_demo.png")
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"Saved plot: {out}")

    env.close()
    return err_final


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--savepath", default="docs")
    args = p.parse_args()
    run(device=args.device, savepath=args.savepath)
