"""Day 1 demo: PyTorch MPPI controlling a 7-DoF arm to reach a target.

Plant and controller's predictive model are both the same double-integrator
here — no sim-to-controller gap. That's intentional for the baseline. Day 2
introduces MuJoCo as the plant and keeps the double-integrator as the
controller's internal model.
"""

import os, time
import torch

from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    forward_kinematics,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)


def run_demo(device: str = "cpu", savepath=None):
    dtype = torch.float32

    # --- Plant (true dynamics) ---
    dt = 0.02
    plant = DoubleIntegratorArm(dt=dt, device=device, dtype=dtype)

    # --- Cost ---
    target_pos = torch.tensor([0.5, 0.3, 0.5], device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=target_pos,
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        device=device,
        dtype=dtype,
    )

    # --- Controller ---
    controller = MPPIController(
        dynamics=plant.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7,
        horizon=24,
        num_samples=1024,
        sigma=2.5,         # acceleration noise std (rad/s^2)
        temperature=1.0,
        u_min=-10.0,       # accel limits (rad/s^2)
        u_max=10.0,
        device=device,
        dtype=dtype,
        seed=0,
    )

    # --- Initial state: home ---
    q = torch.tensor(FRANKA_HOME_Q, device=device, dtype=dtype)
    qdot = torch.zeros(7, device=device, dtype=dtype)
    x = torch.cat([q, qdot])

    n_steps = 200
    state_log = [x.clone().cpu().numpy()]
    action_log = []
    tick_times = []

    for t in range(n_steps):
        t0 = time.perf_counter()
        u = controller.step(x)
        tick_times.append(time.perf_counter() - t0)

        x = plant.step(x, u)
        state_log.append(x.clone().cpu().numpy())
        action_log.append(u.clone().cpu().numpy())

    # --- Report ---
    q_final = x[:7].unsqueeze(0)
    ee_final, _ = forward_kinematics(q_final)
    ee_final = ee_final.squeeze(0)
    err = (ee_final - target_pos).norm().item()

    import numpy as np
    tick_times_ms = np.array(tick_times) * 1000

    print(f"Device:               {device}")
    print(f"K (samples), H (horizon): {controller.K}, {controller.H}")
    print(f"Ran {n_steps} control ticks")
    print(f"Per-tick latency: mean {tick_times_ms.mean():.2f} ms, "
          f"median {np.median(tick_times_ms):.2f} ms, "
          f"p99 {np.percentile(tick_times_ms, 99):.2f} ms")
    print(f"Target EE position:  {target_pos.cpu().numpy()}")
    print(f"Final  EE position:  {ee_final.cpu().numpy()}")
    print(f"Final  position err: {err*1000:.2f} mm")

    if savepath:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            state_log_np = np.array(state_log)
            q_traj = torch.from_numpy(state_log_np[:, :7]).to(device)
            ee_traj, _ = forward_kinematics(q_traj)
            ee_traj = ee_traj.cpu().numpy()

            fig = plt.figure(figsize=(12, 4.5))

            ax1 = fig.add_subplot(131, projection="3d")
            ax1.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], lw=2)
            ax1.scatter(*ee_traj[0], color="green", s=80, label="start")
            ax1.scatter(*target_pos.cpu().numpy(), color="red", s=80, label="target")
            ax1.scatter(*ee_traj[-1], color="blue", s=80, label="final")
            ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
            ax1.set_title("End-effector trajectory")
            ax1.legend(fontsize=8)

            ax2 = fig.add_subplot(132)
            time_axis = np.arange(len(ee_traj)) * dt
            err_traj = np.linalg.norm(ee_traj - target_pos.cpu().numpy(), axis=1)
            ax2.plot(time_axis, err_traj * 1000)
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("EE position error (mm)")
            ax2.set_title("Convergence")
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(133)
            ax3.plot(tick_times_ms)
            ax3.set_xlabel("tick")
            ax3.set_ylabel("controller latency (ms)")
            ax3.set_title(f"MPPI latency (K={controller.K}, H={controller.H})")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            out = os.path.join(savepath, "reach_demo.png")
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(out, dpi=120)
            print(f"Saved plot: {out}")
        except ImportError:
            print("matplotlib not available; skipping plot")

    return err


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--savepath", default=None)
    args = p.parse_args()
    run_demo(device=args.device, savepath=args.savepath)
