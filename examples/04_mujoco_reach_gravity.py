"""Day 3 ablation: does analytical gravity compensation help?

Spoiler: no — and the reason is instructive enough that we keep this
script as documentation. The MuJoCo Franka uses PD-controlled actuators
(kp=4500, kd=450). The PD already absorbs gravity via a small steady-
state position offset `q_target - q ≈ g(q)/kp`. Adding an analytical
gravity bias to the controller's predictive model causes *double*
compensation: the controller commands a big counter-torque, the bridge
translates that into a big position offset, the PD tracks it, the arm
overshoots. Result: error goes from 39 mm to ~184 mm.

The takeaway for the README: gravity comp is correct under torque
control. In a position-PD framework, the actuator already does it; what
we want next is a model that captures the *residual* physics the simple
double-integrator misses — friction, PD tracking lag, off-axis dynamics.
That's a learned dynamics model, which is what Day 3b adds.

This script runs the loop twice (gravity on, gravity off) and saves a
side-by-side plot for direct comparison.
"""

from __future__ import annotations
import os, argparse

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
from mppi_cuda.gravity import compute_gravity_acceleration


def accel_to_position_target(u_accel, q, qdot, dt, lookahead: int = 5):
    qdot_target = qdot + dt * u_accel
    return q + lookahead * dt * qdot_target


def _single_run(device: str, dtype, use_gravity: bool, n_ticks: int = 200):
    env = MujocoFrankaEnv(control_dt=0.02)
    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)

    target_pos = torch.tensor([0.5, 0.3, 0.5], device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=target_pos,
        w_pos=500.0,
        w_u=0.005,
        w_qdot=0.05,
        terminal_scale=20.0,
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        device=device,
        dtype=dtype,
    )
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

    s = env.reset()
    ee_log = [env.ee_position.copy()]

    for _ in range(n_ticks):
        if use_gravity:
            g_bias = compute_gravity_acceleration(env.model, env.data, s[:7].astype(np.float64))
            predictive.gravity_bias = torch.from_numpy(g_bias).to(device=device, dtype=dtype)
        else:
            predictive.gravity_bias = None

        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        u = controller.step(x).cpu().numpy().astype(np.float64)
        q, qdot = s[:7].astype(np.float64), s[7:].astype(np.float64)
        q_target = accel_to_position_target(u, q, qdot, env.control_dt)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)
        s = env.step(q_target)
        ee_log.append(env.ee_position.copy())

    env.close()
    return np.array(ee_log), target_pos.cpu().numpy()


def run(device: str = "cpu", savepath=None):
    dtype = torch.float32

    print("Run A: gravity comp OFF (Day 2 baseline)")
    ee_off, target = _single_run(device, dtype, use_gravity=False)
    err_off = np.linalg.norm(ee_off[-1] - target) * 1000

    print("Run B: gravity comp ON")
    ee_on, _ = _single_run(device, dtype, use_gravity=True)
    err_on = np.linalg.norm(ee_on[-1]  - target) * 1000

    print()
    print(f"  Final error  (gravity OFF):  {err_off:.1f} mm")
    print(f"  Final error  (gravity ON):   {err_on:.1f} mm")
    print(f"  Diagnosis: position-PD actuator already cancels gravity at the")
    print(f"  hardware level; adding gravity comp on top double-counts.")
    print(f"  Next step: learn the *residual* dynamics with an MLP (Day 3b).")

    if savepath:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        t_axis_off = np.arange(len(ee_off)) * 0.02
        t_axis_on  = np.arange(len(ee_on))  * 0.02
        err_off_traj = np.linalg.norm(ee_off - target, axis=1) * 1000
        err_on_traj  = np.linalg.norm(ee_on  - target, axis=1) * 1000

        ax = axes[0]
        ax.plot(t_axis_off, err_off_traj, label=f"gravity OFF ({err_off:.0f} mm final)")
        ax.plot(t_axis_on,  err_on_traj,  label=f"gravity ON ({err_on:.0f} mm final)")
        ax.set_xlabel("sim time (s)")
        ax.set_ylabel("EE error (mm)")
        ax.set_title("Convergence: gravity comp ablation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax = fig.add_subplot(122, projection="3d")
        axes[1].remove()
        ax.plot(ee_off[:, 0], ee_off[:, 1], ee_off[:, 2], label="gravity OFF", lw=2)
        ax.plot(ee_on[:, 0],  ee_on[:, 1],  ee_on[:, 2],  label="gravity ON", lw=2)
        ax.scatter(*ee_off[0], color="green", s=80, label="start")
        ax.scatter(*target, color="red", s=80, label="target")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Trajectories")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out = os.path.join(savepath, "gravity_ablation.png")
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(out, dpi=120)
        print("Saved plot:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--savepath", default="docs")
    args = p.parse_args()
    run(device=args.device, savepath=args.savepath)
