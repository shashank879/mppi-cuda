"""Part A: obstacle avoidance with the PyTorch controller (no kernel yet).

Two yellow spheres form a gate between the home pose and the reach target.
The controller must thread the EE through ~9 cm of clear space at the centre.
This demo only exercises the cost formula and the env's marker injection —
Part B adds the kernel side so the same demo runs with CudaMPPIController.

Usage:
    PYTHONPATH=. python examples/05_obstacle_reach.py
    PYTHONPATH=. python examples/05_obstacle_reach.py --no-obstacles  # ablation
"""

from __future__ import annotations
import os, argparse
import time

import numpy as np
import torch

from mppi_cuda import (
    MPPIController,
    CudaMPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    MujocoFrankaEnv,
    forward_kinematics,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)


def accel_to_position_target(u_accel, q, qdot, dt, lookahead: int = 5):
    qdot_target = qdot + dt * u_accel
    return q + lookahead * dt * qdot_target


# Two-sphere gate, placed roughly on the straight-line midpoint between
# home pose (0.307, 0, 0.59) and target (0.5, 0.3, 0.5). Spheres are at
# y=0.05 and y=0.25 so the gate centre is at y=0.15 with ~9 cm clear after
# inflation by the safety margin.
DEFAULT_OBSTACLES = [
    (0.40, 0.05, 0.55, 0.06),
    (0.40, 0.25, 0.55, 0.06),
]


def run(device: str = "cpu", with_obstacles: bool = True, savepath=None):
    dtype = torch.float32

    target_pos_np = np.array([0.5, 0.3, 0.5])
    obstacles = DEFAULT_OBSTACLES if with_obstacles else []

    env = MujocoFrankaEnv(
        control_dt=0.02,
        target_marker=target_pos_np,
        obstacle_markers=np.asarray(obstacles) if obstacles else None,
    )

    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)

    target_pos = torch.from_numpy(target_pos_np).to(device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=target_pos,
        w_pos=500.0,
        w_u=0.005,
        w_qdot=0.05,
        terminal_scale=20.0,
        obstacles=obstacles,
        w_obs=1000.0,
        obs_margin=0.05,
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        device=device,
        dtype=dtype,
    )

    if args.controller == "torch":
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
    elif args.controller == "cuda":
        controller = CudaMPPIController(
            dynamics=predictive,
            cost=cost,
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
    else:
        raise NotImplementedError(args.controller)

    s = env.reset()
    n_ticks = 200
    ee_log = [env.ee_position.copy()]
    tick_times = []

    for _ in range(n_ticks):
        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        t0 = time.perf_counter()
        u = controller.step(x)
        tick_times.append(time.perf_counter() - t0)
        u_np = u.cpu().numpy().astype(np.float64)

        q = s[:7].astype(np.float64)
        qdot = s[7:].astype(np.float64)
        q_target = accel_to_position_target(u_np, q, qdot, env.control_dt)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)
        s = env.step(q_target)
        ee_log.append(env.ee_position.copy())

    ee_log = np.array(ee_log)
    err = np.linalg.norm(ee_log[-1] - target_pos_np) * 1000
    print(f"Obstacles:           {'2 spheres' if with_obstacles else 'none'}")
    print(f"Final EE position:   {ee_log[-1]}")
    print(f"Final position err:  {err:.2f} mm")

    # Minimum-clearance audit: closest EE-to-obstacle distance along the run.
    if obstacles:
        obs = np.asarray(obstacles)
        # distances: (T, N_obs)
        d = np.linalg.norm(ee_log[:, None, :] - obs[None, :, :3], axis=-1)
        per_obs_min = d.min(axis=0) - obs[:, 3]   # min clearance per obstacle
        for i, (clear_, r) in enumerate(zip(per_obs_min, obs[:, 3])):
            print(f"  obs {i}: r={r*100:.1f} cm, min EE clearance = {clear_*1000:+.1f} mm")

    inside_any = None
    if obstacles:
        obs = np.asarray(obstacles)
        d = np.linalg.norm(ee_log[:, None, :] - obs[None, :, :3], axis=-1)  # (T, N_obs)
        per_obs_min = d.min(axis=0) - obs[:, 3]
        inside_any = (d < obs[None, :, 3]).any(axis=-1)                      # (T,)
        for i, (clear_, r) in enumerate(zip(per_obs_min, obs[:, 3])):
            print(f"  obs {i}: r={r*100:.1f} cm, min EE clearance = {clear_*1000:+.1f} mm")
        if inside_any.any():
            print(f"  EE inside an obstacle for {int(inside_any.sum())}/{len(ee_log)} ticks  (BAD)")

    if savepath:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.lines import Line2D

        if inside_any is None:
            inside_any = np.zeros(len(ee_log), dtype=bool)

        fig = plt.figure(figsize=(12, 4.5))
        ax = fig.add_subplot(121, projection="3d")

        # Per-segment colour: red if either endpoint sat inside a sphere.
        segs = np.stack([ee_log[:-1], ee_log[1:]], axis=1)            # (T-1, 2, 3)
        seg_bad = inside_any[:-1] | inside_any[1:]
        seg_colors = np.where(seg_bad, "red", "tab:blue")
        ax.add_collection(Line3DCollection(segs, colors=seg_colors, linewidths=2))

        ax.scatter(*ee_log[0],    color="green", s=80)
        ax.scatter(*target_pos_np, color="red",   s=80)
        ax.scatter(*ee_log[-1],   color="blue",  s=80)

        # Line3DCollection doesn't autoscale — set limits manually so the
        # trajectory and target are both visible.
        pts = np.vstack([ee_log, [target_pos_np]])
        ax.set_xlim(pts[:, 0].min() - 0.05, pts[:, 0].max() + 0.05)
        ax.set_ylim(pts[:, 1].min() - 0.05, pts[:, 1].max() + 0.05)
        ax.set_zlim(pts[:, 2].min() - 0.05, pts[:, 2].max() + 0.05)

        if obstacles:
            for ox, oy, oz, r in obstacles:
                u_, v_ = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
                sx = ox + r * np.cos(u_) * np.sin(v_)
                sy = oy + r * np.sin(u_) * np.sin(v_)
                sz = oz + r * np.cos(v_)
                ax.plot_surface(sx, sy, sz, color="goldenrod", alpha=0.35)

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        title = "EE trajectory with obstacle gate"
        if inside_any.any():
            title += f"  —  {int(inside_any.sum())} intersecting ticks"
        ax.set_title(title)

        # Build a legend manually — Line3DCollection doesn't make one for us.
        handles = [Line2D([0], [0], color="tab:blue", lw=2, label="EE trajectory")]
        if inside_any.any():
            handles.append(Line2D([0], [0], color="red", lw=2, label="inside obstacle"))
        handles += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="start"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",   markersize=8, label="target"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",  markersize=8, label="final"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

        ax2 = fig.add_subplot(122)
        t_axis = np.arange(len(ee_log)) * env.control_dt
        err_traj = np.linalg.norm(ee_log - target_pos_np, axis=1) * 1000
        ax2.plot(t_axis, err_traj)
        ax2.set_xlabel("sim time (s)"); ax2.set_ylabel("EE error (mm)")
        ax2.set_title("Convergence")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(savepath, f"{args.controller}_obstacle_reach.png" if with_obstacles else f"{args.controller}_obstacle_reach_no_obs.png")
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"Saved plot: {out}")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--controller", default="torch", choices=["torch", "cuda"])
    p.add_argument("--no-obstacles", action="store_true",
                   help="Ablation: run without obstacle cost.")
    p.add_argument("--savepath", default="docs")
    args = p.parse_args()
    run(device=args.device, with_obstacles=not args.no_obstacles, savepath=args.savepath)
