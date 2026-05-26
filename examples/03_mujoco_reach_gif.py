"""Render the Day-2 closed-loop reach as a GIF/MP4.

Same controller and tuning as examples/02_mujoco_reach.py — we just
capture frames during the run and encode them.
"""

from __future__ import annotations
import os

# Set the rendering backend before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import torch
import imageio.v3 as iio
from tqdm import tqdm

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


def main(device="cpu", out_gif: str = "docs/reach.gif", out_mp4: str | None = "docs/reach.mp4"):
    dtype = torch.float32

    target_pos = np.array([0.5, 0.3, 0.5])

    # Env with target marker for visualization
    env = MujocoFrankaEnv(
        control_dt=0.02,
        render_size=(360, 540),
        target_marker=target_pos,
    )

    # Same predictive model + cost + controller as 02_mujoco_reach.py
    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=torch.from_numpy(target_pos).to(device=device, dtype=dtype),
        w_pos=500.0,
        w_u=0.005,
        w_qdot=0.05,
        terminal_scale=20.0,
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
            horizon=10,
            num_samples=4096,
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
            horizon=10,
            num_samples=4096,
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
    frames = []
    capture_every = 2          # render every 2 ticks → 25 fps from 50 Hz control
    ee_log = [env.ee_position.copy()]

    print("Running closed loop + capturing frames...")
    for t in tqdm(range(n_ticks)):
        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        u = controller.step(x)
        u_np = u.cpu().numpy().astype(np.float64)

        q = s[:7].astype(np.float64)
        qdot = s[7:].astype(np.float64)
        q_target = accel_to_position_target(u_np, q, qdot, env.control_dt)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)

        s = env.step(q_target)
        ee_log.append(env.ee_position.copy())

        if t % capture_every == 0:
            frames.append(env.render())

    # Final stats
    ee_log = np.array(ee_log)
    err = np.linalg.norm(ee_log[-1] - target_pos)
    print(f"Final EE error: {err*1000:.2f} mm")
    print(f"Captured {len(frames)} frames at ({frames[0].shape[1]}x{frames[0].shape[0]})")

    # Encode
    fps = 50 // capture_every
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    print(f"Writing {out_gif} ...")
    iio.imwrite(out_gif, np.stack(frames), duration=1000 // fps, loop=0)
    print(f"Wrote {out_gif} ({os.path.getsize(out_gif) / 1024:.0f} KB)")

    if out_mp4:
        print(f"Writing {out_mp4} ...")
        iio.imwrite(out_mp4, np.stack(frames), fps=fps, codec="libx264")
        print(f"Wrote {out_mp4} ({os.path.getsize(out_mp4) / 1024:.0f} KB)")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--controller", default="cuda", choices=["torch", "cuda"])
    p.add_argument("--gif", default="docs/reach.gif")
    p.add_argument("--mp4", default="docs/reach.mp4")
    args = p.parse_args()
    main(device=args.device, out_gif=args.gif, out_mp4=args.mp4)
