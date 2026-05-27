"""Collect a replay buffer for IVL training.

Runs kernel-MPPI in MuJoCo against a stream of randomized parametric
trajectories (circles, figure-eights, waypoint loops with randomized
centres, radii, periods, and planes). Saves the result as a single NPZ.

The collector uses CudaMPPIController by default — collecting the same
volume of data with the PyTorch backend would take ~30× longer, so the
kernel is doing real work here. The script will refuse to start if the
CUDA extension isn't built and you didn't pass --backend pytorch.

NPZ layout (N = number of episodes, T = ticks per episode):
    states        (N, T+1, 14)   q + qdot at every step (incl. initial+final)
    actions       (N, T,   7)    commanded joint accelerations
    targets       (N, T+1, 3)    target EE position at every step
    ee_positions  (N, T+1, 3)    achieved EE position at every step (from env)
    rewards       (N, T)         per-tick reward (against the true current target)
    + scalar metadata: ticks_per_episode, horizon, alpha_u, alpha_du, dt

The (states, actions, targets, ee_positions) arrays are sufficient to
recompute rewards under arbitrary goal relabeling during training.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from mppi_cuda import (
    DoubleIntegratorArm, ReachingCost, MujocoFrankaEnv,
    MPPIController,
    FRANKA_HOME_Q, FRANKA_Q_MIN, FRANKA_Q_MAX,
    trajectories as traj,
)
from mppi_cuda.rewards import tracking_reward


def _build_controller(args, env, predictive, cost):
    if args.backend == "cuda_kernel":
        from mppi_cuda import CudaMPPIController
        if CudaMPPIController is None:
            raise RuntimeError(
                "CUDA kernel not built. Either build it with `pip install -e .` "
                "in a CUDA environment, or pass --backend pytorch (much slower)."
            )
        return CudaMPPIController(
            dynamics=predictive, cost=cost,
            action_dim=7, horizon=args.horizon, num_samples=args.samples,
            sigma=args.sigma, temperature=1.0,
            u_min=-20.0, u_max=20.0,
            device="cuda", dtype=torch.float32, seed=args.seed,
        )
    return MPPIController(
        dynamics=predictive.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7, horizon=args.horizon, num_samples=args.samples,
        sigma=args.sigma, temperature=1.0,
        u_min=-20.0, u_max=20.0,
        device=args.device, dtype=torch.float32, seed=args.seed,
    )


def collect_episode(
    env, controller, cost, traj_fn,
    n_ticks: int, horizon: int,
    alpha_u: float, alpha_du: float,
    device: str,
) -> dict:
    """Run one episode end-to-end and return per-tick arrays.

    The controller is reset() at the start so warm-starts don't leak between
    episodes. The marker is updated each tick so headless renders look right.
    """
    controller.reset()
    s = env.reset()

    states       = [s.copy().astype(np.float32)]
    actions      = []
    targets      = [traj_fn(0.0).astype(np.float32)]
    ee_positions = [env.ee_position.copy().astype(np.float32)]
    rewards      = []

    u_prev = None
    dtype = torch.float32

    for tick in range(n_ticks):
        t_now = tick * env.control_dt

        # Slide the lookahead window; cost reads target_traj[t] inside the kernel.
        target_buf = traj.build_target_traj(
            traj_fn, t_now=t_now, dt=env.control_dt, horizon=horizon,
        )
        cost.set_target_traj(torch.from_numpy(target_buf).to(device=device, dtype=dtype))
        env.data.mocap_pos[0] = target_buf[0]  # visual marker tracking

        # Plan + bridge to position target
        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        u = controller.step(x)
        u_np = u.cpu().numpy().astype(np.float64)

        q = s[:7].astype(np.float64)
        qdot = s[7:].astype(np.float64)
        q_target = q + 5 * env.control_dt * (qdot + env.control_dt * u_np)
        q_target = np.clip(q_target, FRANKA_Q_MIN, FRANKA_Q_MAX)

        # Physics step
        s = env.step(q_target)
        ee_now = env.ee_position.copy()
        target_now = target_buf[0]  # the "true" current target

        r = tracking_reward(
            ee_pos=ee_now, action=u_np, target=target_now,
            action_prev=u_prev, alpha_u=alpha_u, alpha_du=alpha_du,
        )

        states.append(s.copy().astype(np.float32))
        actions.append(u_np.astype(np.float32))
        ee_positions.append(ee_now.astype(np.float32))
        targets.append(traj_fn((tick + 1) * env.control_dt).astype(np.float32))
        rewards.append(np.float32(r))
        u_prev = u_np

    return {
        "states":       np.array(states),
        "actions":      np.array(actions),
        "targets":      np.array(targets),
        "ee_positions": np.array(ee_positions),
        "rewards":      np.array(rewards),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes",        type=int,   default=500)
    p.add_argument("--ticks-per-episode", type=int,   default=300)
    p.add_argument("--horizon",           type=int,   default=10,
                   help="MPPI horizon. Matches what tracking demos use post-tuning.")
    p.add_argument("--samples",           type=int,   default=4096,
                   help="K rollouts per tick. 4096 is the comfortable default.")
    p.add_argument("--sigma",             type=float, default=2.5)
    p.add_argument("--alpha-u",           type=float, default=0.005)
    p.add_argument("--alpha-du",          type=float, default=0.01)
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--backend",           choices=["cuda_kernel", "pytorch"],
                   default="cuda_kernel")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--out",               default="data/ivl_buffer.npz")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the env once. The marker geom is injected at construction time, so
    # we seed it with a placeholder; we update it via mocap_pos every tick.
    placeholder_pos = traj.Circle()(0.0)
    env = MujocoFrankaEnv(control_dt=0.02, target_marker=placeholder_pos)

    predictive = DoubleIntegratorArm(dt=env.control_dt,
                                     device=args.device if args.backend == "cuda_kernel"
                                            else args.device,
                                     dtype=torch.float32)

    # Placeholder cost — we overwrite target_traj every tick anyway. The cost
    # weights don't affect data; they're just the controller's planning prior.
    placeholder_traj = traj.build_target_traj(
        traj.Circle(), t_now=0.0, dt=env.control_dt, horizon=args.horizon,
    )
    cost = ReachingCost(
        target_pos=placeholder_traj,
        w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX,
        device=args.device if args.backend == "cuda_kernel" else args.device,
        dtype=torch.float32,
    )

    controller = _build_controller(args, env, predictive, cost)
    device_for_step = "cuda" if args.backend == "cuda_kernel" else args.device

    print(f"Collecting {args.n_episodes} episodes × {args.ticks_per_episode} ticks")
    print(f"  backend={args.backend}  K={args.samples}  H={args.horizon}")
    print(f"  total transitions: {args.n_episodes * args.ticks_per_episode:,}")
    print(f"  output: {out_path}")

    all_eps = []
    t_start = time.time()
    failed = 0
    for ep in tqdm(range(args.n_episodes)):
        traj_fn = traj.sample_random_trajectory(rng)
        try:
            ep_data = collect_episode(
                env=env, controller=controller, cost=cost,
                traj_fn=traj_fn,
                n_ticks=args.ticks_per_episode, horizon=args.horizon,
                alpha_u=args.alpha_u, alpha_du=args.alpha_du,
                device=device_for_step,
            )
        except Exception as e:
            failed += 1
            print(f"  ep {ep+1}: failed — {type(e).__name__}: {e}")
            continue
        all_eps.append(ep_data)

        if (ep + 1) % 25 == 0 or ep == 0:
            elapsed = time.time() - t_start
            ep_rate = (ep + 1) / max(elapsed, 1e-6)
            eta = (args.n_episodes - ep - 1) / max(ep_rate, 1e-6)
            mean_r = float(np.mean([float(e["rewards"].mean()) for e in all_eps]))
            print(f"  ep {ep+1:4d}/{args.n_episodes}  "
                  f"rate={ep_rate:5.2f} ep/s  eta={eta/60:5.1f} min  "
                  f"mean_r={mean_r:+.4f}")

    n_kept = len(all_eps)
    if n_kept == 0:
        raise RuntimeError("no episodes collected")

    states       = np.stack([e["states"]       for e in all_eps])
    actions      = np.stack([e["actions"]      for e in all_eps])
    targets      = np.stack([e["targets"]      for e in all_eps])
    ee_positions = np.stack([e["ee_positions"] for e in all_eps])
    rewards      = np.stack([e["rewards"]      for e in all_eps])

    np.savez(
        out_path,
        states=states, actions=actions, targets=targets,
        ee_positions=ee_positions, rewards=rewards,
        # Scalar metadata stored as 0-d arrays.
        ticks_per_episode=np.int64(args.ticks_per_episode),
        horizon=np.int64(args.horizon),
        alpha_u=np.float32(args.alpha_u),
        alpha_du=np.float32(args.alpha_du),
        dt=np.float32(env.control_dt),
    )

    total_transitions = states.shape[0] * actions.shape[1]
    elapsed = time.time() - t_start
    track_rms_mm = float(np.sqrt(np.mean((targets - ee_positions) ** 2).sum() * 1000.0))
    # Better RMS estimator: per-tick ||ee - target||, then RMS over all of those.
    per_tick_err = np.linalg.norm(ee_positions - targets, axis=-1)  # (N, T+1)
    rms_mm = float(np.sqrt(np.mean(per_tick_err ** 2)) * 1000.0)

    print()
    print(f"Done in {elapsed/60:.1f} min ({elapsed:.1f} s).")
    print(f"Kept:           {n_kept:>5} / {args.n_episodes} episodes  ({failed} failed)")
    print(f"Transitions:    {total_transitions:,}")
    print(f"Mean reward/tick: {rewards.mean():+.4f}")
    print(f"Tracking RMS:   {rms_mm:.2f} mm  (across all ticks of all episodes)")
    print(f"NPZ size:       {out_path.stat().st_size / (1024**2):.1f} MB")
    print(f"Saved to:       {out_path}")

    env.close()


if __name__ == "__main__":
    main()
