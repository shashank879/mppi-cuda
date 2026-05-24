"""Per-component timing breakdown for the closed-loop reach demo.

Run this when the closed loop seems too slow. It separates:
  - controller.step()    (the MPPI rollout — the thing the kernel will fix)
  - env.step()           (MuJoCo physics)
  - env.render()         (frame capture)
  - torch device transfer overhead

and reports averages over a warm-started loop (so CUDA init / first-tick
costs don't dominate the numbers).
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import time
import numpy as np
import torch

from mppi_cuda import (
    MPPIController, DoubleIntegratorArm, ReachingCost, MujocoFrankaEnv,
    FRANKA_HOME_Q, FRANKA_Q_MIN, FRANKA_Q_MAX,
)


def bench(device: str = "cpu", n_warmup: int = 5, n_measure: int = 50, do_render: bool = True):
    dtype = torch.float32

    target_pos = np.array([0.5, 0.3, 0.5])
    env = MujocoFrankaEnv(
        control_dt=0.02,
        render_size=(360, 540),
        target_marker=target_pos if do_render else None,
    )

    predictive = DoubleIntegratorArm(dt=env.control_dt, device=device, dtype=dtype)
    cost = ReachingCost(
        target_pos=torch.from_numpy(target_pos).to(device=device, dtype=dtype),
        w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX,
        device=device, dtype=dtype,
    )
    controller = MPPIController(
        dynamics=predictive.step, running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7, horizon=40, num_samples=1024,
        sigma=2.5, temperature=1.0, u_min=-20.0, u_max=20.0,
        device=device, dtype=dtype, seed=0,
    )

    s = env.reset()

    # Warmup — important for CUDA init not to skew measurements.
    for _ in range(n_warmup):
        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        u = controller.step(x)
        s = env.step(s[:7] + 0.001 * np.random.randn(7))
        if do_render:
            _ = env.render()
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    t_to_dev = []
    t_ctrl = []
    t_to_host = []
    t_envstep = []
    t_render = []
    for _ in range(n_measure):
        # transfer state to device
        t0 = time.perf_counter()
        x = torch.from_numpy(s).to(device=device, dtype=dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        t_to_dev.append(time.perf_counter() - t0)

        # controller
        t0 = time.perf_counter()
        u = controller.step(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t_ctrl.append(time.perf_counter() - t0)

        # transfer action to host
        t0 = time.perf_counter()
        u_np = u.cpu().numpy().astype(np.float64)
        t_to_host.append(time.perf_counter() - t0)

        # env step (we use a slightly different action so the env evolves)
        target = s[:7] + 0.001 * np.random.randn(7)
        t0 = time.perf_counter()
        s = env.step(target)
        t_envstep.append(time.perf_counter() - t0)

        # render
        if do_render:
            t0 = time.perf_counter()
            _ = env.render()
            t_render.append(time.perf_counter() - t0)

    def stats(name, arr):
        if not arr:
            return
        a = np.array(arr) * 1000  # ms
        print(f"  {name:25s} mean {a.mean():7.2f} ms   "
              f"median {np.median(a):7.2f} ms   "
              f"min {a.min():7.2f} ms   max {a.max():7.2f} ms")

    print(f"Device:           {device}")
    print(f"Torch:            {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device:      {torch.cuda.get_device_name()}")
        print(f"CUDA compute:     {torch.cuda.get_device_capability()}")
    print(f"MuJoCo GL:        {os.environ.get('MUJOCO_GL', '(unset)')}")
    print(f"Threads:          torch.get_num_threads()={torch.get_num_threads()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"K, H, action_dim: {controller.K}, {controller.H}, {controller.m}")
    print(f"Render:           {'on' if do_render else 'off'}  "
          f"({env._render_size if do_render else '-'})")
    print(f"Measured over {n_measure} steps (after {n_warmup} warmup):")
    stats("host -> device", t_to_dev)
    stats("controller.step", t_ctrl)
    stats("device -> host", t_to_host)
    stats("env.step", t_envstep)
    if do_render:
        stats("env.render",  t_render)

    total = (np.mean(t_to_dev) + np.mean(t_ctrl) + np.mean(t_to_host)
             + np.mean(t_envstep) + (np.mean(t_render) if do_render else 0))
    print(f"  {'TOTAL':25s} {total * 1000:7.2f} ms / step "
          f"({1.0 / total:.1f} Hz)")

    env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--n-measure", type=int, default=50)
    args = p.parse_args()
    bench(device=args.device, do_render=not args.no_render, n_measure=args.n_measure)
