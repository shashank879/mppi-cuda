"""Latency benchmark for MPPI rollouts across backends.

Backends are registered through a `Backend` protocol so the CUDA kernel
plugs in without changing the harness:

    cpu_pytorch:  current MPPIController on CPU
    cuda_pytorch: current MPPIController on GPU (no custom kernel yet)
    cuda_kernel:  our fused kernel (added in Day 4)

Output: one CSV row per (backend, K, H, repeat). Stored under
`benchmarks/results/latency_<timestamp>.csv`. The README perf table
is generated from this CSV.

Usage:
    python benchmarks/bench_latency.py --backends cpu_pytorch
    python benchmarks/bench_latency.py --backends cpu_pytorch cuda_pytorch \\
        --K 1024 4096 --H 40 80 --warmup 5 --measure 50
"""

from __future__ import annotations
import argparse
import csv
import datetime
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mppi_cuda import (
    MPPIController, DoubleIntegratorArm, ReachingCost,
    FRANKA_HOME_Q, FRANKA_Q_MIN, FRANKA_Q_MAX,
)


# ---------- Backend protocol ----------

class Backend:
    """A callable that runs one MPPI tick and returns its latency in ms."""
    name: str
    def setup(self, K: int, H: int, device: str): raise NotImplementedError
    def tick(self) -> float: raise NotImplementedError
    def teardown(self): pass


class PyTorchBackend(Backend):
    """Current MPPIController on a chosen device."""
    def __init__(self, device: str):
        self.name = f"pytorch_{device}"
        self.device = device

    def setup(self, K: int, H: int, *, dtype=torch.float32):
        d, dt = self.device, dtype
        self.predictive = DoubleIntegratorArm(dt=0.02, device=d, dtype=dt)
        self.cost = ReachingCost(
            target_pos=[0.5, 0.3, 0.5],
            w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
            q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX, device=d, dtype=dt,
        )
        self.ctrl = MPPIController(
            dynamics=self.predictive.step,
            running_cost=self.cost.running_cost,
            terminal_cost=self.cost.terminal_cost,
            action_dim=7, horizon=H, num_samples=K,
            sigma=2.5, temperature=1.0,
            u_min=-20.0, u_max=20.0, device=d, dtype=dt, seed=0,
        )
        self.x = torch.cat([
            torch.tensor(FRANKA_HOME_Q, device=d, dtype=dt),
            torch.zeros(7, device=d, dtype=dt),
        ])
        self._is_cuda = d.startswith("cuda")

    def tick(self) -> float:
        if self._is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = self.ctrl.step(self.x)
        if self._is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


# Hook for the CUDA kernel — registered once we have it.
# class CudaKernelBackend(Backend):
#     name = "cuda_kernel"
#     ...


BACKENDS: dict[str, Callable[[], Backend]] = {
    "cpu_pytorch": lambda: PyTorchBackend("cpu"),
    "cuda_pytorch": lambda: PyTorchBackend("cuda"),
}


# ---------- Harness ----------

def run_one(backend: Backend, K: int, H: int, warmup: int, measure: int) -> dict:
    backend.setup(K, H)
    for _ in range(warmup):
        backend.tick()
    samples = np.array([backend.tick() for _ in range(measure)])
    backend.teardown()
    return {
        "backend":  backend.name,
        "K":        K,
        "H":        H,
        "mean_ms":  float(samples.mean()),
        "median_ms":float(np.median(samples)),
        "p99_ms":   float(np.percentile(samples, 99)),
        "min_ms":   float(samples.min()),
        "max_ms":   float(samples.max()),
        "n":        len(samples),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+",
                   default=["cuda_pytorch"],
                   choices=list(BACKENDS.keys()))
    p.add_argument("--K", nargs="+", type=int, default=[256,1024,4096])
    p.add_argument("--H", nargs="+", type=int, default=[40])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=20)
    p.add_argument("--out", type=str,
                   default=f"benchmarks/results/latency_"
                           f"{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"host: {platform.node()}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:  {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'backend':<15} {'K':>6} {'H':>4} | "
          f"{'mean':>8} {'med':>8} {'p99':>8}  (ms)")
    print("-" * 60)

    rows = []
    for name in args.backends:
        backend = BACKENDS[name]()
        for K in args.K:
            for H in args.H:
                row = run_one(backend, K, H, args.warmup, args.measure)
                rows.append(row)
                print(f"{row['backend']:<15} {K:>6} {H:>4} | "
                      f"{row['mean_ms']:>8.2f} {row['median_ms']:>8.2f} "
                      f"{row['p99_ms']:>8.2f}")

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
