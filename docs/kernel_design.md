# Kernel design: fused MPPI rollout for 7-DoF manipulation

## Goal

Replace the inner loop of `MPPIController.step()` — the `K × H`-step
rollout that accumulates per-rollout cost — with a single fused CUDA
kernel. Inputs and outputs match the existing PyTorch baseline so the
controller's Python surface is unchanged.

## Workload characterisation

Reference config used in Day 2 / 3:

| Quantity        | Value | Notes                                      |
| --------------- | ----- | ------------------------------------------ |
| K (rollouts)    | 1024  | scales to ~16k once we have headroom       |
| H (horizon)     | 40    | scales to 80–100 once we have headroom     |
| N (state dim)   | 14    | q (7) + qdot (7)                           |
| M (action dim)  | 7     |                                            |
| dt              | 0.02  | predictive model timestep                  |
| n_joints        | 7     | for FK chain inside cost                   |

The two axes of parallelism are very asymmetric: **K is embarrassingly
parallel** (each rollout is independent), **H is strictly sequential**
(state at t depends on state at t-1). Everything below follows from that.

## Parallelism strategy

**One thread per rollout.** Each thread holds its own state in registers,
loops `H` times, accumulates a single fp32 cost into a register, then
writes one fp32 out to global at the end. No cross-thread synchronisation
required.

Why not one block per rollout (threads cooperate on each timestep)?

- The work per timestep is tiny: a double-integrator update plus an FK
  chain plus a few cost terms. Cooperation overhead (`__syncthreads`,
  shared-memory write/read) would dominate.
- The serial dependency between timesteps means we'd need a `__syncthreads`
  every step anyway. That's `H` barriers per rollout — death.
- One thread per rollout means every thread reads sequentially through
  its noise stripe — coalesced, cache-friendly access.

The thread block layout is then just `block(256) × grid(ceil(K/256))`.
256 is a safe default for occupancy; we'll autotune.

## Register budget

Per-thread register usage estimate (NVCC will inline aggressively, so
these are upper-bound; actual usage shown after first compile):

| Item                     | fp32 regs |
| ------------------------ | --------- |
| state x[14]              | 14        |
| local u[7]               | 7         |
| FK accumulator T[16]     | 16        |
| FK per-step T_i[16]      | 16        |
| cost accumulator         | 1         |
| ee_pos[3] for cost       | 3         |
| target_pos[3] (constant) | 3 (const) |
| loop indices, scratch    | ~10       |
| **Total**                | **~70**   |

Ampere/Hopper SMs have 65536 32-bit registers each. At 70 regs/thread
that allows 936 threads per SM; with 256-thread blocks, 3 blocks per SM,
giving 50% occupancy. That's fine for memory-light, compute-light work
like this.

**If actual usage exceeds 128 regs we'll get spills.** Two mitigations
on the table:

1. Store the FK transform stack in shared memory instead of registers
   (4 KB/thread × 256 threads = 1 MB; doesn't fit, so we'd shrink the
   block). Not preferred.
2. Recompute `cos(θ)` and `sin(θ)` on demand instead of caching them per
   joint. Costs 14 transcendentals per timestep but frees registers.
   Preferred fallback.

## Memory layout

All tensors stay in fp32, contiguous, C-major (matching PyTorch's
default). The kernel signature is:

```cuda
__global__ void mppi_rollout_kernel(
    const float* __restrict__ x0,         // (N,)      current state
    const float* __restrict__ U_nominal,  // (H, M)    nominal control sequence
    const float* __restrict__ noise,      // (K, H, M) sampled perturbations
    const float* __restrict__ target_pos, // (3,)      goal in task space
    const float* __restrict__ q_min,      // (M,)      joint limits
    const float* __restrict__ q_max,      // (M,)      joint limits
    const float* __restrict__ qdot_max,   // (M,)      vel limit (for clamp)
    const float* __restrict__ u_min,      // (M,)      ctrl clip
    const float* __restrict__ u_max,      // (M,)      ctrl clip
    float* __restrict__ costs_out,        // (K,)      output
    int K, int H,
    float dt,
    float w_pos, float w_u, float w_qdot, float w_lim, float terminal_scale)
```

Per-tick memory traffic per thread (K=1024, H=40, M=7):

| Op                | Bytes/thread/tick | Notes                            |
| ----------------- | ----------------- | -------------------------------- |
| Load x0           | 56                | (N=14 floats), broadcast         |
| Load U_nominal    | 1120              | (H*M=280 floats), broadcast      |
| Load noise        | 1120              | (H*M=280 floats), unique         |
| Store cost        | 4                 |                                  |
| **Per thread**    | **~2.3 KB**       |                                  |
| **Total (K=1024)**| **~2.3 MB / tick**|                                  |

L2 on A100/H100 is 40–60 MB, so all of `x0`, `U_nominal`, `target_pos`,
and the joint limits fit in cache for the entire grid. Only `noise` is
unique per thread; it's read once sequentially, fully coalesced (thread
`k` reads `noise[k, t, :]` for each `t`).

This kernel is comfortably bandwidth-bound at the noise read, not
compute-bound. Estimated time: K=1024 * 2.3KB / 1.5 TB/s ≈ 1.5 µs of
memory traffic per launch. Compute will be longer per the FLOP estimate
below, but well within real-time targets.

## Compute budget per rollout

For each of H=40 timesteps:

- Dynamics step: `qdot_new = clamp(qdot + dt*u); q_new = q + dt*qdot_new`
  → ~30 FLOPs.
- FK chain: 7 transforms × ~80 FLOPs per 4×4 matmul ≈ 560 FLOPs.
- Cost terms: `||ee - target||²` (6 FLOPs) + `||u||²` (14 FLOPs) +
  `||qdot||²` (14 FLOPs) + barriers (~28 FLOPs) ≈ 60 FLOPs.

Total: **~650 FLOPs per timestep**, **~26000 FLOPs per rollout**,
**~27 MFLOPs per K=1024 launch.**

At A100's 19.5 TFLOPS fp32 peak, that's `27M / 19.5T ≈ 1.4 µs` of pure
math. Memory-bound, not compute-bound.

## Numerics

- **Dtype**: fp32 throughout. The PyTorch baseline uses fp32; we want to
  match exactly for correctness comparison, then explore fp16/bf16 for
  speed in a future iteration.
- **Reduction (softmax of costs)**: lives in a second small kernel,
  `mppi_weighting_kernel`. Standard log-sum-exp pattern with min
  subtraction for stability. K=1024 → trivially fits in one block.
- **Determinism**: the rollout kernel is fully deterministic given the
  same noise tensor. We generate noise on the host (or in a separate
  kernel) with a fixed seed, then both PyTorch baseline and our CUDA
  kernel consume the same `noise` buffer. This makes correctness checks
  bit-exact.

## Phased implementation

Three milestones, each independently runnable + testable against
the PyTorch baseline:

1. **v0 — naive correctness.** One thread per rollout, all logic
   inlined, no shared memory, no FK optimisation, no templating. The
   only requirement is bit-exact agreement with the PyTorch baseline
   (to within fp32 numerical tolerance). Should already be 10–50×
   faster than CPU PyTorch.

2. **v1 — register-tuned, FK as unrolled chain.** Eliminate redundant
   sin/cos, unroll the FK loop manually. Target: ≤ 80 regs/thread.

3. **v2 — templated on H, M, N.** Compile-time constants enable full
   inlining and loop unrolling. Pre-instantiated for our (H=40, 80, 100)
   × (N=14) menu. This is what ships.

## Open questions (deferred)

- **Atomics vs separate weighting kernel.** Could fuse the min-reduction
  into the rollout kernel via shared-memory atomics. Probably not worth
  it — the cost reduction is < 1 µs already.
- **fp16/bf16.** TF32 by default on Ampere matmul-like work, but our
  workload doesn't hit Tensor Cores. fp16 weights with fp32 accumulate
  would save bandwidth on noise reads. Defer.
- **MLP dynamics variant.** Templated on `DynamicsT`. The MLP variant
  stores weights in `__constant__` memory (one MLP, broadcast to all
  threads) — naturally fits the architecture. Defer to post-submission.
- **Triton port.** Same algorithm, different language. Worth doing if
  time permits — gives a fair Triton-vs-CUDA comparison for the README.

## Acceptance criteria

Before merging the kernel:
- Bit-exact match to PyTorch baseline on a fixed-seed noise tensor
  (`max(|cost_cuda - cost_pytorch|) < 1e-3` over K=1024 rollouts).
- `examples/02_mujoco_reach.py` produces identical final EE position
  to within 1 mm when switched to the CUDA backend.
- `benchmarks/bench_latency.py` records per-tick latency at K ∈ {1k, 4k,
  16k}, H ∈ {40, 80}, on whichever GPU is available.
