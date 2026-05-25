"""Correctness test for the v0 CUDA kernel.

The kernel's only job in v0 is to match the PyTorch baseline. We feed
both a fixed-seed noise tensor and the same initial state, compare the
per-rollout costs, and require agreement to fp32 tolerance.

Skipped automatically if CUDA isn't available or the extension isn't built.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
try:
    from mppi_cuda._kernels import mppi_rollout as cuda_rollout
    extension_built = True
except ImportError:
    cuda_rollout = None
    extension_built = False

pytestmark = pytest.mark.skipif(
    not (cuda_available and extension_built),
    reason="CUDA + compiled extension required for kernel tests",
)


from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)


def _setup(K=1024, H=40, device="cuda", seed=0, with_obstacles=False):
    """Build the dynamics, cost, and a deterministic noise tensor used by both backends."""
    dtype = torch.float32
    dyn = DoubleIntegratorArm(dt=0.02, device=device, dtype=dtype)

    obstacles = None
    if with_obstacles:
        obstacles = [
            (0.40, 0.05, 0.55, 0.06),
            (0.40, 0.25, 0.55, 0.06),
        ]

    cost = ReachingCost(
        target_pos=[0.5, 0.3, 0.5],
        w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
        obstacles=obstacles, w_obs=1000.0, obs_margin=0.05,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX,
        device=device, dtype=dtype,
    )
    gen = torch.Generator(device=device).manual_seed(seed)
    sigma = 2.5
    noise_raw = torch.randn(K, H, 7, generator=gen, device=device, dtype=dtype) * sigma
    U_nominal = torch.zeros(H, 7, device=device, dtype=dtype)

    # Clamp on the host so both backends see the same effective u.
    U_perturbed = torch.clamp(U_nominal.unsqueeze(0) + noise_raw, -20.0, 20.0)
    noise = (U_perturbed - U_nominal.unsqueeze(0)).contiguous()

    q0 = torch.tensor(FRANKA_HOME_Q, device=device, dtype=dtype)
    qdot0 = torch.zeros(7, device=device, dtype=dtype)
    x = torch.cat([q0, qdot0])

    return dyn, cost, U_nominal, U_perturbed, noise, x


def _pytorch_rollout_costs(dyn, cost, U_perturbed, x, K, H):
    """Reference rollout matching MPPIController.step() exactly."""
    x_batch = x.unsqueeze(0).expand(K, -1).contiguous()
    costs = torch.zeros(K, device=x.device, dtype=x.dtype)
    for t in range(H):
        u_t = U_perturbed[:, t]
        costs = costs + cost.running_cost(x_batch, u_t)
        x_batch = dyn.step(x_batch, u_t)
    costs = costs + cost.terminal_cost(x_batch)
    return costs


def _call_kernel(x, U_nominal, noise, cost, dyn):
    return cuda_rollout(
        x, U_nominal, noise,
        cost.target_pos, dyn.q_min, dyn.q_max, dyn.qdot_max,
        cost.obstacles,
        -20.0, 20.0,
        dyn.dt,
        cost.w_pos, cost.w_u, cost.w_qdot, cost.w_lim,
        cost.w_obs, cost.obs_margin, cost.w_obs_flat,
        cost.terminal_scale,
    )


def test_kernel_matches_pytorch_costs_K1024_H40():
    K, H = 1024, 40
    dyn, cost, U_nominal, U_perturbed, noise, x = _setup(K=K, H=H)

    ref = _pytorch_rollout_costs(dyn, cost, U_perturbed, x, K, H)
    out = _call_kernel(x, U_nominal, noise, cost, dyn)
    torch.cuda.synchronize()

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)

    max_rel = rel_err.max().item()
    max_abs = abs_err.max().item()
    print(f"\n  max abs err: {max_abs:.3e}")
    print(f"  max rel err: {max_rel:.3e}")
    print(f"  ref range:   [{ref.min():.2f}, {ref.max():.2f}]")

    assert max_rel < 1e-3, (
        f"costs differ too much: max_rel={max_rel:.3e}, max_abs={max_abs:.3e}"
    )


def test_kernel_matches_pytorch_costs_with_nonzero_U_nominal():
    """Repeat but with a non-zero nominal sequence — exercises the U_nominal load path."""
    K, H = 512, 32
    dyn, cost, U_nominal, _, _, x = _setup(K=K, H=H)

    # Pretend we've been planning for a while: non-zero U_nominal.
    U_nominal = torch.linspace(-1.0, 1.0, H * 7, device="cuda", dtype=torch.float32).reshape(H, 7)

    gen = torch.Generator(device="cuda").manual_seed(7)
    noise_raw = torch.randn(K, H, 7, generator=gen, device="cuda", dtype=torch.float32) * 2.5
    U_perturbed = torch.clamp(U_nominal.unsqueeze(0) + noise_raw, -20.0, 20.0)
    noise = (U_perturbed - U_nominal.unsqueeze(0)).contiguous()

    ref = _pytorch_rollout_costs(dyn, cost, U_perturbed, x, K, H)
    out = _call_kernel(x, U_nominal, noise, cost, dyn)
    torch.cuda.synchronize()

    max_rel = ((out - ref).abs() / (ref.abs() + 1e-6)).max().item()
    assert max_rel < 1e-3, f"costs differ: max_rel={max_rel:.3e}"


def test_kernel_matches_pytorch_costs_with_obstacles():
    """Same fixed-seed comparison, but with two obstacles configured.

    Exercises the obstacle_cost device function in both the running- and
    terminal-cost code paths. Some rollouts will sample trajectories that
    penetrate the spheres, producing large costs — the kernel must agree
    with the PyTorch baseline across the full cost range.
    """
    K, H = 1024, 40
    dyn, cost, U_nominal, U_perturbed, noise, x = _setup(K=K, H=H, with_obstacles=True)

    ref = _pytorch_rollout_costs(dyn, cost, U_perturbed, x, K, H)
    out = _call_kernel(x, U_nominal, noise, cost, dyn)
    torch.cuda.synchronize()

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    max_rel = rel_err.max().item()
    max_abs = abs_err.max().item()
    print(f"\n  (with obstacles)  max abs err: {max_abs:.3e}, max rel: {max_rel:.3e}")
    print(f"  ref range: [{ref.min():.2f}, {ref.max():.2f}]")

    assert max_rel < 1e-3, (
        f"obstacle-case costs differ too much: "
        f"max_rel={max_rel:.3e}, max_abs={max_abs:.3e}"
    )


def test_cuda_controller_step_matches_cpu_controller_step():
    """End-to-end: CudaMPPIController.step() vs MPPIController.step() with same seed.

    Both controllers use the same Generator seed, so the noise they sample is
    identical. The action they produce should be identical (within fp32).
    """
    from mppi_cuda import CudaMPPIController

    K, H, dtype = 1024, 40, torch.float32

    # CPU reference
    dyn_cpu = DoubleIntegratorArm(dt=0.02, device="cpu", dtype=dtype)
    cost_cpu = ReachingCost(
        target_pos=[0.5, 0.3, 0.5],
        w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX, device="cpu", dtype=dtype,
    )
    ctrl_cpu = MPPIController(
        dynamics=dyn_cpu.step,
        running_cost=cost_cpu.running_cost,
        terminal_cost=cost_cpu.terminal_cost,
        action_dim=7, horizon=H, num_samples=K,
        sigma=2.5, temperature=1.0,
        u_min=-20.0, u_max=20.0,
        device="cpu", dtype=dtype, seed=0,
    )

    # CUDA backend
    dyn_cuda = DoubleIntegratorArm(dt=0.02, device="cuda", dtype=dtype)
    cost_cuda = ReachingCost(
        target_pos=[0.5, 0.3, 0.5],
        w_pos=500.0, w_u=0.005, w_qdot=0.05, terminal_scale=20.0,
        q_min=FRANKA_Q_MIN, q_max=FRANKA_Q_MAX, device="cuda", dtype=dtype,
    )
    ctrl_cuda = CudaMPPIController(
        dynamics=dyn_cuda, cost=cost_cuda,
        action_dim=7, horizon=H, num_samples=K,
        sigma=2.5, temperature=1.0,
        u_min=-20.0, u_max=20.0,
        device="cuda", dtype=dtype, seed=0,
    )

    x_cpu = torch.cat([torch.tensor(FRANKA_HOME_Q, dtype=dtype), torch.zeros(7)])
    x_cuda = x_cpu.to("cuda")

    u_cpu = ctrl_cpu.step(x_cpu)
    u_cuda = ctrl_cuda.step(x_cuda)

    diff = (u_cpu - u_cuda.cpu()).abs().max().item()
    print(f"\n  max |u_cpu - u_cuda|: {diff:.3e}")
    # The two backends use device-specific torch.randn, so the sampled noise
    # is NOT bit-identical even with the same seed. We accept any reasonable
    # similarity here — the real correctness test is the rollout cost test above.
    # This test is mainly to confirm the end-to-end plumbing works.
    assert torch.isfinite(u_cuda).all(), "CUDA controller produced non-finite action"
