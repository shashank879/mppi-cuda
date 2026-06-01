"""CUDA-backed MPPI controller.

Same external interface as `MPPIController.step(x)` but the inner
rollout-and-cost loop is replaced by a single fused CUDA kernel call.

Constraints (v0):
  - `dynamics` must be a `DoubleIntegratorArm`
  - `cost` must be a `ReachingCost`
  - All tensors live on CUDA

The kernel reads the dynamics/cost parameters directly off these
objects (joint limits, dt, weights, target, etc.) and is hardcoded for
the 7-DoF Franka geometry. Later versions will template on dynamics
type so a learned MLP or different arm can plug in.
"""

from __future__ import annotations

import torch

from .dynamics import DoubleIntegratorArm
from .costs import ReachingCost


class CudaMPPIController:

    def __init__(
        self,
        dynamics: DoubleIntegratorArm,
        cost: ReachingCost,
        action_dim: int,
        horizon: int,
        num_samples: int,
        sigma,
        temperature: float = 1.0,
        u_min: float = -20.0,
        u_max: float = 20.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        if not isinstance(dynamics, DoubleIntegratorArm):
            raise TypeError("CUDA backend currently supports DoubleIntegratorArm only")
        if not isinstance(cost, ReachingCost):
            raise TypeError("CUDA backend currently supports ReachingCost only")
        if not device.startswith("cuda"):
            raise ValueError("CudaMPPIController requires a CUDA device")
        if dtype != torch.float32:
            raise ValueError("v0 kernel is fp32-only")

        # Lazy import so the package is usable without the compiled extension.
        from mppi_cuda._kernels import mppi_rollout as _kernel
        self._kernel = _kernel

        self.dynamics = dynamics
        self.cost = cost
        self.m = action_dim
        self.H = horizon
        self.K = num_samples
        self.lam = temperature
        self.device = device
        self.dtype = dtype

        sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)
        if sigma_t.ndim == 0:
            sigma_t = sigma_t.expand(self.m).clone()
        self.sigma = sigma_t

        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.U = torch.zeros(self.H, self.m, device=device, dtype=dtype)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        # Make sure dynamics/cost tensors live on the right device.
        # (Cheap; just throws if user constructed them on CPU.)
        for name, t in [
            ("dynamics.q_min",    self.dynamics.q_min),
            ("dynamics.q_max",    self.dynamics.q_max),
            ("dynamics.qdot_max", self.dynamics.qdot_max),
            ("cost.target_pos",   self.cost.target_pos),
        ]:
            if not t.is_cuda:
                raise ValueError(f"{name} must be on CUDA for the kernel backend")

    def reset(self) -> None:
        self.U.zero_()

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One MPPI tick. Returns the control to apply (m,)."""
        K, H, m = self.K, self.H, self.m
        d, dt = self.device, self.dtype

        # 1. Sample noise.
        noise = torch.randn(K, H, m, generator=self.generator, device=d, dtype=dt)
        noise = noise * self.sigma

        # 2. Clamp on the host so the weighted update uses the same effective
        #    noise the kernel will see.
        U_perturbed = self.U.unsqueeze(0) + noise
        U_perturbed = torch.clamp(U_perturbed, self.u_min, self.u_max)
        noise = (U_perturbed - self.U.unsqueeze(0)).contiguous()

        # 3. Build the per-step target buffer (H+1, 3).
        # If the cost has an explicit trajectory and it's long enough, use it.
        # Otherwise broadcast the static target_pos. Either way the kernel sees
        # a uniform (H+1, 3) layout, so its time loop just indexes by t.
        if (self.cost.target_traj is not None
                and self.cost.target_traj.shape[0] >= H + 1):
            target_traj = self.cost.target_traj[: H + 1].contiguous()
        else:
            target_traj = self.cost.target_pos.unsqueeze(0).expand(H + 1, 3).contiguous()

        # 4. Fused rollout + cost on GPU.
        # IMPORTANT: the kernel re-applies the same clamp internally, so the
        # noise we pass in is the *raw* (pre-clamp) perturbation. After
        # clamping above, `noise = U_perturbed - U`, so `U + noise` clamps to
        # itself — round-trip identity. The kernel and host therefore agree
        # on the effective u_t at every step. Verified in the correctness test.
        costs, final_state = self._kernel(
            x.contiguous(),
            self.U.contiguous(),
            noise,
            target_traj,
            self.dynamics.q_min,
            self.dynamics.q_max,
            self.dynamics.qdot_max,
            self.cost.obstacles,  # (n_obs, 4); (0, 4) if disabled
            self.u_min, self.u_max,
            self.dynamics.dt,
            self.cost.w_pos * (1. - self.cost.alpha),
            self.cost.w_u * (1. - self.cost.alpha),
            self.cost.w_qdot * (1. - self.cost.alpha),
            self.cost.w_lim * (1. - self.cost.alpha),
            self.cost.w_obs,
            self.cost.obs_margin,
            self.cost.w_obs_flat,
            self.cost.terminal_scale * (1. - self.cost.alpha),
        )
        if self.cost.alpha:
            costs = costs + self.cost.value_cost(final_state, H)

        # 4. Importance weights.
        beta = costs.min()
        w = torch.exp(-(costs - beta) / self.lam)
        w = w / (w.sum() + 1e-9)

        # 5. Weighted update of nominal control.
        weighted_noise = torch.einsum("k,khm->hm", w, noise)
        self.U = self.U + weighted_noise
        self.U = torch.clamp(self.U, self.u_min, self.u_max)

        # 6. Action + shift.
        u_0 = self.U[0].clone()
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0
        return u_0
