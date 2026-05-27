"""MPPI (Model Predictive Path Integral) controller — PyTorch baseline.

Algorithm (per control tick):

    1. Sample noise:   eps ~ N(0, Sigma),  shape (K, H, m)
    2. Roll out K perturbed control sequences from the current state,
       accumulating per-rollout cost.
    3. Compute importance weights via softmax of negative cost.
    4. Update nominal control sequence as the noise-weighted average.
    5. Apply U[0] to the plant, shift U left, zero the new last entry.

This implementation is intentionally close to the structure the CUDA
kernel will use: rollouts are independent across the K dimension, time
is sequential within a rollout, and the state and accumulated cost are
the only things that flow through the time loop.
"""

from __future__ import annotations
import inspect
from typing import Callable, Optional

import torch


def _accepts_step_kwarg(fn) -> bool:
    """True if `fn` exposes a `step` parameter (incl. **kwargs catch-all)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if "step" in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


class MPPIController:
    def __init__(
        self,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        running_cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost: Optional[Callable[[torch.Tensor], torch.Tensor]],
        action_dim: int,
        horizon: int,
        num_samples: int,
        sigma,
        temperature: float = 1.0,
        u_min=None,
        u_max=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        # Cache whether each cost callable accepts the `step` kwarg, so we
        # stay back-compat with user lambdas that don't.
        self._rc_takes_step = _accepts_step_kwarg(running_cost)
        self._tc_takes_step = (
            _accepts_step_kwarg(terminal_cost) if terminal_cost is not None else False
        )
        self.m = action_dim
        self.H = horizon
        self.K = num_samples
        self.lam = temperature
        self.device = device
        self.dtype = dtype

        # Per-dim noise std, broadcast to (m,)
        sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)
        if sigma_t.ndim == 0:
            sigma_t = sigma_t.expand(self.m).clone()
        self.sigma = sigma_t

        self.u_min = (
            torch.as_tensor(u_min, device=device, dtype=dtype) if u_min is not None else None
        )
        self.u_max = (
            torch.as_tensor(u_max, device=device, dtype=dtype) if u_max is not None else None
        )

        # Nominal control sequence (H, m). Warm-started across calls.
        self.U = torch.zeros(self.H, self.m, device=device, dtype=dtype)

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def reset(self) -> None:
        self.U.zero_()

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Run one MPPI iteration and return the control to apply.

        Args:
            x: (state_dim,) current state.

        Returns:
            u_0: (action_dim,) control to apply this tick.
        """
        K, H, m = self.K, self.H, self.m

        # 1. Sample noise (K, H, m) scaled by per-dim sigma.
        noise = torch.randn(K, H, m, generator=self.generator, device=self.device, dtype=self.dtype)
        noise = noise * self.sigma

        # 2. Perturbed control sequences.
        U_perturbed = self.U.unsqueeze(0) + noise  # (K, H, m)
        if self.u_min is not None or self.u_max is not None:
            U_perturbed = torch.clamp(U_perturbed, self.u_min, self.u_max)
            # Re-derive the noise we actually used, so the weighted update
            # reflects the *applied* perturbations after clamping.
            noise = U_perturbed - self.U.unsqueeze(0)

        # 3. Rollout. Time loop is sequential, K loop is vectorized.
        x_batch = x.unsqueeze(0).expand(K, -1).contiguous()
        costs = torch.zeros(K, device=self.device, dtype=self.dtype)
        for t in range(H):
            u_t = U_perturbed[:, t]                       # (K, m)
            if self._rc_takes_step:
                costs = costs + self.running_cost(x_batch, u_t, step=t)
            else:
                costs = costs + self.running_cost(x_batch, u_t)
            x_batch = self.dynamics(x_batch, u_t)
        if self.terminal_cost is not None:
            if self._tc_takes_step:
                costs = costs + self.terminal_cost(x_batch, step=H)
            else:
                costs = costs + self.terminal_cost(x_batch)

        # 4. Importance weights via subtracted-min softmax for stability.
        beta = costs.min()
        w = torch.exp(-(costs - beta) / self.lam)         # (K,)
        w = w / (w.sum() + 1e-9)

        # 5. Weighted update of the nominal control.
        # weighted_noise[t] = sum_k w_k * noise[k, t]
        weighted_noise = torch.einsum("k,khm->hm", w, noise)
        self.U = self.U + weighted_noise
        if self.u_min is not None or self.u_max is not None:
            self.U = torch.clamp(self.U, self.u_min, self.u_max)

        # 6. Extract action, shift nominal sequence.
        u_0 = self.U[0].clone()
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0
        return u_0
