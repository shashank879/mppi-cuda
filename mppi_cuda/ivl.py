"""Implicit V-Learning (IVL) — networks and loss math.

Adapted from OGBench's GCIVLAgent (Park et al., goal-conditioned variant of
Kostrikov et al.'s IQL). The V-only flavour fits our use case exactly:
MPPI's terminal cost wants a scalar value for a final state given a goal —
that's V(s, g), not Q(s, a).

Two networks (V1, V2) and a target network with Polyak averaging. The "double"
trick mitigates over-estimation bias: advantage is computed against the
minimum of the two target V's, while each network's TD target uses its own
target-net counterpart. See Park et al. for the analysis.

Input normalisation is baked into the network as `register_buffer`s so a
saved checkpoint is self-contained — no separate "stats" file to track at
inference time.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn


class GCVNetwork(nn.Module):
    """Goal-conditioned twin V network with optional input normalisation.

    Forward: V1(s, g), V2(s, g) — two scalar value estimates.

    Both nets share the same architecture but have independent parameters,
    so they explore the value landscape from slightly different angles
    and we take their minimum as the conservative estimate during
    advantage computation.
    """

    def __init__(
        self,
        state_dim: int = 14,
        goal_dim: int = 3,
        hidden: Sequence[int] = (256, 256),
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        state_mean=None, state_std=None,
        goal_mean=None,  goal_std=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.hidden = tuple(hidden)
        self.use_layer_norm = use_layer_norm

        # Normalisation stats — saved with the model, applied in forward().
        # Default identity (mean=0, std=1) so a vanilla constructor is a no-op.
        self.register_buffer("state_mean", torch.zeros(state_dim) if state_mean is None
                             else torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std",  torch.ones(state_dim) if state_std is None
                             else torch.as_tensor(state_std,  dtype=torch.float32))
        self.register_buffer("goal_mean",  torch.zeros(goal_dim) if goal_mean is None
                             else torch.as_tensor(goal_mean,  dtype=torch.float32))
        self.register_buffer("goal_std",   torch.ones(goal_dim) if goal_std is None
                             else torch.as_tensor(goal_std,   dtype=torch.float32))

        self.v1 = self._build_mlp(state_dim + goal_dim, hidden, use_layer_norm, dropout=dropout)
        self.v2 = self._build_mlp(state_dim + goal_dim, hidden, use_layer_norm, dropout=dropout)

    @staticmethod
    def _build_mlp(in_dim: int, hidden: Sequence[int], use_layer_norm: bool, dropout: float) -> nn.Sequential:
        layers, prev = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_n = (s - self.state_mean) / self.state_std
        g_n = (g - self.goal_mean) / self.goal_std
        x = torch.cat([s_n, g_n], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)

    def v_min(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Convenience: returns min(V1, V2) for inference (terminal-cost use)."""
        v1, v2 = self(s, g)
        return torch.minimum(v1, v2)

    def v_mean(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Convenience: returns mean(V1, V2)."""
        v1, v2 = self(s, g)
        return 0.5 * (v1 + v2)


def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric weighted squared error.

    weight = expectile when adv >= 0 (under-estimates of value get more weight)
           = (1 - expectile) otherwise

    With expectile=0.5 this reduces to MSE/2. With expectile -> 1, V converges
    to the maximum-Q expectation, which is what we want for evaluating a
    near-optimal policy. The OGBench default is 0.9.
    """
    weight = torch.where(adv >= 0,
                         torch.full_like(adv, expectile),
                         torch.full_like(adv, 1.0 - expectile))
    return weight * (diff ** 2)


def value_loss(
    batch: dict,
    v_net: GCVNetwork,
    target_v_net: GCVNetwork,
    discount: float,
    expectile: float,
):
    """IVL value loss with the OGBench stabilisation tricks.

    The advantage signal is computed from the target network (variance reduction),
    while each V_i's TD target uses its own target-network counterpart V_i_target.
    That's why we don't just use `min(V1_target, V2_target)` as the per-network
    target — that would give all losses the same gradient w.r.t. their V.

    Args:
        batch: dict with keys s (B, Ds), g (B, Dg), r (B,), s_next (B, Ds), mask (B,)
               where mask = 1 - done.

    Returns:
        (loss_tensor, info_dict)
    """
    s       = batch["s"]
    g       = batch["g"]
    r       = batch["r"]
    s_next  = batch["s_next"]
    mask    = batch["mask"]

    with torch.no_grad():
        v1_next_t, v2_next_t = target_v_net(s_next, g)
        v_next_t = torch.minimum(v1_next_t, v2_next_t)
        v1_t,     v2_t     = target_v_net(s, g)
        v_t = 0.5 * (v1_t + v2_t)

        adv = r + discount * mask * v_next_t - v_t
        # Per-network TD targets; each V_i regresses against its own bootstrap.
        q1_target = r + discount * mask * v1_next_t
        q2_target = r + discount * mask * v2_next_t

    v1, v2 = v_net(s, g)
    loss1 = expectile_loss(adv, q1_target - v1, expectile).mean()
    loss2 = expectile_loss(adv, q2_target - v2, expectile).mean()
    loss = loss1 + loss2

    with torch.no_grad():
        v_pred_mean = 0.5 * (v1 + v2).mean().item()
        v_pred_min  = torch.minimum(v1, v2).min().item()
        v_pred_max  = torch.maximum(v1, v2).max().item()
    info = {
        "value_loss": loss.item(),
        "loss1": loss1.item(),
        "loss2": loss2.item(),
        "v_mean": v_pred_mean,
        "v_min":  v_pred_min,
        "v_max":  v_pred_max,
        "adv_mean": adv.mean().item(),
        "adv_std":  adv.std().item(),
        "r_mean":   r.mean().item(),
    }
    return loss, info


@torch.no_grad()
def polyak_update(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """In-place: target_net <- (1 - tau) * target_net + tau * net."""
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
    # Also sync buffers (normalisation stats) — these don't change but cheap to copy
    for b, tb in zip(net.buffers(), target_net.buffers()):
        tb.data.copy_(b.data)


def load_v_network(checkpoint_path: str, device: str = "cuda") -> GCVNetwork:
    """Convenience loader for inference. Returns just the V network."""
    print(f'Loading value network from: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print('Config:\n', cfg)
    v_net = GCVNetwork(
        state_dim=cfg["state_dim"], goal_dim=cfg["goal_dim"],
        hidden=cfg["hidden"], use_layer_norm=cfg["use_layer_norm"], dropout=cfg["dropout"]
    ).to(device)
    v_net.load_state_dict(ckpt["v_net"])
    v_net.eval()
    return v_net
