"""IVL replay-buffer dataset with goal relabeling.

Loads an NPZ written by `scripts/collect_ivl_data.py` and serves batches
of (s, g, r, s_next, mask) with goal relabeling per the OGBench scheme:
  - 20%: g = the target the agent was aiming at (current goal)
  - 50%: g = a future target within the same episode (geometric sampling)
  - 30%: g = a target from a randomly different episode

Reward is recomputed against the relabeled goal so V learns the value of
"reaching g from s," not the value of "tracking the saved trajectory."
"""

from __future__ import annotations

import math

import numpy as np
import torch


class IVLDataset:
    """Goal-conditioned replay buffer for IVL training.

    Stored arrays live on `device` for the lifetime of the dataset; we pay
    one upload at construction time and the per-batch sampling is then all
    GPU-side indexing. For a 500-ep × 300-tick buffer that's ~10 MB on the
    A5000 — peanuts.
    """

    def __init__(
        self,
        npz_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        p_cur_goal:    float = 0.20,
        p_traj_goal:   float = 0.50,
        p_random_goal: float = 0.30,
        geom_sample: bool = True,
        geom_p:      float = 0.02,    # ~50-tick mean lookahead
        alpha_u:  float | None = None, # if None, read from NPZ metadata
        alpha_du: float | None = None,
    ):
        d = np.load(npz_path)

        def _to(name: str) -> torch.Tensor:
            return torch.from_numpy(d[name]).to(device=device, dtype=dtype).contiguous()

        self.states       = torch.cat([_to("ee_positions"), _to("states")], -1)        # (N, T+1, 14)
        self.actions      = _to("actions")       # (N, T,   7)
        self.targets      = _to("targets")       # (N, T+1, 3)
        self.ee_positions = _to("ee_positions")  # (N, T+1, 3)
        self.rewards_orig = _to("rewards")       # (N, T)

        self.alpha_u  = float(d["alpha_u"])  if alpha_u  is None else float(alpha_u)
        self.alpha_du = float(d["alpha_du"]) if alpha_du is None else float(alpha_du)

        self.N, self.T_plus_1 = self.states.shape[:2]
        self.T = self.T_plus_1 - 1            # transitions per episode

        if not abs(p_cur_goal + p_traj_goal + p_random_goal - 1.0) < 1e-6:
            raise ValueError("goal-relabel probs must sum to 1")
        self.p_cur    = p_cur_goal
        self.p_traj   = p_traj_goal
        self.p_random = p_random_goal
        self.geom_sample = geom_sample
        self.geom_p      = geom_p
        self._log_one_minus_p = math.log(1.0 - geom_p)

        self.device = device
        self.dtype = dtype

        self.stats = self.fit_normalisation(device='cuda')

    # -------------- normalisation stats --------------

    def fit_normalisation(self, device='cpu') -> dict:
        """Compute per-dim mean/std over (states, targets). Use to seed GCVNetwork."""
        states_flat = self.states.reshape(-1, self.states.shape[-1])
        targets_flat = self.targets.reshape(-1, self.targets.shape[-1])
        stats = {
            "state_mean": states_flat.mean(dim=0),
            "state_std":  states_flat.std(dim=0).clamp(min=1e-6),
            "state_max": states_flat.max(dim=0)[0],
            "state_min":  states_flat.min(dim=0)[0],
            "goal_mean":  targets_flat.mean(dim=0),
            "goal_std":   targets_flat.std(dim=0).clamp(min=1e-6),
            "goal_max":  targets_flat.max(dim=0)[0],
            "goal_min":  targets_flat.min(dim=0)[0],
        }
        if device == 'cpu':
            stats = {k: v.cpu().numpy() for k,v in stats.items()}
        return stats

    # -------------- sampling --------------

    def _sample_traj_ahead(self, t_idx: torch.Tensor, B: int, gen) -> torch.Tensor:
        """For each item, return `ahead` ticks such that t' = t_idx + ahead, with
        ahead >= 1 and t' <= T (so the target index t' is in [t+1, T])."""
        if self.geom_p:
            u = torch.rand(B, device=self.device, generator=gen)
            u = u.clamp(min=1e-8, max=1.0 - 1e-8)
            ahead = (torch.log1p(-u) / self._log_one_minus_p).floor().long() + 1
        else:
            # Uniform in [1, T - t_idx]
            ahead = torch.randint(1, self.T + 1, (B,), device=self.device, generator=gen)
        ahead_max = (self.T - t_idx).clamp(min=1)
        return torch.minimum(ahead, ahead_max)

    def sample_batch(self, batch_size: int, generator: torch.Generator | None = None) -> dict:
        """Draw a relabeled batch. All tensors returned are on `self.device`."""
        gen = generator
        dev = self.device
        B = batch_size

        ep_idx = torch.randint(0, self.N, (B,), device=dev, generator=gen)
        t_idx  = torch.randint(0, self.T, (B,), device=dev, generator=gen)

        s       = self.states[ep_idx, t_idx]              # (B, 14)
        s_next  = self.states[ep_idx, t_idx + 1]          # (B, 14)
        a       = self.actions[ep_idx, t_idx]             # (B, 7)
        # ee      = self.ee_positions[ep_idx, t_idx]
        # ee_next = self.ee_positions[ep_idx, t_idx + 1]    # (B, 3) — where we landed

        # Previous action for jerk; zero at t=0 to match collector convention.
        t_prev = torch.clamp(t_idx - 1, min=0)
        a_prev = self.actions[ep_idx, t_prev]
        a_prev = torch.where((t_idx > 0).unsqueeze(-1), a_prev, torch.zeros_like(a))

        # ---------------- goal relabel ----------------
        r_choice = torch.rand(B, device=dev, generator=gen)
        is_cur    = r_choice <  self.p_cur
        is_traj   = (r_choice >= self.p_cur) & (r_choice < self.p_cur + self.p_traj)
        # is_random is the complement (>= p_cur + p_traj); used implicitly below.

        g_cur = self.ee_positions[ep_idx, t_idx]  # (B, 3)

        ahead   = self._sample_traj_ahead(t_idx, B, gen)
        traj_t  = t_idx + ahead
        g_traj  = self.ee_positions[ep_idx, traj_t]  # (B, 3)

        rand_ep = torch.randint(0, self.N,          (B,), device=dev, generator=gen)
        rand_t  = torch.randint(0, self.T_plus_1,   (B,), device=dev, generator=gen)
        g_random = self.ee_positions[rand_ep, rand_t]  # (B, 3)

        g = torch.where(is_cur.unsqueeze(-1),  g_cur,
            torch.where(is_traj.unsqueeze(-1), g_traj, g_random))

        # ---------------- relabelled reward ----------------
        # track_err_sq = ((ee_next - g) ** 2).sum(dim=-1)
        # effort_sq    = (a ** 2).sum(dim=-1)
        # jerk_sq      = ((a - a_prev) ** 2).sum(dim=-1)
        # r = -(track_err_sq + self.alpha_u * effort_sq + self.alpha_du * jerk_sq)

        # # Fixed-length episodes → all transitions non-terminal.
        # mask = torch.ones(B, device=dev, dtype=self.dtype)

        success = is_cur.to(self.dtype)
        r       = success - 1.0
        mask    = 1.0 - success

        def add_noise(x, range=1.):
            noise_scale = torch.clamp(range, 0., 1.) * 0.005
            return x + torch.normal(torch.zeros_like(x), torch.zeros_like(x) + noise_scale)

        goal_dist = torch.norm(s[..., :3]-g)

        return {
            "s": add_noise(s, goal_dist),
            "g": add_noise(g, goal_dist),
            "r": r,
            "s_next": add_noise(s_next, goal_dist),
            "mask": mask,
            # Aux for diagnostics — not consumed by value_loss.
            "is_cur": is_cur,
            "is_traj": is_traj,
        }

    # -------------- introspection --------------

    def __len__(self) -> int:
        return self.N * self.T

    def __repr__(self) -> str:
        return (f"IVLDataset(N={self.N}, T={self.T}, "
                f"state_dim={self.states.shape[-1]}, goal_dim={self.targets.shape[-1]}, "
                f"transitions={len(self)}, device={self.device})")
