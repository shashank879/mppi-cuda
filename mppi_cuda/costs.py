"""Cost functions for MPPI.

The reaching cost is the sum of:
  - task-space position error (weight w_pos)
  - control magnitude    (weight w_u)
  - joint velocity magnitude (weight w_qdot)
  - one-sided quadratic barrier for joint limits (weight w_lim)

Terminal cost is a heavier weight on the final position error.
"""

from __future__ import annotations
import torch

from .kinematics import forward_kinematics
from .ivl import GCVNetwork, load_v_network


class ReachingCost:
    def __init__(
        self,
        target_pos,
        w_pos: float = 50.0,
        w_u: float = 0.01,
        w_qdot: float = 0.005,
        w_lim: float = 100.0,
        terminal_scale: float = 10.0,
        # Obstacle avoidance: smooth quadratic in the margin band,
        # plus optional flat penalty for actual intersection.
        obstacles=None,         # list of (x, y, z, r) or (N_obs, 4) tensor; None -> disabled
        w_obs: float = 1000.0,  # weight on smooth violation² sum
        obs_margin: float = 0.05,
        w_obs_flat: float = 0.0,  # weight on count of actual intersections; 0 disables
        q_min=None,
        q_max=None,
        value_fn_pt=None,
        alpha=0.99,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        # Target may be:
        #   - shape (3,):       static reach target (back-compat)
        #   - shape (T, 3):     per-step trajectory (T >= horizon+1)
        # `target_pos` always exposes the t=0 entry so existing static callers
        # keep working unchanged. `target_traj` is None when static and
        # (T, 3) when tracking — the controllers branch on this.
        t = torch.as_tensor(target_pos, device=device, dtype=dtype)
        if t.ndim == 1:
            if t.shape[0] != 3:
                raise ValueError(f"static target_pos must be shape (3,); got {t.shape}")
            self.target_pos = t.contiguous()
            self.target_traj = None
        elif t.ndim == 2:
            if t.shape[1] != 3:
                raise ValueError(
                    f"target_pos as a trajectory must be shape (T, 3); got {t.shape}"
                )
            self.target_traj = t.contiguous()
            self.target_pos = self.target_traj[0]    # t=0 view, for static callers
        else:
            raise ValueError(f"target_pos must be 1D or 2D; got shape {t.shape}")

        self.w_pos = w_pos
        self.w_u = w_u
        self.w_qdot = w_qdot
        self.w_lim = w_lim
        self.terminal_scale = terminal_scale
        self.q_min = (
            torch.as_tensor(q_min, device=device, dtype=dtype) if q_min is not None else None
        )
        self.q_max = (
            torch.as_tensor(q_max, device=device, dtype=dtype) if q_max is not None else None
        )

        # Obstacles stored as a single (N_obs, 4) tensor so kernel passes are
        # one ptr + a length. Empty (0, 4) when disabled.
        if obstacles is None or (hasattr(obstacles, "__len__") and len(obstacles) == 0):
            self.obstacles = torch.zeros((0, 4), device=device, dtype=dtype)
        else:
            obs_t = torch.as_tensor(obstacles, device=device, dtype=dtype)
            if obs_t.ndim != 2 or obs_t.shape[1] != 4:
                raise ValueError(
                    f"obstacles must be shape (N, 4) for (x, y, z, r); got {obs_t.shape}"
                )
            self.obstacles = obs_t.contiguous()
        self.n_obs = int(self.obstacles.shape[0])
        self.w_obs = float(w_obs)
        self.obs_margin = float(obs_margin)
        self.w_obs_flat = float(w_obs_flat)

        if value_fn_pt:
            self.value_net = load_v_network(value_fn_pt, device=device) if value_fn_pt else None
        else:
            assert alpha == 0., f"Non-zero alpha while value_fn_pt={value_fn_pt}"
            self.value_net = None
        self.alpha = alpha

    def set_target_traj(self, traj):
        """Replace the per-step target with a fresh (T, 3) buffer at runtime.

        Called once per outer control tick by the trajectory-tracking demo to
        slide the lookahead window forward.
        """
        t = torch.as_tensor(traj, device=self.target_pos.device, dtype=self.target_pos.dtype)
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"target_traj must be shape (T, 3); got {t.shape}")
        self.target_traj = t.contiguous()
        self.target_pos = self.target_traj[0]

    def _target_at(self, step):
        """Returns the (3,) target for rollout-step index `step`.

        When `target_traj` is unset (static reach), returns `target_pos`
        regardless of step. When it's set, indexes into it with safe clamp.
        """
        if self.target_traj is None or step is None:
            return self.target_pos
        idx = min(int(step), self.target_traj.shape[0] - 1)
        return self.target_traj[idx]

    def _ee_pos_err_sq(self, q: torch.Tensor, step=None) -> torch.Tensor:
        ee_pos, _ = forward_kinematics(q)
        return ((ee_pos - self._target_at(step)) ** 2).sum(-1)

    def _joint_limit_cost(self, q: torch.Tensor) -> torch.Tensor:
        if self.q_min is None or self.q_max is None:
            return torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
        below = torch.clamp(self.q_min - q, min=0.0)
        above = torch.clamp(q - self.q_max, min=0.0)
        return (below ** 2 + above ** 2).sum(-1)

    def _obstacle_cost(self, q: torch.Tensor) -> torch.Tensor:
        """Returns the *already-weighted* obstacle cost.

        Smooth quadratic ramp inside `r + margin`, optionally plus a flat
        penalty when the EE has actually entered the sphere. Returns the
        full weighted sum (w_obs * smooth + w_obs_flat * flat) so the caller
        just adds it to the running cost.
        """
        if self.n_obs == 0:
            return torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
        ee_pos, _ = forward_kinematics(q)         # (..., 3)
        centers = self.obstacles[:, :3]            # (N_obs, 3)
        radii   = self.obstacles[:, 3]             # (N_obs,)

        # Distance from ee to each obstacle center.
        # ee_pos.unsqueeze(-2): (..., 1, 3); centers: (N_obs, 3) → (..., N_obs, 3)
        d = (ee_pos.unsqueeze(-2) - centers).norm(dim=-1)   # (..., N_obs)

        # Smooth quadratic violation inside the inflated radius (r + margin).
        violation = torch.clamp(radii + self.obs_margin - d, min=0.0)
        cost = self.w_obs * (violation ** 2).sum(-1)

        # Optional flat bump for actual intersection (d < r).
        if self.w_obs_flat > 0.0:
            cost = cost + self.w_obs_flat * (d < radii).to(q.dtype).sum(-1)

        return cost

    def running_cost(self, x: torch.Tensor, u: torch.Tensor, step=None) -> torch.Tensor:
        q = x[..., :7]
        qdot = x[..., 7:]
        return (
            self.w_pos * self._ee_pos_err_sq(q, step=step)
            + self.w_u * (u ** 2).sum(-1)
            + self.w_qdot * (qdot ** 2).sum(-1)
            + self.w_lim * self._joint_limit_cost(q)
        ) * (1. - self.alpha) + self._obstacle_cost(q)

    def value_cost(self, x: torch.Tensor, step=None) -> torch.Tensor:
        q = x[..., :7]
        ee_target = self._target_at(step)
        return - self.value_net.v_mean(
            torch.cat([forward_kinematics(q)[0], x], -1),
            ee_target.expand(*x.shape[:-1], ee_target.shape[-1])
        ) * self.alpha

    def terminal_cost(self, x: torch.Tensor, step=None) -> torch.Tensor:
        q = x[..., :7]
        # Heavier reach cost at the end; also keep obstacles active so the
        # planner doesn't "tunnel" to the goal in the last step.
        cost = (
            self.terminal_scale * self.w_pos * self._ee_pos_err_sq(q, step=step)
        ) * (1 - self.alpha) + self._obstacle_cost(q)
        if self.value_net:
            cost = cost + (self.alpha * self.value_cost(x, step))
        return cost

    def __call__(self, x, u, step=None):
        return self.running_cost(x, u, step=step)
