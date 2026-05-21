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


class ReachingCost:
    def __init__(
        self,
        target_pos,
        w_pos: float = 50.0,
        w_u: float = 0.01,
        w_qdot: float = 0.005,
        w_lim: float = 100.0,
        terminal_scale: float = 10.0,
        q_min=None,
        q_max=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.target_pos = torch.as_tensor(target_pos, device=device, dtype=dtype)
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

    def _ee_pos_err_sq(self, q: torch.Tensor) -> torch.Tensor:
        ee_pos, _ = forward_kinematics(q)
        return ((ee_pos - self.target_pos) ** 2).sum(-1)

    def _joint_limit_cost(self, q: torch.Tensor) -> torch.Tensor:
        if self.q_min is None or self.q_max is None:
            return torch.zeros(q.shape[:-1], device=q.device, dtype=q.dtype)
        below = torch.clamp(self.q_min - q, min=0.0)
        above = torch.clamp(q - self.q_max, min=0.0)
        return (below ** 2 + above ** 2).sum(-1)

    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q = x[..., :7]
        qdot = x[..., 7:]
        return (
            self.w_pos * self._ee_pos_err_sq(q)
            + self.w_u * (u ** 2).sum(-1)
            + self.w_qdot * (qdot ** 2).sum(-1)
            + self.w_lim * self._joint_limit_cost(q)
        )

    def terminal_cost(self, x: torch.Tensor) -> torch.Tensor:
        q = x[..., :7]
        return self.terminal_scale * self.w_pos * self._ee_pos_err_sq(q)

    def __call__(self, x, u):
        return self.running_cost(x, u)
