"""Per-tick reward for trajectory tracking.

The reward is the negation of three cost terms:
  - tracking error ||ee - target||²
  - effort         ||u||²            (weight α_u)
  - jerk           ||u - u_prev||²   (weight α_du)

This is intentionally a strict subset of the MPPI running cost — we drop the
joint-limit barrier and obstacle terms because IVL learns V(s, g) where g is
just a target position, and we don't want V to encode obstacle knowledge into
its terminal value (otherwise it can't generalise to new obstacle layouts).

Numpy-only on purpose: the collector runs in numpy after pulling state from
MuJoCo, and the trainer recomputes rewards in PyTorch from saved arrays
during goal relabeling. Two implementations with the same formula is fine —
they're only ever evaluated on saved data, so any drift would be caught by
a unit test.
"""

from __future__ import annotations

import numpy as np


def tracking_reward(
    ee_pos: np.ndarray,
    action: np.ndarray,
    target: np.ndarray,
    action_prev: np.ndarray | None = None,
    alpha_u: float = 0.005,
    alpha_du: float = 0.01,
) -> np.ndarray:
    """Per-tick reward for end-effector trajectory tracking.

    Args:
        ee_pos:      (..., 3) achieved EE position at this tick
        action:      (..., 7) commanded joint accel
        target:      (..., 3) target EE position at this tick
        action_prev: (..., 7) previous tick's action, or None at episode start
                     (None is treated as zeros; this matches the standard
                     "no jerk penalty on the first action" convention).
        alpha_u:     weight on effort²
        alpha_du:    weight on jerk² (matters for smoothness — see README)

    Returns:
        reward of shape (...). Strictly non-positive.
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    track_err_sq = np.sum((ee_pos - target) ** 2, axis=-1)
    effort_sq    = np.sum(action ** 2, axis=-1)
    if action_prev is None:
        jerk_sq = np.zeros_like(effort_sq)
    else:
        action_prev = np.asarray(action_prev, dtype=np.float64)
        jerk_sq = np.sum((action - action_prev) ** 2, axis=-1)
    return -(track_err_sq + alpha_u * effort_sq + alpha_du * jerk_sq)


def recompute_reward_torch(
    ee_pos,
    action,
    target,
    action_prev,
    alpha_u: float = 0.005,
    alpha_du: float = 0.01,
):
    """PyTorch version used during IVL goal-relabeling.

    Same formula as `tracking_reward`. All inputs are torch tensors with a
    leading batch dim; broadcasting is along that dim. `action_prev` must
    be provided (the caller is responsible for zeroing it at episode start).
    """
    import torch
    track_err_sq = ((ee_pos - target) ** 2).sum(dim=-1)
    effort_sq    = (action ** 2).sum(dim=-1)
    jerk_sq      = ((action - action_prev) ** 2).sum(dim=-1)
    return -(track_err_sq + alpha_u * effort_sq + alpha_du * jerk_sq)
