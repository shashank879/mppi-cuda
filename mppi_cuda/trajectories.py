"""Parametric Cartesian trajectories for end-effector tracking.

All trajectory functions have signature `traj_fn(t) -> (3,) np.ndarray` where
`t` is sim time in seconds. They're meant to be composed into the per-tick
target buffer the controller (Python or CUDA) consumes — see
`build_target_traj()` for the lookahead-aware host helper.

We deliberately don't put these on a device: they're called O(H) times per
control tick on the host, then bulk-uploaded once to GPU per tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# -------- parametric primitives --------

@dataclass(frozen=True)
class Circle:
    """Constant-radius circle around `center`, period `period` s."""
    center: tuple[float, float, float] = (0.5, 0.0, 0.5)
    radius: float = 0.10
    period: float = 4.0
    plane: str = "xy"          # which two axes the circle lies in

    def __call__(self, t: float) -> np.ndarray:
        omega = 2.0 * np.pi / self.period
        c0, c1, c2 = self.center
        a = self.radius * np.cos(omega * t)
        b = self.radius * np.sin(omega * t)
        if self.plane == "xy":   return np.array([c0 + a, c1 + b, c2])
        if self.plane == "xz":   return np.array([c0 + a, c1,     c2 + b])
        if self.plane == "yz":   return np.array([c0,     c1 + a, c2 + b])
        raise ValueError(f"unknown plane {self.plane}")


@dataclass(frozen=True)
class FigureEight:
    """Lemniscate of Gerono in `plane`, scale `radius`, period `period`."""
    center: tuple[float, float, float] = (0.5, 0.0, 0.5)
    radius: float = 0.10
    period: float = 6.0
    plane: str = "xy"

    def __call__(self, t: float) -> np.ndarray:
        omega = 2.0 * np.pi / self.period
        c0, c1, c2 = self.center
        s = np.sin(omega * t)
        a = self.radius * s
        b = self.radius * s * np.cos(omega * t)
        if self.plane == "xy":   return np.array([c0 + a, c1 + b, c2])
        if self.plane == "xz":   return np.array([c0 + a, c1,     c2 + b])
        if self.plane == "yz":   return np.array([c0,     c1 + a, c2 + b])
        raise ValueError(f"unknown plane {self.plane}")


@dataclass(frozen=True)
class Waypoints:
    """Linear interpolation between a closed cycle of 3-D waypoints.

    `segment_time` is the time spent on each segment; one full cycle takes
    `len(points) * segment_time`. Cycles forever.
    """
    points: Sequence[tuple[float, float, float]]
    segment_time: float = 2.0

    def __call__(self, t: float) -> np.ndarray:
        n = len(self.points)
        cycle = n * self.segment_time
        u = (t % cycle) / self.segment_time
        i = int(np.floor(u)) % n
        alpha = u - np.floor(u)
        p0 = np.asarray(self.points[i], dtype=float)
        p1 = np.asarray(self.points[(i + 1) % n], dtype=float)
        return (1.0 - alpha) * p0 + alpha * p1


# -------- host-side trajectory buffer builder --------

def build_target_traj(
    traj_fn: Callable[[float], np.ndarray],
    t_now: float,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Sample `traj_fn` at `[t_now, t_now+dt, ..., t_now+H*dt]`.

    Returns shape (H+1, 3). The +1 is the terminal step the rollout reaches
    after H dynamics updates.
    """
    times = t_now + dt * np.arange(horizon + 1)
    return np.stack([np.asarray(traj_fn(float(t)), dtype=np.float32)
                     for t in times], axis=0)


# -------- randomised trajectory sampler for data collection --------

def sample_random_trajectory(
    rng: np.random.Generator | None = None,
    workspace_center: tuple[float, float, float] = (0.45, 0.10, 0.50),
    workspace_half_extent: tuple[float, float, float] = (0.10, 0.15, 0.08),
) -> Callable[[float], np.ndarray]:
    """Return a random trajectory fn for use in data collection.

    Mixes circles, figure-eights, and 3-5 waypoint loops in roughly the
    proportions the cost-tracker is expected to deploy against. Centre and
    radius randomised within a reachable workspace.
    """
    rng = rng if rng is not None else np.random.default_rng()
    kind = rng.choice(["circle", "figure_eight", "waypoints"], p=[0.50, 0.30, 0.20])
    cx = workspace_center[0] + rng.uniform(-1, 1) * workspace_half_extent[0]
    cy = workspace_center[1] + rng.uniform(-1, 1) * workspace_half_extent[1]
    cz = workspace_center[2] + rng.uniform(-1, 1) * workspace_half_extent[2]
    center = (float(cx), float(cy), float(cz))

    if kind == "circle":
        r = float(rng.uniform(0.05, 0.13))
        period = float(rng.uniform(3.0, 6.0))
        plane = rng.choice(["xy", "xz", "yz"])
        return Circle(center=center, radius=r, period=period, plane=plane)
    if kind == "figure_eight":
        r = float(rng.uniform(0.05, 0.13))
        period = float(rng.uniform(4.0, 8.0))
        plane = rng.choice(["xy", "xz", "yz"])
        return FigureEight(center=center, radius=r, period=period, plane=plane)
    # waypoints
    n_wp = int(rng.integers(3, 6))
    pts = []
    for _ in range(n_wp):
        ox = workspace_half_extent[0] * rng.uniform(-1, 1)
        oy = workspace_half_extent[1] * rng.uniform(-1, 1)
        oz = workspace_half_extent[2] * rng.uniform(-1, 1)
        pts.append((center[0] + ox, center[1] + oy, center[2] + oz))
    seg_t = float(rng.uniform(1.0, 2.5))
    return Waypoints(points=tuple(pts), segment_time=seg_t)
