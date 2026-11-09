"""Lidar helpers: beam angles, subsampling, safety sector check and a ray-cast-on-grid simulator."""
from __future__ import annotations

import math

import numpy as np

from .nav_grid import Grid


def beam_angles(angle_min: float, angle_inc: float, n: int) -> np.ndarray:
    return angle_min + angle_inc * np.arange(n)


def subsample_indices(n: int, k: int = 60) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def forward_min(ranges: np.ndarray, angles: np.ndarray, half_width: float = math.radians(30.0)) -> float:
    """Smallest range in the forward sector |angle| <= half_width (inf if none)."""
    sel = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= half_width + 1e-9
    r = ranges[sel]
    r = r[np.isfinite(r)]
    return float(r.min()) if len(r) else math.inf


def raycast(grid: Grid, pose, angles: np.ndarray, range_max: float, step: float | None = None) -> np.ndarray:
    """Ranges from `pose` along `angles` (robot frame) to the first occupied cell; range_max if none.

    Vectorised ray marching with step res/4 (≈1 cm at 5 cm cells): all beams x all samples at once.
    """
    step = step or grid.res / 4.0
    x, y, th = pose
    ts = np.arange(step, range_max + step, step)
    a = th + np.asarray(angles)
    px = x + np.cos(a)[:, None] * ts[None, :]
    py = y + np.sin(a)[:, None] * ts[None, :]
    ix, iy = grid.world_to_cell(px, py)
    inb = grid.in_bounds(ix, iy)
    hit = np.zeros(px.shape, dtype=bool)
    hit[inb] = grid.occ[iy[inb], ix[inb]]
    hit |= ~inb
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    out = np.where(first >= 0, ts[np.maximum(first, 0)] - step / 2.0, range_max)
    return np.minimum(out, range_max)


def scan_points(pose, ranges: np.ndarray, angles: np.ndarray, range_max: float) -> np.ndarray:
    """Map-frame hit points of the valid (< range_max) beams."""
    ok = np.isfinite(ranges) & (ranges < range_max - 1e-3) & (ranges > 0.05)
    x, y, th = pose
    a = th + angles[ok]
    return np.stack([x + ranges[ok] * np.cos(a), y + ranges[ok] * np.sin(a)], axis=1)
