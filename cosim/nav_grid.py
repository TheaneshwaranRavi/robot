"""Occupancy grid, inflation, distance field and planning costs for the navigator (numpy only).

Grid convention: ``occ[iy, ix]`` with cell (ix, iy) covering
[ox + ix*res, ox + (ix+1)*res) x [oy + iy*res, oy + (iy+1)*res) in the map frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

RES = 0.05            # m per cell (contract: 5 cm)
INFLATION_MARGIN = 0.05  # m added to the AMR half-diagonal (contract)
LETHAL_MARGIN = 0.025    # m added to the AMR half-width for the hard (lethal) layer
INFLATED_PENALTY = 25.0  # extra cost per cell inside the inflation ring but outside the lethal core
SOFT_BAND = 0.30         # m of soft cost outside the inflation ring (keeps paths off rack corners)
SOFT_PENALTY = 3.0


@dataclass
class Grid:
    occ: np.ndarray          # bool [ny, nx]
    origin: tuple[float, float]
    res: float = RES

    @property
    def shape(self) -> tuple[int, int]:
        return self.occ.shape

    def world_to_cell(self, x, y):
        """Cell indices (ix, iy) for map coordinates (scalars or arrays, no bounds check)."""
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.res,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.res)

    def in_bounds(self, ix, iy):
        ny, nx = self.occ.shape
        return (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)


def _rects(layout: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    out = []
    for item in list(layout.get("walls", [])) + list(layout.get("obstacles", [])):
        (cx, cy), (sx, sy) = item["center"], item["size"]
        out.append((cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2))
    return out


def build_grid(layout: dict[str, Any], res: float = RES, margin: float = 0.25) -> Grid:
    """Rasterise walls + obstacles: every cell that overlaps a rectangle is occupied."""
    rects = _rects(layout)
    fx, fy = layout.get("floor", {}).get("size", [0.0, 0.0])
    xmin = min([0.0] + [r[0] for r in rects]) - margin
    ymin = min([0.0] + [r[1] for r in rects]) - margin
    xmax = max([fx] + [r[2] for r in rects]) + margin
    ymax = max([fy] + [r[3] for r in rects]) + margin
    # snap the origin to a multiple of res so cell edges line up with "round" coordinates
    ox = math.floor(xmin / res) * res
    oy = math.floor(ymin / res) * res
    nx = int(math.ceil((xmax - ox) / res))
    ny = int(math.ceil((ymax - oy) / res))
    occ = np.zeros((ny, nx), dtype=bool)
    eps = 1e-6
    for x0, y0, x1, y1 in rects:
        ix0 = max(0, int(math.floor((x0 - ox) / res + eps)))
        ix1 = min(nx, int(math.ceil((x1 - ox) / res - eps)))
        iy0 = max(0, int(math.floor((y0 - oy) / res + eps)))
        iy1 = min(ny, int(math.ceil((y1 - oy) / res - eps)))
        occ[iy0:max(iy0 + 1, iy1), ix0:max(ix0 + 1, ix1)] = True
    return Grid(occ=occ, origin=(ox, oy), res=res)


def _shift_or(dst: np.ndarray, src: np.ndarray, dx: int, dy: int) -> None:
    """dst |= src shifted by (dx, dy) cells (no wrap-around)."""
    ny, nx = src.shape
    xs0, xs1 = max(0, -dx), min(nx, nx - dx)
    ys0, ys1 = max(0, -dy), min(ny, ny - dy)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    dst[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] |= src[ys0:ys1, xs0:xs1]


def inflate(occ: np.ndarray, radius_m: float, res: float = RES) -> np.ndarray:
    """Exact disk dilation: a cell is inflated if an occupied cell centre lies within radius."""
    r = radius_m / res
    n = int(math.floor(r))
    out = occ.copy()
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r + 1e-9:
                _shift_or(out, occ, dx, dy)
    return out


_CHAMFER = [(dx, dy, math.hypot(dx, dy)) for dx in range(-2, 3) for dy in range(-2, 3)
            if (dx or dy) and not (abs(dx) == 2 and abs(dy) in (0, 2))]


def distance_field(occ: np.ndarray, res: float = RES, max_dist: float = 2.0) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest occupied cell centre, capped at max_dist.

    scipy is not available, so this is a 5x5 chamfer transform (weights 1, sqrt2, sqrt5; < 2 % error
    vs. Euclidean) solved by vectorised Jacobi relaxation: each sweep is a handful of numpy minimum
    operations, and it converges after about max_dist/res sweeps.
    """
    cap = max_dist / res
    d = np.where(occ, 0.0, cap)
    for _ in range(int(cap) + 4):
        new = d.copy()
        ny, nx = d.shape
        for dx, dy, w in _CHAMFER:
            xs0, xs1 = max(0, -dx), min(nx, nx - dx)
            ys0, ys1 = max(0, -dy), min(ny, ny - dy)
            np.minimum(new[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx], d[ys0:ys1, xs0:xs1] + w,
                       out=new[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx])
        if np.array_equal(new, d):
            break
        d = new
    return np.minimum(d, cap) * res


@dataclass
class NavMap:
    """Everything the navigator derives from layout.json once at startup."""
    grid: Grid                 # raw occupancy (walls + obstacles)
    inflated: np.ndarray       # bool: occupancy inflated by half-diagonal + 5 cm (contract)
    lethal: np.ndarray         # bool: occupancy inflated by half-width + 2.5 cm (hard limit)
    dist: np.ndarray           # m: distance field to raw occupancy (likelihood field + costs)
    cost: np.ndarray           # float: per-cell traversal cost, inf where lethal
    inflation_radius: float
    lethal_radius: float


def build_navmap(layout: dict[str, Any], res: float = RES) -> NavMap:
    grid = build_grid(layout, res)
    fx, fy = layout["amr"]["footprint"]
    infl_r = math.hypot(fx / 2, fy / 2) + INFLATION_MARGIN
    lethal_r = min(fx, fy) / 2 + LETHAL_MARGIN
    dist = distance_field(grid.occ, res, max_dist=2.0)
    inflated = inflate(grid.occ, infl_r, res)
    lethal = inflate(grid.occ, lethal_r, res)
    cost = make_cost(inflated, lethal, dist, infl_r)
    return NavMap(grid, inflated, lethal, dist, cost, infl_r, lethal_r)


def make_cost(inflated: np.ndarray, lethal: np.ndarray, dist: np.ndarray, infl_r: float) -> np.ndarray:
    """Cost layers: free = 1 (+ soft band near the inflation ring), inflated ring = 1 + penalty, lethal = inf.

    Paths stay outside the contract's inflated grid whenever possible; the ring is only crossed
    when the start or goal is inside it (e.g. LOAD, which docks 0.35 m from the arm pedestal).
    """
    cost = np.ones(inflated.shape)
    soft = np.clip(1.0 - (dist - infl_r) / SOFT_BAND, 0.0, 1.0)
    cost += SOFT_PENALTY * soft
    cost[inflated] = 1.0 + SOFT_PENALTY + INFLATED_PENALTY
    cost[lethal] = np.inf
    return cost


def add_obstacle_points(navmap: NavMap, points_xy: np.ndarray) -> np.ndarray:
    """Cost grid with extra (dynamic) obstacle points marked lethal/inflated."""
    g = navmap.grid
    extra = np.zeros(g.occ.shape, dtype=bool)
    if len(points_xy):
        ix, iy = g.world_to_cell(points_xy[:, 0], points_xy[:, 1])
        ok = g.in_bounds(ix, iy)
        extra[iy[ok], ix[ok]] = True
    if not extra.any():
        return navmap.cost
    cost = navmap.cost.copy()
    ring = inflate(extra, navmap.inflation_radius, g.res)
    cost[ring & np.isfinite(cost)] = 1.0 + SOFT_PENALTY + INFLATED_PENALTY
    cost[inflate(extra, navmap.lethal_radius, g.res)] = np.inf
    return cost
