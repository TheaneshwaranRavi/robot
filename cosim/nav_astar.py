"""8-connected A* on a cost grid, path shortcutting and docking-approach planning (numpy + heapq)."""
from __future__ import annotations

import heapq
import math

import numpy as np

from .nav_grid import Grid

SQRT2 = math.sqrt(2.0)


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """A* from start=(ix, iy) to goal=(ix, iy). Step cost = step length x cost of the entered cell.

    Diagonal moves may not cut lethal corners. Returns the list of cells (start..goal) or None.
    """
    ny, nx = cost.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    flat = cost.ravel().tolist()
    s, g = sy * nx + sx, gy * nx + gx
    if not math.isfinite(flat[g]) or not math.isfinite(flat[s]):
        return None
    cmin = float(np.min(cost[np.isfinite(cost)])) if np.isfinite(cost).any() else 1.0
    inf = math.inf
    gscore = [inf] * (nx * ny)
    parent = [-1] * (nx * ny)
    closed = bytearray(nx * ny)
    gscore[s] = 0.0
    straight = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))
    diag = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    isfinite = math.isfinite

    def h(ix: int, iy: int) -> float:
        dx, dy = abs(ix - gx), abs(iy - gy)
        return cmin * ((dx + dy) + (SQRT2 - 2.0) * min(dx, dy))

    heap = [(h(sx, sy), 0.0, s)]
    while heap:
        _, gs, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = 1
        if cur == g:
            break
        cy, cx = divmod(cur, nx)
        for dx, dy, step in straight:
            x, y = cx + dx, cy + dy
            if 0 <= x < nx and 0 <= y < ny:
                n = y * nx + x
                c = flat[n]
                if closed[n] or not isfinite(c):
                    continue
                ng = gs + step * c
                if ng < gscore[n]:
                    gscore[n] = ng
                    parent[n] = cur
                    heapq.heappush(heap, (ng + h(x, y), ng, n))
        for dx, dy in diag:
            x, y = cx + dx, cy + dy
            if 0 <= x < nx and 0 <= y < ny:
                n = y * nx + x
                c = flat[n]
                if closed[n] or not isfinite(c):
                    continue
                if not (isfinite(flat[cy * nx + x]) and isfinite(flat[y * nx + cx])):
                    continue  # no corner cutting
                ng = gs + SQRT2 * c
                if ng < gscore[n]:
                    gscore[n] = ng
                    parent[n] = cur
                    heapq.heappush(heap, (ng + h(x, y), ng, n))
    if not closed[g]:
        return None
    out = []
    cur = g
    while cur != -1:
        cy, cx = divmod(cur, nx)
        out.append((cx, cy))
        cur = parent[cur]
    out.reverse()
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    xs = np.rint(np.linspace(a[0], b[0], n + 1)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n + 1)).astype(int)
    return xs, ys


def _line_max_cost(cost: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    xs, ys = _line_cells(a, b)
    return float(cost[ys, xs].max())


def _line_cost(cost: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """Approximate integral of cell cost along the straight segment a-b (in cell units)."""
    xs, ys = _line_cells(a, b)
    c = cost[ys, xs]
    if not np.all(np.isfinite(c)):
        return math.inf
    return float(c.mean()) * math.hypot(b[0] - a[0], b[1] - a[1])


def shortcut(cost: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy string pulling: replace a run of cells by a straight line if the line crosses no
    lethal cell and is not more expensive (integrated cost) than the A* run it replaces, so
    shortcuts never trade path length for time inside the inflation ring."""
    if len(cells) <= 2:
        return list(cells)
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0) * float(cost[y1, x1]))
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = i + 1
        while j + 1 < len(cells) and \
                _line_cost(cost, cells[i], cells[j + 1]) <= (cum[j + 1] - cum[i]) * 1.02 + 1e-9:
            j += 1
        out.append(cells[j])
        i = j
    return out


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline so consecutive points are at most `spacing` apart (keeps vertices)."""
    if len(points) < 2:
        return np.asarray(points, dtype=float).reshape(-1, 2)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def plan(grid: Grid, cost: np.ndarray, start_xy, goal_xy, spacing: float = 0.05) -> np.ndarray | None:
    """Plan in map coordinates. Returns an (N, 2) polyline from start to the exact goal, or None.

    If the start cell is lethal (localisation noise near a wall) A* starts from the nearest
    non-lethal cell within 0.3 m.
    """
    sx, sy = (int(v) for v in grid.world_to_cell(*start_xy))
    gx, gy = (int(v) for v in grid.world_to_cell(*goal_xy))
    if not (grid.in_bounds(sx, sy) and grid.in_bounds(gx, gy)):
        return None
    if not np.isfinite(cost[sy, sx]):
        near = _nearest_free(cost, sx, sy, int(0.3 / grid.res))
        if near is None:
            return None
        sx, sy = near
    cells = astar(cost, (sx, sy), (gx, gy))
    if cells is None:
        return None
    cells = shortcut(cost, cells)
    pts = np.array([grid.cell_to_world(x, y) for x, y in cells], dtype=float).reshape(-1, 2)
    pts[0] = start_xy
    pts[-1] = goal_xy
    return densify(pts, spacing)


def _nearest_free(cost: np.ndarray, ix: int, iy: int, r: int) -> tuple[int, int] | None:
    ny, nx = cost.shape
    y0, y1, x0, x1 = max(0, iy - r), min(ny, iy + r + 1), max(0, ix - r), min(nx, ix + r + 1)
    win = cost[y0:y1, x0:x1]
    ys, xs = np.nonzero(np.isfinite(win))
    if len(xs) == 0:
        return None
    k = int(np.argmin((xs + x0 - ix) ** 2 + (ys + y0 - iy) ** 2))
    return int(xs[k] + x0), int(ys[k] + y0)


def segment_clear(grid: Grid, cost: np.ndarray, a_xy, b_xy) -> bool:
    a = tuple(int(v) for v in grid.world_to_cell(*a_xy))
    b = tuple(int(v) for v in grid.world_to_cell(*b_xy))
    if not (grid.in_bounds(*a) and grid.in_bounds(*b)):
        return False
    return math.isfinite(_line_max_cost(cost, a, b))


def plan_to_pose(grid: Grid, cost: np.ndarray, start_xy, goal_pose, approach: float = 0.5,
                 spacing: float = 0.05) -> tuple[np.ndarray, int | None] | None:
    """Plan to a docking pose: A* to a pre-goal point `approach` m behind the goal along its yaw,
    then a straight final segment, so the AMR arrives already aligned with the goal yaw.

    This matters at LOAD: the AMR docks 0.35 m from the arm pedestal, closer than its 0.375 m
    half-diagonal, so turning in place there would sweep a corner into the pedestal.
    Falls back to a direct plan if the approach point or segment is blocked, or if the AMR is
    already near the goal.

    Returns (points, dock_index) where points[dock_index] is the pre-goal point (None for a
    direct path), or None if there is no path.
    """
    gx, gy, gyaw = goal_pose
    pre = (gx - approach * math.cos(gyaw), gy - approach * math.sin(gyaw))
    if math.hypot(start_xy[0] - gx, start_xy[1] - gy) > approach + 0.1:
        pix, piy = (int(v) for v in grid.world_to_cell(*pre))
        if grid.in_bounds(pix, piy) and np.isfinite(cost[piy, pix]) and segment_clear(grid, cost, pre, (gx, gy)):
            head = plan(grid, cost, start_xy, pre, spacing)
            if head is not None:
                tail = densify(np.array([pre, (gx, gy)], dtype=float), spacing)
                return np.vstack([head, tail[1:]]), len(head) - 1
    pts = plan(grid, cost, start_xy, (gx, gy), spacing)
    return None if pts is None else (pts, None)
