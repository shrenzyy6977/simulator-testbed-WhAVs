"""A* global planner on an occupancy grid, with path smoothing."""
from __future__ import annotations

import heapq

import numpy as np

from ..world.grid_map import GridMap
from .path import Path, path_from_points

_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8-connected A* with Euclidean heuristic. grid[row, col] True = blocked.
    Returns a list of (row, col) cells from start to goal, or None if unreachable."""
    H, W = grid.shape
    if grid[start] or grid[goal]:
        return None
    h = lambda c: float(np.hypot(c[0] - goal[0], c[1] - goal[1]))
    g = {start: 0.0}
    parent = {start: None}
    open_ = [(h(start), 0.0, start)]
    closed = set()
    while open_:
        _, gc, cur = heapq.heappop(open_)
        if cur in closed:
            continue
        if cur == goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = parent[cur]
            return out[::-1]
        closed.add(cur)
        for dr, dc, cost in _MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or grid[nb] or nb in closed:
                continue
            if dr and dc and (grid[cur[0] + dr, cur[1]] or grid[cur[0], cur[1] + dc]):
                continue  # no corner cutting
            ng = gc + cost
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_, (ng + h(nb), ng, nb))
    return None


def _line_free(grid: np.ndarray, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rr = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return not grid[rr, cc].any()


def shortcut(grid: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight pruning: keep only the waypoints needed to avoid obstacles."""
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(grid, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def densify(pts: np.ndarray, spacing: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def plan_path(world: GridMap, start_xy, goal_xy, clearance: float = 1.5, v_target: float = 5.0,
              a_lat_max: float = 3.0) -> Path:
    """Full global-planning pipeline:
    inflate obstacles -> A* -> line-of-sight shortcut -> densify -> cubic spline -> speed profile."""
    cmap = world.inflated(clearance)
    s = tuple(int(v) for v in cmap.world_to_cell(*start_xy))
    g = tuple(int(v) for v in cmap.world_to_cell(*goal_xy))
    cells = astar(cmap.grid, s, g)
    if cells is None:
        raise RuntimeError(f"A*: no path from {start_xy} to {goal_xy} with clearance {clearance} m")
    cells = shortcut(cmap.grid, cells)
    xs, ys = cmap.cell_to_world(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
    pts = np.column_stack([xs, ys])
    pts[0], pts[-1] = start_xy, goal_xy
    pts = densify(pts, spacing=3.0)
    # Laplacian smoothing of the polyline corners (endpoints fixed), then a cubic spline.
    # Smoothing cuts corners, so we verify the result against a slightly smaller inflated map
    # and fall back to less smoothing until the final path is collision-free.
    check = world.inflated(0.8 * clearance)
    for iters in (20, 10, 5, 2, 0):
        q = pts.copy()
        for _ in range(iters):
            q[1:-1] += 0.3 * (0.5 * (q[:-2] + q[2:]) - q[1:-1])
        path = path_from_points(q[:, 0], q[:, 1], v_target=v_target, a_lat_max=a_lat_max, name="astar")
        if all(check.is_free(x, y) for x, y in zip(path.x[::5], path.y[::5])):
            return path
    return path
