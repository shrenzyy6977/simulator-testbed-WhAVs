"""2D occupancy-grid world: load from YAML shapes or a PNG, collision checks, ray casting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class GridMap:
    """Occupancy grid. grid[row, col] is True when the cell is occupied.

    World (x, y) in metres maps to col = (x - origin_x)/res, row = (y - origin_y)/res.
    """

    def __init__(self, grid: np.ndarray, resolution: float = 0.25, origin=(0.0, 0.0)):
        self.grid = grid.astype(bool)
        self.res = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.height, self.width = self.grid.shape

    # ------------------------------------------------------------------ builders
    @classmethod
    def empty(cls, width_m: float, height_m: float, resolution: float = 0.25, origin=(0.0, 0.0)) -> "GridMap":
        g = np.zeros((int(round(height_m / resolution)), int(round(width_m / resolution))), dtype=bool)
        return cls(g, resolution, origin)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GridMap":
        """YAML format:
            width: 60   height: 40   resolution: 0.25   origin: [0, 0]
            border: true
            obstacles:
              - {type: rect, x: 10, y: 5, w: 4, h: 12}      # lower-left corner + size
              - {type: circle, x: 30, y: 20, r: 3}
            image: optional PNG path (dark pixels = occupied), overrides shapes
        """
        path = Path(path)
        cfg = yaml.safe_load(path.read_text())
        if "image" in cfg:
            return cls.from_png(path.parent / cfg["image"], cfg.get("resolution", 0.25), cfg.get("origin", (0, 0)))
        m = cls.empty(cfg["width"], cfg["height"], cfg.get("resolution", 0.25), cfg.get("origin", (0, 0)))
        if cfg.get("border", True):
            m.grid[0, :] = m.grid[-1, :] = m.grid[:, 0] = m.grid[:, -1] = True
        for ob in cfg.get("obstacles", []):
            m.add_obstacle(ob)
        return m

    @classmethod
    def from_png(cls, path: str | Path, resolution: float = 0.25, origin=(0.0, 0.0)) -> "GridMap":
        import matplotlib.image as mpimg

        img = mpimg.imread(str(path))
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        if img.max() > 1.0:
            img = img / 255.0
        return cls(np.flipud(img < 0.5), resolution, origin)  # image row 0 is the top

    def add_obstacle(self, ob: dict) -> None:
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        cx = self.origin[0] + (xs + 0.5) * self.res
        cy = self.origin[1] + (ys + 0.5) * self.res
        if ob["type"] == "rect":
            mask = (cx >= ob["x"]) & (cx <= ob["x"] + ob["w"]) & (cy >= ob["y"]) & (cy <= ob["y"] + ob["h"])
        elif ob["type"] == "circle":
            mask = (cx - ob["x"]) ** 2 + (cy - ob["y"]) ** 2 <= ob["r"] ** 2
        else:
            raise ValueError(f"Unknown obstacle type {ob['type']}")
        self.grid |= mask

    # ------------------------------------------------------------------ queries
    @property
    def extent(self) -> tuple[float, float, float, float]:
        x0, y0 = self.origin
        return x0, x0 + self.width * self.res, y0, y0 + self.height * self.res

    def world_to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int)
        return row, col

    def cell_to_world(self, row, col):
        return (self.origin[0] + (np.asarray(col) + 0.5) * self.res,
                self.origin[1] + (np.asarray(row) + 0.5) * self.res)

    def in_bounds(self, row, col):
        return (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)

    def is_free(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        return bool(self.in_bounds(r, c) and not self.grid[r, c])

    def collides(self, x: float, y: float, yaw: float, length: float, width: float) -> bool:
        """Check the vehicle footprint (rectangle centred on (x, y)) against the grid."""
        ls = np.linspace(-length / 2, length / 2, max(3, int(length / self.res) + 1))
        ws = np.linspace(-width / 2, width / 2, max(3, int(width / self.res) + 1))
        L, W = np.meshgrid(ls, ws)
        px = x + L * np.cos(yaw) - W * np.sin(yaw)
        py = y + L * np.sin(yaw) + W * np.cos(yaw)
        r, c = self.world_to_cell(px.ravel(), py.ravel())
        inside = self.in_bounds(r, c)
        if not inside.all():
            return True
        return bool(self.grid[r, c].any())

    def inflated(self, radius_m: float) -> "GridMap":
        """Return a copy with obstacles grown by radius_m (configuration-space for planning)."""
        from scipy.ndimage import binary_dilation

        k = int(np.ceil(radius_m / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = xx ** 2 + yy ** 2 <= k ** 2
        return GridMap(binary_dilation(self.grid, structure=disk), self.res, self.origin)

    def raycast(self, x: float, y: float, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Vectorised ray casting. Returns the range to the first occupied cell for each angle
        (max_range if nothing is hit). Samples every res/2 along each ray."""
        step = self.res / 2
        dists = np.arange(step, max_range + step, step)
        px = x + np.outer(np.cos(angles), dists)
        py = y + np.outer(np.sin(angles), dists)
        r, c = self.world_to_cell(px, py)
        inside = self.in_bounds(r, c)
        hit = np.ones_like(inside)          # out of map counts as a hit (wall)
        hit[inside] = self.grid[r[inside], c[inside]]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        out = np.full(len(angles), max_range)
        ok = first >= 0
        out[ok] = dists[first[ok]]
        return np.minimum(out, max_range)
