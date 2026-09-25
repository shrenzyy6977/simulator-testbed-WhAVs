"""Reference path representation, standard test paths, and speed profiling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from ..vehicles.base import wrap_angle


@dataclass
class Path:
    """Densely sampled reference path. All arrays have the same length N."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    k: np.ndarray        # signed curvature [1/m]
    s: np.ndarray        # arc length [m]
    v_ref: np.ndarray    # reference speed [m/s]
    name: str = "path"

    def __len__(self) -> int:
        return len(self.x)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def nearest_index(self, x: float, y: float, start: int = 0, window: int | None = 200) -> int:
        """Index of the closest path point. Searching forward from `start` inside a window
        prevents jumping to a different lap on closed or self-crossing paths."""
        end = len(self) if window is None else min(len(self), start + window)
        start = max(0, min(start, len(self) - 1))
        d2 = (self.x[start:end] - x) ** 2 + (self.y[start:end] - y) ** 2
        return start + int(np.argmin(d2))

    def cross_track_error(self, x: float, y: float, idx: int) -> float:
        """Signed lateral error (positive = vehicle left of the path)."""
        dx, dy = x - self.x[idx], y - self.y[idx]
        return float(-np.sin(self.yaw[idx]) * dx + np.cos(self.yaw[idx]) * dy)

    def index_ahead(self, idx: int, distance: float) -> int:
        target = self.s[idx] + distance
        return int(min(np.searchsorted(self.s, target), len(self) - 1))


def path_from_points(px, py, ds: float = 0.1, v_target: float = 5.0, a_lat_max: float = 3.0,
                     a_long_max: float = 2.0, name: str = "path", smooth: bool = True) -> Path:
    """Fit a cubic spline through waypoints (parameterised by arc length), resample every ds
    metres, and compute yaw, curvature and a feasible speed profile."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    keep = np.r_[True, np.hypot(np.diff(px), np.diff(py)) > 1e-6]
    px, py = px[keep], py[keep]
    seg = np.hypot(np.diff(px), np.diff(py))
    s_pts = np.r_[0.0, np.cumsum(seg)]
    if smooth and len(px) >= 3:
        sx, sy = CubicSpline(s_pts, px), CubicSpline(s_pts, py)
        s = np.arange(0.0, s_pts[-1], ds)
        x, y = sx(s), sy(s)
        dx, dy, ddx, ddy = sx(s, 1), sy(s, 1), sx(s, 2), sy(s, 2)
    else:
        s = np.arange(0.0, s_pts[-1], ds)
        x, y = np.interp(s, s_pts, px), np.interp(s, s_pts, py)
        dx, dy = np.gradient(x, s), np.gradient(y, s)
        ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    yaw = np.arctan2(dy, dx)
    k = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    # recompute exact arc length of the resampled curve
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    v = speed_profile(s, k, v_target, a_lat_max, a_long_max)
    return Path(x, y, yaw, k, s, v, name)


def speed_profile(s, k, v_target, a_lat_max=3.0, a_long_max=2.0, v_end=0.0) -> np.ndarray:
    """v(s) = min(v_target, sqrt(a_lat/|k|)), then forward/backward passes to respect
    acceleration limits and stop at the end of the path."""
    v = np.minimum(v_target, np.sqrt(a_lat_max / np.maximum(np.abs(k), 1e-6)))
    v[-1] = v_end
    for i in range(len(v) - 2, -1, -1):          # braking pass
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long_max * (s[i + 1] - s[i])))
    return v


# --------------------------------------------------------------------------- test paths
def straight(length=80.0, **kw) -> Path:
    x = np.linspace(0, length, 50)
    return path_from_points(x, np.zeros_like(x), name="straight", smooth=False, **kw)


def circle(radius=20.0, **kw) -> Path:
    t = np.linspace(-np.pi / 2, 1.5 * np.pi - 1e-3, 400)   # one full lap starting at (0, -R) heading +x
    return path_from_points(radius * np.cos(t), radius * np.sin(t) + radius, name="circle", **kw)


def s_curve(length=120.0, amplitude=8.0, wavelength=60.0, **kw) -> Path:
    x = np.linspace(0, length, 400)
    return path_from_points(x, amplitude * np.sin(2 * np.pi * x / wavelength), name="s_curve", **kw)


def lane_change(length=150.0, width=3.5, **kw) -> Path:
    """Double lane change (ISO 3888-style): move left by `width`, hold, come back."""
    x = np.linspace(0, length, 600)
    shape = lambda x0: 0.5 * (1 + np.tanh((x - x0) / 5.0))
    y = width * (shape(40.0) - shape(95.0))
    return path_from_points(x, y, name="lane_change", **kw)


TEST_PATHS = {"straight": straight, "circle": circle, "s_curve": s_curve, "lane_change": lane_change}


def make_path(name: str, **kw) -> Path:
    return TEST_PATHS[name](**kw)


__all__ = ["Path", "path_from_points", "speed_profile", "make_path", "TEST_PATHS", "wrap_angle"]
