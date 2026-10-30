"""Geometric path-tracking controllers: Pure Pursuit and Stanley."""
from __future__ import annotations

import numpy as np

from ..planning.path import Path
from ..vehicles.base import State, wrap_angle
from .base import Controller, SpeedPID


class PurePursuit(Controller):
    """Pure Pursuit (Coulter 1992), applied at the rear axle.

        Ld    = k v + Ld0                          (speed-scaled look-ahead)
        alpha = atan2(ty - ry, tx - rx) - yaw      (angle to the look-ahead point)
        delta = atan(2 L sin(alpha) / Ld)
    """

    name = "pure_pursuit"

    def __init__(self, k: float = 0.5, ld_min: float = 2.0, ld_max: float = 20.0, speed: SpeedPID | None = None):
        super().__init__(speed)
        self.k, self.ld_min, self.ld_max = k, ld_min, ld_max

    def steer(self, s: State, path: Path, dt: float) -> float:
        rx, ry = self.rear_axle(s)
        ld = float(np.clip(self.k * s.v + self.ld_min, self.ld_min, self.ld_max))
        j = path.index_ahead(self.idx, ld)
        alpha = wrap_angle(np.arctan2(path.y[j] - ry, path.x[j] - rx) - s.yaw)
        ld_actual = max(np.hypot(path.x[j] - rx, path.y[j] - ry), 1e-3)   # shorter near the path end
        return float(np.arctan2(2.0 * self.p.wheelbase * np.sin(alpha), ld_actual))


class Stanley(Controller):
    """Stanley (Thrun et al. 2006), applied at the front axle.

        delta = -theta_e - atan(k e / (ks + v))
    theta_e = yaw - path_yaw and e = cross-track error of the front axle (positive = left).
    ks keeps the gain finite at low speed.
    """

    name = "stanley"

    def __init__(self, k: float = 1.0, ks: float = 1.0, speed: SpeedPID | None = None):
        super().__init__(speed)
        self.k, self.ks = k, ks
        self._fidx = 0

    def reset(self) -> None:
        super().reset()
        self._fidx = 0

    def steer(self, s: State, path: Path, dt: float) -> float:
        fx, fy = self.front_axle(s)
        self._fidx = path.nearest_index(fx, fy, start=max(self._fidx, self.idx))
        e = path.cross_track_error(fx, fy, self._fidx)
        theta_e = wrap_angle(s.yaw - path.yaw[self._fidx])
        return float(-theta_e - np.arctan2(self.k * e, self.ks + max(s.v, 0.0)))
