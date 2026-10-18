"""Simulated sensors: 2D LiDAR, wheel odometry and GPS, each with Gaussian noise."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vehicles.base import State, wrap_angle
from ..world.grid_map import GridMap


class Sensor:
    name = "sensor"
    rate_hz: float = 20.0

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(0)

    def read(self, world: GridMap | None, state: State, dt: float):
        raise NotImplementedError


@dataclass
class LidarScan:
    angles: np.ndarray   # beam angles in the world frame [rad]
    ranges: np.ndarray   # measured ranges [m]


class Lidar2D(Sensor):
    """Planar LiDAR by ray casting on the occupancy grid (like MVSim's 2D laser scanner)."""

    name = "lidar"

    def __init__(self, n_beams: int = 180, fov: float = 2 * np.pi, max_range: float = 20.0,
                 noise_std: float = 0.02, rng=None):
        super().__init__(rng)
        self.n_beams, self.fov, self.max_range, self.noise_std = n_beams, fov, max_range, noise_std
        self.rel_angles = np.linspace(-fov / 2, fov / 2, n_beams, endpoint=fov < 2 * np.pi)

    def read(self, world, state, dt=0.0) -> LidarScan:
        angles = state.yaw + self.rel_angles
        if world is None:
            return LidarScan(angles, np.full(self.n_beams, self.max_range))
        r = world.raycast(state.x, state.y, angles, self.max_range)
        r = np.clip(r + self.rng.normal(0, self.noise_std, r.shape), 0, self.max_range)
        return LidarScan(angles, r)


class Odometry(Sensor):
    """Dead-reckoning from noisy speed and yaw-rate. Drift grows over time, as on a real robot."""

    name = "odometry"

    def __init__(self, v_std: float = 0.05, yaw_rate_std: float = 0.01, rng=None):
        super().__init__(rng)
        self.v_std, self.yaw_rate_std = v_std, yaw_rate_std
        self.est: State | None = None
        self._prev: State | None = None

    def read(self, world, state, dt):
        if self.est is None:
            self.est, self._prev = state.copy(), state.copy()
            return self.est.copy()
        yaw_rate = wrap_angle(state.yaw - self._prev.yaw) / dt if dt > 0 else 0.0
        v = state.v + self.rng.normal(0, self.v_std)
        w = yaw_rate + self.rng.normal(0, self.yaw_rate_std)
        e = self.est
        e.yaw = wrap_angle(e.yaw + w * dt)
        e.x += v * np.cos(e.yaw) * dt
        e.y += v * np.sin(e.yaw) * dt
        e.v = v
        self._prev = state.copy()
        return e.copy()


class GPS(Sensor):
    """Absolute position with Gaussian noise (no drift)."""

    name = "gps"

    def __init__(self, pos_std: float = 0.3, rng=None):
        super().__init__(rng)
        self.pos_std = pos_std

    def read(self, world, state, dt=0.0):
        return np.array([state.x, state.y]) + self.rng.normal(0, self.pos_std, 2)


SENSORS = {"lidar": Lidar2D, "odometry": Odometry, "gps": GPS}


def make_sensor(name: str, **kw) -> Sensor:
    return SENSORS[name](**kw)
