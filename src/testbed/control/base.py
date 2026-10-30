"""Controller interface and the shared longitudinal (speed) controller."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..planning.path import Path
from ..vehicles.base import State, VehicleModel, VehicleParams, wrap_angle


class SpeedPID:
    """PI(D) speed controller tracking the path's reference speed.

        a = kp e_v + ki int(e_v) + kd de_v/dt,   e_v = v_ref - v
    Shared by every lateral controller so that comparisons isolate the steering law.
    The output is clipped to [a_min, a_max] and the integrator only runs while the output
    is not saturated (conditional-integration anti-windup).
    """

    def __init__(self, kp: float = 1.5, ki: float = 0.2, kd: float = 0.0, i_max: float = 5.0,
                 a_min: float = -5.0, a_max: float = 3.0):
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self.a_min, self.a_max = a_min, a_max
        self.reset()

    def reset(self) -> None:
        self._i = 0.0
        self._prev: float | None = None

    def __call__(self, v_ref: float, v: float, dt: float) -> float:
        e = v_ref - v
        d = 0.0 if self._prev is None or dt <= 0 else (e - self._prev) / dt
        self._prev = e
        i_new = float(np.clip(self._i + e * dt, -self.i_max, self.i_max))
        a = self.kp * e + self.ki * i_new + self.kd * d
        if self.a_min < a < self.a_max:
            self._i = i_new
        else:
            a = self.kp * e + self.ki * self._i + self.kd * d
        return float(np.clip(a, self.a_min, self.a_max))


class Controller(ABC):
    """A path-tracking controller maps (state, path) to (steering angle [rad], accel [m/s^2]).

    Call `bind(vehicle)` before use: it gives the controller the vehicle parameters and the
    offset from the state reference point to the rear axle (0 for the kinematic models,
    lr for the dynamic bicycle, whose state is at the CoG).
    """

    name: str = "base"

    def __init__(self, speed: SpeedPID | None = None):
        self.speed = speed or SpeedPID()
        self.p = VehicleParams()
        self.rear_offset = 0.0
        self.idx = 0            # progress along the path (nearest index of the reference point)

    def bind(self, vehicle: VehicleModel) -> "Controller":
        self.p = vehicle.p
        self.speed.a_min, self.speed.a_max = -vehicle.p.max_decel, vehicle.p.max_accel
        self.rear_offset = vehicle.p.lr if vehicle.name == "dynamic_bicycle" else 0.0
        return self

    def reset(self) -> None:
        self.idx = 0
        self.speed.reset()

    # ------------------------------------------------------------------ helpers
    def rear_axle(self, s: State) -> tuple[float, float]:
        return s.x - self.rear_offset * np.cos(s.yaw), s.y - self.rear_offset * np.sin(s.yaw)

    def front_axle(self, s: State) -> tuple[float, float]:
        rx, ry = self.rear_axle(s)
        return rx + self.p.wheelbase * np.cos(s.yaw), ry + self.p.wheelbase * np.sin(s.yaw)

    def track_errors(self, path: Path, x: float, y: float, yaw: float) -> tuple[int, float, float]:
        """Nearest index, signed cross-track error (positive = left) and heading error
        (vehicle yaw minus path yaw) for the point (x, y)."""
        i = path.nearest_index(x, y, start=self.idx)
        return i, path.cross_track_error(x, y, i), wrap_angle(yaw - path.yaw[i])

    def accel(self, s: State, path: Path, dt: float) -> float:
        return self.speed(float(path.v_ref[self.idx]), s.v, dt)

    def compute(self, state: State, path: Path, dt: float) -> tuple[float, float]:
        """Return (steer, accel). Updates the progress index first."""
        rx, ry = self.rear_axle(state)
        self.idx = path.nearest_index(rx, ry, start=self.idx)
        steer = float(np.clip(self.steer(state, path, dt), -self.p.max_steer, self.p.max_steer))
        return steer, self.accel(state, path, dt)

    @abstractmethod
    def steer(self, state: State, path: Path, dt: float) -> float:
        """Lateral control law: steering angle command [rad]."""
