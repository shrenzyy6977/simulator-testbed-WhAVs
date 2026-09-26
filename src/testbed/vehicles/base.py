"""Common vehicle state and the abstract VehicleModel interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class State:
    """Planar vehicle state. Units: m, rad, m/s, rad/s.

    x, y   : position of the rear-axle centre (kinematic models) or CoG (dynamic model)
    yaw    : heading angle
    v      : longitudinal speed
    vy, r  : lateral speed and yaw rate (only used by the dynamic model)
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    vy: float = 0.0
    r: float = 0.0

    def copy(self) -> "State":
        return State(self.x, self.y, self.yaw, self.v, self.vy, self.r)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.vy, self.r])


@dataclass
class VehicleParams:
    """Physical and actuator limits. Defaults approximate a small passenger car."""

    wheelbase: float = 2.5          # L [m]
    width: float = 1.6              # [m] (drawing / collision)
    length: float = 3.8             # [m] (drawing)
    max_steer: float = np.deg2rad(35.0)
    max_steer_rate: float = np.deg2rad(60.0)   # [rad/s]
    max_accel: float = 3.0          # [m/s^2]
    max_decel: float = 5.0          # [m/s^2]
    max_speed: float = 15.0         # [m/s]
    # dynamic bicycle only
    mass: float = 1500.0            # [kg]
    Iz: float = 2250.0              # [kg m^2]
    lf: float = 1.2                 # CoG -> front axle [m]
    lr: float = 1.3                 # CoG -> rear axle [m]
    Cf: float = 80000.0             # front cornering stiffness [N/rad]
    Cr: float = 80000.0             # rear cornering stiffness [N/rad]
    # differential drive only
    track: float = 1.6              # distance between wheels [m]
    max_wheel_speed: float = 16.0   # [m/s]


class VehicleModel(ABC):
    """A vehicle model integrates a State forward given (steer, accel) commands.

    All models take the same command (steering angle [rad], acceleration [m/s^2])
    so any controller can drive any vehicle.
    """

    name: str = "base"

    def __init__(self, params: VehicleParams | None = None):
        self.p = params or VehicleParams()
        self.steer = 0.0  # actual (rate-limited) steering angle

    def reset(self) -> None:
        self.steer = 0.0

    def _apply_limits(self, steer_cmd: float, accel_cmd: float, dt: float) -> tuple[float, float]:
        p = self.p
        steer_cmd = float(np.clip(steer_cmd, -p.max_steer, p.max_steer))
        max_d = p.max_steer_rate * dt
        self.steer += float(np.clip(steer_cmd - self.steer, -max_d, max_d))
        accel = float(np.clip(accel_cmd, -p.max_decel, p.max_accel))
        return self.steer, accel

    def step(self, s: State, steer_cmd: float, accel_cmd: float, dt: float) -> State:
        delta, a = self._apply_limits(steer_cmd, accel_cmd, dt)
        new = self._integrate(s, delta, a, dt)
        new.v = float(np.clip(new.v, 0.0, self.p.max_speed))
        new.yaw = wrap_angle(new.yaw)
        return new

    @abstractmethod
    def derivatives(self, s: np.ndarray, delta: float, a: float) -> np.ndarray:
        """Return d/dt of the state vector [x, y, yaw, v, vy, r]."""

    def _integrate(self, s: State, delta: float, a: float, dt: float) -> State:
        """4th-order Runge-Kutta integration."""
        x = s.as_array()
        k1 = self.derivatives(x, delta, a)
        k2 = self.derivatives(x + 0.5 * dt * k1, delta, a)
        k3 = self.derivatives(x + 0.5 * dt * k2, delta, a)
        k4 = self.derivatives(x + dt * k3, delta, a)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return State(*x)
