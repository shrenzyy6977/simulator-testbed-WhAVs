"""Closed-loop simulation: vehicle + controller + reference path (+ optional world)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..control.base import Controller
from ..planning.path import Path
from ..vehicles.base import State, VehicleModel, wrap_angle
from ..world.grid_map import GridMap


@dataclass
class SimConfig:
    dt: float = 0.02               # integration and control step [s]
    t_max: float | None = None     # default: 3x the time needed at the mean reference speed, + 20 s
    goal_tol: float = 1.0          # [m] distance to the last path point that counts as arrived
    max_cte: float = 5.0           # [m] abort when the vehicle is this far from the path
    init_offset: float = 0.0       # [m] initial lateral offset (positive = left)
    init_yaw_offset: float = 0.0   # [rad] initial heading offset
    init_speed: float = 0.0        # [m/s]


@dataclass
class SimLog:
    """Time series of one run. Errors are measured at the vehicle state reference point."""

    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    v: list = field(default_factory=list)
    v_ref: list = field(default_factory=list)
    steer: list = field(default_factory=list)
    accel: list = field(default_factory=list)
    cte: list = field(default_factory=list)
    heading_err: list = field(default_factory=list)
    s: list = field(default_factory=list)
    compute_time: list = field(default_factory=list)   # wall-clock time of controller.compute [s]
    status: str = "running"                            # finished | off_track | collision | timeout
    controller: str = ""
    vehicle: str = ""
    path: str = ""

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items() if isinstance(v, list)}


def initial_state(path: Path, cfg: SimConfig) -> State:
    n = np.array([-np.sin(path.yaw[0]), np.cos(path.yaw[0])])
    return State(x=path.x[0] + cfg.init_offset * n[0], y=path.y[0] + cfg.init_offset * n[1],
                 yaw=wrap_angle(path.yaw[0] + cfg.init_yaw_offset), v=cfg.init_speed)


def simulate(vehicle: VehicleModel, controller: Controller, path: Path, cfg: SimConfig | None = None,
             world: GridMap | None = None) -> SimLog:
    """Run the closed loop until the vehicle reaches the end of the path, leaves it, collides
    or times out. Each step: measure errors -> controller.compute (timed) -> vehicle.step."""
    cfg = cfg or SimConfig()
    t_max = cfg.t_max or 3.0 * path.length / max(float(np.mean(path.v_ref)), 0.5) + 20.0
    vehicle.reset()
    controller.bind(vehicle).reset()
    s = initial_state(path, cfg)
    log = SimLog(controller=controller.name, vehicle=vehicle.name, path=path.name)
    idx, t = 0, 0.0
    while True:
        idx = path.nearest_index(s.x, s.y, start=idx)
        cte = path.cross_track_error(s.x, s.y, idx)
        t0 = time.perf_counter()
        steer, accel = controller.compute(s, path, cfg.dt)
        ct = time.perf_counter() - t0
        for k, val in (("t", t), ("x", s.x), ("y", s.y), ("yaw", s.yaw), ("v", s.v), ("v_ref", path.v_ref[idx]),
                       ("steer", vehicle.steer), ("accel", accel), ("cte", cte),
                       ("heading_err", wrap_angle(s.yaw - path.yaw[idx])), ("s", path.s[idx]), ("compute_time", ct)):
            getattr(log, k).append(float(val))

        if idx >= len(path) - 2 or np.hypot(s.x - path.x[-1], s.y - path.y[-1]) < cfg.goal_tol and idx > len(path) // 2:
            log.status = "finished"
            break
        if abs(cte) > cfg.max_cte:
            log.status = "off_track"
            break
        if world is not None and world.collides(s.x, s.y, s.yaw, vehicle.p.length, vehicle.p.width):
            log.status = "collision"
            break
        if t >= t_max:
            log.status = "timeout"
            break
        s = vehicle.step(s, steer, accel, cfg.dt)
        t += cfg.dt
    return log
