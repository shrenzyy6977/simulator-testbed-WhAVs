"""Run the 2D testbed's path-tracking controllers on a MuJoCo robot driving over 3D terrain.

The controllers are unchanged. A skid-steer robot has no steering angle, so the controller's
steering output is turned into a yaw-rate command through a virtual wheelbase Lv, exactly as
the 2D DiffDrive model does:

    omega_cmd = v_cmd tan(delta) / Lv,      v_cmd = v_ref(s)

The physics then decides how much of that command the robot really achieves (wheel slip,
skid-steer scrubbing, slopes). Errors are measured in the horizontal (x, y) plane.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from testbed.control.base import Controller
from testbed.planning.path import Path
from testbed.sim.simulator import SimLog
from testbed.vehicles.base import State, VehicleParams, wrap_angle
from testbed.vehicles.models import DiffDrive

from .terrain import Terrain
from .world import RobotSpec, WheeledRobot


@dataclass
class TrackConfig:
    control_hz: float = 20.0
    virtual_wheelbase: float = 0.30   # Lv [m]
    max_steer_deg: float = 70.0       # caps the commanded curvature at tan(70 deg)/Lv
    max_cte: float = 2.0              # [m] abort threshold
    max_tilt_deg: float = 60.0        # roll/pitch beyond this counts as flipped
    t_max: float | None = None
    gps_noise: float = 0.0            # [m] std of the position fed to the controller
    delay_steps: int = 0              # control delay in control periods
    seed: int = 0


@dataclass
class TerrainLog(SimLog):
    z: list = field(default_factory=list)
    roll: list = field(default_factory=list)
    pitch: list = field(default_factory=list)
    slip: list = field(default_factory=list)
    omega_cmd: list = field(default_factory=list)
    omega: list = field(default_factory=list)
    ground: list = field(default_factory=list)       # ground-class name under the robot

    def arrays(self) -> dict[str, np.ndarray]:
        out = super().arrays()
        out.pop("ground", None)
        return out


def shift_path(path: Path, dx: float, dy: float) -> Path:
    return Path(path.x + dx, path.y + dy, path.yaw, path.k, path.s, path.v_ref, path.name)


def track(terrain: Terrain, controller: Controller, path: Path, cfg: TrackConfig | None = None,
          robot: RobotSpec | None = None) -> TerrainLog:
    """Drive `path` on `terrain` with `controller`; returns a log at the control rate."""
    cfg = cfg or TrackConfig()
    rob = WheeledRobot(terrain, robot, start=(path.x[0], path.y[0]), start_yaw=float(path.yaw[0]))
    rob.step(int(0.5 / rob.dt))                                   # settle on the ground
    params = VehicleParams(wheelbase=cfg.virtual_wheelbase, max_steer=np.deg2rad(cfg.max_steer_deg),
                           max_speed=rob.robot.max_wheel_speed * rob.robot.wheel_radius)
    controller.bind(DiffDrive(params)).reset()
    rng = np.random.default_rng(cfg.seed)
    n_sub = max(1, int(round(1.0 / (cfg.control_hz * rob.dt))))
    dt = n_sub * rob.dt
    t_max = cfg.t_max or 3.0 * path.length / max(float(np.mean(path.v_ref)), 0.2) + 10.0
    queue = [(0.0, 0.0)] * cfg.delay_steps
    log = TerrainLog(controller=controller.name, vehicle=rob.robot.urdf.stem, path=path.name)
    idx, t = 0, 0.0
    while True:
        x, y, z, roll, pitch, yaw = rob.pose()
        v, _, omega = rob.body_velocity()
        idx = path.nearest_index(x, y, start=idx)
        cte = path.cross_track_error(x, y, idx)
        mx, my = (x, y) if cfg.gps_noise <= 0 else (x + rng.normal(0, cfg.gps_noise), y + rng.normal(0, cfg.gps_noise))
        t0 = time.perf_counter()
        steer, _ = controller.compute(State(x=mx, y=my, yaw=yaw, v=v), path, dt)
        ct = time.perf_counter() - t0
        v_cmd = float(path.v_ref[controller.idx])
        w_cmd = v_cmd * np.tan(steer) / cfg.virtual_wheelbase
        queue.append((v_cmd, w_cmd))
        v_app, w_app = queue.pop(0)
        for k, val in (("t", t), ("x", x), ("y", y), ("yaw", yaw), ("v", v), ("v_ref", path.v_ref[idx]),
                       ("steer", steer), ("accel", 0.0), ("cte", cte), ("heading_err", wrap_angle(yaw - path.yaw[idx])),
                       ("s", path.s[idx]), ("compute_time", ct), ("z", z), ("roll", roll), ("pitch", pitch),
                       ("slip", rob.slip_ratio()), ("omega_cmd", w_cmd), ("omega", omega)):
            getattr(log, k).append(float(val))
        log.ground.append(terrain.ground_at(x, y).name)

        if idx >= len(path) - 2 or (np.hypot(x - path.x[-1], y - path.y[-1]) < 0.3 and idx > len(path) // 2):
            log.status = "finished"
            break
        if abs(cte) > cfg.max_cte:
            log.status = "off_track"
            break
        if max(abs(roll), abs(pitch)) > np.deg2rad(cfg.max_tilt_deg):
            log.status = "flipped"
            break
        if t >= t_max:
            log.status = "timeout"
            break
        rob.command(v_app, w_app)
        rob.step(n_sub)
        t += dt
    return log
