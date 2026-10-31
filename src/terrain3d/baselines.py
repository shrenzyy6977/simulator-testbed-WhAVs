"""Classical baseline policies for RoverNavEnv, with the same predict() interface as SB3 models."""
from __future__ import annotations

import numpy as np

from testbed.control.base import Controller
from testbed.planning.path import path_from_points
from testbed.vehicles.base import State, VehicleParams, wrap_angle
from testbed.vehicles.models import DiffDrive


class GoToGoal:
    """Proportional bearing controller on the (noisy) GPS-relative goal, the usual classical
    baseline for GNSS waypoint missions:

        omega = k_w * bearing                      (bearing = atan2(goal_y, goal_x) in the body frame)
        v     = max(cos(bearing), v_min)           (slow down while turning towards the goal)
    Both are in the env's normalised action units ([-1, 1], scaled by v_max / omega_max).
    """

    def __init__(self, k_w: float = 2.0, v_min: float = 0.2, omega_max: float = 2.0):
        self.k_w, self.v_min, self.omega_max = k_w, v_min, omega_max

    def predict(self, obs, deterministic: bool = True):
        sin_b, cos_b = float(obs[3]), float(obs[4])
        bearing = np.arctan2(sin_b, cos_b)
        v = max(cos_b, self.v_min)
        omega = np.clip(self.k_w * bearing / self.omega_max, -1.0, 1.0)
        return np.array([v, omega], dtype=np.float32), None


class PathTracker:
    """A 2D-testbed path-tracking controller flying a checkpoint mission one leg at a time: the
    path is the straight line from the latest GPS fix to the next checkpoint, and the controller
    tracks it from the latest fix (stale between fixes, like the observation) and the heading. The steering
    output becomes a yaw-rate command through a virtual wheelbase Lv, as in `tracking.track`, and
    the speed is regulated by the heading error e = psi - psi_path (as skid-steer path followers
    such as Nav2's Regulated Pure Pursuit do):

        v = v_ref(s) max(cos e, v_min),      omega = v_ref(s) tan(delta) / Lv      if |e| <= e_rot
        v = 0,                               omega = -sign(e) omega_max            otherwise (rotate on the spot)
    The leg is re-planned from the current fix when a checkpoint is reached or when the leg ends
    without reaching it. The fix and heading are the same information the
    observation carries (the heading follows from the body-frame goal vector and the fix), so this
    is not a privileged policy.
    """

    def __init__(self, controller: Controller, v_target: float = 1.0, a_lat_max: float = 1.0,
                 v_min: float = 0.2, rotate_above_deg: float = 90.0, virtual_wheelbase: float = 0.30,
                 max_steer_deg: float = 70.0):
        self.ctrl, self.v_target, self.a_lat_max, self.v_min = controller, v_target, a_lat_max, v_min
        self.e_rot = np.deg2rad(rotate_above_deg)
        self.lv, self.max_steer = virtual_wheelbase, np.deg2rad(max_steer_deg)
        self.env = None

    def reset(self, env) -> "PathTracker":
        """Call after env.reset(): plans the first path of the episode."""
        self.env = env
        self.ctrl.bind(DiffDrive(VehicleParams(wheelbase=self.lv, max_steer=self.max_steer, max_speed=env.cfg.v_max)))
        self._plan()
        return self

    def _plan(self) -> None:
        env = self.env
        start, goal = env.fix, env.checkpoints[env.target]
        if np.hypot(*(goal - start)) < 0.2:             # fix already on the checkpoint: keep a short leg
            start = goal - [0.2, 0.0]
        pts = np.vstack([start, goal])
        self.path = path_from_points(pts[:, 0], pts[:, 1], ds=0.05, v_target=self.v_target,
                                     a_lat_max=self.a_lat_max, name="checkpoints", smooth=False)
        self.ctrl.reset()
        self.planned_for = env.target

    def predict(self, obs, deterministic: bool = True):
        env = self.env
        if env.target != self.planned_for or self.ctrl.idx >= len(self.path) - 2:
            self._plan()
        *_, yaw = env.robot.pose()
        x, y = env.fix
        steer, _ = self.ctrl.compute(State(x=x, y=y, yaw=yaw, v=float(obs[5])), self.path, 1.0 / env.cfg.control_hz)
        i = self.ctrl.idx
        e = wrap_angle(yaw - self.path.yaw[i])
        if abs(e) > self.e_rot:
            return np.array([0.0, -np.sign(e)], dtype=np.float32), None
        v_ref = max(float(self.path.v_ref[i]), self.v_min)      # the path ends at v = 0; keep creeping in
        v = v_ref * max(np.cos(e), self.v_min)
        omega = v_ref * np.tan(steer) / self.lv                 # turn rate from the path speed, so it can turn on the spot
        return np.array([v / env.cfg.v_max, np.clip(omega / env.cfg.omega_max, -1, 1)], dtype=np.float32), None
