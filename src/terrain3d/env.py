"""Gymnasium environment: ERC/URC-style checkpoint navigation over 3D terrain.

The robot must reach a sequence of GNSS checkpoints. Like the Earth Rover Challenge interface,
the policy only sees a noisy GPS fix, the heading, its own speeds and the last command, at a
fixed control rate and with a configurable control delay.

    action  a in [-1, 1]^2   ->  v = a0 v_max,  omega = a1 omega_max
    obs     [goal_x, goal_y] in the body frame from noisy GPS (m), distance (m),
            sin/cos of the bearing error, forward speed (m/s), yaw rate (rad/s),
            roll, pitch (rad), previous action (2)                                  -> 11 values
    reward  10 * (d_prev - d)  - 0.01 per step  + 10 per checkpoint  - 10 if flipped / out of map
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .terrain import GROUND_CLASSES, Terrain, make_terrain
from .world import RobotSpec, WheeledRobot


@dataclass
class NavConfig:
    terrain: dict = field(default_factory=lambda: {"size": 20.0, "n": 65,
                                                   "elevation": {"type": "hills", "amplitude": 0.15, "wavelength": 6.0}})
    grounds: list = field(default_factory=lambda: list(GROUND_CLASSES))   # sampled per episode
    tiles: int = 4                    # patches of different ground per map
    n_checkpoints: int = 3
    checkpoint_radius: float = 0.5    # [m]
    checkpoint_spacing: tuple = (3.0, 6.0)
    v_max: float = 1.0                # [m/s]
    omega_max: float = 2.0            # [rad/s]
    control_hz: float = 10.0
    delay_steps: int = 1              # control delay in control periods
    gps_noise: float = 0.3            # [m]
    max_steps: int = 600
    new_map_every: int = 10           # episodes between rebuilding the map (compiling costs ~ms)
    robot: RobotSpec = field(default_factory=RobotSpec)


class RoverNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: NavConfig | None = None, render_mode: str | None = None):
        self.cfg = config or NavConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.robot: WheeledRobot | None = None
        self._episodes = 0
        self._renderer = None

    # ------------------------------------------------------------------ helpers
    def _new_map(self) -> None:
        c = self.cfg
        tc = dict(c.terrain)
        tc["elevation"] = {**tc.get("elevation", {}), "seed": int(self.np_random.integers(1 << 30))}
        tc["ground"] = {"patches": c.grounds, "tiles": c.tiles, "seed": int(self.np_random.integers(1 << 30))}
        if tc["elevation"]["type"] == "flat":
            tc["elevation"].pop("seed")
        self.terrain: Terrain = make_terrain(tc)
        self.robot = WheeledRobot(self.terrain, c.robot, start=(0.0, 0.0))

    def _place(self) -> None:
        c, rob = self.cfg, self.robot
        half = self.terrain.size / 2 - 1.5
        start = self.np_random.uniform(-half, half, 2)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rob.reset()
        q = rob.data.qpos
        q[0:2] = start
        q[2] = self.terrain.height(*start) + c.robot.wheel_radius + 0.08
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        rob.step(int(0.3 / rob.dt))
        pts, p = [], start
        for _ in range(c.n_checkpoints):
            for _ in range(50):
                ang = self.np_random.uniform(-np.pi, np.pi)
                cand = p + self.np_random.uniform(*c.checkpoint_spacing) * np.array([np.cos(ang), np.sin(ang)])
                if np.all(np.abs(cand) < half):
                    break
            pts.append(cand)
            p = cand
        self.checkpoints = np.array(pts)

    def _gps(self) -> np.ndarray:
        x, y, *_ = self.robot.pose()
        return np.array([x, y]) + self.np_random.normal(0, self.cfg.gps_noise, 2)

    def _obs(self) -> np.ndarray:
        rob = self.robot
        _, _, _, roll, pitch, yaw = rob.pose()
        v, _, omega = rob.body_velocity()
        d = self.checkpoints[self.target] - self._gps()
        c, s = np.cos(yaw), np.sin(yaw)
        gx, gy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        bearing = np.arctan2(gy, gx)
        return np.array([gx, gy, np.hypot(gx, gy), np.sin(bearing), np.cos(bearing), v, omega, roll, pitch,
                         *self.last_action], dtype=np.float32)

    def _true_dist(self) -> float:
        x, y, *_ = self.robot.pose()
        return float(np.hypot(*(self.checkpoints[self.target] - [x, y])))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.robot is None or seed is not None or self._episodes % self.cfg.new_map_every == 0:
            self._new_map()            # a seeded reset always rebuilds, so it is reproducible
        self._episodes += 1
        self._place()
        self.target, self.steps = 0, 0
        self.last_action = np.zeros(2)
        self.queue = [np.zeros(2)] * self.cfg.delay_steps
        self.prev_dist = self._true_dist()
        return self._obs(), {"ground": self.robot.ground().name}

    def step(self, action):
        c, rob = self.cfg, self.robot
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.queue.append(a)
        applied = self.queue.pop(0)
        rob.command(applied[0] * c.v_max, applied[1] * c.omega_max)
        rob.step(max(1, int(round(1.0 / (c.control_hz * rob.dt)))))
        self.last_action, self.steps = a, self.steps + 1

        d = self._true_dist()
        reward = 10.0 * (self.prev_dist - d) - 0.01
        terminated = False
        if d < c.checkpoint_radius:
            reward += 10.0
            self.target += 1
            if self.target == len(self.checkpoints):
                terminated = True
                self.target -= 1
            d = self._true_dist()
        self.prev_dist = d
        x, y, _, roll, pitch, _ = rob.pose()
        failed = max(abs(roll), abs(pitch)) > np.deg2rad(60) or max(abs(x), abs(y)) > self.terrain.size / 2 - 0.3
        if failed:
            reward -= 10.0
            terminated = True
        truncated = self.steps >= c.max_steps
        info = {"ground": rob.ground().name, "checkpoints_reached": self.target + int(terminated and not failed),
                "slip": rob.slip_ratio(), "success": bool(terminated and not failed)}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.robot.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.robot.data.xpos[self.robot.body_id]
        cam.distance, cam.elevation, cam.azimuth = 4.0, -30.0, 135.0
        self._renderer.update_scene(self.robot.data, cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def register() -> None:
    if "terrain3d/RoverNav-v0" not in gym.registry:
        gym.register("terrain3d/RoverNav-v0", entry_point=RoverNavEnv)
