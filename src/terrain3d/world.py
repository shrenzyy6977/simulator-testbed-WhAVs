"""Assemble a MuJoCo world (terrain + any wheeled robot loaded from URDF) and drive it.

The robot is loaded with MjSpec from its URDF, attached under a free joint, and every wheel
joint gets a velocity actuator (like the velocity-controlled hub motors of small rovers).
Wheel joints are found by name (`wheel_<side>...`): names containing `l` / `r` after `wheel_`
choose the side for skid steering, e.g. wheel_fl_joint, wheel_rl_joint, wheel_left_joint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .terrain import Terrain

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "robots" / "rover4wd.urdf"


@dataclass
class RobotSpec:
    urdf: Path = DEFAULT_URDF
    base_link: str = "base_link"
    wheel_radius: float = 0.065     # [m]
    track: float = 0.30             # distance between left and right wheels [m]
    kv: float = 2.0                 # velocity actuator gain [N m s/rad]
    max_torque: float = 2.0         # [N m] per wheel
    max_wheel_speed: float = 25.0   # [rad/s]


def _wheel_side(joint_name: str) -> int | None:
    """+1 for a left wheel, -1 for a right wheel, None if the joint is not a wheel."""
    name = joint_name.lower()
    if "wheel" not in name:
        return None
    if "left" in name:
        return +1
    if "right" in name:
        return -1
    tag = re.search(r"wheel_?([a-z]*)", name).group(1)      # e.g. fl, rr, ml
    if tag.endswith("l"):
        return +1
    if tag.endswith("r"):
        return -1
    return None


def build_model(terrain: Terrain, robot: RobotSpec | None = None, start=(0.0, 0.0), start_yaw: float = 0.0,
                timestep: float = 0.002) -> tuple[mujoco.MjModel, list[tuple[str, int]]]:
    """Compile terrain + robot. Returns the model and [(wheel actuator name, side)], side = +1 left, -1 right."""
    robot = robot or RobotSpec()
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.worldbody.add_light(pos=[0, 0, 30], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    terrain.add_to_spec(spec)

    rspec = mujoco.MjSpec.from_file(str(robot.urdf))
    joint_names = [j.name for j in rspec.joints]          # read before attach (attach renames)
    z = terrain.height(*start) + robot.wheel_radius + 0.08
    frame = spec.worldbody.add_frame(pos=[start[0], start[1], z],
                                     quat=[np.cos(start_yaw / 2), 0, 0, np.sin(start_yaw / 2)])
    base = frame.attach_body(rspec.body(robot.base_link), "robot/", "")
    base.add_freejoint(name="robot/root")
    base.add_site(name="robot/imu", pos=[0, 0, 0])

    wheels = []
    for jn in joint_names:
        side = _wheel_side(jn)
        if side is None:
            continue
        a = spec.add_actuator(name=f"robot/act_{jn}", target=f"robot/{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=robot.kv)
        a.forcelimited = 1
        a.forcerange = [-robot.max_torque, robot.max_torque]
        wheels.append((a.name, side))
    if not wheels:
        raise ValueError(f"no wheel joints (named wheel_*) found in {robot.urdf}")
    spec.add_sensor(name="robot/gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="robot/imu")
    spec.add_sensor(name="robot/accel", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="robot/imu")
    return spec.compile(), wheels


class WheeledRobot:
    """Runtime handle: skid-steer commands in, planar pose / velocities / wheel slip out."""

    def __init__(self, terrain: Terrain, robot: RobotSpec | None = None, start=(0.0, 0.0), start_yaw: float = 0.0,
                 timestep: float = 0.002):
        self.terrain, self.robot = terrain, robot or RobotSpec()
        self.model, wheels = build_model(terrain, self.robot, start, start_yaw, timestep)
        self.data = mujoco.MjData(self.model)
        self.act_ids = np.array([self.model.actuator(n).id for n, _ in wheels])
        self.sides = np.array([s for _, s in wheels], dtype=float)
        self.wheel_dof = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in self.act_ids])
        self.body_id = self.model.body("robot/" + self.robot.base_link).id
        self.qpos0 = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos0
        mujoco.mj_forward(self.model, self.data)

    def command(self, v: float, omega: float) -> None:
        """Skid-steer mixing: left = (v - omega track/2)/r, right = (v + omega track/2)/r, saturated."""
        r = self.robot
        w = (v - self.sides * omega * r.track / 2) / r.wheel_radius
        self.data.ctrl[self.act_ids] = np.clip(w, -r.max_wheel_speed, r.max_wheel_speed)

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ------------------------------------------------------------------ state
    def pose(self) -> tuple[float, float, float, float, float, float]:
        """x, y, z, roll, pitch, yaw of the base."""
        x, y, z = self.data.xpos[self.body_id]
        R = self.data.xmat[self.body_id].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)

    def body_velocity(self) -> tuple[float, float, float]:
        """Forward speed, lateral speed (body frame) and yaw rate."""
        # flg_local=1 would use the body's principal inertia axes, not the body frame,
        # so take the world-frame velocity and rotate it with xmat.
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, v, 0)
        R = self.data.xmat[self.body_id].reshape(3, 3)
        lin, ang = R.T @ v[3:], R.T @ v[:3]
        return float(lin[0]), float(lin[1]), float(ang[2])

    def wheel_speeds(self) -> np.ndarray:
        return self.data.qvel[self.wheel_dof].copy()

    def slip_ratio(self) -> float:
        """Longitudinal slip, 1 - v / (mean |wheel speed| r); 0 = pure rolling, 1 = spinning in place."""
        v, _, _ = self.body_velocity()
        w = float(np.mean(np.abs(self.wheel_speeds()))) * self.robot.wheel_radius
        return float(np.clip(1.0 - abs(v) / w, -1.0, 1.0)) if w > 0.05 else 0.0

    def ground(self):
        x, y, *_ = self.pose()
        return self.terrain.ground_at(x, y)
