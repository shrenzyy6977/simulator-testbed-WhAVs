"""Three wheeled-vehicle models: kinematic bicycle, differential drive, dynamic bicycle."""
from __future__ import annotations

import numpy as np

from .base import State, VehicleModel, VehicleParams


class KinematicBicycle(VehicleModel):
    """Ackermann car as a kinematic bicycle (reference point = rear axle).

        x' = v cos(yaw)    y' = v sin(yaw)    yaw' = v/L tan(delta)    v' = a
    Valid at low/moderate speed where tyre slip is negligible (Liu et al. 2021).
    """

    name = "kinematic_bicycle"

    def derivatives(self, s, delta, a):
        _, _, yaw, v, _, _ = s
        L = self.p.wheelbase
        return np.array([v * np.cos(yaw), v * np.sin(yaw), v / L * np.tan(delta), a, 0.0, 0.0])


class DiffDrive(VehicleModel):
    """Differential-drive (unicycle) robot.

    To share the controller interface, the steering command is mapped to a yaw
    rate through a virtual wheelbase: omega = v/L tan(delta). Wheel speeds
    v_l, v_r = v -/+ omega*track/2 are then saturated at max_wheel_speed,
    which is how a real diff-drive robot limits turning.
    """

    name = "diff_drive"

    def derivatives(self, s, delta, a):
        _, _, yaw, v, _, _ = s
        p = self.p
        omega = v / p.wheelbase * np.tan(delta)
        vl, vr = v - omega * p.track / 2, v + omega * p.track / 2
        vl, vr = np.clip([vl, vr], -p.max_wheel_speed, p.max_wheel_speed)
        v_eff, omega = (vl + vr) / 2, (vr - vl) / p.track
        return np.array([v_eff * np.cos(yaw), v_eff * np.sin(yaw), omega, a, 0.0, 0.0])


class DynamicBicycle(VehicleModel):
    """Dynamic bicycle with linear tyres (reference point = centre of gravity).

    States: x, y, yaw, vx (=v), vy, r.  Slip angles:
        alpha_f = delta - atan((vy + lf r)/vx),   alpha_r = -atan((vy - lr r)/vx)
        Fyf = Cf alpha_f,  Fyr = Cr alpha_r
        vy' = (Fyf cos(delta) + Fyr)/m - vx r
        r'  = (lf Fyf cos(delta) - lr Fyr)/Iz
    Below 1 m/s the tyre model is singular, so we blend to kinematic behaviour.
    """

    name = "dynamic_bicycle"

    def derivatives(self, s, delta, a):
        _, _, yaw, vx, vy, r = s
        p = self.p
        if vx < 1.0:  # kinematic fallback at very low speed
            beta = np.arctan(p.lr / (p.lf + p.lr) * np.tan(delta))
            r_k = vx / p.lr * np.sin(beta)
            vy_k = vx * np.tan(beta)
            return np.array([
                vx * np.cos(yaw) - vy_k * np.sin(yaw),
                vx * np.sin(yaw) + vy_k * np.cos(yaw),
                r_k, a, (vy_k - vy) / 0.05, (r_k - r) / 0.05,
            ])
        af = delta - np.arctan2(vy + p.lf * r, vx)
        ar = -np.arctan2(vy - p.lr * r, vx)
        Fyf, Fyr = p.Cf * af, p.Cr * ar
        return np.array([
            vx * np.cos(yaw) - vy * np.sin(yaw),
            vx * np.sin(yaw) + vy * np.cos(yaw),
            r,
            a + vy * r,
            (Fyf * np.cos(delta) + Fyr) / p.mass - vx * r,
            (p.lf * Fyf * np.cos(delta) - p.lr * Fyr) / p.Iz,
        ])


MODELS = {m.name: m for m in (KinematicBicycle, DiffDrive, DynamicBicycle)}


def make_vehicle(name: str, params: VehicleParams | None = None) -> VehicleModel:
    try:
        return MODELS[name](params)
    except KeyError:
        raise ValueError(f"Unknown vehicle model '{name}'. Options: {list(MODELS)}") from None


__all__ = ["State", "VehicleParams", "KinematicBicycle", "DiffDrive", "DynamicBicycle", "make_vehicle", "MODELS"]
