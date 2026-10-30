"""Classical baseline policies for RoverNavEnv, with the same predict() interface as SB3 models."""
from __future__ import annotations

import numpy as np


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
