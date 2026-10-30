"""Model-based path-tracking controllers: LQR and linear MPC.

Both use the kinematic lateral-error model at the rear axle, with u = tan(delta) as input
so that the yaw-error equation is exactly linear in the input:

    e'       = v sin(theta_e)  ~ v theta_e
    theta_e' = v/L u - v kappa

Discretised with step dt (forward Euler):
    x_{k+1} = A x_k + B u_k + c_k,   A = [[1, v dt], [0, 1]],  B = [0, v dt / L]^T,
    c_k = [0, -v kappa_k dt]^T,      x = [e, theta_e]
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import lsq_linear

from ..planning.path import Path
from ..vehicles.base import State
from .base import Controller, SpeedPID


def error_model(v: float, L: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[1.0, v * dt], [0.0, 1.0]])
    B = np.array([[0.0], [v * dt / L]])
    return A, B


class LQR(Controller):
    """Discrete infinite-horizon LQR on the lateral-error model, plus curvature feed-forward.

        u = L kappa - K x,   K = (R + B^T P B)^-1 B^T P A,   P solves the DARE
        delta = atan(u)
    K depends on speed, so it is recomputed (and cached on a 0.1 m/s grid) as v changes.
    """

    name = "lqr"

    def __init__(self, q_e: float = 1.0, q_theta: float = 1.0, r: float = 5.0, speed: SpeedPID | None = None):
        super().__init__(speed)
        self.Q = np.diag([q_e, q_theta])
        self.R = np.array([[r]])
        self._cache: dict[tuple[float, float], np.ndarray] = {}

    def gain(self, v: float, dt: float) -> np.ndarray:
        key = (round(max(v, 0.5), 1), dt)
        if key not in self._cache:
            A, B = error_model(key[0], self.p.wheelbase, dt)
            P = solve_discrete_are(A, B, self.Q, self.R)
            self._cache[key] = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        return self._cache[key]

    def steer(self, s: State, path: Path, dt: float) -> float:
        rx, ry = self.rear_axle(s)
        _, e, theta_e = self.track_errors(path, rx, ry, s.yaw)
        K = self.gain(s.v, dt)
        u = self.p.wheelbase * path.k[self.idx] - float((K @ np.array([e, theta_e]))[0])
        return float(np.arctan(u))


class LinearMPC(Controller):
    """Linear time-varying MPC on the lateral-error model (condensed, box-constrained QP).

        min_U  sum_{k=1..N} x_k^T Q x_k  +  r sum (u_k - L kappa_k)^2  +  r_d sum (u_k - u_{k-1})^2
        s.t.   x_{k+1} = A x_k + B u_k + c_k,    |u_k| <= tan(max_steer)
    The path curvature kappa_k is previewed at the arc length the vehicle reaches after k steps.
    The QP is written as a bounded least-squares problem and solved with scipy's BVLS, so no
    extra solver dependency is needed. Only u_0 is applied (receding horizon).
    """

    name = "mpc"

    def __init__(self, horizon: int = 20, dt_pred: float = 0.1, q_e: float = 1.0, q_theta: float = 1.0,
                 q_terminal: float = 5.0, r: float = 0.5, r_d: float = 20.0, speed: SpeedPID | None = None):
        super().__init__(speed)
        self.N, self.dt_pred = horizon, dt_pred
        self.q = np.array([q_e, q_theta])
        self.q_terminal, self.r, self.r_d = q_terminal, r, r_d
        self._u_prev = 0.0

    def reset(self) -> None:
        super().reset()
        self._u_prev = 0.0

    def preview_curvature(self, path: Path, v: float) -> np.ndarray:
        s_k = path.s[self.idx] + v * self.dt_pred * np.arange(self.N)
        idx = np.minimum(np.searchsorted(path.s, s_k), len(path) - 1)
        return path.k[idx]

    def steer(self, s: State, path: Path, dt: float) -> float:
        rx, ry = self.rear_axle(s)
        _, e, theta_e = self.track_errors(path, rx, ry, s.yaw)
        v, L, N = max(s.v, 0.5), self.p.wheelbase, self.N
        A, B = error_model(v, L, self.dt_pred)
        kappa = self.preview_curvature(path, v)
        u_ff = L * kappa

        # Condensed prediction: X = Sx x0 + Su U + Sc, X = [x_1; ...; x_N]
        Sx = np.zeros((2 * N, 2))
        Su = np.zeros((2 * N, N))
        Sc = np.zeros(2 * N)
        AkB = [B.ravel()]                     # A^i B for i = 0..N-1
        for _ in range(N - 1):
            AkB.append(A @ AkB[-1])
        Ak = np.eye(2)
        c_acc = np.zeros(2)
        for k in range(N):
            Ak = A @ Ak
            Sx[2 * k:2 * k + 2] = Ak
            c_acc = A @ c_acc + np.array([0.0, -v * kappa[k] * self.dt_pred])
            Sc[2 * k:2 * k + 2] = c_acc
            for j in range(k + 1):
                Su[2 * k:2 * k + 2, j] = AkB[k - j]
        free = Sx @ np.array([e, theta_e]) + Sc

        w = np.tile(np.sqrt(self.q), N)
        w[-2:] *= np.sqrt(self.q_terminal)
        D = np.eye(N) - np.eye(N, k=-1)
        d0 = np.zeros(N)
        d0[0] = self._u_prev
        M = np.vstack([w[:, None] * Su, np.sqrt(self.r) * np.eye(N), np.sqrt(self.r_d) * D])
        b = np.concatenate([-w * free, np.sqrt(self.r) * u_ff, np.sqrt(self.r_d) * d0])
        u_max = np.tan(self.p.max_steer)
        sol = lsq_linear(M, b, bounds=(-u_max, u_max), method="bvls")
        self._u_prev = float(sol.x[0])
        return float(np.arctan(self._u_prev))
