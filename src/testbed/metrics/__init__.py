"""Tracking metrics computed from a SimLog."""
from __future__ import annotations

import numpy as np

from ..sim.simulator import SimLog


def compute_metrics(log: SimLog, settle_time: float = 0.0) -> dict[str, float | str]:
    """Summary metrics of one run.

    rms_cte / max_cte      : lateral error [m]      (after `settle_time`, to skip an initial offset)
    rms_heading_err        : heading error [rad]
    rms_speed_err          : |v_ref - v| RMS [m/s]
    mean_abs_steer_rate    : steering smoothness [rad/s]
    mean/p95/max_compute_ms: controller wall-clock time per call [ms]
    """
    a = log.arrays()
    t = a["t"]
    m = t >= settle_time if len(t) else np.zeros(0, bool)
    rms = lambda z: float(np.sqrt(np.mean(np.square(z)))) if len(z) else float("nan")
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    ct_ms = a["compute_time"] * 1e3
    return {
        "controller": log.controller,
        "vehicle": log.vehicle,
        "path": log.path,
        "status": log.status,
        "time_s": float(t[-1]) if len(t) else 0.0,
        "rms_cte": rms(a["cte"][m]),
        "max_cte": float(np.max(np.abs(a["cte"][m]))) if m.any() else float("nan"),
        "rms_heading_err": rms(a["heading_err"][m]),
        "rms_speed_err": rms((a["v_ref"] - a["v"])[m]),
        "mean_abs_steer_rate": float(np.mean(np.abs(np.diff(a["steer"])))) / dt if len(t) > 1 else 0.0,
        "mean_compute_ms": float(np.mean(ct_ms)) if len(ct_ms) else 0.0,
        "p95_compute_ms": float(np.percentile(ct_ms, 95)) if len(ct_ms) else 0.0,
        "max_compute_ms": float(np.max(ct_ms)) if len(ct_ms) else 0.0,
    }


__all__ = ["compute_metrics"]
