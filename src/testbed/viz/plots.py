"""Matplotlib plots for runs and benchmarks. Colour follows the controller, never its rank."""
from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..planning.path import Path
from ..sim.simulator import SimLog
from ..world.grid_map import GridMap

CONTROLLER_COLORS = {
    "pure_pursuit": "#2a78d6",
    "stanley": "#eb6834",
    "lqr": "#1baf7a",
    "mpc": "#eda100",
    "go_to_goal": "#e87ba4",       # navigation policies (terrain3d) continue the fixed slot order
    "ppo_nominal": "#008300",
    "ppo_randomized": "#6250d6",
}
CONTROLLER_LABELS = {"pure_pursuit": "Pure Pursuit", "stanley": "Stanley", "lqr": "LQR", "mpc": "MPC",
                     "go_to_goal": "Go-to-goal", "ppo_nominal": "PPO (nominal)", "ppo_randomized": "PPO (randomised)"}
REFERENCE_COLOR = "#8a8984"
TEXT = "#52514e"
GRID = "#e4e3df"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def color(name: str) -> str:
    return CONTROLLER_COLORS.get(name, REFERENCE_COLOR)


def label(name: str) -> str:
    return CONTROLLER_LABELS.get(name, name)


def plot_runs(path: Path, logs: list[SimLog], out: str | FsPath, world: GridMap | None = None,
              title: str | None = None) -> None:
    """Trajectory (left) and cross-track error over time (right) for one or more runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    if world is not None:
        x0, x1, y0, y1 = world.extent
        ax1.imshow(world.grid, origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", alpha=0.5)
    ax1.plot(path.x, path.y, "--", color=REFERENCE_COLOR, linewidth=1.5, label="Reference")
    for lg in logs:
        a = lg.arrays()
        ax1.plot(a["x"], a["y"], color=color(lg.controller), linewidth=2, label=label(lg.controller))
        ax2.plot(a["t"], a["cte"], color=color(lg.controller), linewidth=2, label=label(lg.controller))
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=9, labelcolor=TEXT, loc="best")
    ax2.axhline(0, color=REFERENCE_COLOR, linewidth=1)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("cross-track error [m]  (+ = left)")
    for ax in (ax1, ax2):
        _style(ax)
    fig.suptitle(title or f"{path.name}: {logs[0].vehicle}" if logs else path.name, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_benchmark_bars(rows: list[dict], metric: str, out: str | FsPath, ylabel: str, title: str) -> None:
    """Grouped bars: one group per path, one bar per controller (fixed controller order)."""
    paths = list(dict.fromkeys(r["path"] for r in rows))
    ctrls = [c for c in CONTROLLER_COLORS if any(r["controller"] == c for r in rows)]
    width = 0.8 / max(len(ctrls), 1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, c in enumerate(ctrls):
        vals = [next((r[metric] for r in rows if r["path"] == p and r["controller"] == c), np.nan) for p in paths]
        xs = np.arange(len(paths)) + (i - (len(ctrls) - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=color(c), label=label(c), edgecolor="white", linewidth=1)
    ax.set_xticks(np.arange(len(paths)), paths)
    ax.set_ylabel(ylabel)
    fig.suptitle(title, color="#0b0b0b", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=len(ctrls), labelcolor=TEXT, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
