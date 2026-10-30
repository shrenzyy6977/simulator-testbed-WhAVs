"""Mixed-terrain demo: all four controllers drive one path across patches of different ground.

    python scripts/terrain_demo.py [--out experiments/results/terrain_demo]

Saves map.png (ground classes, elevation contours, trajectories), cte.png and render.png (3D view).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from terrain3d import GROUND_CLASSES, WheeledRobot, make_terrain  # noqa: E402
from terrain3d.tracking import shift_path, track  # noqa: E402
from testbed.control import make_controller  # noqa: E402
from testbed.metrics import compute_metrics  # noqa: E402
from testbed.planning.path import make_path  # noqa: E402
from testbed.viz.plots import REFERENCE_COLOR, _style, color, label  # noqa: E402

GRID = [["grass", "gravel", "snow", "grass"],
        ["sand", "asphalt", "mud", "sand"],
        ["grass", "mud", "sand", "gravel"],
        ["asphalt", "grass", "gravel", "snow"]]
CONTROLLERS = [{"name": "pure_pursuit", "k": 0.3, "ld_min": 0.5, "ld_max": 2.0}, {"name": "stanley", "ks": 0.5},
               {"name": "lqr"}, {"name": "mpc"}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "terrain_demo"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terrain = make_terrain({"size": 20.0, "n": 129, "elevation": {"type": "hills", "amplitude": 0.25, "wavelength": 6.0, "seed": 4},
                            "ground": {"grid": GRID}})
    path = shift_path(make_path("s_curve", length=16.0, amplitude=3.0, wavelength=11.0, v_target=0.8, a_lat_max=0.5, ds=0.05), -8, 0)
    logs = [track(terrain, make_controller(**c), path) for c in CONTROLLERS]

    # top-down map
    fig, ax = plt.subplots(figsize=(7.5, 7))
    t, h = terrain.tiles, terrain.size / 2
    for i in range(t):
        for j in range(t):
            g = GROUND_CLASSES[terrain.classes[i, j]]
            ax.add_patch(plt.Rectangle((-h + j * 2 * h / t, -h + i * 2 * h / t), 2 * h / t, 2 * h / t,
                                       color=g.rgba[:3], alpha=0.55, linewidth=0))
    xs = np.linspace(-h, h, terrain.elevation.shape[0])
    ax.contour(xs, xs, terrain.elevation, levels=8, colors="#52514e", linewidths=0.5, alpha=0.6)
    ax.plot(path.x, path.y, "--", color=REFERENCE_COLOR, linewidth=1.5, label="Reference")
    for lg in logs:
        a = lg.arrays()
        ax.plot(a["x"], a["y"], color=color(lg.controller), linewidth=2, label=label(lg.controller))
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=GROUND_CLASSES[g].rgba[:3], alpha=0.55) for g in GROUND_CLASSES]
    labels += list(GROUND_CLASSES)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _style(ax)
    ax.set_title("Path tracking across ground classes (contours: elevation, 0.25 m hills)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # cross-track error over arc length
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for lg in logs:
        a = lg.arrays()
        ax.plot(a["s"], a["cte"], color=color(lg.controller), linewidth=2, label=label(lg.controller))
    ax.axhline(0, color=REFERENCE_COLOR, linewidth=1)
    ax.set_xlabel("distance along path [m]")
    ax.set_ylabel("cross-track error [m]")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cte.png", dpi=150)
    plt.close(fig)

    # 3D render from behind the start
    rob = WheeledRobot(terrain, start=(path.x[0], path.y[0]), start_yaw=float(path.yaw[0]))
    rob.step(200)
    with mujoco.Renderer(rob.model, 540, 960) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [path.x[0] + 2.0, path.y[0] + 1.0, 0.0]
        cam.distance, cam.elevation, cam.azimuth = 6.0, -25.0, -135.0
        r.update_scene(rob.data, cam)
        plt.imsave(out / "render.png", r.render())

    for lg in logs:
        m = compute_metrics(lg, 2.0)
        grounds = sorted(set(lg.ground))
        print(f"{m['controller']:13s} {m['status']:9s} rms_cte={m['rms_cte']:.3f} m  max={m['max_cte']:.3f} m  grounds crossed: {', '.join(grounds)}")
    print(f"Saved map.png, cte.png, render.png in {out}")


if __name__ == "__main__":
    main()
