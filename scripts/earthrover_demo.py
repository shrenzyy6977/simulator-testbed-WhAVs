"""Earth Rover Mini+ demo: one checkpoint mission on a real swissALTI3D terrain patch, with the
Earth Rover's sensing (noisy, biased 1 Hz GPS, control delay).

    python scripts/earthrover_demo.py [--policy go_to_goal | path/to/ppo.zip] [--seed 3] [--out DIR]

Saves map.png (elevation contours, checkpoints, driven path, GPS fixes) and render.png (3D view)
in DIR (default experiments/results/dem/).
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
import numpy as np  # noqa: E402

from terrain3d.baselines import GoToGoal  # noqa: E402
from terrain3d.env import NavConfig, RoverNavEnv  # noqa: E402
from terrain3d.world import RobotSpec  # noqa: E402
from testbed.viz.plots import CONTROLLER_COLORS, REFERENCE_COLOR, TEXT, _style, label  # noqa: E402

DEM = "experiments/maps/swissalti3d_bern.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="go_to_goal", help="go_to_goal, or the path of a PPO model (.zip)")
    ap.add_argument("--seed", type=int, default=4, help="mission seed (map patch, checkpoints, start)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "dem"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    robot = RobotSpec(urdf=ROOT / "robots" / "earthrover_mini_plus.urdf", wheel_radius=0.0475, track=0.14, max_wheel_speed=30.0)
    env = RoverNavEnv(NavConfig(terrain={"size": 20.0, "n": 65, "elevation": {"type": "dem", "path": DEM}},
                                grounds=["asphalt", "grass", "gravel"], n_checkpoints=4, checkpoint_spacing=(4.0, 8.0),
                                checkpoint_radius=0.75, max_steps=900, min_avg_speed=0.6, gps_noise=0.5, gps_bias=0.3,
                                gps_hz=1.0, delay_steps=2, robot=robot))
    if args.policy == "go_to_goal":
        policy, name = GoToGoal(), "go_to_goal"
    else:
        from stable_baselines3 import PPO

        policy, name = PPO.load(args.policy, device="cpu"), Path(args.policy).stem.rsplit("_s", 1)[0]
    obs, _ = env.reset(seed=args.seed)
    xy, fixes, done = [env.robot.pose()[:2]], [env.fix.copy()], False
    while not done:
        obs, _, term, trunc, info = env.step(policy.predict(obs, deterministic=True)[0])
        xy.append(env.robot.pose()[:2])
        if env.fix_age == 0:
            fixes.append(env.fix.copy())
        done = term or trunc
    xy, fixes = np.array(xy), np.array(fixes)
    status = "success" if info["success"] else f"{info['checkpoints_reached']} of 4 checkpoints"
    print(f"{name}: {status} in {env.steps / 10:.1f} s (budget {env.max_steps / 10:.1f} s)")

    t = env.terrain
    g = np.linspace(-t.size / 2, t.size / 2, t.elevation.shape[0])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cs = ax.contourf(g, g, t.elevation, levels=14, cmap="Greys", alpha=0.55)
    ax.contour(g, g, t.elevation, levels=14, colors=REFERENCE_COLOR, linewidths=0.5)
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03).set_label("height [m]", color=TEXT)
    ax.plot(*fixes.T, "x", color=REFERENCE_COLOR, ms=5, label="GPS fixes (1 Hz, noisy)")
    c = CONTROLLER_COLORS.get(name, TEXT)
    ax.plot(*xy.T, "-", color=c, lw=2, label=f"{label(name)}: {status}")
    for i, (cx, cy) in enumerate(env.checkpoints):
        ax.add_patch(plt.Circle((cx, cy), env.cfg.checkpoint_radius, fill=False, ec=TEXT, lw=1.5))
        ax.annotate(str(i + 1), (cx, cy), ha="center", va="center", fontsize=9, color=TEXT)
    ax.plot(*xy[0], "o", color=c, ms=8, mec="white", mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Earth Rover Mini+ on real terrain (swissALTI3D (c) swisstopo)", fontsize=10, color=TEXT, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    import mujoco

    env.reset(seed=args.seed)                         # 3D view of the start, close up (the rover is 25 cm long)
    for _ in range(20):
        env.step(np.zeros(2))
    rob = env.robot
    cam = mujoco.MjvCamera()
    cam.lookat[:] = rob.data.xpos[rob.body_id]
    cam.distance, cam.elevation, cam.azimuth = 1.6, -18.0, np.degrees(rob.pose()[5]) + 150.0
    with mujoco.Renderer(rob.model, 720, 1080) as r:
        r.update_scene(rob.data, cam)
        plt.imsave(out / "render.png", r.render())
    print(f"Saved map.png and render.png in {out}")


if __name__ == "__main__":
    main()
