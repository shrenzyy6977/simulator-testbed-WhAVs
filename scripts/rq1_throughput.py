"""RQ1 experiment: RoverNavEnv throughput as ground classes, sensor noise, randomisation and a
second robot are added, plus heightfield resolution, multi-process scaling and PPO training speed.

    python scripts/rq1_throughput.py              # settings: experiments/configs/rq1_throughput.yaml
    python scripts/rq1_throughput.py --quick      # few steps, to check the script runs

Timings depend on the machine and on what else is running; close other heavy programs first.
Outputs in experiments/results/rq1/: throughput.csv, summary.md, throughput.png
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terrain3d.env import NavConfig, RoverNavEnv  # noqa: E402
from terrain3d.world import RobotSpec  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402

OUT = ROOT / "experiments" / "results" / "rq1"
CONFIG = ROOT / "experiments" / "configs" / "rq1_throughput.yaml"


def nav_config(kw: dict) -> NavConfig:
    kw = dict(kw)
    for k in ("amplitude_range", "gps_noise_range", "delay_range"):
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    if isinstance(kw.get("robot"), dict):
        r = dict(kw["robot"])
        r["urdf"] = ROOT / r["urdf"]
        kw["robot"] = RobotSpec(**r)
    return NavConfig(**kw)


def levels(cfg: dict) -> dict[str, dict]:
    """Cumulative NavConfig kwargs per ladder level."""
    out, kw = {}, dict(cfg["base"])
    for level in cfg["ladder"]:
        kw = {**kw, **{k: v for k, v in level.items() if k != "name"}}
        out[level["name"]] = kw
    return out


def env_steps_per_s(kw: dict, steps: int, repeats: int) -> tuple[float, float]:
    """Best-of-`repeats` single-process steps/s with random actions (resets included), and the
    mean time of a reset that rebuilds the map [ms]."""
    best = 0.0
    for rep in range(repeats):
        env = RoverNavEnv(nav_config(kw))
        env.reset(seed=rep)
        rng = np.random.default_rng(rep)
        t0 = time.perf_counter()
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            if term or trunc:
                env.reset()
        best = max(best, steps / (time.perf_counter() - t0))
    t0 = time.perf_counter()
    for s in range(10):
        env.reset(seed=1000 + s)                 # a seeded reset always rebuilds the map
    return best, 1000 * (time.perf_counter() - t0) / 10


def vec_steps_per_s(kw: dict, n_proc: int, steps_per_env: int) -> float:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([lambda: RoverNavEnv(nav_config(kw)) for _ in range(n_proc)])
    venv.seed(0)
    venv.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):                          # warm up the worker processes
        venv.step(rng.uniform(-1, 1, (n_proc, 2)).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(steps_per_env):
        venv.step(rng.uniform(-1, 1, (n_proc, 2)).astype(np.float32))   # auto-resets finished envs
    rate = n_proc * steps_per_env / (time.perf_counter() - t0)
    venv.close()
    return rate


def ppo_steps_per_s(kw: dict, n_proc: int, timesteps: int) -> float:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    venv = VecMonitor(SubprocVecEnv([lambda: RoverNavEnv(nav_config(kw)) for _ in range(n_proc)]))
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, learning_rate=3e-4, seed=0, device="cpu", verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    rate = timesteps / (time.perf_counter() - t0)
    venv.close()
    return rate


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from testbed.viz.plots import CONTROLLER_COLORS, REFERENCE_COLOR, TEXT, _style

    blue = CONTROLLER_COLORS["pure_pursuit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    lad = df[df.experiment == "ladder"]
    y = np.arange(len(lad))[::-1]
    ax1.barh(y, lad.steps_per_s, height=0.6, color=blue)
    for yi, v in zip(y, lad.steps_per_s):
        ax1.text(v, yi, f" {v:,.0f}", va="center", fontsize=9, color=TEXT)
    ax1.set_yticks(y, lad.setting, fontsize=9, color=TEXT)
    ax1.set_xlabel("env steps/s, one process (10 steps = 1 s simulated)")
    ax1.set_xlim(0, lad.steps_per_s.max() * 1.18)
    ax1.set_title("Cost of each feature (cumulative)", fontsize=10, color=TEXT, loc="left")
    _style(ax1)
    ax1.grid(False, axis="y")

    sc = df[df.experiment == "processes"]
    n = sc.setting.astype(int).to_numpy()
    ax2.plot(n, sc.steps_per_s / 1000, "-o", color=blue, lw=2, ms=8)
    ax2.plot(n, n * sc.steps_per_s.iloc[0] / 1000, "--", color=REFERENCE_COLOR, lw=1.2)
    ax2.annotate("measured", (n[-1], sc.steps_per_s.iloc[-1] / 1000), xytext=(-8, -14), textcoords="offset points",
                 ha="right", fontsize=9, color=TEXT)
    ax2.annotate("linear", (n[-1], n[-1] * sc.steps_per_s.iloc[0] / 1000), xytext=(-8, 4), textcoords="offset points",
                 ha="right", fontsize=9, color=TEXT)
    ax2.set_xticks(n)
    ax2.set_xlabel("worker processes")
    ax2.set_ylabel("env steps/s [thousands]", color=TEXT)
    ax2.set_ylim(0, None)
    ax2.set_title("Scaling, full-feature env", fontsize=10, color=TEXT, loc="left")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true", help="few steps, to check the script runs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps, repeats, ppo_steps = (200, 1, 4096 * 2) if args.quick else (cfg["steps"], cfg["repeats"], cfg["ppo_timesteps"])
    lv = levels(cfg)
    ref = lv[cfg["reference"]]
    rows = []

    def add(experiment, setting, rate, reset_ms=np.nan):
        rows.append({"experiment": experiment, "setting": setting, "steps_per_s": rate,
                     "real_time_factor": rate / NavConfig().control_hz, "reset_ms": reset_ms})
        print(f"{experiment:10s} {setting:45s} {rate:8.0f} steps/s" + ("" if np.isnan(reset_ms) else f"   map rebuild {reset_ms:5.1f} ms"))

    for name, kw in lv.items():
        add("ladder", name, *env_steps_per_s(kw, steps, repeats))
    for n in cfg["resolution"]:
        add("resolution", str(n), *env_steps_per_s({**ref, "terrain": {**ref["terrain"], "n": n}}, steps, repeats))
    for n in cfg["processes"]:
        add("processes", str(n), vec_steps_per_s(ref, n, max(steps // 2, 50)))
    add("ppo", f"PPO training, {cfg['processes'][-1]} processes", ppo_steps_per_s(ref, cfg["processes"][-1], ppo_steps))

    df = pd.DataFrame(rows)
    df.to_csv(out / "throughput.csv", index=False)
    plot(df, out / "throughput.png")

    def table(exp, col):
        t = df[df.experiment == exp][["setting", "steps_per_s", "real_time_factor", "reset_ms"]].copy()
        t.columns = [col, "env steps/s", "x real time", "map rebuild [ms]"]
        t = t.round({"env steps/s": 0, "x real time": 0, "map rebuild [ms]": 1}).dropna(axis=1, how="all")
        return t.to_markdown(index=False)

    ppo = df[df.experiment == "ppo"].iloc[0]
    lad = df[df.experiment == "ladder"].steps_per_s.to_numpy()
    lines = ["# RQ1: throughput as simulator features are added", "",
             f"Machine: {platform.machine()} {platform.processor() or ''}, {platform.system()} {platform.release()}; "
             f"MuJoCo {mujoco.__version__}, physics 500 Hz, control 10 Hz (one env step = 0.1 s simulated). "
             f"Random actions, resets included, best of {repeats} runs of {steps} steps.", "",
             "## Feature ladder (one process, cumulative)", "", table("ladder", "env"), "",
             f"All features together keep {100 * lad[-2] / lad[0]:.0f}% of the simplest env's throughput "
             f"(4-wheel rover); the 6-wheel rover keeps {100 * lad[-1] / lad[0]:.0f}%.", "",
             f"## Heightfield resolution (`{cfg['reference']}`)", "", table("resolution", "samples per side"), "",
             "## Worker processes (SubprocVecEnv)", "", table("processes", "processes"), "",
             "## PPO training", "",
             f"{ppo.setting}: {ppo.steps_per_s:,.0f} env steps/s including gradient updates, so 1M steps take "
             f"{1e6 / ppo.steps_per_s / 60:.1f} min.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
