"""RQ2 experiment: does evaluating in the upgraded simulator change which navigation policies rank best?

Every policy (go-to-goal, the four 2D-testbed path trackers, PPO trained in the simple and in the
randomised simulator) flies the same checkpoint missions in a simple simulator, with each upgrade
on its own, and with all upgrades; the rankings by SCT are compared with Kendall's tau.

    python scripts/rq2_ranking.py               # eval + report (settings: experiments/configs/rq2_ranking.yaml)
    python scripts/rq2_ranking.py report        # summary.md + ranking.png from an existing eval.csv
    python scripts/rq2_ranking.py --episodes 5  # quick check

Outputs in experiments/results/rq2/: eval.csv, summary.md, ranking.png
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terrain3d.baselines import GoToGoal, PathTracker  # noqa: E402
from terrain3d.env import NavConfig, RoverNavEnv  # noqa: E402
from terrain3d.ranking import bootstrap, ranks, sct, score_cube, scores, tau  # noqa: E402
from terrain3d.world import RobotSpec  # noqa: E402
from testbed.control import make_controller  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402

OUT = ROOT / "experiments" / "results" / "rq2"
CONFIG = ROOT / "experiments" / "configs" / "rq2_ranking.yaml"
LEVEL_LABELS = {"simple": "simple", "hills": "+ hills", "ground_classes": "+ 6 ground classes",
                "gps_noise": "+ GPS noise, bias", "gps_1hz": "+ 1 Hz GPS", "control_delay": "+ control delay",
                "rover6wd": "6-wheel rover", "all_upgrades": "all upgrades",
                "simple_retest": "simple, new missions (control)"}


def nav_config(cfg: dict, level: str) -> NavConfig:
    kw = {**cfg["mission"], **cfg["simple"], **cfg["levels"][level], "new_map_every": 1}
    kw.pop("episode_seed", None)
    for k in ("checkpoint_spacing", "amplitude_range", "gps_noise_range", "delay_range"):
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    if isinstance(kw.get("robot"), dict):
        kw["robot"] = RobotSpec(**{**kw["robot"], "urdf": ROOT / kw["robot"]["urdf"]})
    return NavConfig(**kw)


def make_policy(cfg: dict, name: str, seed: int):
    spec = cfg["policies"][name]
    if name == "go_to_goal":
        return GoToGoal()
    if spec == "rq3":
        from stable_baselines3 import PPO

        return PPO.load(ROOT / cfg["models"] / f"{name}_s{seed}.zip", device="cpu")
    return PathTracker(make_controller(name, **spec), **cfg["tracker"])


def _evaluate(job: tuple) -> list[dict]:
    cfg, name, seed, level = job
    policy = make_policy(cfg, name, seed)
    env = RoverNavEnv(nav_config(cfg, level))
    first = cfg["levels"][level].get("episode_seed", 100_000)
    rows = []
    for ep in range(cfg["episodes"]):
        obs, info = env.reset(seed=first + ep)            # same missions for every policy
        if hasattr(policy, "reset"):
            policy.reset(env)
        x, y, *_ = env.robot.pose()
        route = float(np.sum(np.hypot(*np.diff(np.r_[[[x, y]], env.checkpoints], axis=0).T)))
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        rows.append({"policy": name, "seed": seed, "level": level, "episode": ep, "success": info["success"],
                     "checkpoints": info["checkpoints_reached"], "time_s": env.steps / env.cfg.control_hz,
                     "t_opt": route / env.cfg.v_max})
    return rows


def evaluate(cfg: dict) -> pd.DataFrame:
    jobs = []
    for name, spec in cfg["policies"].items():
        for seed in (cfg["seeds"] if spec == "rq3" else [0]):
            if spec == "rq3" and not (ROOT / cfg["models"] / f"{name}_s{seed}.zip").exists():
                raise FileNotFoundError(f"{name}_s{seed}.zip missing: run python scripts/rq3_robustness.py train")
            jobs += [(cfg, name, seed, level) for level in cfg["levels"]]
    jobs.sort(key=lambda j: j[1] != "mpc")                   # slowest first, for better load balance
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        rows = [r for res in ex.map(_evaluate, jobs) for r in res]
    df = pd.DataFrame(rows)
    df["sct"] = sct(df.success, df.time_s, df.t_opt)
    df.to_csv(OUT / "eval.csv", index=False)
    print(f"{len(df)} evaluation episodes in {time.time() - t0:.0f} s")
    return df


def plot(policies: list, levels: list, rk: np.ndarray, boot_tau: np.ndarray, point_tau: np.ndarray, ref: int,
         control: int | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from testbed.viz.plots import CONTROLLER_COLORS, GRID, REFERENCE_COLOR, TEXT, _style, label

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    sims = [j for j in range(len(levels)) if j != control]          # the control is not a simulator
    x = np.arange(len(sims))
    for i, p in enumerate(policies):
        ax1.plot(x, rk[i, sims], "-o", color=CONTROLLER_COLORS[p], lw=2, ms=8, mec="white", mew=2, label=label(p))
        ax1.annotate(label(p), (x[-1], rk[i, sims[-1]]), xytext=(10, 0), textcoords="offset points", va="center",
                     fontsize=9, color=TEXT)
    ax1.set_xticks(x, [LEVEL_LABELS.get(levels[j], levels[j]) for j in sims], rotation=30, ha="right", fontsize=9)
    ax1.set_yticks(range(1, len(policies) + 1))
    ax1.set_ylim(len(policies) + 0.5, 0.5)
    ax1.set_xlim(-0.3, x[-1] + 1.6)
    ax1.set_ylabel("rank by SCT (1 = best)")
    ax1.set_title("Rank of each policy in each simulator", fontsize=10, color=TEXT, loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=4, frameon=False, fontsize=9)
    _style(ax1)
    ax1.grid(False, axis="x")

    others = [j for j in range(len(levels)) if j not in (ref, control)] + ([control] if control is not None else [])
    y = np.arange(len(others))[::-1]
    lo, hi = np.percentile(boot_tau[:, others], [2.5, 97.5], axis=0)
    t = point_tau[others]
    colors = [REFERENCE_COLOR if j == control else TEXT for j in others]   # not a policy colour
    for yi, ti, l, h, c in zip(y, t, lo, hi, colors):
        ax2.errorbar(ti, yi, xerr=[[ti - l], [h - ti]], fmt="o", color=c, ms=8, lw=2, capsize=0)
    if control is not None:                       # how far re-testing alone moves tau
        k = others.index(control)
        ax2.axvspan(lo[k], hi[k], color=GRID, alpha=0.6, lw=0)
    ax2.axvline(1.0, color=TEXT, lw=0.8, ls="--")
    ax2.set_yticks(y, [LEVEL_LABELS.get(levels[j], levels[j]) for j in others], fontsize=9)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_xlabel("Kendall's tau vs the simple simulator (95% CI)")
    ax2.set_title("Agreement with the simple simulator's ranking", fontsize=10, color=TEXT, loc="left")
    _style(ax2)
    ax2.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(cfg: dict) -> None:
    from testbed.viz.plots import label

    df = pd.read_csv(OUT / "eval.csv")
    policies, levels, cube = score_cube(df, list(cfg["policies"]), list(cfg["levels"]))
    ref = levels.index(cfg["reference"])
    s = scores(cube)
    rk = ranks(s)
    point_tau = np.array([tau(s[:, ref], s[:, j]) for j in range(len(levels))])
    boot = bootstrap(cube, ref, cfg["bootstrap"])
    names = [label(p) for p in policies]
    cols = [LEVEL_LABELS.get(lv, lv) for lv in levels]

    score_tab = pd.DataFrame([[f"{s[i, j]:.3f} ({rk[i, j]})" for j in range(len(levels))] for i in range(len(policies))],
                             index=names, columns=cols)
    succ = df.groupby(["policy", "seed", "level"]).success.mean().groupby(["policy", "level"]).mean().unstack()
    succ_tab = (100 * succ.loc[policies, levels]).round(0).astype(int).astype(str) + "%"
    succ_tab.index, succ_tab.columns = names, cols
    lo, hi = np.percentile(boot["tau"], [2.5, 97.5], axis=0)
    best_ref = s[:, ref].argmax()
    agree = pd.DataFrame({
        "simulator": cols,
        "Kendall's tau vs simple": [f"{point_tau[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]" for j in range(len(levels))],
        "best policy": [names[s[:, j].argmax()] for j in range(len(levels))],
        "same best as simple (bootstrap)": [f"{100 * np.mean(boot['best'][:, j] == boot['best'][:, ref]):.0f}%"
                                            for j in range(len(levels))],
    }).iloc[[j for j in range(len(levels)) if j != ref]]
    n_seeds = len(cfg["seeds"])
    lines = ["# RQ2: does the simulator change which navigation policies rank best?", "",
             f"Settings: `{CONFIG.name}`. {cfg['episodes']} identical checkpoint missions per (policy, simulator); "
             f"PPO scores are averaged over {n_seeds} training seeds (the RQ3 models). Score: SCT, success weighted by "
             f"completion time (1 = success at v_max on the straight route, 0 = failure). 95% intervals from "
             f"{cfg['bootstrap']} paired bootstrap resamples of missions and seeds.", "",
             "## Score (rank) per simulator", "", score_tab.to_markdown(), "",
             "## Success rate per simulator", "", succ_tab.to_markdown(), "",
             f"## Agreement with the simple simulator (best there: {names[best_ref]})", "",
             agree.to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    control = levels.index("simple_retest") if "simple_retest" in levels else None
    plot(policies, levels, rk, boot["tau"], point_tau, ref, control, OUT / "ranking.png")
    print("\n".join(lines))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["all", "report"], default="all")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--episodes", type=int, help="override episodes (quick checks)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.episodes:
        cfg["episodes"], cfg["bootstrap"] = args.episodes, 200
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        evaluate(cfg)
    report(cfg)


if __name__ == "__main__":
    main()
