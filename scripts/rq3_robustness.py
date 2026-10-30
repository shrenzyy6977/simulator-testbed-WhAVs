"""RQ3 experiment: nominal vs domain-randomised PPO vs a classical go-to-goal baseline,
evaluated on the training conditions and on held-out terrain and noise.

    python scripts/rq3_robustness.py train        # PPO policies x seeds (settings: experiments/configs/rq3_robustness.yaml)
    python scripts/rq3_robustness.py eval         # every policy on every eval condition
    python scripts/rq3_robustness.py report       # summary.md + success.png from eval.csv
    python scripts/rq3_robustness.py all

Outputs in experiments/results/rq3/: models/, eval.csv, summary.md, success.png
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

from terrain3d.baselines import GoToGoal  # noqa: E402
from terrain3d.env import NavConfig, RoverNavEnv  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402

OUT = ROOT / "experiments" / "results" / "rq3"
CONFIG = ROOT / "experiments" / "configs" / "rq3_robustness.yaml"


def nav_config(cfg: dict, condition: str, **extra) -> NavConfig:
    kw = {**cfg["mission"], **cfg["conditions"][condition], **extra}
    for k in ("checkpoint_spacing", "amplitude_range", "gps_noise_range", "delay_range"):
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    return NavConfig(**kw)


def make_env(cfg: dict, condition: str):
    return lambda: RoverNavEnv(nav_config(cfg, condition))


def train(cfg: dict) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for name, condition in cfg["train"].items():
        for seed in cfg["seeds"]:
            path = OUT / "models" / f"{name}_s{seed}.zip"
            if path.exists():
                print(f"skip {path.name} (exists)")
                continue
            venv = VecMonitor(SubprocVecEnv([make_env(cfg, condition) for _ in range(cfg["envs"])]))
            venv.seed(seed)
            model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, learning_rate=3e-4, gamma=0.99,
                        seed=seed, device="cpu", tensorboard_log=str(OUT / "tb"), verbose=0)
            t0 = time.time()
            model.learn(total_timesteps=cfg["timesteps"], tb_log_name=f"{name}_s{seed}")
            model.save(path)
            venv.close()
            print(f"trained {name} seed {seed}: {cfg['timesteps']} steps in {time.time() - t0:.0f} s")


def _evaluate(job: tuple) -> list[dict]:
    cfg, policy_name, seed, condition, out = job
    if policy_name == "go_to_goal":
        policy = GoToGoal()
    else:
        from stable_baselines3 import PPO

        policy = PPO.load(Path(out) / "models" / f"{policy_name}_s{seed}.zip", device="cpu")
    env = RoverNavEnv(nav_config(cfg, condition, new_map_every=1))
    rows = []
    for ep in range(cfg["eval"]["episodes"]):
        obs, info = env.reset(seed=100_000 + ep)          # same episodes for every policy
        done, ret, slips = False, 0.0, []
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            slips.append(info["slip"])
            done = term or trunc
        rows.append({"policy": policy_name, "seed": seed, "condition": condition, "episode": ep,
                     "success": info["success"], "checkpoints": info["checkpoints_reached"],
                     "time_s": env.steps / env.cfg.control_hz, "return": ret, "mean_slip": float(np.mean(slips))})
    return rows


def evaluate(cfg: dict) -> None:
    jobs = [(cfg, "go_to_goal", 0, c, str(OUT)) for c in cfg["eval"]["conditions"]]
    jobs += [(cfg, name, seed, c, str(OUT)) for name in cfg["train"] for seed in cfg["seeds"] for c in cfg["eval"]["conditions"]
             if (OUT / "models" / f"{name}_s{seed}.zip").exists()]          # untrained policies are skipped
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg["envs"]) as ex:
        rows = [r for res in ex.map(_evaluate, jobs) for r in res]
    pd.DataFrame(rows).to_csv(OUT / "eval.csv", index=False)
    print(f"{len(rows)} evaluation episodes in {time.time() - t0:.0f} s")


def report(cfg: dict) -> None:
    from testbed.viz import plot_benchmark_bars

    df = pd.read_csv(OUT / "eval.csv")
    conds = cfg["eval"]["conditions"]
    policies = [p for p in ["go_to_goal", *cfg["train"]] if p in set(df.policy)]
    # per-seed success rate first, then mean +/- std over seeds
    per_seed = df.groupby(["policy", "seed", "condition"])["success"].mean().reset_index()
    agg = per_seed.groupby(["policy", "condition"])["success"].agg(["mean", "std"]).reset_index()
    cell = agg.apply(lambda r: f"{100 * r['mean']:.0f}%" + ("" if np.isnan(r["std"]) else f" ± {100 * r['std']:.0f}"), axis=1)
    tab = agg.assign(cell=cell).pivot(index="policy", columns="condition", values="cell").loc[policies, conds]
    cps = df.groupby(["policy", "condition"])["checkpoints"].mean().unstack().loc[policies, conds].round(2)
    tmean = df[df.success].groupby(["policy", "condition"])["time_s"].mean().unstack().reindex(index=policies, columns=conds).round(1)
    n_ep = cfg["eval"]["episodes"]
    lines = ["# RQ3: robustness to held-out terrain and noise", "",
             f"Settings: `{CONFIG.name}`. {n_ep} episodes per (policy, seed, condition), identical episodes for every "
             f"policy; PPO trained {cfg['timesteps']:,} steps, seeds {cfg['seeds']} (mean ± std over seeds).", "",
             "## Success rate (all 4 checkpoints reached)", "", tab.to_markdown(), "",
             "## Checkpoints reached per episode (of 4)", "", cps.to_markdown(), "",
             "## Mean time of successful episodes [s]", "", tmean.to_markdown(), ""]
    if {"ppo_nominal", "ppo_randomized"} <= set(policies) and len(cfg["seeds"]) > 2:
        from scipy import stats

        # seeds are the independent units: Welch's t-test on the per-seed success rates
        rows = []
        for c in conds:
            a, b = (per_seed[(per_seed.policy == p) & (per_seed.condition == c)].success.to_numpy()
                    for p in ("ppo_nominal", "ppo_randomized"))
            rows.append({"condition": c, "randomised - nominal [points]": round(100 * (b.mean() - a.mean()), 1),
                         "Welch p": "-" if np.isclose(a.std() + b.std(), 0) else
                         f"{stats.ttest_ind(b, a, equal_var=False).pvalue:.3f}"})      # both always succeed: no test
        lines += ["## PPO randomised vs nominal (per-seed success rates, Welch's t-test)", "",
                  pd.DataFrame(rows).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    bars = [{"path": c, "controller": p, "success": 100 * float(agg[(agg.policy == p) & (agg.condition == c)]["mean"].iloc[0])}
            for c in conds for p in policies]
    plot_benchmark_bars(bars, "success", OUT / "success.png", "success rate [%]",
                        "Checkpoint missions: success by evaluation condition")
    print("\n".join(lines))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "eval", "report", "all"])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--episodes", type=int, help="override eval episodes (quick checks)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.episodes:
        cfg["eval"]["episodes"] = args.episodes
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("train", "all"):
        train(cfg)
    if args.stage in ("eval", "all"):
        evaluate(cfg)
    if args.stage in ("report", "all"):
        report(cfg)


if __name__ == "__main__":
    main()
