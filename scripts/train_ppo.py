"""Train a PPO navigation policy on RoverNavEnv (Stable-Baselines3) and evaluate it.

    python scripts/train_ppo.py --timesteps 500000 --envs 8 [--out experiments/results/ppo]

Saves the policy (model.zip), TensorBoard logs and eval.csv (success rate per ground class).
PPO's small MLP runs faster on the CPU than on the M-series GPU, so device defaults to cpu.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from terrain3d.env import NavConfig, RoverNavEnv  # noqa: E402


def evaluate(model, episodes: int, seed: int) -> pd.DataFrame:
    env = RoverNavEnv(NavConfig(new_map_every=1))
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ground, done, ret, steps = info["ground"], False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret, steps, done = ret + r, steps + 1, term or trunc
        rows.append({"episode": ep, "start_ground": ground, "return": ret, "steps": steps,
                     "success": info["success"], "checkpoints": info["checkpoints_reached"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "ppo"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(RoverNavEnv, n_envs=args.envs, seed=args.seed,
                        vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, learning_rate=3e-4, gamma=0.99,
                seed=args.seed, device=args.device, tensorboard_log=str(out / "tb"), verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    wall = time.time() - t0
    model.save(out / "model")
    venv.close()
    print(f"trained {args.timesteps} steps in {wall:.0f} s ({args.timesteps / wall:.0f} steps/s with {args.envs} envs)")

    df = evaluate(model, args.eval_episodes, seed=10_000)
    df.to_csv(out / "eval.csv", index=False)
    print(f"eval: success {df.success.mean():.0%}, checkpoints/episode {df.checkpoints.mean():.2f}, return {df['return'].mean():.1f}")
    print(df.groupby("start_ground")[["success", "checkpoints"]].mean().round(2).to_markdown())


if __name__ == "__main__":
    main()
