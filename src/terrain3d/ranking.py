"""Ranking statistics for RQ2: does the simulator a policy is evaluated in change its rank?

Episodes are scored with SCT, Success weighted by Completion Time (Yokoyama et al., 2021):

    SCT = S * t_opt / max(t, t_opt),      t_opt = route length / v_max
so a failed episode scores 0 and a successful one scores more the faster it is. A policy's score
in a simulator is its mean SCT over episodes (and over training seeds for learned policies), and
two simulators agree on the ranking to the extent that Kendall's tau between their scores is 1.
Uncertainty comes from a paired bootstrap: episode indices (the same missions for every policy)
and training seeds are resampled together for every policy and simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def sct(success: np.ndarray, time_s: np.ndarray, t_opt: np.ndarray) -> np.ndarray:
    """Success weighted by Completion Time, per episode."""
    success, time_s, t_opt = (np.asarray(a, float) for a in (success, time_s, t_opt))
    return success * t_opt / np.maximum(time_s, t_opt)


def score_cube(df: pd.DataFrame, policies: list | None = None, levels: list | None = None,
               metric: str = "sct") -> tuple[list, list, dict[str, np.ndarray]]:
    """Per policy, an array [seed, level, episode] of `metric` (rows of df: policy, seed, level, episode)."""
    policies = [p for p in (policies or dict.fromkeys(df.policy)) if p in set(df.policy)]
    levels = [lv for lv in (levels or dict.fromkeys(df.level)) if lv in set(df.level)]
    episodes = sorted(df.episode.unique())
    cube = {}
    for p in policies:
        d = df[df.policy == p]
        seeds = sorted(d.seed.unique())
        idx = pd.MultiIndex.from_product([seeds, levels, episodes], names=["seed", "level", "episode"])
        v = d.set_index(["seed", "level", "episode"])[metric].reindex(idx).to_numpy(float)
        cube[p] = v.reshape(len(seeds), len(levels), len(episodes))
    return policies, levels, cube


def scores(cube: dict[str, np.ndarray], seed_idx: dict | None = None, ep_idx: np.ndarray | None = None) -> np.ndarray:
    """Mean score [policy, level], optionally on resampled seeds / episodes."""
    out = []
    for p, v in cube.items():
        if seed_idx is not None:
            v = v[seed_idx[p]]
        if ep_idx is not None:
            v = v[:, :, ep_idx]
        out.append(np.nanmean(v, axis=(0, 2)))
    return np.array(out)


def ranks(score: np.ndarray) -> np.ndarray:
    """Rank 1 = best (highest score) per column of a [policy, level] score matrix."""
    return (-score).argsort(axis=0).argsort(axis=0) + 1


def tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-b between two score vectors (1 = same order, -1 = reversed)."""
    t = kendalltau(a, b).statistic
    return float(t) if np.isfinite(t) else 1.0      # all tied in both: no disagreement


def bootstrap(cube: dict[str, np.ndarray], reference: int, n_boot: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    """Paired bootstrap of the score matrix, the tau of every level against `reference` and the best policy.

    Returns arrays with a leading bootstrap axis: score [b, policy, level], tau [b, level], best [b, level].
    """
    rng = np.random.default_rng(seed)
    n_ep = next(iter(cube.values())).shape[2]
    S, T, B = [], [], []
    for _ in range(n_boot):
        ep = rng.integers(0, n_ep, n_ep)
        si = {p: rng.integers(0, v.shape[0], v.shape[0]) for p, v in cube.items()}
        s = scores(cube, si, ep)
        S.append(s)
        T.append([tau(s[:, reference], s[:, j]) for j in range(s.shape[1])])
        B.append(s.argmax(axis=0))
    return {"score": np.array(S), "tau": np.array(T), "best": np.array(B)}
