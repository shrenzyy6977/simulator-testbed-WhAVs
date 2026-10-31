import numpy as np
import pandas as pd

from terrain3d.ranking import bootstrap, ranks, sct, score_cube, scores, tau


def test_sct_is_zero_on_failure_and_one_at_the_optimal_time():
    assert np.allclose(sct([1, 1, 0, 1], [10.0, 20.0, 10.0, 5.0], [10.0, 10.0, 10.0, 10.0]), [1.0, 0.5, 0.0, 1.0])


def test_ranks_and_kendall_tau():
    s = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])        # [policy, level]: level 1 reverses level 0
    assert ranks(s).tolist() == [[1, 3], [2, 2], [3, 1]]
    assert tau(s[:, 0], s[:, 0]) == 1.0
    assert tau(s[:, 0], s[:, 1]) == -1.0


def _frame(level_scores: dict, n_seeds: dict, episodes: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [{"policy": p, "seed": k, "level": lv, "episode": e, "sct": m + 0.01 * rng.normal()}
            for p, by_level in level_scores.items() for k in range(n_seeds[p])
            for lv, m in by_level.items() for e in range(episodes)]
    return pd.DataFrame(rows)


def test_score_cube_averages_seeds_and_bootstrap_detects_a_reordering():
    df = _frame({"a": {"simple": 0.9, "same": 0.9, "flipped": 0.2},
                 "b": {"simple": 0.6, "same": 0.6, "flipped": 0.5},
                 "c": {"simple": 0.3, "same": 0.3, "flipped": 0.8}}, {"a": 1, "b": 3, "c": 10})
    policies, levels, cube = score_cube(df, ["a", "b", "c"], ["simple", "same", "flipped"])
    assert [cube[p].shape for p in policies] == [(1, 3, 20), (3, 3, 20), (10, 3, 20)]
    assert np.allclose(scores(cube)[:, 0], [0.9, 0.6, 0.3], atol=0.01)
    boot = bootstrap(cube, reference=0, n_boot=200)
    assert np.all(boot["tau"][:, 1] == 1.0) and np.all(boot["tau"][:, 2] == -1.0)
    assert np.all(boot["best"][:, 2] == 2)
