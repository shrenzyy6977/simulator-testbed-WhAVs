# RQ3: robustness to held-out terrain and noise

Settings: `rq3_robustness.yaml`. 50 episodes per (policy, seed, condition), identical episodes for every policy; PPO trained 1,000,000 steps, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (mean ± std over seeds).

## Success rate (all 4 checkpoints reached)

| policy         | nominal   | randomized   | held_out_ground   | steep_hills   | high_noise   |
|:---------------|:----------|:-------------|:------------------|:--------------|:-------------|
| go_to_goal     | 100%      | 94%          | 18%               | 66%           | 36%          |
| ppo_nominal    | 100% ± 0  | 97% ± 3      | 28% ± 7           | 70% ± 3       | 49% ± 7      |
| ppo_randomized | 100% ± 1  | 99% ± 1      | 23% ± 6           | 73% ± 4       | 55% ± 7      |

## Checkpoints reached per episode (of 4)

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |      4    |         3.94 |              2.82 |          3.38 |         3.04 |
| ppo_nominal    |      4    |         3.97 |              2.81 |          3.39 |         3.26 |
| ppo_randomized |      3.99 |         3.98 |              2.77 |          3.42 |         3.34 |

## Mean time of successful episodes [s]

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |      27.6 |         31.9 |              39.1 |          33.2 |         35.6 |
| ppo_nominal    |      25.2 |         28.8 |              36.8 |          31.4 |         34.1 |
| ppo_randomized |      25.4 |         28.7 |              36.3 |          31.5 |         34.6 |

## PPO randomised vs nominal (per-seed success rates, Welch's t-test)

| condition       |   randomised - nominal [points] |   Welch p |
|:----------------|--------------------------------:|----------:|
| nominal         |                            -0.4 |     0.343 |
| randomized      |                             1.2 |     0.227 |
| held_out_ground |                            -4.6 |     0.136 |
| steep_hills     |                             2.2 |     0.164 |
| high_noise      |                             5.2 |     0.112 |
