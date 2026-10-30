# RQ3: robustness to held-out terrain and noise

Settings: `rq3_robustness.yaml`. 50 episodes per (policy, seed, condition), identical episodes for every policy; PPO trained 1,000,000 steps, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (mean ± std over seeds).

## Success rate (all 4 checkpoints reached)

| policy         | nominal   | randomized   | held_out_ground   | steep_hills   | high_noise   |
|:---------------|:----------|:-------------|:------------------|:--------------|:-------------|
| go_to_goal     | 100%      | 98%          | 26%               | 62%           | 62%          |
| ppo_nominal    | 100% ± 0  | 98% ± 2      | 20% ± 7           | 70% ± 4       | 63% ± 8      |
| ppo_randomized | 100% ± 0  | 99% ± 1      | 26% ± 7           | 73% ± 2       | 71% ± 7      |

## Checkpoints reached per episode (of 4)

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |         4 |         3.98 |              2.76 |          3.18 |         3.36 |
| ppo_nominal    |         4 |         3.97 |              2.49 |          3.38 |         3.26 |
| ppo_randomized |         4 |         3.98 |              2.64 |          3.41 |         3.52 |

## Mean time of successful episodes [s]

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |      27.6 |         31.5 |              38.2 |          33.9 |         34.4 |
| ppo_nominal    |      25.2 |         28.3 |              36.5 |          31.5 |         32.3 |
| ppo_randomized |      24.6 |         27.1 |              35.8 |          30.7 |         31   |

## PPO randomised vs nominal (per-seed success rates, Welch's t-test)

| condition       |   randomised - nominal [points] | Welch p   |
|:----------------|--------------------------------:|:----------|
| nominal         |                             0   | -         |
| randomized      |                             1   | 0.244     |
| held_out_ground |                             6.2 | 0.057     |
| steep_hills     |                             3   | 0.063     |
| high_noise      |                             8.8 | 0.019     |
