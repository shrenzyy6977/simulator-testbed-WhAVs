# RQ3: robustness to held-out terrain and noise

Settings: `rq3_robustness.yaml`. 50 episodes per (policy, seed, condition), identical episodes for every policy; PPO trained 1,000,000 steps, seeds [0, 1, 2] (mean ± std over seeds).

## Success rate (all 4 checkpoints reached)

| policy         | nominal   | randomized   | held_out_ground   | steep_hills   | high_noise   |
|:---------------|:----------|:-------------|:------------------|:--------------|:-------------|
| go_to_goal     | 100%      | 98%          | 26%               | 62%           | 62%          |
| ppo_nominal    | 100% ± 0  | 100% ± 0     | 18% ± 11          | 71% ± 3       | 61% ± 15     |
| ppo_randomized | 100% ± 0  | 100% ± 0     | 28% ± 5           | 73% ± 2       | 73% ± 10     |

## Checkpoints reached per episode (of 4)

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |         4 |         3.98 |              2.76 |          3.18 |         3.36 |
| ppo_nominal    |         4 |         4    |              2.44 |          3.4  |         3.24 |
| ppo_randomized |         4 |         4    |              2.73 |          3.39 |         3.55 |

## Mean time of successful episodes [s]

| policy         |   nominal |   randomized |   held_out_ground |   steep_hills |   high_noise |
|:---------------|----------:|-------------:|------------------:|--------------:|-------------:|
| go_to_goal     |      27.6 |         31.5 |              38.2 |          33.9 |         34.4 |
| ppo_nominal    |      25.3 |         28.3 |              36.8 |          31.7 |         33.2 |
| ppo_randomized |      24.9 |         27.4 |              35.8 |          31.1 |         31.7 |
