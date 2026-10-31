# RQ1: throughput as simulator features are added

Machine: arm64 arm, Darwin 24.6.0; MuJoCo 3.17.0, physics 500 Hz, control 10 Hz (one env step = 0.1 s simulated). Random actions, resets included, best of 3 runs of 3000 steps.

## Feature ladder (one process, cumulative)

| env                                    |   env steps/s |   x real time |   map rebuild [ms] |
|:---------------------------------------|--------------:|--------------:|-------------------:|
| flat, one ground                       |          2071 |           207 |                2.4 |
| + hills (65 x 65 heightfield)          |          1879 |           188 |                2.6 |
| + 16 ground-class tiles                |          1738 |           174 |                3.3 |
| + GPS noise, bias, 1 Hz rate, delay    |          1743 |           174 |                3.3 |
| + randomisation, new map every episode |          1635 |           163 |                3.7 |
| + 6-wheel rover (second URDF)          |          1164 |           116 |                4.2 |

All features together keep 79% of the simplest env's throughput (4-wheel rover); the 6-wheel rover keeps 56%.

## Heightfield resolution (`+ randomisation, new map every episode`)

|   samples per side |   env steps/s |   x real time |   map rebuild [ms] |
|-------------------:|--------------:|--------------:|-------------------:|
|                 65 |          1643 |           164 |                3.6 |
|                129 |          1409 |           141 |                4.9 |
|                257 |          1094 |           109 |                9   |

## Worker processes (SubprocVecEnv)

|   processes |   env steps/s |   x real time |
|------------:|--------------:|--------------:|
|           1 |          1498 |           150 |
|           2 |          2653 |           265 |
|           4 |          4448 |           445 |
|           8 |          5885 |           589 |

## PPO training

PPO training, 8 processes: 4,753 env steps/s including gradient updates, so 1M steps take 3.5 min.
