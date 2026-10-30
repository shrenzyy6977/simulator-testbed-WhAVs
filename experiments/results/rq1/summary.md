# RQ1: throughput as simulator features are added

Machine: arm64 arm, Darwin 24.6.0; MuJoCo 3.17.0, physics 500 Hz, control 10 Hz (one env step = 0.1 s simulated). Random actions, resets included, best of 3 runs of 3000 steps.

## Feature ladder (one process, cumulative)

| env                                    |   env steps/s |   x real time |   map rebuild [ms] |
|:---------------------------------------|--------------:|--------------:|-------------------:|
| flat, one ground                       |          2045 |           205 |                2.4 |
| + hills (65 x 65 heightfield)          |          1854 |           185 |                2.6 |
| + 16 ground-class tiles                |          1724 |           172 |                3.3 |
| + GPS noise, bias, control delay       |          1688 |           169 |                3.3 |
| + randomisation, new map every episode |          1618 |           162 |                3.6 |
| + 6-wheel rover (second URDF)          |          1157 |           116 |                4.1 |

All features together keep 79% of the simplest env's throughput (4-wheel rover); the 6-wheel rover keeps 57%.

## Heightfield resolution (`+ randomisation, new map every episode`)

|   samples per side |   env steps/s |   x real time |   map rebuild [ms] |
|-------------------:|--------------:|--------------:|-------------------:|
|                 65 |          1625 |           162 |                3.6 |
|                129 |          1388 |           139 |                4.9 |
|                257 |          1118 |           112 |                8.9 |

## Worker processes (SubprocVecEnv)

|   processes |   env steps/s |   x real time |
|------------:|--------------:|--------------:|
|           1 |          1519 |           152 |
|           2 |          2639 |           264 |
|           4 |          4382 |           438 |
|           8 |          5154 |           515 |

## PPO training

PPO training, 8 processes: 4,451 env steps/s including gradient updates, so 1M steps take 3.7 min.
