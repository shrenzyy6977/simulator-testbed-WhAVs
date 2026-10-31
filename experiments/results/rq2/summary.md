# RQ2: does the simulator change which navigation policies rank best?

Settings: `rq2_ranking.yaml`. 50 identical checkpoint missions per (policy, simulator); PPO scores are averaged over 10 training seeds (the RQ3 models). Score: SCT, success weighted by completion time (1 = success at v_max on the straight route, 0 = failure). 95% intervals from 2000 paired bootstrap resamples of missions and seeds.

## Score (rank) per simulator

|                  | simple    | + hills   | + 6 ground classes   | + GPS noise, bias   | + 1 Hz GPS   | + control delay   | 6-wheel rover   | all upgrades   | real terrain   | Earth Rover Mini+   | ERC-like   | simple, new missions (control)   |
|:-----------------|:----------|:----------|:---------------------|:--------------------|:-------------|:------------------|:----------------|:---------------|:---------------|:--------------------|:-----------|:---------------------------------|
| Go-to-goal       | 0.876 (7) | 0.853 (7) | 0.781 (7)            | 0.844 (5)           | 0.874 (6)    | 0.846 (5)         | 0.815 (7)       | 0.510 (3)      | 0.796 (7)      | 0.844 (4)           | 0.303 (3)  | 0.870 (7)                        |
| Pure Pursuit     | 0.885 (6) | 0.857 (6) | 0.790 (6)            | 0.791 (7)           | 0.839 (7)    | 0.841 (7)         | 0.827 (6)       | 0.226 (7)      | 0.817 (6)      | 0.813 (5)           | 0.168 (7)  | 0.881 (6)                        |
| Stanley          | 0.897 (3) | 0.872 (3) | 0.813 (3)            | 0.867 (3)           | 0.888 (3)    | 0.855 (3)         | 0.841 (3)       | 0.412 (4)      | 0.826 (3)      | 0.878 (3)           | 0.191 (6)  | 0.892 (3)                        |
| LQR              | 0.891 (4) | 0.866 (4) | 0.799 (5)            | 0.859 (4)           | 0.875 (4)    | 0.850 (4)         | 0.836 (4)       | 0.385 (5)      | 0.821 (5)      | 0.778 (7)           | 0.217 (4)  | 0.885 (4)                        |
| MPC              | 0.890 (5) | 0.866 (5) | 0.802 (4)            | 0.842 (6)           | 0.874 (5)    | 0.845 (6)         | 0.834 (5)       | 0.371 (6)      | 0.825 (4)      | 0.787 (6)           | 0.199 (5)  | 0.885 (5)                        |
| PPO (nominal)    | 0.955 (1) | 0.939 (1) | 0.856 (1)            | 0.911 (1)           | 0.954 (1)    | 0.905 (2)         | 0.905 (1)       | 0.543 (2)      | 0.889 (1)      | 0.923 (2)           | 0.334 (2)  | 0.951 (1)                        |
| PPO (randomised) | 0.945 (2) | 0.930 (2) | 0.845 (2)            | 0.907 (2)           | 0.951 (2)    | 0.910 (1)         | 0.862 (2)       | 0.585 (1)      | 0.882 (2)      | 0.931 (1)           | 0.364 (1)  | 0.945 (2)                        |

## Success rate per simulator

|                  | simple   | + hills   | + 6 ground classes   | + GPS noise, bias   | + 1 Hz GPS   | + control delay   | 6-wheel rover   | all upgrades   | real terrain   | Earth Rover Mini+   | ERC-like   | simple, new missions (control)   |
|:-----------------|:---------|:----------|:---------------------|:--------------------|:-------------|:------------------|:----------------|:---------------|:---------------|:--------------------|:-----------|:---------------------------------|
| Go-to-goal       | 100%     | 100%      | 100%                 | 100%                | 100%         | 100%              | 100%            | 72%            | 96%            | 100%                | 40%        | 100%                             |
| Pure Pursuit     | 100%     | 100%      | 100%                 | 98%                 | 100%         | 100%              | 100%            | 34%            | 98%            | 98%                 | 24%        | 100%                             |
| Stanley          | 100%     | 100%      | 100%                 | 100%                | 100%         | 100%              | 100%            | 60%            | 98%            | 100%                | 26%        | 100%                             |
| LQR              | 100%     | 100%      | 100%                 | 100%                | 100%         | 100%              | 100%            | 56%            | 98%            | 98%                 | 30%        | 100%                             |
| MPC              | 100%     | 100%      | 100%                 | 98%                 | 100%         | 100%              | 100%            | 54%            | 98%            | 98%                 | 26%        | 100%                             |
| PPO (nominal)    | 100%     | 100%      | 99%                  | 99%                 | 100%         | 100%              | 99%             | 73%            | 98%            | 97%                 | 42%        | 100%                             |
| PPO (randomised) | 100%     | 100%      | 99%                  | 100%                | 100%         | 100%              | 96%             | 78%            | 98%            | 98%                 | 45%        | 100%                             |

## Agreement with the simple simulator (best there: PPO (nominal))

| simulator                      | Kendall's tau vs simple   | best policy      | same best as simple (bootstrap)   |
|:-------------------------------|:--------------------------|:-----------------|:----------------------------------|
| + hills                        | 1.00 [0.81, 1.00]         | PPO (nominal)    | 91%                               |
| + 6 ground classes             | 0.90 [0.81, 1.00]         | PPO (nominal)    | 80%                               |
| + GPS noise, bias              | 0.81 [0.62, 0.90]         | PPO (nominal)    | 72%                               |
| + 1 Hz GPS                     | 0.90 [0.62, 0.90]         | PPO (nominal)    | 79%                               |
| + control delay                | 0.71 [0.62, 0.90]         | PPO (randomised) | 48%                               |
| 6-wheel rover                  | 1.00 [0.52, 1.00]         | PPO (nominal)    | 86%                               |
| all upgrades                   | 0.52 [0.24, 0.62]         | PPO (randomised) | 25%                               |
| real terrain                   | 0.90 [0.62, 1.00]         | PPO (nominal)    | 84%                               |
| Earth Rover Mini+              | 0.33 [0.33, 0.62]         | PPO (randomised) | 44%                               |
| ERC-like                       | 0.33 [0.05, 0.52]         | PPO (randomised) | 25%                               |
| simple, new missions (control) | 1.00 [0.81, 1.00]         | PPO (nominal)    | 82%                               |
