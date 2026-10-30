# Benchmark: dynamic_bicycle

Settings: `experiments/configs/benchmark.yaml`. Errors measured at the vehicle reference point after 5.0 s (initial lateral offset 1.0 m).

## RMS cross-track error [m], v_target = 5 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.084 |     0.026 | 0.071 | 0.076 |
| figure_eight |          0.221 |     0.054 | 0.147 | 0.144 |
| lane_change  |          0.05  |     0.02  | 0.032 | 0.021 |
| s_curve      |          0.109 |     0.038 | 0.072 | 0.073 |
| straight     |          0.001 |     0.006 | 0     | 0     |

## RMS cross-track error [m], v_target = 10 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.255 |     0.313 | 0.131 | 0.183 |
| figure_eight |          0.309 |     0.168 | 0.166 | 0.191 |
| lane_change  |          0.137 |     0.073 | 0.089 | 0.079 |
| s_curve      |          0.235 |     0.189 | 0.128 | 0.162 |
| straight     |          0.002 |     0.004 | 0     | 0     |

## RMS cross-track error [m], v_target = 15 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.255 |     0.313 | 0.131 | 0.183 |
| figure_eight |          0.311 |     0.169 | 0.167 | 0.192 |
| lane_change  |          0.154 |     0.088 | 0.098 | 0.089 |
| s_curve      |          0.233 |     0.187 | 0.128 | 0.163 |
| straight     |          0.002 |     0.004 | 0     | 0     |

## Controller compute time per call [ms] (mean over all runs)

| controller   |   mean_compute_ms |   p95_compute_ms |   max_compute_ms |
|:-------------|------------------:|-----------------:|-----------------:|
| pure_pursuit |             0.011 |            0.012 |            0.023 |
| stanley      |             0.011 |            0.011 |            0.023 |
| lqr          |             0.03  |            0.134 |            0.299 |
| mpc          |             0.179 |            0.19  |            0.266 |
