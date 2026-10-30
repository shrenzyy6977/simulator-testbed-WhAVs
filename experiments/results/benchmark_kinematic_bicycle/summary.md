# Benchmark: kinematic_bicycle

Settings: `experiments/configs/benchmark.yaml`. Errors measured at the vehicle reference point after 5.0 s (initial lateral offset 1.0 m).

## RMS cross-track error [m], v_target = 5 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.001 |     0.158 | 0.003 | 0.003 |
| figure_eight |          0.155 |     0.281 | 0.004 | 0.045 |
| lane_change  |          0.054 |     0.048 | 0.002 | 0.019 |
| s_curve      |          0.077 |     0.145 | 0.005 | 0.02  |
| straight     |          0.001 |     0.009 | 0     | 0     |

## RMS cross-track error [m], v_target = 10 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.001 |     0.159 | 0.001 | 0     |
| figure_eight |          0.217 |     0.274 | 0.004 | 0.047 |
| lane_change  |          0.118 |     0.048 | 0.003 | 0.028 |
| s_curve      |          0.199 |     0.145 | 0.003 | 0.024 |
| straight     |          0.001 |     0.008 | 0     | 0     |

## RMS cross-track error [m], v_target = 15 m/s

| path         |   pure_pursuit |   stanley |   lqr |   mpc |
|:-------------|---------------:|----------:|------:|------:|
| circle       |          0.001 |     0.159 | 0.001 | 0     |
| figure_eight |          0.218 |     0.276 | 0.004 | 0.047 |
| lane_change  |          0.127 |     0.048 | 0.002 | 0.029 |
| s_curve      |          0.206 |     0.146 | 0.003 | 0.024 |
| straight     |          0.002 |     0.008 | 0     | 0     |

## Controller compute time per call [ms] (mean over all runs)

| controller   |   mean_compute_ms |   p95_compute_ms |   max_compute_ms |
|:-------------|------------------:|-----------------:|-----------------:|
| pure_pursuit |             0.011 |            0.012 |            0.025 |
| stanley      |             0.011 |            0.012 |            0.023 |
| lqr          |             0.029 |            0.133 |            0.302 |
| mpc          |             0.179 |            0.19  |            0.27  |
