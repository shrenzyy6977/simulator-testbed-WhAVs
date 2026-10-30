# Terrain benchmark (MuJoCo)

Settings: `terrain_benchmark.yaml`. Robot `robots/rover4wd.urdf`, hills of +/-0.15 m, v_target 0.8 m/s, control 20 Hz. Errors after 2.0 s.

## s_curve: RMS cross-track error [m]

| ground   |   pure_pursuit |   stanley |   lqr |   mpc |   mean_slip |
|:---------|---------------:|----------:|------:|------:|------------:|
| asphalt  |          0.995 |     0.359 | 0.697 | 0.665 |       -0    |
| grass    |          0.076 |     0.05  | 0.101 | 0.1   |        0.04 |
| gravel   |          0.096 |     0.077 | 0.153 | 0.135 |        0.02 |
| sand     |          0.168 |     0.141 | 0.186 | 0.18  |        0.2  |
| snow     |          0.193 |     0.154 | 0.207 | 0.222 |        0.14 |
| mud      |          0.36  |     0.21  | 0.303 | 0.295 |        0.34 |

## circle: RMS cross-track error [m]

| ground   |   pure_pursuit |   stanley |   lqr |   mpc |   mean_slip |
|:---------|---------------:|----------:|------:|------:|------------:|
| asphalt  |          0.396 |     0.341 | 0.565 | 0.536 |        0    |
| grass    |          0.093 |     0.115 | 0.169 | 0.164 |        0.04 |
| gravel   |          0.073 |     0.086 | 0.139 | 0.133 |        0.02 |
| sand     |          0.198 |     0.261 | 0.319 | 0.328 |        0.22 |
| snow     |          0.248 |     0.316 | 0.355 | 0.377 |        0.16 |
| mud      |          0.287 |     0.372 | 0.433 | 0.45  |        0.33 |

## figure_eight: RMS cross-track error [m]

| ground   | pure_pursuit      |   stanley |   lqr |   mpc |   mean_slip |
|:---------|:------------------|----------:|------:|------:|------------:|
| asphalt  | 1.242 (off_track) |     0.359 | 0.617 | 0.593 |        0.01 |
| grass    | 0.093             |     0.072 | 0.129 | 0.121 |        0.04 |
| gravel   | 0.095             |     0.064 | 0.134 | 0.12  |        0.02 |
| sand     | 0.220             |     0.218 | 0.252 | 0.253 |        0.21 |
| snow     | 0.273             |     0.244 | 0.257 | 0.366 |        0.14 |
| mud      | 0.602             |     0.412 | 0.49  | 0.466 |        0.36 |

## Speed

72 runs, real-time factor median 113x (min 73x, max 133x), one CPU core, physics at 500 Hz.
