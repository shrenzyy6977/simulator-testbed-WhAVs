# Simulator Testbed for Wheeled Autonomous Vehicles (WhAVs)

Foundations of Robotics — course project, **Group 1**.

## Team

| Name | Roll No. | GitHub | Program |
|---|---|---|---|
| Shranay Malhotra | IIT2023093 | shrenzyy6977 | BTech |
| Harkirat Chadha | IIT2023096 | hchadha28 | BTech |
| Ujjwal Mishra | IIT2023177 | UjjWd | BTech |
| Sparsh Garg | IIT2023186 | sparshga | BTech |
| Malay Kumar Jain | MRM2025003 | jainmalaykumar | MTech |

## Project summary

We are building a software simulator / testbed for **Wheeled Autonomous Vehicles (WhAVs)**.
Topic and base work were confirmed with our TA.

- **Base paper:** J.-L. Blanco-Claraco et al., "MultiVehicle Simulator (MVSim): Lightweight dynamics simulator for multiagents and mobile robotics research", SoftwareX 23 (2023) 101443, https://doi.org/10.1016/j.softx.2023.101443
- **Base code:** https://github.com/MRPT/mvsim (docs: https://mvsimulator.readthedocs.io)
- **Scope / our contribution:** TODO (fill in once agreed with the TA)

## Repository layout

```
external/mvsim/   upstream MVSim (git submodule, pinned to a known commit)  [if added]
worlds/           our own MVSim world files (.xml)
scripts/          our Python clients and experiment scripts
src/testbed/      our own code
experiments/      experiment configs and results
docs/             literature notes, simulator comparison, weekly progress logs
report/           submitted progress reports (PDF)
```

## Getting started (Ubuntu or WSL2 on Windows)

MVSim's documented install path is via ROS 2 packages (see https://mvsimulator.readthedocs.io/en/latest/install.html):

```bash
# after installing ROS 2 for your Ubuntu version
sudo apt install ros-$ROS_DISTRO-mvsim
ros2 launch mvsim demo_warehouse.launch.py
```

Building from source is described on the same page. Record the exact steps that worked
for us (OS, versions, problems) in `docs/progress/`.

## Python testbed (`src/testbed`)

A lightweight 2D Python testbed that follows MVSim's architecture (world, vehicles, sensors,
simulation loop). It lets us develop and benchmark planners and path-tracking controllers on any
laptop, without a ROS install. Work in progress.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/run_all.py          # or: pytest -q
```

With conda instead (also installs MuJoCo / PyTorch / Stable-Baselines3 for the 3D work):

```bash
conda env create -f environment.yml && conda activate whavs
```

### Path-tracking controllers

`src/testbed/control/` has four lateral controllers behind one interface
(`compute(state, path, dt) -> (steer, accel)`), all sharing the same PI speed controller so a
comparison isolates the steering law:

| Controller | File | Law |
|---|---|---|
| Pure Pursuit | `control/geometric.py` | `delta = atan(2 L sin(alpha) / Ld)`, `Ld = k v + Ld0`, rear axle |
| Stanley | `control/geometric.py` | `delta = -theta_e - atan(k e / (ks + v))`, front axle |
| LQR | `control/optimal.py` | DARE on the kinematic lateral-error model + curvature feed-forward |
| Linear MPC | `control/optimal.py` | 20-step condensed QP with steering bounds, solved by scipy BVLS (no extra solver) |

```bash
python scripts/run_scenario.py experiments/configs/figure_eight.yaml       # one scenario, all controllers
python scripts/run_scenario.py experiments/configs/warehouse_astar.yaml    # A* path + collision checking
python scripts/benchmark.py --model dynamic_bicycle                        # full benchmark
```

Outputs go to `experiments/results/` (plots, CSV, `summary.md`).

### Benchmark results

Settings in `experiments/configs/benchmark.yaml`: 5 paths x 3 target speeds (5, 10, 15 m/s, capped on
curves by a lateral acceleration of 4 m/s^2), 1 m initial lateral offset, errors measured after 5 s.
RMS cross-track error [m] at `v_target` = 15 m/s:

| Path | Model | Pure Pursuit | Stanley | LQR | MPC |
|---|---|---|---|---|---|
| s_curve | kinematic bicycle | 0.206 | 0.146 | **0.003** | 0.024 |
| figure_eight | kinematic bicycle | 0.218 | 0.276 | **0.004** | 0.047 |
| lane_change | kinematic bicycle | 0.127 | 0.048 | **0.002** | 0.029 |
| s_curve | dynamic bicycle | 0.233 | 0.187 | **0.128** | 0.163 |
| figure_eight | dynamic bicycle | 0.311 | 0.169 | **0.167** | 0.192 |
| lane_change | dynamic bicycle | 0.154 | **0.088** | 0.098 | 0.089 |

Mean controller time per call: Pure Pursuit 0.011 ms, Stanley 0.011 ms, LQR 0.029 ms, MPC 0.18 ms
(Apple M4, single thread). Full tables: `experiments/results/benchmark_*/summary.md`.

Observations:
- On the kinematic bicycle, LQR and MPC use the same model as the plant, so they track almost
  perfectly. Once tyre slip is simulated (dynamic bicycle) the gap shrinks and Stanley becomes
  best on the lane change: **the ranking changes with the vehicle model**, which is the question
  raised in our literature review.
- Stanley's error on the circle (0.16 m on the kinematic model) is geometric: it holds the front
  axle on the path, while errors are measured at the rear axle, giving `R - sqrt(R^2 - L^2)`.
- Differential drive gives the same numbers as the kinematic bicycle because its wheel-speed limit
  is never reached at these speeds.

## Workflow

- Work on a branch named `feature/<name>-<task>` and open a PR into `main`.
- Every PR gets at least one review from another member.
- Track tasks in GitHub Issues; milestones are `Week 1`, `Week 2`, ...
- Weekly logs live in `docs/progress/`.
