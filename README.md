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
src/testbed/      our own code: 2D testbed (numpy/scipy only)
src/terrain3d/    our own code: 3D terrain simulator on MuJoCo
robots/           robot URDFs
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

## 3D terrain simulator (`src/terrain3d`, MuJoCo)

First prototype of Option B from the progress review: a map-based 3D simulator in MuJoCo with
ground classes, robots loaded from URDF, and an ERC-style Gymnasium interface. It lives in its own
package because it needs MuJoCo; `src/testbed` stays plain numpy/scipy.

| Piece | File | What it does |
|---|---|---|
| Terrain | `terrain3d/terrain.py` | Elevation grid (flat, hills, slope, or a `.npy` / PNG elevation map) cut into tiles; each tile is a heightfield with its own ground class |
| Ground classes | `terrain3d/terrain.py` | asphalt, grass, gravel, sand, snow, mud: sliding / torsional / rolling friction and contact softness (sinkage proxy). Values are placeholders to calibrate |
| Robot | `robots/rover4wd.urdf`, `terrain3d/world.py` | Any URDF with joints named `wheel_*` is loaded with MjSpec, gets a free base and velocity-controlled wheel motors (skid steering) |
| Classical baselines | `terrain3d/tracking.py` | The four controllers from `src/testbed` drive the robot unchanged (steering output -> yaw-rate command) |
| RL interface | `terrain3d/env.py` | `RoverNavEnv`: reach GNSS checkpoints with noisy GPS (0.3 m), 10 Hz control, 1-step control delay; random map and ground patches per episode |

```bash
python scripts/terrain_benchmark.py          # 4 controllers x 6 ground classes x 3 paths
python scripts/terrain_demo.py               # mixed-terrain figure + 3D render
python scripts/train_ppo.py --timesteps 500000 --envs 8
```

First results (`experiments/results/terrain_benchmark/summary.md`), s-curve at 0.8 m/s over
+/-0.15 m hills, RMS cross-track error [m]:

| Ground | Pure Pursuit | Stanley | LQR | MPC | mean wheel slip |
|---|---|---|---|---|---|
| asphalt | 0.995 | **0.359** | 0.697 | 0.665 | 0.00 |
| grass | 0.076 | **0.050** | 0.101 | 0.100 | 0.04 |
| gravel | 0.096 | **0.077** | 0.153 | 0.135 | 0.02 |
| sand | 0.168 | **0.141** | 0.186 | 0.180 | 0.20 |
| snow | 0.193 | **0.154** | 0.207 | 0.222 | 0.14 |
| mud | 0.360 | **0.210** | 0.303 | 0.295 | 0.34 |

- In 2D, LQR was best on the s-curve; on 3D terrain with wheel slip, Stanley is best on every
  ground class, and on the circle Pure Pursuit is best on most grounds. The ranking depends on the
  ground and the path, which is RQ2 of the review.
- Speed: median ~130x real time per run on one CPU core (physics at 500 Hz); the Gym env runs
  ~1,300 steps/s single-threaded and ~5,000 steps/s with 8 processes on an M4.
- PPO baseline: 500k steps (105 s, 8 processes) reach all 3 checkpoints in 60/60 evaluation
  episodes on unseen random maps (`experiments/results/ppo/eval.csv`). The current mission is
  clearly too easy (checkpoints 3-6 m apart, gentle hills); harder ERC/URC-style missions are next.

Known limitations (to fix or calibrate next):
- Skid-steer turning: with MuJoCo's contact model the rover reaches only part of the commanded yaw
  rate, worst on high-friction ground (asphalt), which is why every controller does badly there. Needs
  calibration against real data (e.g. FrodoBots logs) before drawing conclusions about asphalt.
- Ground parameters are plausible placeholders, not measured values; sinkage is approximated by soft
  contacts, there is no deformable soil.
- The rover URDF is a generic small 4WD skid-steer, not the real Earth Rover Mini+ geometry.

## Workflow

- Work on a branch named `feature/<name>-<task>` and open a PR into `main`.
- Every PR gets at least one review from another member.
- Track tasks in GitHub Issues; milestones are `Week 1`, `Week 2`, ...
- Weekly logs live in `docs/progress/`.
