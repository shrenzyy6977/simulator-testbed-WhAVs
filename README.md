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

## Workflow

- Work on a branch named `feature/<name>-<task>` and open a PR into `main`.
- Every PR gets at least one review from another member.
- Track tasks in GitHub Issues; milestones are `Week 1`, `Week 2`, ...
- Weekly logs live in `docs/progress/`.
