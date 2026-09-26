# CLAUDE.md: rules for working on WhAVs testbed with Claude Code

## Project
Group repo for the WhAVs simulator testbed (base: MVSim). This file covers the lightweight Python testbed in `src/testbed/`. See README.md for the architecture.

## Conventions
- SI units everywhere: metres, seconds, radians (convert to degrees only for display).
- Angles wrapped with `testbed.vehicles.base.wrap_angle`.
- Vehicle reference point = rear axle for kinematic models, CoG for the dynamic model.
- Cross-track error sign: positive = vehicle is left of the path.
- Every controller subclasses `Controller` and implements `compute(state, path, dt) -> (steer, accel)`.
- Every vehicle subclasses `VehicleModel` and implements `derivatives(x, delta, a)`.
- Type hints + a docstring with the governing equation on every new class.
- New features come with a test in `tests/` and a YAML config in `experiments/configs/` if they are user-facing.
- Plots use the fixed controller colours in `src/testbed/viz/plots.py` (colour follows the controller, never its rank).

## Commands
- Tests: `pytest -q` (or `python tests/run_all.py`)
- One scenario: `python scripts/run_scenario.py experiments/configs/<file>.yaml`
- Benchmark: `python scripts/benchmark.py [--model dynamic_bicycle]`

## Do not
- Add heavy dependencies (ROS, cvxpy, torch) to `src/testbed` without discussing; the Python testbed must run on plain pip installs (MVSim / ROS work lives in `worlds/*.xml`, `scripts/` and `external/`).
- Change benchmark settings without regenerating `experiments/results/` and updating README results.
