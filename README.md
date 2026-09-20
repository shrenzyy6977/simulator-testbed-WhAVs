# Simulator Testbed (WhAVs) — Foundations of Robotics, Group 1

Course project for **Foundations of Robotics**. Topic: **Simulator testbed (WhAVs)**.

## Team

| Name | Roll No. | GitHub | Program |
|---|---|---|---|
| Shranay Malhotra | IIT2023093 | shrenzyy6977 | BTech |
| Harkirat Chadha | IIT2023096 | hchadha28 | BTech |
| Ujjwal Mishra | IIT2023177 | UjjWd | BTech |
| Sparsh Garg | IIT2023186 | sparshga | BTech |
| Malay Kumar Jain | MRM2025003 | jainmalaykumar | MTech |

## Goal

TODO: 2-3 sentences on what this testbed will do (fill in from the project brief).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/hello_sim.py         # headless baseline check
python scripts/hello_sim.py --gui   # with 3D viewer
```

Outputs (trajectory CSV/plot and a snapshot) are written to `experiments/results/`.

## Repository structure

```
scripts/       runnable entry points (hello_sim.py = baseline environment check)
src/testbed/   testbed code (vehicle models, scenarios, logging)
experiments/   experiment configs and results
docs/          literature notes, simulator comparison, weekly progress logs
report/        submitted progress reports (PDF)
```

## Workflow

- Work on a branch named `feature/<name>-<task>` and open a PR into `main`.
- Every PR gets at least one review from another member.
- Track tasks in GitHub Issues; milestones are `Week 1`, `Week 2`, ...
- Weekly logs live in `docs/progress/`.
