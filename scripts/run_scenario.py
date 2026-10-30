"""Run one scenario config: every listed controller on the same vehicle, path and world.

    python scripts/run_scenario.py experiments/configs/figure_eight.yaml [--out experiments/results]

Writes <name>.png (trajectories + cross-track error) and <name>.csv (metrics) and prints a table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from testbed.metrics import compute_metrics  # noqa: E402
from testbed.sim import simulate  # noqa: E402
from testbed.sim.scenario import build_controllers, build_path, build_sim, build_vehicle, build_world, load_config  # noqa: E402
from testbed.viz import plot_runs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--model", help="override the vehicle model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.setdefault("vehicle", {})["model"] = args.model
    world = build_world(cfg)
    path = build_path(cfg, world)
    sim = build_sim(cfg)
    logs, rows = [], []
    for ctrl in build_controllers(cfg):
        log = simulate(build_vehicle(cfg), ctrl, path, sim, world)
        logs.append(log)
        rows.append(compute_metrics(log, cfg.get("settle_time", 0.0)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.get('name', Path(args.config).stem)}_{logs[0].vehicle}"
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}.csv", index=False)
    plot_runs(path, logs, out / f"{name}.png", world)
    cols = ["controller", "status", "time_s", "rms_cte", "max_cte", "rms_heading_err", "mean_compute_ms"]
    print(df[cols].to_markdown(index=False, floatfmt=".3f"))
    print(f"\nSaved {out / (name + '.png')} and {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
