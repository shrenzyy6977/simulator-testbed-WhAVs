"""Benchmark every controller on every test path and speed (settings: experiments/configs/benchmark.yaml).

    python scripts/benchmark.py [--model kinematic_bicycle|dynamic_bicycle|diff_drive] [--out DIR]

Outputs in DIR (default experiments/results/benchmark_<model>/):
    results.csv          one row per (path, speed, controller)
    summary.md           RMS cross-track error tables per speed + compute time
    rms_cte_<v>.png      grouped bars of RMS cross-track error per path
    compute_time.png     mean controller time per call
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from testbed.control import make_controller  # noqa: E402
from testbed.metrics import compute_metrics  # noqa: E402
from testbed.planning.path import make_path  # noqa: E402
from testbed.sim import SimConfig, simulate  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402
from testbed.vehicles.models import make_vehicle  # noqa: E402
from testbed.viz import plot_benchmark_bars  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="kinematic_bicycle")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "benchmark.yaml"))
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or ROOT / "experiments" / "results" / f"benchmark_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    sim = SimConfig(**cfg.get("sim", {}))
    rows = []
    t0 = time.time()
    for v in cfg["speeds"]:
        for pc in cfg["paths"]:
            pc = dict(pc)
            path = make_path(pc.pop("name"), v_target=v, a_lat_max=cfg.get("a_lat_max", 3.0), **pc)
            for cc in cfg["controllers"]:
                log = simulate(make_vehicle(args.model), make_controller(**cc), path, sim)
                rows.append({"v_target": v, **compute_metrics(log, cfg.get("settle_time", 0.0))})
                r = rows[-1]
                print(f"v={v:4.1f}  {r['path']:13s} {r['controller']:13s} {r['status']:9s} "
                      f"rms_cte={r['rms_cte']:.3f} m  max={r['max_cte']:.3f} m  {r['mean_compute_ms']:.3f} ms")
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    lines = [f"# Benchmark: {args.model}", "",
             f"Settings: `{Path(args.config).relative_to(ROOT) if Path(args.config).is_relative_to(ROOT) else args.config}`. "
             f"Errors measured at the vehicle reference point after {cfg.get('settle_time', 0.0)} s "
             f"(initial lateral offset {sim.init_offset} m).", ""]
    for v, d in df.groupby("v_target"):
        lines += [f"## RMS cross-track error [m], v_target = {v:g} m/s", ""]
        cell = d.apply(lambda r: f"{r.rms_cte:.3f}" + ("" if r.status == "finished" else f" ({r.status})"), axis=1)
        tab = d.assign(cell=cell).pivot(index="path", columns="controller", values="cell")
        lines += [tab[[c["name"] for c in cfg["controllers"]]].to_markdown(), ""]
        plot_benchmark_bars(d.to_dict("records"), "rms_cte", out / f"rms_cte_{v:g}.png", "RMS cross-track error [m]",
                            f"{args.model}, v_target = {v:g} m/s")
    ct = df.groupby("controller")[["mean_compute_ms", "p95_compute_ms", "max_compute_ms"]].mean()
    lines += ["## Controller compute time per call [ms] (mean over all runs)", "",
              ct.loc[[c["name"] for c in cfg["controllers"]]].round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    ct_rows = [{"path": "all runs", "controller": c, "mean_compute_ms": ct.loc[c, "mean_compute_ms"]} for c in ct.index]
    plot_benchmark_bars(ct_rows, "mean_compute_ms", out / "compute_time.png", "time per call [ms]",
                        "Mean controller compute time")
    print(f"\n{len(df)} runs in {time.time() - t0:.1f} s. Results in {out}")


if __name__ == "__main__":
    main()
