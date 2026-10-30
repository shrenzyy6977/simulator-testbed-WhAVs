"""3D terrain benchmark: every controller x ground class x path on the MuJoCo rover.

    python scripts/terrain_benchmark.py [--config experiments/configs/terrain_benchmark.yaml] [--out DIR]

Outputs in DIR (default experiments/results/terrain_benchmark/):
    results.csv, summary.md, rms_cte_<path>.png (grouped bars: ground class x controller)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terrain3d import RobotSpec, make_terrain  # noqa: E402
from terrain3d.tracking import TrackConfig, shift_path, track  # noqa: E402
from testbed.control import make_controller  # noqa: E402
from testbed.metrics import compute_metrics  # noqa: E402
from testbed.planning.path import make_path  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402
from testbed.viz import plot_benchmark_bars  # noqa: E402


def build_path(pc: dict, v_target: float, a_lat_max: float):
    pc = dict(pc)
    dx, dy = pc.pop("offset", (0.0, 0.0))
    return shift_path(make_path(pc.pop("name"), v_target=v_target, a_lat_max=a_lat_max, ds=0.05, **pc), dx, dy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "terrain_benchmark.yaml"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "terrain_benchmark"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    robot = RobotSpec(urdf=ROOT / cfg["robot"])
    tcfg = TrackConfig(**cfg.get("track", {}))
    rows = []
    t_all = time.time()
    for ground in cfg["grounds"]:
        terrain = make_terrain({**cfg["terrain"], "ground": ground})
        for pc in cfg["paths"]:
            path = build_path(pc, cfg["v_target"], cfg["a_lat_max"])
            for cc in cfg["controllers"]:
                t0 = time.perf_counter()
                log = track(terrain, make_controller(**cc), path, tcfg, robot)
                wall = time.perf_counter() - t0
                a = log.arrays()
                r = {"ground": ground, **compute_metrics(log, cfg.get("settle_time", 0.0)),
                     "mean_slip": float(np.mean(a["slip"])), "real_time_factor": float(a["t"][-1] / wall)}
                rows.append(r)
                print(f"{ground:8s} {r['path']:13s} {r['controller']:13s} {r['status']:9s} "
                      f"rms_cte={r['rms_cte']:.3f} m  slip={r['mean_slip']:.2f}  {r['real_time_factor']:.0f}x real time")
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    order = [c["name"] for c in cfg["controllers"]]
    lines = ["# Terrain benchmark (MuJoCo)", "",
             f"Settings: `{Path(args.config).name}`. Robot `{cfg['robot']}`, hills of +/-{cfg['terrain']['elevation'].get('amplitude', 0)} m, "
             f"v_target {cfg['v_target']} m/s, control {tcfg.control_hz:g} Hz. Errors after {cfg.get('settle_time', 0)} s.", ""]
    for p, d in df.groupby("path", sort=False):
        cell = d.apply(lambda r: f"{r.rms_cte:.3f}" + ("" if r.status == "finished" else f" ({r.status})"), axis=1)
        tab = d.assign(cell=cell).pivot(index="ground", columns="controller", values="cell").loc[cfg["grounds"], order]
        slip = d.groupby("ground")["mean_slip"].mean().loc[cfg["grounds"]].round(2)
        lines += [f"## {p}: RMS cross-track error [m]", "", tab.assign(mean_slip=slip).to_markdown(), ""]
        bars = d.rename(columns={"path": "_path", "ground": "path"}).to_dict("records")   # one bar group per ground class
        plot_benchmark_bars(bars, "rms_cte", out / f"rms_cte_{p}.png", "RMS cross-track error [m]",
                            f"{p} on 3D terrain: RMS cross-track error by ground class")
    rtf = df["real_time_factor"]
    lines += ["## Speed", "", f"{len(df)} runs, real-time factor median {rtf.median():.0f}x "
              f"(min {rtf.min():.0f}x, max {rtf.max():.0f}x), one CPU core, physics at 500 Hz.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"\n{len(df)} runs in {time.time() - t_all:.0f} s. Results in {out}")


if __name__ == "__main__":
    main()
