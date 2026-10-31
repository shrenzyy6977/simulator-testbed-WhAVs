"""Calibrate the Earth Rover Mini+ model's skid-steer turning against a real Mini+ log.

    python scripts/calibrate_turning.py [--config experiments/configs/turning_calibration.yaml]

1. Real robot: spin-in-place bursts (linear 0, angular c) from a LeRobot log of a Mini+; the yaw
   rate of a burst is its compass heading change (read `lag_s` after the command ends) over its
   duration. Per unit command this gives the real yaw rate k_real.
2. Firmware: the Mini+ firmware drives each side at v_w = 0.3 * 0.7 * |c| * full_speed when
   spinning, so pure rolling would give omega_kin = 2 v_w / B. The skid factor is
   chi = omega_kin / omega_real (1 = no skidding).
3. Simulation: the model with `lateral_friction` = rho flies the same bursts on hard ground; rho is
   interpolated so that the simulated k matches k_real (and its quartiles, for the uncertainty).

Outputs in experiments/results/turning/: bursts.csv, sweep.csv, summary.md, calibration.png
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terrain3d import Terrain, WheeledRobot  # noqa: E402
from terrain3d.terrain import flat  # noqa: E402
from terrain3d.world import RobotSpec  # noqa: E402
from testbed.sim.scenario import load_config  # noqa: E402

STATE = ["linear.vel", "angular.vel", "battery.level", "orientation.deg", "gps.latitude", "gps.longitude",
         "gps.signal", "signal.level", "vibration", "lamp.state", "voltage", "current", "imu.accel.x",
         "imu.accel.y", "imu.accel.z", "imu.gyro.x", "imu.gyro.y", "imu.gyro.z"]


def load_log(raw: Path, url: str) -> pd.DataFrame:
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, raw)
    d = pd.read_parquet(raw)                                  # needs pyarrow >= 20
    a, s = np.stack(d.action.to_numpy()), np.stack(d["observation.state"].to_numpy())
    return pd.DataFrame({"episode": d.episode_index, "cmd_linear": a[:, 0], "cmd_angular": a[:, 1],
                         "heading_deg": s[:, STATE.index("orientation.deg")]})


def real_bursts(log: pd.DataFrame, lag_s: float, min_turn_deg: float) -> pd.DataFrame:
    """One row per spin burst; `clean` = turned the commanded way by >= min_turn_deg, nothing else commanded in the lag."""
    lag = int(round(lag_s * 10))
    rows = []
    for ep, g in log.groupby("episode"):
        g = g.reset_index(drop=True)
        h = np.degrees(np.unwrap(np.radians(g.heading_deg.to_numpy())))
        spin = (g.cmd_linear == 0) & (g.cmd_angular.abs() > 0.1)
        key = ((spin != spin.shift()) | (g.cmd_angular != g.cmd_angular.shift())).cumsum()
        for _, b in g[spin].groupby(key[spin]):
            i0, i1 = b.index[0], b.index[-1]
            j1 = min(i1 + lag, len(g) - 1)
            c, dur, dh = float(b.cmd_angular.iloc[0]), (i1 - i0 + 1) / 10, float(h[j1] - h[i0])
            quiet = bool((g.cmd_angular.iloc[i1 + 1: j1 + 1] == 0).all())
            right_way = np.sign(dh) == -np.sign(c)               # compass heading grows clockwise; +c = counter-clockwise
            rows.append({"episode": ep, "cmd": c, "duration_s": dur, "turn_deg": dh,
                         "yaw_rate": np.radians(abs(dh)) / dur, "clean": quiet and right_way and abs(dh) >= min_turn_deg})
    return pd.DataFrame(rows)


def sim_rate(spec: RobotSpec, ground: str, cmd: float, omega_full: float, burst_s: float, lag_s: float) -> float:
    """Yaw rate of one simulated burst, measured like the real ones (heading change incl. lag / duration)."""
    rob = WheeledRobot(Terrain.uniform(ground, flat(65), 20.0), spec, start=(0.0, 0.0))
    rob.step(250)
    yaws = [rob.pose()[5]]
    for k in range(int(round((burst_s + lag_s) * 10))):
        rob.command(0.0, cmd * omega_full if k < burst_s * 10 else 0.0)
        rob.step(50)
        yaws.append(rob.pose()[5])
    return float(abs(np.unwrap(yaws)[-1] - yaws[0]) / burst_s)


def plot(bursts: pd.DataFrame, sweep: pd.DataFrame, k: dict, rho: dict, path: Path, attribution: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from testbed.viz.plots import CONTROLLER_COLORS, GRID, REFERENCE_COLOR, TEXT, _style

    blue = CONTROLLER_COLORS["pure_pursuit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    clean = bursts[bursts.clean]
    ax1.plot(bursts[~bursts.clean].cmd.abs(), bursts[~bursts.clean].yaw_rate, "x", color=REFERENCE_COLOR, ms=6,
             label="real bursts, excluded")
    ax1.plot(clean.cmd.abs(), clean.yaw_rate, "o", color=TEXT, ms=6, label="real bursts, clean")
    c = np.linspace(0.3, 0.8, 2)
    ax1.fill_between(c, k["q25"] * c, k["q75"] * c, color=GRID, alpha=0.7, lw=0, label="real, quartiles")
    for r, ls in ((1.0, ":"), (rho["median"], "-")):
        s = sweep[np.isclose(sweep.ratio, r)] if r in set(sweep.ratio) else None
        if s is None:                                          # calibrated ratio: interpolate the sweep per command
            s = pd.DataFrame([{"cmd": cc, "rate": np.interp(r, *zip(*sorted(zip(sweep[sweep.cmd == cc].ratio,
                                                                                   sweep[sweep.cmd == cc].rate))))}
                              for cc in sorted(sweep.cmd.unique())])
        ax1.plot(s.cmd, s.rate, ls, color=blue, lw=2,
                 label=f"model, sideways/forward friction {r:.2f}" + (" (calibrated)" if ls == "-" else " (isotropic)"))
    ax1.set_xlabel("spin command |c|")
    ax1.set_ylabel("yaw rate [rad/s]")
    ax1.set_title("Turning on the spot: real Mini+ vs model", fontsize=10, color=TEXT, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1)

    per = sweep.groupby("ratio").k.mean()
    ax2.plot(per.index, per.values, "-o", color=blue, lw=2, ms=6)
    ax2.axhspan(k["q25"], k["q75"], color=GRID, alpha=0.7, lw=0)
    ax2.axhline(k["median"], color=TEXT, lw=1, ls="--")
    ax2.annotate(f"real median {k['median']:.2f}", (per.index.max(), k["median"]), xytext=(-4, 4),
                 textcoords="offset points", ha="right", fontsize=8, color=TEXT)
    ax2.axvline(rho["median"], color=TEXT, lw=0.8)
    ax2.set_xlabel("sideways / forward tyre friction")
    ax2.set_ylabel("yaw rate per unit command [rad/s]")
    ax2.set_title("Model response vs friction ratio", fontsize=10, color=TEXT, loc="left")
    _style(ax2)
    fig.text(0.01, 0.005, f"Log: {attribution}", fontsize=7, color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "turning_calibration.yaml"))
    cfg = load_config(ap.parse_args().config)
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)

    bc, fw = cfg["bursts"], cfg["firmware"]
    bursts = real_bursts(load_log(ROOT / cfg["log"]["raw"], cfg["log"]["url"]), bc["lag_s"], bc["min_turn_deg"])
    bursts.to_csv(out / "bursts.csv", index=False)
    per_cmd = (bursts[bursts.clean].yaw_rate / bursts[bursts.clean].cmd.abs()).to_numpy()
    k = {"median": float(np.median(per_cmd)), "q25": float(np.percentile(per_cmd, 25)), "q75": float(np.percentile(per_cmd, 75))}

    rob = dict(cfg["robot"])
    rob["urdf"] = ROOT / rob["urdf"]
    track = rob["track"]
    omega_full = 2 * fw["spin_target_scale"] * fw["spin_duty_per_cmd"] * fw["full_speed"] / track   # firmware, full command
    rows = []
    for ratio in cfg["ratios"]:
        spec = RobotSpec(**rob, lateral_friction=ratio)
        for c in cfg["commands"]:
            r = sim_rate(spec, cfg["ground"], c, omega_full, cfg["burst_s"], bc["lag_s"])
            rows.append({"ratio": ratio, "cmd": c, "rate": r, "k": r / c})
        print(f"ratio {ratio:.2f}: yaw rate per unit command {np.mean([x['k'] for x in rows[-len(cfg['commands']):]]):.2f} rad/s")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "sweep.csv", index=False)
    per = sweep.groupby("ratio").k.mean().sort_values()                 # k grows as the ratio falls
    rho = {q: float(np.interp(k[q], per.values, per.index)) for q in ("median", "q25", "q75")}
    span = {q: ("< " if k[q] > per.max() else "> " if k[q] < per.min() else "") + f"{rho[q]:.2f}" for q in rho}   # beyond the sweep
    chi = {q: omega_full / k[q] for q in k}

    n_clean = int(bursts.clean.sum())
    lines = ["# Skid-steer turning calibration (Earth Rover Mini+)", "",
             f"Log: {cfg['log']['attribution']}. {len(bursts)} spin-in-place bursts, {n_clean} clean (turned the "
             f"commanded way by >= {bc['min_turn_deg']} deg, no other turn command within {bc['lag_s']} s after).", "",
             "| | median | quartiles |", "|---|---|---|",
             f"| real yaw rate per unit spin command [rad/s] | {k['median']:.2f} | {k['q25']:.2f}-{k['q75']:.2f} |",
             f"| pure-rolling yaw rate per unit command (firmware) [rad/s] | {omega_full:.2f} | |",
             f"| skid factor chi = rolling / real | {chi['median']:.1f} | {chi['q75']:.1f}-{chi['q25']:.1f} |",
             f"| model sideways / forward tyre friction that reproduces it ({cfg['ground']}) | **{rho['median']:.2f}** | "
             f"{span['q25']} to {span['q75']} |", "",
             f"With isotropic friction (1.0) the model turns at {per.loc[1.0]:.2f} rad/s per unit command, "
             f"{k['median'] / per.loc[1.0]:.1f}x slower than the real robot.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    plot(bursts, sweep, k, rho, out / "calibration.png", cfg["log"]["attribution"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
