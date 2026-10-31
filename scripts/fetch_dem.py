"""Cut real elevation patches for the 3D simulator from a swissALTI3D tile.

    python scripts/fetch_dem.py [--config experiments/configs/dem_swissalti3d.yaml]

Downloads the tile once into data/dem/ (gitignored), grids its XYZ points, picks random
`patch_size` squares whose 95th-percentile slope and height range are in the configured bounds,
resamples each to n x n, removes its mean height and saves them as one .npz for
`make_terrain({"elevation": {"type": "dem", "path": ...}})`. Also draws a figure of the tile with
the chosen patches.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from testbed.sim.scenario import load_config  # noqa: E402


def load_tile(raw: Path, url: str) -> tuple[np.ndarray, float, float, float]:
    """Elevation grid [row = north, col = east] (row 0 = south edge), its SW corner (E, N) and spacing."""
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, raw)
    with zipfile.ZipFile(raw) as zf:
        with zf.open(zf.namelist()[0]) as f:
            pts = pd.read_csv(f, sep=" ", dtype=float).to_numpy()
    e, n = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    z = np.full((len(n), len(e)), np.nan)
    z[np.searchsorted(n, pts[:, 1]), np.searchsorted(e, pts[:, 0])] = pts[:, 2]
    return z, float(e[0]), float(n[0]), float(e[1] - e[0])


def slope_deg(z: np.ndarray, spacing: float) -> np.ndarray:
    gy, gx = np.gradient(z, spacing)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def plot(z, slope, boxes, keep, spacing, path: Path, attribution: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    from testbed.viz.plots import TEXT, _style

    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_axes([0.04, 0.1, 0.42, 0.82])
    shade = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=2, dx=spacing, dy=spacing)
    ext = [0, z.shape[1] * spacing, 0, z.shape[0] * spacing]
    ax.imshow(shade, cmap="gray", origin="lower", extent=ext)
    for (r, c, w) in boxes:
        ax.add_patch(plt.Rectangle((c * spacing, r * spacing), w * spacing, w * spacing, fill=False, lw=1.2, ec="#eb6834"))
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(f"{len(boxes)} patches (orange) on the 1 km tile", fontsize=10, color=TEXT, loc="left")
    _style(ax)
    ax.grid(False)
    show = keep[:6]
    lim = max(np.abs(z_p).max() for z_p in show)                  # one colour scale for every panel
    for i, z_p in enumerate(show):
        a = fig.add_axes([0.50 + 0.15 * (i % 3), 0.53 - 0.43 * (i // 3), 0.135, 0.37])
        im = a.imshow(z_p, cmap="viridis", origin="lower", extent=[0, 20, 0, 20], vmin=-lim, vmax=lim)
        a.set_title(f"patch {i}: {np.ptp(z_p):.1f} m relief", fontsize=8, color=TEXT)
        a.set_xticks([])
        a.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_axes([0.95, 0.1, 0.008, 0.8]))
    cb.set_label("height about the patch mean [m]", fontsize=8, color=TEXT)
    fig.text(0.04, 0.01, attribution, fontsize=7, color=TEXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "dem_swissalti3d.yaml"))
    cfg = load_config(ap.parse_args().config)
    z, e0, n0, spacing = load_tile(ROOT / cfg["raw"], cfg["source"])
    slope = slope_deg(z, spacing)
    w = int(round(cfg["patch_size"] / spacing)) + 1                 # samples per patch side
    rng = np.random.default_rng(cfg["seed"])
    taken = np.zeros_like(z, dtype=bool)
    keep, boxes, meta = [], [], []
    for _ in range(20000):
        if len(keep) == cfg["count"]:
            break
        r, c = rng.integers(0, z.shape[0] - w), rng.integers(0, z.shape[1] - w)
        win = (slice(r, r + w), slice(c, c + w))
        if taken[win].any() or np.isnan(z[win]).any():
            continue                                               # no overlap: patches are distinct ground
        s95, relief = float(np.percentile(slope[win], 95)), float(np.ptp(z[win]))
        if s95 > cfg["max_slope_deg"] or relief < cfg["min_relief"]:
            continue
        u = np.linspace(0.0, 1.0, w)
        g = np.linspace(0.0, 1.0, cfg["n"])
        p = RegularGridInterpolator((u, u), z[win])(np.stack(np.meshgrid(g, g, indexing="ij"), -1))
        keep.append(p - p.mean())
        boxes.append((r, c, w))
        meta.append((e0 + c * spacing, n0 + r * spacing, s95, relief))
        taken[win] = True
    meta = np.array(meta)
    out = ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, elevation=np.array(keep, dtype=np.float32), patch_size=cfg["patch_size"],
                        sw_corner_lv95=meta[:, :2], slope95_deg=meta[:, 2], relief_m=meta[:, 3],
                        attribution=cfg["attribution"], source=cfg["source"])
    plot(z, slope, boxes, keep, spacing, ROOT / cfg["figure"], cfg["attribution"])
    print(f"{len(keep)} patches -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB); tile slope median "
          f"{np.nanmedian(slope):.0f} deg; kept 95th-pct slope {meta[:, 2].min():.0f}-{meta[:, 2].max():.0f} deg, "
          f"relief {meta[:, 3].min():.1f}-{meta[:, 3].max():.1f} m (median {np.median(meta[:, 3]):.1f} m)")


if __name__ == "__main__":
    main()
