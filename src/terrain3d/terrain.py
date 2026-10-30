"""Map-based 3D terrain for MuJoCo: an elevation grid split into tiles, each tile carrying a
ground class with its own physical properties.

MuJoCo assigns contact parameters per geom, so one heightfield can only have one friction value.
We therefore cut the elevation grid into `tiles x tiles` heightfields that share their boundary
samples (no seams) and give each tile the parameters of its ground class.

Ground-class parameters map onto MuJoCo contact parameters:
    friction = [sliding mu, torsional, rolling]   (condim 6, so rolling resistance is active)
    solref   = [timeconst, dampratio]             larger timeconst = softer ground = deeper
                                                 penetration, our first-order stand-in for sinkage
Terrain geoms get priority 1, so their parameters win over the wheel's in every contact.
The values in GROUND_CLASSES are placeholders in plausible ranges, to be calibrated against data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class GroundClass:
    name: str
    mu: float               # Coulomb sliding friction coefficient [-]
    torsional: float        # torsional friction [m]
    rolling: float          # rolling friction (rolling resistance lever arm) [m]
    timeconst: float        # contact stiffness time constant [s] (softness / sinkage proxy)
    dampratio: float = 1.0
    rgba: tuple = (0.5, 0.5, 0.5, 1.0)


# MuJoCo's rolling friction is a lever arm [m]: the rolling-resistance coefficient is
# Crr = rolling / wheel_radius. Values below give Crr ~0.01 (asphalt) to ~0.15 (sand, mud) for
# a 0.065 m wheel; Crr must stay below mu or the robot cannot move at all.
GROUND_CLASSES: dict[str, GroundClass] = {g.name: g for g in [
    GroundClass("asphalt", 0.90, 0.005, 0.0008, 0.004, rgba=(0.35, 0.35, 0.37, 1)),
    GroundClass("grass", 0.60, 0.010, 0.0040, 0.010, rgba=(0.33, 0.55, 0.25, 1)),
    GroundClass("gravel", 0.55, 0.010, 0.0025, 0.008, rgba=(0.60, 0.57, 0.52, 1)),
    GroundClass("sand", 0.45, 0.020, 0.0100, 0.030, rgba=(0.86, 0.76, 0.52, 1)),
    GroundClass("snow", 0.30, 0.010, 0.0060, 0.025, rgba=(0.93, 0.95, 0.98, 1)),
    GroundClass("mud", 0.30, 0.020, 0.0100, 0.040, rgba=(0.40, 0.30, 0.20, 1)),
]}


# --------------------------------------------------------------------------- elevation sources
def flat(n: int = 129) -> np.ndarray:
    return np.zeros((n, n))


def hills(n: int = 129, size: float = 20.0, amplitude: float = 0.4, wavelength: float = 6.0,
          seed: int = 0) -> np.ndarray:
    """Smooth random hills: a sum of random-phase cosines with wavelengths around `wavelength`.
    Peak-to-peak height is about 2 * amplitude."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(x, x)
    z = np.zeros_like(X)
    for _ in range(12):
        k = 2 * np.pi / (wavelength * rng.uniform(0.6, 1.6))
        th = rng.uniform(0, 2 * np.pi)
        z += np.cos(k * (np.cos(th) * X + np.sin(th) * Y) + rng.uniform(0, 2 * np.pi))
    z *= amplitude / np.abs(z).max()
    return z


def slope(n: int = 129, size: float = 20.0, grade_deg: float = 10.0) -> np.ndarray:
    """Planar incline rising along +x."""
    x = np.linspace(-size / 2, size / 2, n)
    return np.tile(np.tan(np.deg2rad(grade_deg)) * (x - x[0]), (n, 1))


def from_file(path: str | Path) -> np.ndarray:
    """Elevation in metres from a .npy array, or a grayscale PNG (needs `z_range` scaling by the
    caller). Row 0 of the array is the -y edge of the map."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(float)
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return np.flipud(img.astype(float))


ELEVATIONS = {"flat": flat, "hills": hills, "slope": slope}


# --------------------------------------------------------------------------- terrain
@dataclass
class Terrain:
    """Square terrain of side `size` [m] centred on the origin.

    elevation : (n, n) heights [m]; row index = y, column index = x
    classes   : (tiles, tiles) ground-class names; classes[i, j] covers tile row i (y), col j (x)
    """

    elevation: np.ndarray
    size: float
    classes: np.ndarray

    def __post_init__(self):
        n = self.elevation.shape[0]
        t = self.classes.shape[0]
        if self.elevation.shape != (n, n) or self.classes.shape != (t, t) or (n - 1) % t:
            raise ValueError(f"elevation must be square with (n-1) divisible by tiles; got {self.elevation.shape}, {t}")
        unknown = set(self.classes.ravel()) - set(GROUND_CLASSES)
        if unknown:
            raise ValueError(f"unknown ground classes {unknown}; options: {list(GROUND_CLASSES)}")

    @classmethod
    def uniform(cls, ground: str, elevation: np.ndarray, size: float, tiles: int = 1) -> "Terrain":
        return cls(elevation, size, np.full((tiles, tiles), ground, dtype=object))

    @classmethod
    def random_patches(cls, elevation: np.ndarray, size: float, tiles: int, grounds: list[str],
                       seed: int = 0) -> "Terrain":
        rng = np.random.default_rng(seed)
        return cls(elevation, size, rng.choice(np.array(grounds, dtype=object), (tiles, tiles)))

    @property
    def tiles(self) -> int:
        return self.classes.shape[0]

    def height(self, x: float, y: float) -> float:
        """Bilinear interpolation of the elevation at world (x, y)."""
        n = self.elevation.shape[0]
        u = np.clip((x / self.size + 0.5) * (n - 1), 0, n - 1 - 1e-9)
        v = np.clip((y / self.size + 0.5) * (n - 1), 0, n - 1 - 1e-9)
        j, i = int(u), int(v)
        fu, fv = u - j, v - i
        e = self.elevation
        return float((1 - fv) * ((1 - fu) * e[i, j] + fu * e[i, j + 1]) + fv * ((1 - fu) * e[i + 1, j] + fu * e[i + 1, j + 1]))

    def ground_at(self, x: float, y: float) -> GroundClass:
        t = self.tiles
        j = int(np.clip((x / self.size + 0.5) * t, 0, t - 1))
        i = int(np.clip((y / self.size + 0.5) * t, 0, t - 1))
        return GROUND_CLASSES[self.classes[i, j]]

    def add_to_spec(self, spec: mujoco.MjSpec, prefix: str = "terrain") -> None:
        """Add one heightfield geom per tile. Tiles share edge samples, so the surface is continuous.

        MuJoCo rescales every heightfield's data to [0, 1] when compiling, so each tile gets its
        own vertical scale (size[2] = tile height range) and offset (geom z = tile minimum).
        """
        e = self.elevation
        n = e.shape[0]
        t = self.tiles
        m = (n - 1) // t                       # samples per tile edge, minus one
        half = self.size / t / 2
        for i in range(t):
            for j in range(t):
                g = GROUND_CLASSES[self.classes[i, j]]
                tile = e[i * m:(i + 1) * m + 1, j * m:(j + 1) * m + 1]
                z0, dz = float(tile.min()), float(tile.max() - tile.min())
                if dz < 1e-6:                  # flat tile: MuJoCo cannot normalise a constant array
                    z0, dz = z0 - 1e-4, 1e-4
                    data = np.ones_like(tile)
                else:
                    data = (tile - z0) / dz
                name = f"{prefix}_{i}_{j}"
                spec.add_hfield(name=name, size=[half, half, dz, 0.1], nrow=m + 1, ncol=m + 1,
                                userdata=data.ravel().tolist())
                cx = -self.size / 2 + (j + 0.5) * 2 * half
                cy = -self.size / 2 + (i + 0.5) * 2 * half
                spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                                        pos=[cx, cy, z0], friction=[g.mu, g.torsional, g.rolling],
                                        solref=[g.timeconst, g.dampratio], condim=6, priority=1, margin=0.001, rgba=list(g.rgba))


def make_terrain(cfg: dict) -> Terrain:
    """Build a Terrain from a config dict, e.g.
        {size: 20, n: 129, elevation: {type: hills, amplitude: 0.4, seed: 1},
         ground: sand}                                  # uniform
      or ground: {patches: [grass, sand, mud], tiles: 4, seed: 0}
      or ground: {grid: [[asphalt, grass], [sand, mud]]}
    """
    size, n = float(cfg.get("size", 20.0)), int(cfg.get("n", 129))
    ec = dict(cfg.get("elevation", {"type": "flat"}))
    kind = ec.pop("type")
    if kind == "file":
        elev = from_file(ec.pop("path")) * float(ec.pop("z_scale", 1.0))
    elif kind == "flat":
        elev = flat(n)
    else:
        elev = ELEVATIONS[kind](n=n, size=size, **ec)
    g = cfg.get("ground", "asphalt")
    if isinstance(g, str):
        return Terrain.uniform(g, elev, size, int(cfg.get("tiles", 1)))
    if "grid" in g:
        return Terrain(elev, size, np.array(g["grid"], dtype=object)[::-1])   # first row = top (+y)
    return Terrain.random_patches(elev, size, int(g.get("tiles", 4)), list(g["patches"]), int(g.get("seed", 0)))
