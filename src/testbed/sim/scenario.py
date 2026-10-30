"""Build vehicles, paths, controllers and worlds from a YAML scenario config.

Example (see experiments/configs/):

    name: figure_eight
    path: {name: figure_eight, v_target: 8.0}        # or: {astar: {start: [4, 5], goal: [55, 35], clearance: 2.5}}
    world: worlds/grid/warehouse.yaml                 # optional, needed for astar paths
    vehicle: {model: kinematic_bicycle, params: {max_speed: 15.0}}
    controllers:
      - {name: pure_pursuit, k: 0.5}
      - {name: mpc, horizon: 20}
    sim: {dt: 0.02, init_offset: 1.0}
    settle_time: 5.0                                  # metrics ignore the first seconds
"""
from __future__ import annotations

from pathlib import Path as FsPath

import yaml

from ..control import Controller, make_controller
from ..planning.astar import plan_path
from ..planning.path import Path, make_path
from ..vehicles.base import VehicleParams
from ..vehicles.models import VehicleModel, make_vehicle
from ..world.grid_map import GridMap
from .simulator import SimConfig

ROOT = FsPath(__file__).resolve().parents[3]


def load_config(path: str | FsPath) -> dict:
    return yaml.safe_load(FsPath(path).read_text())


def build_world(cfg: dict) -> GridMap | None:
    w = cfg.get("world")
    return GridMap.from_yaml(ROOT / w) if w else None


def build_path(cfg: dict, world: GridMap | None = None) -> Path:
    pc = dict(cfg["path"])
    if "astar" in pc:
        if world is None:
            raise ValueError("an astar path needs a 'world' entry in the scenario")
        a = pc.pop("astar")
        return plan_path(world, tuple(a["start"]), tuple(a["goal"]), clearance=a.get("clearance", 1.5), **pc)
    return make_path(pc.pop("name"), **pc)


def build_vehicle(cfg: dict) -> VehicleModel:
    vc = cfg.get("vehicle", {})
    return make_vehicle(vc.get("model", "kinematic_bicycle"), VehicleParams(**vc.get("params", {})))


def build_controllers(cfg: dict) -> list[Controller]:
    speed = cfg.get("speed_pid")
    return [make_controller(**{"speed": speed, **c}) for c in cfg["controllers"]]


def build_sim(cfg: dict) -> SimConfig:
    return SimConfig(**cfg.get("sim", {}))
