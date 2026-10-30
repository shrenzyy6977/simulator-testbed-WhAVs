from pathlib import Path

import numpy as np
import pytest

from testbed.control import CONTROLLERS, LQR, make_controller
from testbed.control.optimal import error_model
from testbed.metrics import compute_metrics
from testbed.planning.path import make_path
from testbed.sim import SimConfig, simulate
from testbed.sim.scenario import build_controllers, build_path, build_sim, build_vehicle, build_world, load_config
from testbed.vehicles.models import make_vehicle

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_controller_converges_from_lateral_offset(name):
    log = simulate(make_vehicle("kinematic_bicycle"), make_controller(name), make_path("straight", v_target=5.0),
                   SimConfig(init_offset=1.5))
    cte = np.abs(log.arrays()["cte"])
    assert log.status == "finished"
    assert cte[0] > 1.4 and cte[-1] < 0.05


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_controller_tracks_figure_eight(name):
    log = simulate(make_vehicle("kinematic_bicycle"), make_controller(name), make_path("figure_eight", v_target=6.0))
    m = compute_metrics(log)
    assert m["status"] == "finished"
    assert m["max_cte"] < 0.6
    assert np.all(np.abs(log.arrays()["steer"]) <= make_vehicle("kinematic_bicycle").p.max_steer + 1e-9)


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_controller_finishes_on_dynamic_bicycle(name):
    log = simulate(make_vehicle("dynamic_bicycle"), make_controller(name), make_path("lane_change", v_target=12.0))
    assert log.status == "finished" and compute_metrics(log)["max_cte"] < 0.6


def test_lqr_closed_loop_is_stable_over_speed_range():
    lqr = LQR().bind(make_vehicle("kinematic_bicycle"))
    for v in (0.5, 2.0, 5.0, 10.0, 20.0):
        A, B = error_model(v, lqr.p.wheelbase, 0.02)
        eig = np.linalg.eigvals(A - B @ lqr.gain(v, 0.02))
        assert np.all(np.abs(eig) < 1.0)


def test_speed_controller_reaches_reference_speed():
    log = simulate(make_vehicle("kinematic_bicycle"), make_controller("pure_pursuit"), make_path("straight", length=200.0, v_target=8.0))
    a = log.arrays()
    mid = (a["s"] > 80) & (a["s"] < 120)
    assert np.all(np.abs(a["v"][mid] - 8.0) < 0.2)


def test_warehouse_scenario_is_collision_free():
    cfg = load_config(ROOT / "experiments" / "configs" / "warehouse_astar.yaml")
    world = build_world(cfg)
    path = build_path(cfg, world)
    for ctrl in build_controllers(cfg):
        assert simulate(build_vehicle(cfg), ctrl, path, build_sim(cfg), world).status == "finished"
