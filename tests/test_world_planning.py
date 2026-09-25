from pathlib import Path

import numpy as np

from testbed.planning.path import make_path
from testbed.world.grid_map import GridMap

ROOT = Path(__file__).resolve().parents[1]


def test_raycast_hits_wall_at_expected_distance():
    m = GridMap.empty(20, 20, 0.1)
    m.add_obstacle({"type": "rect", "x": 15, "y": 0, "w": 1, "h": 20})
    r = m.raycast(5.0, 10.0, np.array([0.0]), 30.0)[0]
    assert abs(r - 10.0) < 0.15


def test_map_yaml_loads_and_has_obstacles():
    m = GridMap.from_yaml(ROOT / "worlds" / "grid" / "warehouse.yaml")
    assert m.grid.any() and not m.is_free(13.5, 10) and m.is_free(5, 5)


def test_test_paths_have_consistent_geometry():
    for name in ["straight", "circle", "s_curve", "lane_change"]:
        p = make_path(name, v_target=6.0)
        assert len(p) > 100 and p.v_ref.max() <= 6.0 + 1e-9 and p.v_ref[-1] == 0.0
    c = make_path("circle", radius=20.0)
    assert abs(np.median(np.abs(c.k)) - 1 / 20.0) < 1e-3
