from pathlib import Path

import numpy as np

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
