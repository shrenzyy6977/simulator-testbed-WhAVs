"""terrain3d: map-based 3D terrain simulator on MuJoCo (ground classes, URDF robots, Gymnasium env).

Kept separate from `testbed` because it needs MuJoCo; `testbed` stays plain numpy/scipy.
"""
from .terrain import GROUND_CLASSES, GroundClass, Terrain, make_terrain
from .world import RobotSpec, WheeledRobot, build_model

__all__ = ["GROUND_CLASSES", "GroundClass", "Terrain", "make_terrain", "RobotSpec", "WheeledRobot", "build_model"]
