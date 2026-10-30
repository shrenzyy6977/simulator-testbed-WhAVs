import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from terrain3d import GROUND_CLASSES, Terrain, WheeledRobot, make_terrain  # noqa: E402
from terrain3d.env import RoverNavEnv  # noqa: E402
from terrain3d.terrain import flat, hills  # noqa: E402
from terrain3d.tracking import shift_path, track  # noqa: E402
from testbed.control import make_controller  # noqa: E402
from testbed.planning.path import make_path  # noqa: E402


def drive(ground: str, v=0.8, omega=0.0, seconds=4.0, elevation=None):
    terrain = Terrain.uniform(ground, flat(65) if elevation is None else elevation, 20.0)
    rob = WheeledRobot(terrain, start=(-6.0, 0.0))
    rob.step(250)
    x0 = rob.pose()[0]
    for _ in range(int(seconds * 50)):
        rob.command(v, omega)
        rob.step(10)
    return rob, rob.pose()[0] - x0


def _surface_z(rob, x, y):
    geomid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(rob.model, rob.data, np.array([x, y, 5.0]), np.array([0, 0, -1.0]), None, 1, rob.body_id, geomid)
    return 5.0 - dist


def test_heightfield_matches_terrain_height():
    e = np.zeros((65, 65))
    e[50:60, 5:15] = 1.0                       # bump at x < 0, y > 0
    terrain = Terrain.uniform("asphalt", e, 10.0, tiles=4)
    rob = WheeledRobot(terrain, start=(4.0, -4.0))
    for x, y in [(-3.0, 3.5), (3.0, 3.5), (-3.0, -3.5)]:
        assert abs(_surface_z(rob, x, y) - terrain.height(x, y)) < 0.02


def test_tiled_hills_have_no_seams():
    terrain = make_terrain({"size": 20, "n": 129, "elevation": {"type": "hills", "amplitude": 0.25, "seed": 4},
                            "ground": {"grid": [["grass"] * 4] * 4}})
    rob = WheeledRobot(terrain, start=(-8.0, 0.0))
    pts = np.linspace(-9.5, 9.5, 39)
    err = [abs(_surface_z(rob, x, y) - terrain.height(x, y)) for x in pts for y in pts]
    assert max(err) < 0.005


def test_tiles_carry_their_ground_class():
    t = make_terrain({"size": 20, "n": 65, "ground": {"grid": [["sand", "mud"], ["asphalt", "grass"]]}})
    assert t.ground_at(-5, 5).name == "sand" and t.ground_at(5, 5).name == "mud"
    assert t.ground_at(-5, -5).name == "asphalt" and t.ground_at(5, -5).name == "grass"
    rob = WheeledRobot(t)
    mus = sorted(rob.model.geom_friction[: t.tiles ** 2, 0])
    assert mus == sorted(GROUND_CLASSES[g].mu for g in ["sand", "mud", "asphalt", "grass"])


def test_urdf_robot_drives_straight_at_commanded_speed():
    rob, dx = drive("grass", v=0.8)
    assert abs(rob.body_velocity()[0] - 0.8) < 0.05
    assert abs(rob.pose()[1]) < 0.1 and dx > 2.5


def test_turn_direction_follows_command_sign():
    rob, _ = drive("grass", v=0.0, omega=1.0, seconds=2.0)
    assert rob.body_velocity()[2] > 0.1
    rob, _ = drive("grass", v=0.0, omega=-1.0, seconds=2.0)
    assert rob.body_velocity()[2] < -0.1


def test_soft_low_friction_ground_slips_more_uphill():
    slope = hills(65, 20.0, 0.15, 6.0, seed=1)
    slips = {}
    for g in ["asphalt", "sand"]:
        rob = WheeledRobot(Terrain.uniform(g, slope, 20.0), start=(-6.0, 0.0))
        rob.step(250)
        s = []
        for _ in range(150):
            rob.command(0.8, 0.0)
            rob.step(10)
            s.append(rob.slip_ratio())
        slips[g] = np.mean(s)
    assert slips["sand"] > slips["asphalt"] + 0.05


@pytest.mark.parametrize("name", ["pure_pursuit", "stanley", "lqr", "mpc"])
def test_controllers_track_path_on_terrain(name):
    terrain = Terrain.uniform("grass", hills(65, 20.0, 0.1, 6.0, seed=1), 20.0)
    path = shift_path(make_path("s_curve", length=12.0, amplitude=1.0, wavelength=8.0, v_target=0.8, a_lat_max=0.5, ds=0.05), -6, 0)
    gains = {"pure_pursuit": dict(k=0.3, ld_min=0.5, ld_max=2.0), "stanley": dict(ks=0.5)}.get(name, {})
    log = track(terrain, make_controller(name, **gains), path)
    assert log.status == "finished"
    assert np.max(np.abs(log.arrays()["cte"])) < 0.5


def test_nav_env_api_and_determinism():
    from gymnasium.utils.env_checker import check_env

    env = RoverNavEnv()
    check_env(env, skip_render_check=True)
    o1, _ = env.reset(seed=3)
    o2, _ = RoverNavEnv().reset(seed=3)
    assert o1.shape == (11,) and np.allclose(o1, o2)
