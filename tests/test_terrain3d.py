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
    assert o1.shape == (12,) and np.allclose(o1, o2)


def test_nav_env_randomisation_and_time_budget():
    from terrain3d.env import NavConfig

    cfg = NavConfig(grounds=["sand"], amplitude_range=(0.0, 0.0), gps_noise_range=(0.2, 0.4), gps_bias=0.5,
                    delay_range=(0, 2), min_avg_speed=0.5)
    env = RoverNavEnv(cfg)
    stds, delays = set(), set()
    for s in range(8):
        env.reset(seed=s)
        stds.add(round(env.gps_std, 3))
        delays.add(env.delay)
        assert np.ptp(env.terrain.elevation) == 0.0 and env.robot.ground().name == "sand"
        assert all(0.2 <= v <= 0.4 for v in stds) and delays <= {0, 1, 2}
        route = np.r_[[env.robot.pose()[:2]], env.checkpoints]
        assert env.max_steps == int(np.ceil(np.sum(np.hypot(*np.diff(route, axis=0).T)) / 0.5 * cfg.control_hz))
    assert len(stds) > 1


def test_go_to_goal_baseline_completes_easy_mission():
    from terrain3d.baselines import GoToGoal
    from terrain3d.env import NavConfig

    env = RoverNavEnv(NavConfig(grounds=["grass"], amplitude_range=(0.0, 0.0), gps_noise=0.0, delay_steps=0))
    policy = GoToGoal()
    obs, _ = env.reset(seed=7)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy.predict(obs)[0])
        done = term or trunc
    assert info["success"]


@pytest.mark.parametrize("name", ["pure_pursuit", "stanley", "lqr", "mpc"])
def test_path_trackers_fly_checkpoint_mission(name):
    from terrain3d.baselines import PathTracker
    from terrain3d.env import NavConfig

    cfg = NavConfig(grounds=["grass"], amplitude_range=(0.0, 0.0), gps_noise=0.0, delay_steps=0, n_checkpoints=4,
                    checkpoint_spacing=(4.0, 8.0), checkpoint_radius=0.75, min_avg_speed=0.6, max_steps=900)
    env = RoverNavEnv(cfg)
    obs, _ = env.reset(seed=100_012)          # starts facing ~156 deg away from the first leg
    policy = PathTracker(make_controller(name)).reset(env)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy.predict(obs)[0])
        done = term or trunc
    assert info["success"]


def test_second_urdf_robot_loads_and_drives():
    from terrain3d.world import ROOT, RobotSpec

    spec = RobotSpec(urdf=ROOT / "robots" / "rover6wd.urdf", track=0.36, max_torque=3.0)
    rob = WheeledRobot(Terrain.uniform("grass", flat(65), 20.0), spec, start=(-6.0, 0.0))
    assert sorted(rob.sides) == [-1, -1, -1, 1, 1, 1]        # left/right read from "wheel_left_*" names
    rob.step(250)
    for _ in range(150):
        rob.command(0.8, 0.0)
        rob.step(10)
    assert abs(rob.body_velocity()[0] - 0.8) < 0.1 and abs(rob.pose()[1]) < 0.1


def test_gps_fix_updates_at_gps_rate():
    from terrain3d.env import NavConfig

    env = RoverNavEnv(NavConfig(gps_noise=0.5, gps_hz=1.0, control_hz=10.0, delay_steps=0))
    env.reset(seed=2)
    fixes, ages = [env.fix.copy()], []
    for _ in range(20):
        obs, *_ = env.step(np.array([1.0, 0.0], np.float32))
        fixes.append(env.fix.copy())
        ages.append(round(float(obs[-1]), 3))
    changed = [i for i in range(1, 21) if not np.allclose(fixes[i], fixes[i - 1])]
    assert changed == [10, 20]                                 # a new fix every 10 control steps
    assert ages[:10] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0]
