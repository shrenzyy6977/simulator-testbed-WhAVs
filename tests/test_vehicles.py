import numpy as np

from testbed.vehicles.base import State, VehicleParams
from testbed.vehicles.models import DiffDrive, DynamicBicycle, KinematicBicycle


def drive_circle(model, delta, v=3.0, t=40.0, dt=0.01):
    s = State(v=v)
    model.steer = delta                      # skip the steering-rate transient
    xs, ys = [], []
    for _ in range(int(t / dt)):
        s = model.step(s, delta, 0.0, dt)
        xs.append(s.x); ys.append(s.y)
    return np.array(xs), np.array(ys)


def test_kinematic_circle_radius_matches_theory():
    p = VehicleParams()
    delta = np.deg2rad(10)
    x, y = drive_circle(KinematicBicycle(p), delta)
    R_theory = p.wheelbase / np.tan(delta)
    R_measured = (y.max() - y.min()) / 2
    assert abs(R_measured - R_theory) < 0.05 * R_theory


def test_diff_drive_straight_line_and_speed_limit():
    p = VehicleParams(max_speed=2.0)
    m = DiffDrive(p)
    s = State()
    for _ in range(500):
        s = m.step(s, 0.0, 1.0, 0.02)
    assert abs(s.y) < 1e-9 and abs(s.yaw) < 1e-9
    assert s.v <= 2.0 + 1e-9


def test_dynamic_bicycle_low_speed_close_to_kinematic():
    delta = np.deg2rad(5)
    xk, yk = drive_circle(KinematicBicycle(), delta, v=2.0, t=10)
    xd, yd = drive_circle(DynamicBicycle(), delta, v=2.0, t=10)
    # same turning direction and similar radius at low speed
    assert np.sign(yk[-100]) == np.sign(yd[-100])
    assert abs((yk.max() - yk.min()) - (yd.max() - yd.min())) < 0.25 * (yk.max() - yk.min())


def test_steering_rate_limit():
    p = VehicleParams()
    m = KinematicBicycle(p)
    m.step(State(v=1.0), p.max_steer, 0.0, 0.05)
    assert abs(m.steer - p.max_steer_rate * 0.05) < 1e-9
