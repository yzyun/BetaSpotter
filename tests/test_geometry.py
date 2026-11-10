import numpy as np
import pytest

from betaspotter.analysis import geometry as G
from synth import good_pose, bent_pose


def test_angle_deg():
    assert G.angle_deg(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])) == pytest.approx(90)
    assert G.angle_deg(np.array([1.0, 0]), np.array([0.0, 0]), np.array([-1.0, 0])) == pytest.approx(180)


def test_elbow_angles_of_synthetic_poses():
    assert G.elbow_angle(good_pose(), "left") > 160
    assert G.elbow_angle(bent_pose(), "left") < 100


def test_com_is_inside_the_body_and_near_the_trunk():
    kp = good_pose()
    c = G.com(kp)
    assert kp[11, 1] < c[1] < kp[27, 1]
    assert abs(c[0] - 300) < 5  # symmetric pose


def test_com_offset_sign():
    assert G.com_offset(50, [100, 200], 100) == pytest.approx(-0.5)
    assert G.com_offset(150, [100, 200], 100) == 0
    assert G.com_offset(230, [100, 200], 100) == pytest.approx(0.3)


def test_runs():
    mask = np.array([0, 1, 1, 0, 1, 1, 1], bool)
    assert G.runs(mask, 1) == [(1, 3), (4, 7)]
    assert G.runs(mask, 3) == [(4, 7)]


def test_poly_distance():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert G.poly_distance(square, (5, 5)) == 0
    assert G.poly_distance(square, (15, 5)) == pytest.approx(5)
