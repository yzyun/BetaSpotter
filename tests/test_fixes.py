import numpy as np
import pytest

import synth
from boulderai.analysis import geometry as G
from boulderai.fixes.beta import suggest_beta
from boulderai.fixes.corrections import build_problem, suggest_fix
from boulderai.pose import landmarks as L
from boulderai.rules import bent_arms, body_tension, footwork, pull_direction
from boulderai.rules.pull_direction import pull_angle


def check_body_kept(climb, finding, fix, tol=0.02):
    """Bone lengths within tol of the solver's targets, and placed limbs still on their holds."""
    pins, bones, _ = build_problem(climb, finding)
    for (a, b), length in bones.items():
        got = np.linalg.norm(fix.keypoints[a] - fix.keypoints[b])
        assert got == pytest.approx(length, rel=tol), f"bone {a}-{b}"
    scale = climb.scale[finding.frame]
    for joint, target in pins.items():
        assert np.linalg.norm(fix.keypoints[joint] - target) < 0.03 * scale, f"pin {joint}"


def test_bent_arms_fix_straightens_arms():
    climb = synth.scenario_bent_arms()
    f = bent_arms.check(climb)[0]
    fix = suggest_fix(climb, f)
    for side in ("left", "right"):
        assert G.elbow_angle(fix.keypoints, side) > climb.cfg["bent_arms"]["max_elbow_deg"]
    check_body_kept(climb, f, fix)
    assert G.hip_mid(fix.keypoints)[1] > G.hip_mid(climb.kp[f.frame])[1]  # hips sink
    assert "lower" in fix.tip


def test_off_balance_fix_brings_com_inside():
    climb = synth.scenario_off_balance()
    f = footwork.check_com(climb)[0]
    fix = suggest_fix(climb, f)
    xs = G.limb_points(fix.keypoints)[:, 0]
    off = G.com_offset(G.com(fix.keypoints)[0], xs, climb.scale[f.frame])
    assert abs(off) <= climb.cfg["footwork"]["com_margin"]
    check_body_kept(climb, f, fix)


def test_pull_direction_fix_lines_up_forearm():
    climb = synth.scenario_wrong_pull()
    f = pull_direction.check(climb)[0]
    fix = suggest_fix(climb, f)
    hold = climb.hold(f.evidence["hold_id"])
    assert pull_angle(fix.keypoints, "left", hold.direction) < climb.cfg["pull_direction"]["max_angle_deg"]
    check_body_kept(climb, f, fix, tol=0.04)


def test_feet_cut_fix_puts_feet_back():
    climb = synth.scenario_feet_cut()
    f = body_tension.check_cut(climb)[0]
    fix = suggest_fix(climb, f)
    for limb, target in f.evidence["foot_targets"].items():
        assert np.linalg.norm(fix.keypoints[L.LIMB_END_JOINT[limb]] - target) < 5


def test_hips_fix_lengthens_torso():
    climb = synth.scenario_hips_out()
    f = body_tension.check_hips(climb)[0]
    fix = suggest_fix(climb, f)
    assert G.torso_length(fix.keypoints) > 1.3 * G.torso_length(climb.kp[f.frame])


def test_beta_picks_the_better_foothold():
    climb = synth.scenario_off_balance(extra_foothold=True)
    f = footwork.check_com(climb)[0]
    fix = suggest_fix(climb, f)
    beta = suggest_beta(climb, f, fix)
    assert beta is not None
    assert beta.hold_id == 4 and beta.limb == 3  # right foot onto the hold out right
    assert "right foot" in beta.tip and "#4" in beta.tip
    xs = G.limb_points(beta.keypoints)[:, 0]
    assert G.com_offset(G.com(beta.keypoints)[0], xs, climb.scale[f.frame]) == pytest.approx(0, abs=0.15)


def test_no_beta_without_a_better_hold():
    climb = synth.scenario_off_balance()
    f = footwork.check_com(climb)[0]
    assert suggest_beta(climb, f, suggest_fix(climb, f)) is None
