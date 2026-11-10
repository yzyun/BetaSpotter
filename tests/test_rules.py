import numpy as np

import synth
from boulderai.rules import bent_arms, body_tension, footwork, pull_direction, run_rules
from boulderai.types import NO_CONTACT


def rules_hit(climb):
    return {f.rule for f in run_rules(climb)}


def test_good_pose_is_clean():
    kp = synth.good_pose()
    climb = synth.static_climb(kp, synth.holds_for(kp))
    assert (climb.contacts[10] == [0, 1, 2, 3]).all()
    assert run_rules(climb) == []


def test_bent_arms_flagged():
    findings = bent_arms.check(synth.scenario_bent_arms())
    assert len(findings) == 1
    f = findings[0]
    assert f.limb == "both" and f.severity == "severe"
    assert all(a < 100 for a in f.evidence["sides"].values())


def test_bent_arm_lockoff_before_a_reach_is_not_flagged():
    bent = synth.bent_pose()
    reach = synth.make_pose((300, 340), synth.HANDS[0], (345, 120), *synth.FEET)
    poses = [bent] * 18 + synth.lerp_poses(bent, reach, 10) + [reach] * 5
    climb = synth.sequence_climb(poses, synth.holds_for(bent))
    assert bent_arms.check(climb) == []


def test_off_balance_flagged():
    findings = footwork.check_com(synth.scenario_off_balance())
    assert len(findings) == 1
    assert findings[0].evidence["side"] == "right"


def test_wrong_pull_direction_flagged_only_when_misaligned():
    assert len(pull_direction.check(synth.scenario_wrong_pull())) == 1
    assert pull_direction.check(synth.scenario_wrong_pull(direction=(0, 1))) == []


def test_feet_cut_flagged():
    climb = synth.scenario_feet_cut()
    findings = body_tension.check_cut(climb)
    assert len(findings) == 1
    assert (climb.contacts[findings[0].frame, 2:] == NO_CONTACT).all()
    assert set(findings[0].evidence["foot_targets"]) == {2, 3}


def test_hips_off_wall_flagged():
    findings = body_tension.check_hips(synth.scenario_hips_out())
    assert len(findings) == 1
    assert findings[0].evidence["ratio"] < 0.7


def test_swing_flagged():
    assert len(body_tension.check_swing(synth.scenario_swing())) >= 1


def test_foot_readjust_flagged():
    findings = footwork.check_readjust(synth.scenario_foot_readjust())
    assert len(findings) == 1
    assert findings[0].limb == "left" and findings[0].evidence["count"] == 3


def test_ground_frames_are_ignored():
    # Standing on the floor with bent arms on the start holds, then climbing up.
    floor = synth.make_pose((300, 640), (255, 476), (345, 476), (270, 780), (330, 780))
    up = synth.good_pose()
    poses = [floor] * 45 + synth.lerp_poses(floor, up, 30) + [up] * 60
    climb = synth.sequence_climb(poses)
    assert not climb.on_wall[:45].any() and climb.on_wall[-30:].all()
