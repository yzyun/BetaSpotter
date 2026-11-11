import json

import numpy as np
import pytest

import synth
import synth_side as S
from betaspotter.analysis import geometry as G
from betaspotter.analysis.wall import WallError, estimate_wall
from betaspotter.fixes.corrections import suggest_fix
from betaspotter.pipeline import AnalysisError, analyse_side
from betaspotter.report import frame_picker
from betaspotter.report.annotate import render_clean, render_finding
from betaspotter.rules.side import hip_distance, hip_limit, run_side_rules
from test_report import write_video


def rules_of(climb):
    return [f.rule for f in frame_picker.select(run_side_rules(climb), climb.cfg)]


# ---------------------------------------------------------------- the wall line

@pytest.mark.parametrize("angle", [-10, 0, 15, 30, 45, 60])
@pytest.mark.parametrize("side", [1, -1])
def test_wall_angle_is_estimated_from_placements(angle, side):
    wall = S.scenario_route(angle, side).wall
    assert wall.angle_deg == pytest.approx(angle, abs=3)
    assert np.sign(wall.n[0]) == side  # normal points at the climber
    assert wall.source == "estimated" and wall.n_points >= 6


def test_hip_distance_is_measured_from_the_wall_surface():
    climb = S.scenario_hips_out(30, hip_d=1.4)
    assert float(np.median(hip_distance(climb))) == pytest.approx(1.4, abs=0.12)


def test_given_wall_angle_overrides_the_estimate():
    climb = S.climb(S.hold(S.pose(S.SideWall(30))), wall_angle=30)
    assert climb.wall.source == "given" and climb.wall.angle_deg == pytest.approx(30)
    assert climb.wall.n[0] > 0
    assert S.climb(S.hold(S.pose(S.SideWall(30, side=-1))), wall_angle=30).wall.n[0] < 0
    with pytest.raises(WallError):
        estimate_wall(climb, 120)


def test_wall_needs_placements_spread_along_it():
    kp = S.pose(S.SideWall(30))
    poses = []
    for i in range(60):  # feet never still: only the hands are placed, all at one spot
        moving = kp.copy()
        moving[[27, 28, 29, 30, 31, 32]] += [60 * np.sin(0.5 * i), 60 * np.cos(0.5 * i)]
        poses.append(moving)
    with pytest.raises(WallError, match="--wall-angle"):
        S.climb(poses)
    assert S.climb(poses, wall_angle=30).wall.angle_deg == pytest.approx(30)


# ---------------------------------------------------------------- the rules

@pytest.mark.parametrize("angle", [0, 30, 45])
def test_good_position_is_not_flagged(angle):
    assert rules_of(S.scenario_good(angle)) == []
    assert rules_of(S.scenario_route(angle)) == []


SCENARIOS = {
    "hips-far-from-wall": S.scenario_hips_out,
    "bent-arms": S.scenario_bent_arms,
    "passive-shoulders": S.scenario_passive_shoulders,
    "feet-cut": S.scenario_feet_cut,
    "square-reach": S.scenario_square_reach,
    "swing-out": S.scenario_swing,
}


@pytest.mark.parametrize("rule", sorted(SCENARIOS))
def test_each_error_is_flagged_alone(rule):
    assert rules_of(SCENARIOS[rule]()) == [rule]


def test_hip_limit_follows_the_wall_angle():
    limits = [hip_limit(S.scenario_good(a)) for a in (0, 30, 45)]
    assert limits[0] < limits[1] < limits[2]
    # the same hip distance is too far out on a vertical wall but fine on a steep one
    assert rules_of(S.scenario_hips_out(0, hip_d=0.95)) == ["hips-far-from-wall"]
    assert rules_of(S.scenario_hips_out(45, hip_d=0.95)) == []


def test_slab_allows_hips_out():
    assert "hips-far-from-wall" not in rules_of(S.scenario_hips_out(-15, hip_d=1.2))


def test_overhang_only_checks_stay_off_on_a_vertical_wall():
    assert rules_of(S.scenario_passive_shoulders(0)) == []
    assert "square-reach" not in rules_of(S.scenario_square_reach(0, hip_d=0.65))


def test_turned_hips_on_a_reach_are_fine():
    assert rules_of(S.scenario_square_reach(twist=40.0)) == []


def test_small_wobble_is_not_a_swing():
    assert rules_of(S.scenario_swing(out=0.2)) == []


def test_hips_fix_brings_the_hips_in_on_the_same_holds():
    climb = S.scenario_hips_out()
    f = run_side_rules(climb)[0]
    fix = suggest_fix(climb, f)
    before = climb.wall.distance(G.hip_mid(climb.kp[f.frame])) / climb.scale[f.frame]
    after = climb.wall.distance(G.hip_mid(fix.keypoints)) / climb.scale[f.frame]
    assert after < hip_limit(climb) < before
    for joint in (15, 16, 27, 28):  # hands and feet stay put
        assert np.linalg.norm(fix.keypoints[joint] - climb.kp[f.frame, joint]) < 0.1 * synth.TORSO
    assert climb.wall.distance(fix.keypoints[[11, 12, 23, 24, 25, 26]]).min() > -0.1 * synth.TORSO
    assert "in towards the wall" in fix.tip


# ---------------------------------------------------------------- output

@pytest.mark.parametrize("rule", sorted(SCENARIOS))
@pytest.mark.parametrize("layout", ["panels", "overlay"])
def test_every_side_rule_renders(rule, layout):
    climb = SCENARIOS[rule]()
    f = [f for f in run_side_rules(climb) if f.rule == rule][0]
    fix = suggest_fix(climb, f)
    assert (fix is None) == (rule == "passive-shoulders")
    frame = np.full((960, 1280, 3), 150, np.uint8)
    image = render_finding(frame, climb, f, fix, None, layout=layout)
    panels = 1 if layout == "overlay" or fix is None else 2
    assert image.shape[1] >= panels * frame.shape[1] and image.shape[0] > frame.shape[0]
    assert render_clean(frame, climb, f.frame).shape[0] > frame.shape[0]


def test_end_to_end_side_video(tmp_path):
    wall = S.SideWall(30)
    good = S.pose(wall, feet_s=280.0)
    out_pose = S.pose(wall, hip_d=1.4, shoulder_d=0.9, feet_s=280.0)
    poses = S.hold(good, 2.0) + synth.lerp_poses(good, out_pose, 15) + S.hold(out_pose, 2.0)
    video = tmp_path / "side.mp4"
    write_video(video, np.full((960, 1280, 3), 150, np.uint8), poses)
    logs = []

    out = analyse_side(video, tmp_path / "out", estimator=synth.FakeEstimator(poses), log=logs.append)

    index = json.loads((out / "findings.json").read_text())
    assert [item["rule"] for item in index] == ["hips-far-from-wall"]
    assert index[0]["fix"] and (out / index[0]["file"]).exists()
    assert any(line.startswith("Wall:") and "deg overhang" in line for line in logs)


def test_end_to_end_clean_side_video_and_bad_angle(tmp_path):
    poses = S.hold(S.pose(S.SideWall(30)), 2.0)
    video = tmp_path / "clean.mp4"
    write_video(video, np.full((960, 1280, 3), 150, np.uint8), poses)
    out = analyse_side(video, tmp_path / "out", wall_angle=30, estimator=synth.FakeEstimator(poses),
                       log=lambda m: None)
    assert [p.name for p in out.glob("*.png")] == ["00_no-errors.png"]
    with pytest.raises(AnalysisError):
        analyse_side(video, tmp_path / "out2", wall_angle=200, estimator=synth.FakeEstimator(poses),
                     log=lambda m: None)
