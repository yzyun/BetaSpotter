import json

import cv2
import numpy as np
import pytest

import synth
from betaspotter.fixes.beta import suggest_beta
from betaspotter.fixes.corrections import suggest_fix
from betaspotter.pipeline import analyse
from betaspotter.report import frame_picker
from betaspotter.report.annotate import render_finding
from betaspotter.report.writer import filename
from betaspotter.rules import run_rules

SCENARIOS = {
    "bent-arms": synth.scenario_bent_arms,
    "com-outside-support": lambda: synth.scenario_off_balance(extra_foothold=True),
    "wrong-pull-direction": synth.scenario_wrong_pull,
    "feet-cut": synth.scenario_feet_cut,
    "hips-off-wall": synth.scenario_hips_out,
    "body-swing": synth.scenario_swing,
    "foot-readjust": synth.scenario_foot_readjust,
}


@pytest.mark.parametrize("rule", sorted(SCENARIOS))
@pytest.mark.parametrize("layout", ["panels", "overlay"])
def test_every_rule_renders(rule, layout):
    climb = SCENARIOS[rule]()
    findings = [f for f in frame_picker.select(run_rules(climb), climb.cfg) if f.rule == rule]
    assert findings, f"scenario did not trigger {rule}"
    f = findings[0]
    fix = suggest_fix(climb, f)
    beta = suggest_beta(climb, f, fix)
    frame = synth.draw_wall(holds=climb.holds)
    image = render_finding(frame, climb, f, fix, beta, layout=layout)
    panels = 1 if layout == "overlay" else 1 + (fix is not None) + (beta is not None)
    assert image.shape[1] >= panels * frame.shape[1]
    assert image.shape[0] > frame.shape[0]  # caption banner added
    assert filename(1, climb.t[f.frame], f).startswith("01_00m") and rule in filename(1, 0.0, f)


def test_merging_keeps_one_screenshot_per_error():
    climb = synth.scenario_swing()  # several swings close together
    raw = [f for f in run_rules(climb) if f.rule == "body-swing"]
    picked = [f for f in frame_picker.select(raw, climb.cfg)]
    assert len(raw) > 1 and len(picked) == 1


def write_video(path, wall, poses, fps=30):
    h, w = wall.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for kp in poses:
        frame = wall.copy()
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24),
                     (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)):
            cv2.line(frame, tuple(map(int, kp[a])), tuple(map(int, kp[b])), (40, 40, 40), 12)
        writer.write(frame)
    writer.release()


def test_end_to_end_on_a_synthetic_video(tmp_path):
    bent, good = synth.bent_pose(), synth.good_pose()
    poses = [bent] * 60 + synth.lerp_poses(bent, good, 15) + [good] * 60
    holds = synth.holds_for(good)
    video = tmp_path / "climb.mp4"
    write_video(video, synth.draw_wall(holds=holds), poses)

    out = analyse(video, "yellow", tmp_path / "out", estimator=synth.FakeEstimator(poses), log=lambda m: None)

    index = json.loads((out / "findings.json").read_text())
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert [item["file"] for item in index] == pngs
    assert len(pngs) == 1 and pngs[0].startswith("01_") and "bent-arms_both" in pngs[0]
    assert index[0]["fix"] and index[0]["severity"] == "severe"
    assert out.name.startswith("climb_")
    assert cv2.imread(str(out / pngs[0])) is not None


def test_end_to_end_clean_climb(tmp_path):
    good = synth.good_pose()
    poses = [good] * 60
    video = tmp_path / "clean.mp4"
    write_video(video, synth.draw_wall(holds=synth.holds_for(good)), poses)
    out = analyse(video, "yellow", tmp_path / "out", estimator=synth.FakeEstimator(poses), log=lambda m: None)
    assert [p.name for p in out.glob("*.png")] == ["00_no-errors.png"]
    assert json.loads((out / "findings.json").read_text()) == []
