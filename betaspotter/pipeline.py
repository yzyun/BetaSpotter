"""End-to-end: video -> pose track -> holds -> rules -> fixes -> screenshot folder."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .analysis.climb import Climb
from .analysis.geometry import runs
from .analysis.wall import WallError, estimate_wall
from .config import load_config
from .fixes.beta import suggest_beta
from .fixes.corrections import suggest_fix
from .holds.orientation import estimate_orientations, load_holds
from .holds.segment import DISPLAY_BGR, HSV_PRESETS, build_background, parse_hsv, segment_holds
from .io.video import grab_frames, read_frames, video_info
from .pose.smoothing import smooth_track
from .report import frame_picker
from .report.writer import make_out_dir, write_clean, write_report
from .rules import run_rules
from .rules.side import run_side_rules

BG_MAX_DIM = 960
BG_MAX_SAMPLES = 40


class AnalysisError(RuntimeError):
    pass


def track_video(path, estimator, stride: int = 1, log: Callable = print):
    """Run pose estimation over the video and collect frames for the wall background.

    Returns dict with fps (of the processed frames), file_idx, t, kp, vis,
    bg_frames (downscaled), bg_boxes, bg_factor.
    """
    fps, count = video_info(path)
    kps, viss, file_idx, bg_frames, bg_boxes = [], [], [], [], []
    bg_step = max(1, (count // max(stride, 1)) // 25) if count else 10
    factor = None
    for n, (i, frame) in enumerate(read_frames(path, stride)):
        kp, vis = estimator.estimate(frame, i / fps)
        kps.append(kp)
        viss.append(vis)
        file_idx.append(i)
        if n % bg_step == 0:
            if factor is None:
                factor = min(1.0, BG_MAX_DIM / max(frame.shape[:2]))
            small = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) \
                if factor < 1.0 else frame.copy()
            box = None
            if vis.mean() >= 0.3:
                lo, hi = kp.min(axis=0), kp.max(axis=0)
                pad = 0.25 * (hi - lo) + 10
                box = tuple(int(v * factor) for v in (*(lo - pad), *(hi + pad)))
            bg_frames.append(small)
            bg_boxes.append(box)
            if len(bg_frames) > BG_MAX_SAMPLES:
                bg_frames, bg_boxes = bg_frames[::2], bg_boxes[::2]
                bg_step *= 2
        if count and n % max(1, (count // stride) // 10) == 0:
            log(f"  pose: {min(100, 100 * i // max(count, 1))}%")
    if not kps:
        raise AnalysisError(f"no frames could be read from {path}")
    file_idx = np.array(file_idx)
    return {
        "fps": fps / stride, "file_idx": file_idx, "t": file_idx / fps,
        "kp": np.array(kps), "vis": np.array(viss),
        "bg_frames": bg_frames, "bg_boxes": bg_boxes, "bg_factor": factor or 1.0,
    }


def find_holds(track: dict, colour: str, hsv: str | None = None):
    """Segment holds from the climber-free background. Returns (holds, background image)."""
    if hsv:
        ranges = parse_hsv(hsv)
    elif colour in HSV_PRESETS:
        ranges = HSV_PRESETS[colour]
    else:
        raise AnalysisError(f"unknown hold colour '{colour}'. Choose from "
                            f"{', '.join(sorted(HSV_PRESETS))} or pass --hsv.")
    bg = build_background(track["bg_frames"], track["bg_boxes"])
    holds = segment_holds(bg, ranges)
    f = track["bg_factor"]
    for h in holds:  # back to full-resolution pixels
        h.polygon = h.polygon / f
        h.centroid = h.centroid / f
        h.area = h.area / (f * f)
    estimate_orientations(holds)
    return holds, bg


def _clean_frame(climb) -> int:
    """A representative frame for the 'no errors' image: middle of the longest static spell."""
    spells = runs(climb.static & climb.on_wall, 1) or runs(climb.valid, 1)
    s, e = max(spells, key=lambda r: r[1] - r[0])
    return (s + e) // 2


def analyse(video, hold_colour: str = "yellow", out="out", config=None, holds_file=None,
            hsv: str | None = None, layout: str = "panels", stride: int = 1,
            model: str = "full", estimator=None, log: Callable = print) -> Path:
    """Analyse a climbing video and write a folder of annotated screenshots. Returns the folder."""
    cfg = load_config(config)
    track, kp, valid = _tracked(video, cfg, stride, model, estimator, log, "from behind")

    if holds_file:
        holds, hold_colour = load_holds(holds_file)
        log(f"Loaded {len(holds)} holds from {holds_file}")
    else:
        holds, _ = find_holds(track, hold_colour, hsv)
        log(f"Found {len(holds)} {hold_colour} holds")
        if not holds:
            log("  warning: no holds found, so pull direction and alternative beta are skipped. "
                "Try another --hold-colour or --hsv.")
    hold_bgr = DISPLAY_BGR.get(hold_colour, (0, 230, 255))

    climb = Climb.build(track["t"], kp, valid, holds, cfg, fps=track["fps"])
    findings = frame_picker.select(run_rules(climb), cfg)
    log(f"Flagged {len(findings)} technique errors")

    items = []
    for f in findings:
        fix = suggest_fix(climb, f)
        beta = suggest_beta(climb, f, fix)
        items.append((f, fix, beta))

    return _write(video, out, track, climb, items, hold_bgr, layout, log)


def _tracked(video, cfg, stride, model, estimator, log, view: str):
    """Pose-track the video and clean the track up. Returns (track, keypoints, valid)."""
    own_estimator = estimator is None
    if own_estimator:
        from .pose.estimator import PoseEstimator
        log("Loading pose model...")
        estimator = PoseEstimator(model)
    try:
        log("Tracking the climber...")
        track = track_video(video, estimator, stride, log)
    finally:
        if own_estimator:
            estimator.close()
    kp, valid = smooth_track(track["kp"], track["vis"], track["fps"], cfg)
    if valid.sum() < max(5, int(track["fps"])):
        raise AnalysisError("Could not find a climber in this video. Check that one person is "
                            f"clearly visible, filmed {view} with a steady camera.")
    return track, kp, valid


def _write(video, out, track, climb, items, hold_bgr, layout, log) -> Path:
    findings = [item[0] for item in items]
    out_dir = make_out_dir(out, video)
    if items:
        wanted = sorted({int(track["file_idx"][f.frame]) for f in findings})
        raw = grab_frames(video, wanted)
        frames = {f.frame: raw[int(track["file_idx"][f.frame])] for f in findings}
        write_report(out_dir, climb, frames, items, hold_bgr, layout)
    else:
        i = _clean_frame(climb)
        idx = int(track["file_idx"][i])
        write_clean(out_dir, climb, grab_frames(video, [idx])[idx], i, hold_bgr)
    log(f"Wrote {max(1, len(items))} screenshot(s) to {out_dir}")
    return out_dir


def analyse_side(video, out="out", config=None, wall_angle: float | None = None,
                 layout: str = "panels", stride: int = 1, model: str = "full",
                 estimator=None, log: Callable = print) -> Path:
    """Analyse a climb filmed from the side: body position relative to the wall.

    wall_angle: degrees past vertical (positive = overhang) to use instead of the
    angle fitted through the hand and foot placements.
    """
    cfg = load_config(config)
    track, kp, valid = _tracked(video, cfg, stride, model, estimator, log, "from the side")
    climb = Climb.build(track["t"], kp, valid, [], cfg, fps=track["fps"])
    try:
        climb.wall = wall = estimate_wall(climb, wall_angle)
    except WallError as err:
        raise AnalysisError(str(err)) from err
    how = "as given" if wall.source == "given" else f"estimated from {wall.n_points} hand and foot placements"
    log(f"Wall: {wall.describe()} ({how})")
    if wall.rms > cfg["side"]["wall_max_rms"]:
        log("  warning: the placements do not line up well, so the wall line is uncertain. "
            "Check it in the screenshots and pass --wall-angle if it is off.")

    findings = frame_picker.select(run_side_rules(climb), cfg)
    log(f"Flagged {len(findings)} technique errors")
    items = [(f, suggest_fix(climb, f), None) for f in findings]
    return _write(video, out, track, climb, items, (0, 230, 255), layout, log)
