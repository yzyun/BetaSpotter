"""Low body tension: feet cutting loose, hips sagging away from the wall, swinging."""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..pose import landmarks as L
from ..types import NO_CONTACT, Finding
from .common import in_contact, severity


def torso_ratio(climb) -> tuple[np.ndarray, float]:
    """Apparent torso length relative to shoulder width, and its normal value.

    Filmed from behind, hips sagging off the wall tilt the torso towards the camera,
    which makes it look shorter. Dividing by shoulder width cancels camera distance.
    """
    ratio = G.torso_length(climb.kp) / np.maximum(G.shoulder_width(climb.kp), 1e-3)
    usable = climb.valid & climb.on_wall
    ref = float(np.median(ratio[usable])) if usable.any() else 1.0
    return ratio, ref


def check_hips(climb) -> list[Finding]:
    cfg = climb.cfg["body_tension"]
    ratio, ref = torso_ratio(climb)
    rel = ratio / ref
    mask = climb.static & climb.on_wall & (rel < cfg["torso_ratio"])
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["min_duration_s"])):
        frame = s + int(np.argmin(rel[s:e]))
        worst = float(rel[frame])
        findings.append(Finding(
            rule="hips-off-wall", title="HIPS OFF THE WALL",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(cfg["torso_ratio"] - worst, 0.1, 0.2), limb=None,
            measured=f"torso looks {worst * 100:.0f}% of its usual length, so your hips are sagging "
                     f"away from the wall (aim for over {cfg['torso_ratio'] * 100:.0f}%)",
            tip="Squeeze your core and push through your toes to pull your hips in towards the wall.",
            evidence={"ratio": worst, "lengthen": 1.0 / max(worst, 0.3)},
            score=cfg["torso_ratio"] - worst,
        ))
    return findings


def check_cut(climb) -> list[Finding]:
    cfg = climb.cfg["body_tension"]
    placed = in_contact(climb)
    hands_on = placed[:, 0] | placed[:, 1]
    feet_off = ~placed[:, 2] & ~placed[:, 3]
    mask = climb.on_wall & hands_on & feet_off
    look_back = climb.frames(cfg["cut_lookback_s"])
    look_ahead = climb.frames(cfg["drop_lookahead_s"])
    T = len(mask)
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["cut_min_s"])):
        before = climb.contacts[max(0, s - look_back):s, 2:]
        if not (before >= 0).any():  # feet were not on holds just before
            continue
        after = hands_on[e:min(T, e + look_ahead)]
        if e + look_ahead >= T or (len(after) and not after.any()):
            continue  # dropping off / end of the clip
        targets = {}
        for limb in (2, 3):
            prior = np.flatnonzero(climb.contacts[:s, limb] != NO_CONTACT)
            if len(prior):
                targets[limb] = climb.kp[prior[-1], L.LIMB_END_JOINT[limb]].tolist()
        frame = (s + e) // 2
        dur = float(climb.t[e - 1] - climb.t[s])
        findings.append(Finding(
            rule="feet-cut", title="FEET CUT LOOSE",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(dur, 0.6, 1.2), limb=None,
            measured=f"both feet off the wall for {dur:.1f}s while hanging on your hands",
            tip="Keep tension through your toes as you move: press into the footholds and tighten your core before reaching.",
            evidence={"foot_targets": targets},
            score=dur,
        ))
    return findings


def check_swing(climb) -> list[Finding]:
    cfg = climb.cfg["body_tension"]
    placed = in_contact(climb)
    mask = climb.on_wall & placed.all(axis=1) & (climb.hip_speed > cfg["swing_speed"])
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["swing_min_s"])):
        frame = s + int(np.argmax(climb.hip_speed[s:e]))
        worst = float(climb.hip_speed[frame])
        findings.append(Finding(
            rule="body-swing", title="UNCONTROLLED SWING",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(worst - cfg["swing_speed"], 0.5, 1.0), limb=None,
            measured=f"hips moving at {worst:.1f} torso lengths/s with all four limbs placed "
                     f"(aim for under {cfg['swing_speed']:.1f})",
            tip="Set your hips over your feet before you let go with a hand, so the body does not swing out.",
            evidence={"speed": worst},
            score=worst - cfg["swing_speed"],
        ))
    return findings


def check(climb) -> list[Finding]:
    return check_hips(climb) + check_cut(climb) + check_swing(climb)
