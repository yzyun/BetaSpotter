"""Side-view rules: body position relative to the wall surface, which only a side camera sees.

Thresholds follow the wall angle: an overhang allows the hips further out than a
vertical wall, and the overhang-only checks stay off on vertical walls and slabs.
The camera sees both sides of the body on top of each other, so nothing here relies
on telling left from right.
"""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..pose import landmarks as L
from ..types import Finding
from . import bent_arms, body_tension
from .common import in_contact, severity


def hip_distance(climb) -> np.ndarray:
    """[T] how far the hips are out from the wall, in torso lengths."""
    return climb.wall.distance(G.hip_mid(climb.kp)) / climb.scale


def hip_limit(climb) -> float:
    """Allowed hip distance for this wall's angle."""
    cfg = climb.cfg["side"]
    extra = cfg["hip_dist_per_deg"] * max(climb.wall.angle_deg, 0.0)
    return float(min(cfg["hip_dist_vertical"] + extra, cfg["hip_dist_max"]))


def is_slab(climb) -> bool:
    return climb.wall.angle_deg < climb.cfg["side"]["slab_max_deg"]


def is_overhang(climb) -> bool:
    return climb.wall.angle_deg >= climb.cfg["side"]["overhang_min_deg"]


def latches(climb) -> list[int]:
    """Frames where a hand lands while the other hand is on."""
    placed = in_contact(climb)
    out = set()
    for hand in (0, 1):
        on = placed[:, hand]
        for e in np.flatnonzero(~on[:-1] & on[1:]) + 1:
            if climb.on_wall[e] and placed[e, 1 - hand]:
                out.add(int(e))
    return sorted(out)


def check_hips(climb) -> list[Finding]:
    if is_slab(climb):
        return []
    cfg = climb.cfg["side"]
    placed = in_contact(climb)
    dist, limit = hip_distance(climb), hip_limit(climb)
    mask = (climb.static & climb.on_wall & placed[:, :2].any(axis=1) & placed[:, 2:].any(axis=1)
            & (dist > limit))
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["hip_min_duration_s"])):
        frame = s + int(np.argmax(dist[s:e]))
        worst = float(dist[frame])
        findings.append(Finding(
            rule="hips-far-from-wall", title="HIPS FAR FROM THE WALL",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(worst - limit, 0.15, 0.3), limb=None,
            measured=f"hips {worst:.2f} torso lengths out from the wall for "
                     f"{climb.t[e - 1] - climb.t[s]:.1f}s (aim for under {limit:.2f} on a "
                     f"{climb.wall.describe()} wall)",
            tip="The further your hips hang out, the more of your weight goes onto your arms: "
                "tighten your core, pull with your toes and bring your hips in.",
            evidence={"dist": worst, "limit": limit, "target": 0.8 * limit},
            score=worst - limit,
        ))
    return findings


def check_shoulders(climb) -> list[Finding]:
    """Hanging on straight arms with the shoulders shrugged up to the ears (overhangs only)."""
    if not is_overhang(climb):
        return []
    cfg = climb.cfg["side"]
    kp = climb.kp
    ears = 0.5 * (kp[:, L.L_EAR] + kp[:, L.R_EAR])
    gap = np.linalg.norm(ears - G.shoulder_mid(kp), axis=-1) / climb.scale
    usable = climb.valid & climb.on_wall
    if not usable.any():
        return []
    ref = float(np.percentile(gap[usable], 80))
    placed = in_contact(climb)
    hanging = np.zeros(len(gap), dtype=bool)
    for limb, side in enumerate(("left", "right")):
        shoulder, _, wrist = L.ARM[side]
        hanging |= (placed[:, limb] & (kp[:, wrist, 1] < kp[:, shoulder, 1] - 0.1 * climb.scale)
                    & (G.elbow_angle(kp, side) >= climb.cfg["bent_arms"]["max_elbow_deg"]))
    rel = gap / max(ref, 1e-6)
    mask = climb.static & climb.on_wall & hanging & (rel < cfg["shoulder_ratio"])
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["shoulder_min_duration_s"])):
        frame = s + int(np.argmin(rel[s:e]))
        worst = float(rel[frame])
        findings.append(Finding(
            rule="passive-shoulders", title="PASSIVE SHOULDERS",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(cfg["shoulder_ratio"] - worst, 0.1, 0.2), limb=None,
            measured=f"the gap between your ears and shoulders is {worst * 100:.0f}% of your usual "
                     f"(aim for over {cfg['shoulder_ratio'] * 100:.0f}%), so you are hanging "
                     f"off your joints",
            tip="Keep your arms straight but pull your shoulder blades down and back, so the "
                "muscles carry you and not the shoulder joint.",
            evidence={"ratio": worst},
            score=cfg["shoulder_ratio"] - worst,
        ))
    return findings


def check_square_reach(climb) -> list[Finding]:
    """A hand lands with the hips square to the wall and far out from it (overhangs only)."""
    if not is_overhang(climb):
        return []
    cfg = climb.cfg["side"]
    dist = hip_distance(climb)
    far = cfg["reach_hip_fraction"] * hip_limit(climb)
    width = np.linalg.norm(climb.kp[:, L.L_HIP] - climb.kp[:, L.R_HIP], axis=-1) / climb.scale
    win = climb.frames(cfg["reach_window_s"])
    findings = []
    for e in latches(climb):
        end = min(len(dist), e + win)
        d, w = float(np.median(dist[e:end])), float(np.median(width[e:end]))
        if d <= far or w >= cfg["square_hip_width"]:
            continue
        frame = e + int(np.argmax(dist[e:end]))
        findings.append(Finding(
            rule="square-reach", title="SQUARE HIPS ON THE REACH",
            t_start=float(climb.t[e]), t_end=float(climb.t[end - 1]), frame=frame,
            severity=severity(d - far, 0.15, 0.3), limb=None,
            measured=f"you reached with your hips square to the wall and {d:.2f} torso lengths "
                     f"out from it (aim for under {far:.2f} on the reach)",
            tip="Turn the hip on the reaching side in to the wall and drop that knee: it brings "
                "your hips in and makes the same reach shorter.",
            evidence={"dist": d, "limit": far, "width": w, "target": 0.8 * far},
            score=d - far,
        ))
    return findings


def check_swing(climb) -> list[Finding]:
    """Hips swinging out from the wall, and back, after a hand lands."""
    if is_slab(climb):
        return []
    cfg = climb.cfg["side"]
    dist = hip_distance(climb)
    hands_on = in_contact(climb)[:, :2].any(axis=1) & climb.on_wall
    win = climb.frames(cfg["swing_window_s"])
    findings = []
    for e in latches(climb):
        end = min(len(dist), e + win)
        gone = np.flatnonzero(~hands_on[e:end])
        if len(gone):
            end = e + int(gone[0])
        if end - e < 3:
            continue
        peak = e + int(np.argmax(dist[e:end]))
        out = float(dist[peak] - dist[e])
        back = float(dist[peak] - dist[peak:end].min())
        if out <= cfg["swing_out"] or back < 0.5 * out:  # must come back: otherwise it is a new position
            continue
        calm = np.flatnonzero(climb.hip_speed[peak:end] < cfg["swing_settle_speed"])
        settle = float(climb.t[peak + int(calm[0]) if len(calm) else end - 1] - climb.t[e])
        findings.append(Finding(
            rule="swing-out", title="SWINGING OUT AFTER THE MOVE",
            t_start=float(climb.t[e]), t_end=float(climb.t[end - 1]), frame=peak,
            severity=severity(out - cfg["swing_out"], 0.2, 0.5), limb=None,
            measured=f"your hips swung {out:.2f} torso lengths out from the wall after the move "
                     f"and took {settle:.1f}s to settle (aim for under {cfg['swing_out']:.2f})",
            tip="Hold tension through your toes and core as the hand lands, so the wall-side "
                "foot stops the swing before it starts.",
            evidence={"out": out, "from_frame": int(e), "target": float(dist[e])},
            score=out - cfg["swing_out"],
        ))
    return findings


def run_side_rules(climb) -> list[Finding]:
    """All checks for side-view footage. Needs climb.wall."""
    if climb.wall is None:
        raise ValueError("side-view rules need the wall line (climb.wall)")
    findings = (bent_arms.check(climb) + body_tension.check_cut(climb, need_hold=False)
                + check_hips(climb) + check_shoulders(climb) + check_square_reach(climb)
                + check_swing(climb))
    findings.sort(key=lambda f: f.t_start)
    return findings
