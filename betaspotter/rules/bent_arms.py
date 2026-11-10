"""Bent arms: hanging on a bent arm while resting, instead of on a straight one."""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..pose import landmarks as L
from ..types import Finding
from .common import in_contact, severity

SIDES = ("left", "right")


def bent_mask(climb, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(frames where this arm is wrongly bent, elbow angle per frame)."""
    cfg = climb.cfg["bent_arms"]
    limb = SIDES.index(side)
    placed = in_contact(climb)
    shoulder, _, wrist = L.ARM[side]
    angle = G.elbow_angle(climb.kp, side)
    hanging = climb.kp[:, wrist, 1] < climb.kp[:, shoulder, 1] - 0.1 * climb.scale

    # A lock-off just before either hand reaches is correct technique: skip it.
    look = climb.frames(cfg["lockoff_lookahead_s"])
    about_to_reach = np.zeros(len(angle), dtype=bool)
    for hand in (0, 1):
        on = placed[:, hand]
        for d in np.flatnonzero(on[:-1] & ~on[1:]) + 1:
            about_to_reach[max(0, d - look):d + 1] = True

    mask = (climb.static & climb.on_wall & placed[:, limb] & hanging
            & (angle < cfg["max_elbow_deg"]) & ~about_to_reach)
    return mask, angle


def check(climb) -> list[Finding]:
    cfg = climb.cfg["bent_arms"]
    masks, angles = {}, {}
    for side in SIDES:
        masks[side], angles[side] = bent_mask(climb, side)
    either = masks["left"] | masks["right"]
    findings = []
    for s, e in G.runs(either, climb.frames(cfg["min_duration_s"])):
        sides = [sd for sd in SIDES if masks[sd][s:e].mean() >= 0.5]
        if not sides:
            sides = [max(SIDES, key=lambda sd: masks[sd][s:e].sum())]
        # worst frame: smallest elbow angle among the flagged arms
        worst_angle = np.min([np.where(masks[sd][s:e], angles[sd][s:e], 999.0) for sd in sides], axis=0)
        frame = s + int(np.argmin(worst_angle))
        measured = {sd: float(angles[sd][frame]) for sd in sides}
        smallest = min(measured.values())
        target = cfg["max_elbow_deg"]
        findings.append(Finding(
            rule="bent-arms", title="BENT ARMS",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(target - smallest, target - cfg["moderate_deg"], target - cfg["severe_deg"]),
            limb="both" if len(sides) == 2 else sides[0],
            measured=", ".join(f"{sd} elbow {a:.0f} deg" for sd, a in measured.items())
                     + f" for {climb.t[e - 1] - climb.t[s]:.1f}s (aim for more than {target:.0f} deg)",
            tip="Hang off straight arms and let your skeleton take the load: sink your hips and bend your knees instead of your elbows.",
            evidence={"sides": measured},
            score=float(target - smallest),
        ))
    return findings
