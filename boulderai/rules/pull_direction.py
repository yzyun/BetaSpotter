"""Wrong direction of pull: the forearm is not lined up with the way the hold wants loading."""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..analysis.contacts import episodes
from ..types import Finding
from .common import severity

SIDES = ("left", "right")


def pull_angle(kp, side: str, direction: np.ndarray) -> np.ndarray:
    """Angle (deg) between the forearm's pull and the hold's usable direction."""
    return G.angle_between(G.pull_vector(kp, side), direction)


def threshold_for(hold, cfg) -> float:
    return cfg["max_angle_deg"] if hold.confidence >= 0.5 else cfg["low_confidence_angle_deg"]


def check(climb) -> list[Finding]:
    cfg = climb.cfg["pull_direction"]
    findings = []
    for limb, side in enumerate(SIDES):
        for s, e, hold_id in episodes(climb.contacts, limb):
            hold = climb.hold(hold_id) if hold_id >= 0 else None
            if hold is None:
                continue
            thr = threshold_for(hold, cfg)
            ang = pull_angle(climb.kp[s:e], side, hold.direction)
            mask = climb.static[s:e] & climb.on_wall[s:e] & (ang > thr)
            for rs, re in G.runs(mask, climb.frames(cfg["min_duration_s"])):
                frame = s + rs + int(np.argmax(ang[rs:re]))
                worst = float(ang[frame - s])
                guess = "" if hold.confidence >= 0.5 else " (hold direction is a guess: check holds.yaml)"
                findings.append(Finding(
                    rule="wrong-pull-direction", title="WRONG DIRECTION OF PULL",
                    t_start=float(climb.t[s + rs]), t_end=float(climb.t[s + re - 1]), frame=frame,
                    severity=severity(worst - thr, 20, 45), limb=side,
                    measured=f"{side} forearm is {worst:.0f} deg off the hold's pull direction "
                             f"(aim for under {thr:.0f} deg){guess}",
                    tip="Move your body so your forearm points along the direction the hold is good in: "
                        "get opposite a sidepull, underneath an edge.",
                    evidence={"side": side, "hold_id": hold.id, "angle": worst,
                              "confident": bool(hold.confidence >= 0.5)},
                    score=worst - thr,
                ))
    return findings
