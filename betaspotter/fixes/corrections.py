"""Per-rule corrections: the same holds, a better body position."""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..pose import landmarks as L
from ..types import LIMBS, NO_CONTACT, Finding, Suggestion
from .optimizer import Objective, solve_pose

SIDES = ("left", "right")
SIDE_HIP_RULES = ("hips-far-from-wall", "square-reach", "swing-out")


def contact_pins(climb, frame: int) -> dict:
    """Pin every placed limb where it is."""
    return {L.LIMB_END_JOINT[limb]: climb.kp[frame, L.LIMB_END_JOINT[limb]].copy()
            for limb in range(4) if climb.contacts[frame, limb] != NO_CONTACT}


def com_range(climb, frame: int, pins: dict, kp: np.ndarray | None = None):
    """Sideways span the centre of mass should stay inside, from the pinned limbs."""
    kp = climb.kp[frame] if kp is None else kp
    xs = []
    for limb in range(4):
        joint = L.LIMB_END_JOINT[limb]
        if joint in pins:
            # contact point = pinned joint + the hand/toe offset from it
            offset = climb.limb_pts[frame, limb] - climb.kp[frame, joint]
            xs.append(float(np.asarray(pins[joint])[0] + offset[0]))
    if len(xs) < 2:
        return None
    return (min(xs), max(xs))


def build_problem(climb, finding: Finding):
    """Return (pins, bones, objective) for a finding, or None if no pose fix applies."""
    frame = finding.frame
    cfg = climb.cfg
    pins = contact_pins(climb, frame)
    bones = climb.bone_targets(frame)
    obj = Objective()
    rule = finding.rule
    if climb.wall is not None:
        obj.wall = (climb.wall.p0, climb.wall.n)

    if rule == "bent-arms":
        for side in finding.evidence["sides"]:
            obj.elbow_min[side] = cfg["fixes"]["target_elbow_deg"]
    elif rule == "wrong-pull-direction":
        hold = climb.hold(finding.evidence["hold_id"])
        obj.forearm_dir[finding.evidence["side"]] = hold.direction
    elif rule == "com-outside-support":
        obj.com_x = com_range(climb, frame, pins)
    elif rule == "hips-off-wall":
        grow = float(np.clip(finding.evidence["lengthen"], 1.0, 1.6))
        for bone in [(11, 23), (12, 24), (11, 24), (12, 23)]:
            bones[bone] *= grow
    elif rule == "feet-cut":
        for limb, target in finding.evidence["foot_targets"].items():
            pins[L.LIMB_END_JOINT[int(limb)]] = np.asarray(target, dtype=float)
        obj.com_x = com_range(climb, frame, pins)
    elif rule == "body-swing":
        obj.com_x = com_range(climb, frame, pins)
    elif rule in SIDE_HIP_RULES and climb.wall is not None:
        obj.hip_max = float(finding.evidence["target"]) * float(climb.scale[frame])
    else:
        return None
    return pins, bones, obj


def describe_change(kp0: np.ndarray, kp1: np.ndarray, scale: float, wall=None) -> str:
    """Plain-words description of how the hips move between two poses.

    Seen from the side (wall given), sideways in the picture means in or out from the wall.
    """
    d = (G.hip_mid(kp1) - G.hip_mid(kp0)) / scale
    parts = []
    if wall is not None:
        out = float(d @ wall.n)
        if abs(out) > 0.1:
            parts.append("in towards the wall" if out < 0 else "out from the wall")
        if abs(d[1]) > 0.1 and abs(out) <= 0.1:
            parts.append("lower" if d[1] > 0 else "higher")
        return "hips " + " and ".join(parts) if parts else ""
    if abs(d[1]) > 0.1:
        parts.append("lower" if d[1] > 0 else "higher")
    if abs(d[0]) > 0.1:
        parts.append("to the right" if d[0] > 0 else "to the left")
    if not parts:
        return ""
    return "hips " + " and ".join(parts)


_FIX_TIPS = {
    "bent-arms": "Straighten your arms and sink onto them: {move}, knees bent.",
    "wrong-pull-direction": "Shift so your forearm lines up with the hold's good direction: {move}.",
    "com-outside-support": "Bring your weight back between your hands and feet: {move}.",
    "hips-off-wall": "Drive your hips in to the wall so your body is long and flat against it.",
    "feet-cut": "Keep both feet pressed on their holds with your weight between them.",
    "body-swing": "Settle your weight between your holds before moving: {move}.",
    "hips-far-from-wall": "Bring your hips in on the same hands and feet: {move}.",
    "square-reach": "Turn side-on and reach with your hips in: {move}.",
    "swing-out": "Stay tight as the hand lands so your hips stay where they were: {move}.",
}


def suggest_fix(climb, finding: Finding) -> Suggestion | None:
    problem = build_problem(climb, finding)
    if problem is None:
        return None
    pins, bones, obj = problem
    kp0 = climb.kp[finding.frame]
    scale = float(climb.scale[finding.frame])
    kp1 = solve_pose(kp0, pins, bones, obj, scale)
    move = describe_change(kp0, kp1, scale, climb.wall) or "small adjustment"
    tip = _FIX_TIPS[finding.rule].format(move=move)
    return Suggestion(kind="fix", keypoints=kp1, tip=tip)


def limb_name(limb: int) -> str:
    return LIMBS[limb].replace("_", " ")
