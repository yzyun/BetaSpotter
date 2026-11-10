"""Alternative beta: would using a different hold solve the problem better?"""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..pose import landmarks as L
from ..types import Finding, Suggestion
from .corrections import build_problem, com_range, limb_name
from .optimizer import solve_pose

SIDES = ("left", "right")


def pose_cost(climb, kp0, kp, placed: dict, scale: float) -> float:
    """Lower is better. placed: limb index -> hold id (or SMEAR) for limbs on the wall.

    Adds up: centre of mass outside the contacts, bent hanging arms, pull misalignment,
    and how far the body had to move from what the climber actually did.
    """
    cost = 0.0
    pts = G.limb_points(kp)
    if len(placed) >= 2:
        xs = [pts[limb, 0] for limb in placed]
        cost += 3.0 * abs(G.com_offset(float(G.com(kp)[0]), xs, scale))
    for limb in (0, 1):
        if limb not in placed:
            continue
        side = SIDES[limb]
        s, _, w = L.ARM[side]
        if kp[w, 1] < kp[s, 1] - 0.1 * scale:
            cost += max(0.0, 150.0 - float(G.elbow_angle(kp, side))) / 90.0
        hold = climb.hold(placed[limb]) if placed[limb] >= 0 else None
        if hold is not None and hold.confidence >= 0.5:
            off = float(G.angle_between(G.pull_vector(kp, side), hold.direction))
            cost += max(0.0, off - 45.0) / 90.0
    # Crossed limbs are awkward: prefer the limb on the same side as the hold.
    for left, right, penalty in ((2, 3, 0.3), (0, 1, 0.15)):
        if left in placed and right in placed and pts[left, 0] > pts[right, 0] + 0.1 * scale:
            cost += penalty
    # Shifting the whole body costs far more than re-placing one hand or foot.
    dist = np.linalg.norm(kp - kp0, axis=1) / scale
    torso = dist[[L.L_SHOULDER, L.R_SHOULDER, L.L_HIP, L.R_HIP]].mean()
    limbs = dist[[j for j in L.MAIN_JOINTS if j not in (11, 12, 23, 24)]].mean()
    return cost + torso + 0.15 * limbs


def _limb_reach(bones: dict, limb: int) -> float:
    root, mid_, end = (L.ARM if limb < 2 else L.LEG)[SIDES[limb % 2]][:3]
    return bones[(root, mid_)] + bones[(mid_, end)]


def _where(kp0, limb: int, hold, scale: float) -> str:
    d = (hold.centroid - G.limb_points(kp0)[limb]) / scale
    words = []
    if abs(d[1]) > 0.15:
        words.append("below" if d[1] > 0 else "above")
    if abs(d[0]) > 0.15:
        words.append("right" if d[0] > 0 else "left")
    return " and ".join(words) + " of where it is now" if words else "next to where it is now"


def suggest_beta(climb, finding: Finding, fix: Suggestion | None) -> Suggestion | None:
    """Try moving one limb to another hold within reach; keep it if clearly better than the fix."""
    problem = build_problem(climb, finding)
    if problem is None or not climb.holds:
        return None
    pins0, bones, obj0 = problem
    frame = finding.frame
    kp0 = climb.kp[frame]
    scale = float(climb.scale[frame])
    cfg = climb.cfg["fixes"]

    placed0 = {limb: int(climb.contacts[frame, limb]) for limb in range(4)
               if L.LIMB_END_JOINT[limb] in pins0}
    fix_kp = fix.keypoints if fix is not None else kp0
    baseline = pose_cost(climb, kp0, fix_kp, placed0, scale)

    limbs = [2, 3]
    if finding.limb in SIDES and finding.rule in ("bent-arms", "wrong-pull-direction"):
        limbs.insert(0, SIDES.index(finding.limb))
    used = {h for h in placed0.values() if h >= 0}
    hips_y = float(G.hip_mid(kp0)[1])

    candidates = []
    for limb in limbs:
        root = kp0[L.LIMB_ROOT_JOINT[limb]]
        reach = 0.95 * _limb_reach(bones, limb)
        for hold in climb.holds:
            if hold.id in used or hold.id == placed0.get(limb):
                continue
            if limb >= 2 and hold.centroid[1] < hips_y:  # feet stay below the hips
                continue
            if limb < 2 and hold.centroid[1] > hips_y:  # hands stay above them
                continue
            dist = float(np.linalg.norm(hold.centroid - root))
            if dist <= reach:
                candidates.append((dist, limb, hold))
    candidates.sort(key=lambda c: c[0])

    best = None
    for _, limb, hold in candidates[:cfg["max_candidates"]]:
        joint = L.LIMB_END_JOINT[limb]
        offset = kp0[joint] - G.limb_points(kp0)[limb]  # keep hand/toe centred on the hold
        pins = dict(pins0)
        pins[joint] = hold.centroid + offset
        obj = type(obj0)(elbow_min=dict(obj0.elbow_min), forearm_dir=dict(obj0.forearm_dir),
                         com_x=obj0.com_x)
        if limb < 2:
            obj.forearm_dir.pop(SIDES[limb], None)
            if hold.confidence >= 0.5:
                obj.forearm_dir[SIDES[limb]] = hold.direction
        if obj.com_x is not None:
            obj.com_x = com_range(climb, frame, pins)
        kp1 = solve_pose(kp0, pins, bones, obj, scale)
        placed = dict(placed0)
        placed[limb] = hold.id
        cost = pose_cost(climb, kp0, kp1, placed, scale)
        if best is None or cost < best.cost:
            tip = (f"Try your {limb_name(limb)} on hold #{hold.id} "
                   f"({_where(kp0, limb, hold, scale)}).")
            best = Suggestion(kind="beta", keypoints=kp1, tip=tip, cost=cost,
                              limb=limb, hold_id=hold.id)
    if best is not None and best.cost < baseline - cfg["beta_margin"]:
        return best
    return None
