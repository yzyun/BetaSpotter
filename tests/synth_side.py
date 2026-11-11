"""Synthetic side-view climbers: positions are given relative to a tilted wall."""
from __future__ import annotations

import numpy as np

import synth
from betaspotter.analysis.wall import estimate_wall, normal_for

TORSO = synth.TORSO
REACH_STRAIGHT = 103.5  # shoulder to wrist with the elbow at about 160 degrees
REACH_BENT = 75.0  # about 90 degrees


class SideWall:
    """A wall leaning `angle` degrees past vertical, climber on the +x (side=1) or -x side."""

    def __init__(self, angle=30.0, side=1, base=(300.0, 900.0)):
        self.angle, self.side = angle, side
        self.base = np.array(base if side == 1 else (1280 - base[0], base[1]), float)
        self.n = normal_for(angle, side)
        a = np.radians(angle)
        self.u = np.array([side * np.sin(a), -np.cos(a)])

    def point(self, s, d):
        """s pixels up the wall from its base, d pixels out from the surface."""
        return self.base + s * self.u + d * self.n


def pose(wall, hands_s=500.0, hip_d=0.6, shoulder_d=0.8, reach=REACH_STRAIGHT, feet_s=None,
         feet_d=12.0, lead_hand_s=None, twist=6.0, neck=30.0):
    """Climber hanging below `hands_s`. hip_d / shoulder_d: torso lengths out from the wall.

    lead_hand_s puts the left hand somewhere else up the wall; twist is the apparent
    hip width in pixels (small = hips square to the wall); neck is the ear-shoulder gap.
    """
    wrist_d = 20.0  # the palm, a hand length past the wrist, then sits on the wall
    sd, hd = shoulder_d * TORSO, hip_d * TORSO
    sh_s = hands_s - np.sqrt(reach ** 2 - (sd - wrist_d) ** 2)
    hip_s = sh_s - np.sqrt(TORSO ** 2 - (hd - sd) ** 2)
    feet_s = hip_s - 110.0 if feet_s is None else feet_s
    shoulder, hip = wall.point(sh_s, sd), wall.point(hip_s, hd)
    along = (shoulder - hip) / TORSO

    def limb(root, target, l1, l2):  # joint in the middle bends away from the wall
        options = [synth._ik(root, target, l1, l2, bend) for bend in (-1, 1)]
        return max(options, key=lambda o: float((o[0] - wall.base) @ wall.n))

    kp = np.zeros((33, 2))
    kp[11], kp[12] = shoulder - 4 * wall.u, shoulder + 4 * wall.u
    kp[23], kp[24] = hip - 0.5 * twist * wall.u, hip + 0.5 * twist * wall.u
    lead = hands_s if lead_hand_s is None else lead_hand_s
    kp[13], kp[15] = limb(kp[11], wall.point(lead - 4, wrist_d), synth.UPPER_ARM, synth.FOREARM)
    kp[14], kp[16] = limb(kp[12], wall.point(hands_s + 4, wrist_d), synth.UPPER_ARM, synth.FOREARM)
    kp[25], kp[27] = limb(kp[23], wall.point(feet_s, feet_d), synth.THIGH, synth.SHANK)
    kp[26], kp[28] = limb(kp[24], wall.point(feet_s + 3, feet_d), synth.THIGH, synth.SHANK)
    kp[0:11] = shoulder + neck * along - 10 * wall.n  # face, turned to the wall
    kp[7] = kp[8] = shoulder + neck * along  # ears
    for wrist, elbow, pts in ((15, 13, (17, 19, 21)), (16, 14, (18, 20, 22))):
        kp[list(pts)] = kp[wrist] + 0.3 * (kp[wrist] - kp[elbow])
    for ankle, heel, toe in ((27, 29, 31), (28, 30, 32)):
        kp[heel], kp[toe] = kp[ankle] + 6 * wall.n, kp[ankle] - 8 * wall.n
    return kp


def climb(poses, fps=30.0, cfg=None, wall_angle=None):
    c = synth.sequence_climb(poses, (), fps, cfg)
    c.wall = estimate_wall(c, wall_angle)
    return c


def hold(kp, seconds=2.0, fps=30.0):
    return [kp] * int(seconds * fps)


# ---------------------------------------------------------------- ready-made scenarios

def scenario_good(angle=30.0, side=1):
    return climb(hold(pose(SideWall(angle, side))))


def scenario_route(angle=30.0, side=1):
    """Four positions up the wall, moving between them."""
    wall = SideWall(angle, side)
    poses = []
    stops = [pose(wall, hands_s=s) for s in (430, 500, 570, 640)]
    for a, b in zip(stops, stops[1:]):
        poses += hold(a, 1.0) + synth.lerp_poses(a, b, 10)
    return climb(poses + hold(stops[-1], 1.0))


def scenario_hips_out(angle=30.0, hip_d=1.4):
    return climb(hold(pose(SideWall(angle), hip_d=hip_d, shoulder_d=0.9)))


def scenario_bent_arms(angle=30.0):
    return climb(hold(pose(SideWall(angle), reach=REACH_BENT)))


def scenario_passive_shoulders(angle=30.0):
    wall = SideWall(angle)
    return climb(hold(pose(wall), 3.0) + hold(pose(wall, neck=15.0), 1.5))


def scenario_feet_cut(angle=30.0):
    wall = SideWall(angle)
    on = pose(wall)
    off = pose(wall, feet_d=90.0, feet_s=260.0)
    poses = hold(on, 1.0) + synth.lerp_poses(on, off, 15) + synth.lerp_poses(off, on, 15) + hold(on, 2.5)
    return climb(poses)


def _reach(wall, after, **kw):
    """Left hand goes from beside the right hand to a higher hold, then `after` plays."""
    start, end = pose(wall, **kw), pose(wall, lead_hand_s=545.0, **kw)
    return hold(start, 1.0) + synth.lerp_poses(start, end, 10) + after(end)


def scenario_square_reach(angle=30.0, twist=6.0, hip_d=0.9):
    wall = SideWall(angle)
    return climb(_reach(wall, lambda end: hold(end, 1.5), twist=twist, hip_d=hip_d, shoulder_d=0.9))


def scenario_swing(angle=30.0, out=0.5, fps=30.0):
    wall = SideWall(angle)

    def swing(end):
        n = int(0.8 * fps)
        bump = [pose(wall, lead_hand_s=545.0, hip_d=0.6 + out * np.sin(np.pi * i / n)) for i in range(n)]
        return hold(end, 0.1) + bump + hold(end, 1.5)

    return climb(_reach(wall, swing))
