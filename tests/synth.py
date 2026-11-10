"""Synthetic climbers and walls for tests (no real footage needed)."""
from __future__ import annotations

import cv2
import numpy as np

from boulderai.config import load_config
from boulderai.analysis.climb import Climb
from boulderai.types import Hold

TORSO, SHOULDER_W, HIP_W = 100.0, 70.0, 45.0
UPPER_ARM, FOREARM, THIGH, SHANK = 55.0, 50.0, 80.0, 75.0
HAND = 0.35 * FOREARM  # geometry.hand_point extrapolates this far past the wrist


def _ik(root, target, l1, l2, bend):
    """Two-bone IK. Returns (mid joint, end joint); the end is clamped to reach."""
    root, target = np.asarray(root, float), np.asarray(target, float)
    d = target - root
    dist = np.linalg.norm(d)
    reach = min(dist, 0.999 * (l1 + l2))
    reach = max(reach, abs(l1 - l2) + 1e-3)
    u = d / max(dist, 1e-9)
    end = root + u * reach
    a = (l1 ** 2 - l2 ** 2 + reach ** 2) / (2 * reach)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    perp = np.array([-u[1], u[0]])
    return root + u * a + bend * h * perp, end


def make_pose(hip, lh, rh, lf, rf):
    """Climber seen from behind. lh/rh are wrist targets, lf/rf ankle targets."""
    hip = np.asarray(hip, float)
    kp = np.zeros((33, 2))
    kp[23], kp[24] = hip + [-HIP_W / 2, 0], hip + [HIP_W / 2, 0]
    kp[11], kp[12] = hip + [-SHOULDER_W / 2, -TORSO], hip + [SHOULDER_W / 2, -TORSO]
    kp[13], kp[15] = _ik(kp[11], lh, UPPER_ARM, FOREARM, bend=-1)  # elbows point outwards
    kp[14], kp[16] = _ik(kp[12], rh, UPPER_ARM, FOREARM, bend=+1)
    kp[25], kp[27] = _ik(kp[23], lf, THIGH, SHANK, bend=+1)  # knees point outwards
    kp[26], kp[28] = _ik(kp[24], rf, THIGH, SHANK, bend=-1)
    kp[0:11] = hip + [0, -TORSO - 35]
    for wrist, elbow, pts in ((15, 13, (17, 19, 21)), (16, 14, (18, 20, 22))):
        kp[list(pts)] = kp[wrist] + 0.3 * (kp[wrist] - kp[elbow])
    kp[29], kp[31] = kp[27] + [0, 8], kp[27] + [-6, 6]
    kp[30], kp[32] = kp[28] + [0, 8], kp[28] + [6, 6]
    return kp


def blob(centre, rx=22, ry=22, n=24):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([centre[0] + rx * np.cos(ang), centre[1] + ry * np.sin(ang)], axis=1)


def make_hold(hold_id, centre, rx=22, ry=22, direction=(0, 1), confidence=0.3):
    poly = blob(centre, rx, ry)
    return Hold(id=hold_id, polygon=poly, centroid=np.asarray(centre, float),
                area=float(np.pi * rx * ry), direction=np.asarray(direction, float),
                confidence=confidence)


def hold_under(kp, limb):
    """Where to put a hold so that limb (0..3) of this pose is on it."""
    from boulderai.analysis.geometry import limb_points
    return limb_points(kp)[limb]


def static_climb(kp, holds=(), seconds=2.0, fps=30.0, cfg=None):
    """A climb where the climber holds one pose."""
    n = int(seconds * fps)
    return sequence_climb([kp] * n, holds, fps, cfg)


def sequence_climb(poses, holds=(), fps=30.0, cfg=None):
    poses = np.asarray(poses, float)
    t = np.arange(len(poses)) / fps
    return Climb.build(t, poses, np.ones(len(poses), bool), list(holds), cfg or load_config(), fps=fps)


def lerp_poses(a, b, n):
    return [a + (b - a) * i / max(n - 1, 1) for i in range(n)]


def draw_wall(size=(960, 640), holds=(), colour=(0, 220, 255)):
    """Grey wall picture with filled hold blobs (default colour: yellow in BGR)."""
    h, w = size
    img = np.full((h, w, 3), 150, np.uint8)
    for hold in holds:
        cv2.fillPoly(img, [hold.polygon.astype(np.int32)], colour)
    return img


class FakeEstimator:
    """Stands in for the pose model: replays known keypoints frame by frame."""

    def __init__(self, poses):
        self.poses = poses
        self.i = 0

    def estimate(self, frame, t):
        kp = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        return kp.copy(), np.ones(33)

    def close(self):
        pass


# ---------------------------------------------------------------- ready-made scenarios

HANDS = ((255, 176), (345, 176))
FEET = ((270, 480), (330, 480))


def good_pose():
    return make_pose((300, 380), *HANDS, *FEET)


def bent_pose():
    return make_pose((300, 340), *HANDS, *FEET)


def holds_for(kp, overrides=None):
    """One round hold under each limb of the pose; overrides: hold id -> Hold kwargs."""
    overrides = overrides or {}
    out = []
    for limb in range(4):
        out.append(make_hold(limb, hold_under(kp, limb), **overrides.get(limb, {})))
    return out


def scenario_bent_arms():
    kp = bent_pose()
    return static_climb(kp, holds_for(kp))


def scenario_off_balance(extra_foothold=False):
    kp = make_pose((400, 380), (300, 210), (350, 200), (310, 500), (340, 500))
    holds = holds_for(kp)
    if extra_foothold:
        holds.append(make_hold(4, (430, 500)))
    return static_climb(kp, holds)


def scenario_wrong_pull(direction=(0.866, 0.5)):
    kp = good_pose()
    holds = holds_for(kp, {0: dict(rx=10, ry=30, direction=direction, confidence=0.9)})
    return static_climb(kp, holds)


def scenario_feet_cut(fps=30.0):
    on = good_pose()
    off = make_pose((300, 380), *HANDS, (330, 440), (400, 430))
    poses = [on] * 30 + lerp_poses(on, off, 15) + lerp_poses(off, on, 15) + [on] * 75
    return sequence_climb(poses, holds_for(on), fps)


def scenario_hips_out(fps=30.0):
    normal = good_pose()
    sagging = normal.copy()
    for j in (11, 12, 13, 14):  # shoulders and elbows drop: torso looks 65% as long
        sagging[j] = sagging[j] + [0, 35]
    sagging[0:11] += [0, 35]
    return sequence_climb([normal] * 90 + [sagging] * 45, holds_for(normal), fps)


def scenario_swing(fps=30.0):
    poses = []
    for i in range(90):
        x = 300 + 25 * np.sin(2 * np.pi * 1.2 * i / fps)
        poses.append(make_pose((x, 370), *HANDS, *FEET))
    return sequence_climb(poses, holds_for(good_pose()), fps)


def scenario_foot_readjust(fps=30.0):
    on = good_pose()
    up = make_pose((300, 380), *HANDS, (270, 430), FEET[1])
    hop = lerp_poses(on, up, 6) + lerp_poses(up, on, 6)
    poses = [on] * 45 + hop + [on] * 12 + hop + [on] * 12 + hop + [on] * 30
    return sequence_climb(poses, holds_for(on), fps)
