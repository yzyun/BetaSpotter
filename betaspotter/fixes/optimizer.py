"""A small 2D pose solver.

Moves the 12 main joints so that a rule-specific objective is met while the
climber's bone lengths are kept, placed hands/feet stay on their holds, and the
pose stays as close as possible to what they were actually doing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..analysis import geometry as G
from ..pose import landmarks as L

# Points that ride along rigidly with a solved joint.
_ATTACHED = {
    L.L_WRIST: [17, 19, 21], L.R_WRIST: [18, 20, 22],
    L.L_ANKLE: [29, 31], L.R_ANKLE: [30, 32],
}
_FACE = list(range(0, 11))

W_BONE, W_PIN, W_PRIOR = 8.0, 8.0, 0.25
W_ELBOW, W_LIMIT, W_DIR, W_COM, W_HANG, W_WALL = 3.0, 3.0, 3.0, 6.0, 1.0, 4.0
MIN_JOINT_DEG = 25.0


@dataclass
class Objective:
    elbow_min: dict = field(default_factory=dict)  # side -> minimum elbow angle (deg)
    forearm_dir: dict = field(default_factory=dict)  # side -> unit vector to pull along
    com_x: tuple | None = None  # (lo, hi) the centre of mass should sit between
    wall: tuple | None = None  # side view: (point on the wall, normal towards the climber)
    hip_max: float | None = None  # side view: furthest the hips may be from the wall (pixels)


def bones_from_pose(kp: np.ndarray) -> dict:
    return {(a, b): float(np.linalg.norm(kp[a] - kp[b])) for a, b in L.BONES}


def _unpack(x: np.ndarray, kp0: np.ndarray) -> np.ndarray:
    kp = kp0.copy()
    kp[L.MAIN_JOINTS] = x.reshape(-1, 2)
    for anchor, pts in _ATTACHED.items():
        kp[pts] += kp[anchor] - kp0[anchor]
    kp[_FACE] += G.shoulder_mid(kp) - G.shoulder_mid(kp0)
    return kp


def _hinge_below(value: float, minimum: float) -> float:
    return max(0.0, minimum - value)


def solve_pose(kp0: np.ndarray, pins: dict, bones: dict, objective: Objective,
               scale: float) -> np.ndarray:
    """Return corrected keypoints [33, 2].

    pins: joint index -> [x, y] it must stay at (hands and feet on holds).
    bones: (a, b) -> length to keep.
    """
    kp0 = np.asarray(kp0, dtype=float)
    x0 = kp0[L.MAIN_JOINTS].ravel()

    # Where a hanging shoulder should end up: an arm's length from the hand, mostly below it.
    hang_targets = {}
    for side in objective.elbow_min:
        s, e, w = L.ARM[side]
        wrist = np.asarray(pins.get(w, kp0[w]), dtype=float)
        direction = G.unit(G.unit(kp0[s] - wrist) + np.array([0.0, 0.5]))
        reach = 0.98 * (bones[(s, e)] + bones[(e, w)])
        hang_targets[side] = wrist + reach * direction

    def residuals(x):
        kp = _unpack(x, kp0)
        res = [W_PRIOR * (x - x0) / scale]
        res.append(np.array([W_BONE * (np.linalg.norm(kp[a] - kp[b]) / max(length, 1e-6) - 1.0)
                             for (a, b), length in bones.items()]))
        for joint, target in pins.items():
            res.append(W_PIN * (kp[joint] - np.asarray(target, dtype=float)) / scale)
        limits = []
        for side in ("left", "right"):
            limits.append(W_LIMIT * _hinge_below(float(G.elbow_angle(kp, side)), MIN_JOINT_DEG) / 30.0)
            limits.append(W_LIMIT * _hinge_below(float(G.knee_angle(kp, side)), MIN_JOINT_DEG) / 30.0)
        res.append(np.array(limits))
        for side, minimum in objective.elbow_min.items():
            ang = float(G.elbow_angle(kp, side))
            res.append(np.array([W_ELBOW * _hinge_below(ang, minimum) / 30.0]))
            res.append(W_HANG * (kp[L.ARM[side][0]] - hang_targets[side]) / scale)
        for side, direction in objective.forearm_dir.items():
            s, e, w = L.ARM[side]
            d = np.asarray(direction, dtype=float)
            res.append(W_DIR * (G.unit(kp[e] - kp[w]) - d))
            res.append(0.5 * W_DIR * (G.unit(kp[s] - kp[w]) - d))
        if objective.com_x is not None:
            lo, hi = objective.com_x
            cx = float(G.com(kp)[0])
            res.append(np.array([W_COM * (max(0.0, lo - cx) + max(0.0, cx - hi)) / scale]))
        if objective.wall is not None:
            p0, n = objective.wall
            free = [j for j in L.MAIN_JOINTS if j not in pins]
            res.append(W_WALL * np.minimum((kp[free] - p0) @ n, 0.0) / scale)  # stay out of the wall
            if objective.hip_max is not None:
                out = float((G.hip_mid(kp) - p0) @ n)
                res.append(np.array([W_COM * max(0.0, out - objective.hip_max) / scale]))
        return np.concatenate([np.atleast_1d(r).ravel() for r in res])

    sol = least_squares(residuals, x0, x_scale=max(scale, 1e-3), max_nfev=300)
    return _unpack(sol.x, kp0)
