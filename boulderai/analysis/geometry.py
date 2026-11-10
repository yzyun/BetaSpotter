"""Shared geometry: joint angles, centre of mass, limb end points, run detection."""
from __future__ import annotations

import cv2
import numpy as np

from ..pose import landmarks as L


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) between b->a and b->c. Works on [..., 2] arrays."""
    u, v = unit(a - b), unit(c - b)
    cos = np.clip((u * v).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = np.clip((unit(u) * unit(v)).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def shoulder_mid(kp):
    return 0.5 * (kp[..., L.L_SHOULDER, :] + kp[..., L.R_SHOULDER, :])


def hip_mid(kp):
    return 0.5 * (kp[..., L.L_HIP, :] + kp[..., L.R_HIP, :])


def torso_length(kp) -> np.ndarray:
    return np.linalg.norm(shoulder_mid(kp) - hip_mid(kp), axis=-1)


def shoulder_width(kp) -> np.ndarray:
    return np.linalg.norm(kp[..., L.L_SHOULDER, :] - kp[..., L.R_SHOULDER, :], axis=-1)


def elbow_angle(kp, side: str) -> np.ndarray:
    s, e, w = L.ARM[side]
    return angle_deg(kp[..., s, :], kp[..., e, :], kp[..., w, :])


def knee_angle(kp, side: str) -> np.ndarray:
    h, k, a, _ = L.LEG[side]
    return angle_deg(kp[..., h, :], kp[..., k, :], kp[..., a, :])


def pull_vector(kp, side: str) -> np.ndarray:
    """Direction the hand loads its hold: from the wrist towards the elbow."""
    _, e, w = L.ARM[side]
    return unit(kp[..., e, :] - kp[..., w, :])


def hand_point(kp, side: str) -> np.ndarray:
    """Centre of the palm, extrapolated along the forearm (steadier than finger landmarks)."""
    _, e, w = L.ARM[side]
    return kp[..., w, :] + 0.35 * (kp[..., w, :] - kp[..., e, :])


def foot_point(kp, side: str) -> np.ndarray:
    return kp[..., L.LEG[side][3], :]


def limb_points(kp) -> np.ndarray:
    """[..., 4, 2]: left hand, right hand, left foot, right foot contact points."""
    return np.stack(
        [hand_point(kp, "left"), hand_point(kp, "right"),
         foot_point(kp, "left"), foot_point(kp, "right")],
        axis=-2,
    )


# Body segment mass fractions (sum to 1): simple model good enough for 2D balance.
_SEGMENTS = [
    (0.08, (L.NOSE, L.NOSE)),
    (0.027, (L.L_SHOULDER, L.L_ELBOW)), (0.027, (L.R_SHOULDER, L.R_ELBOW)),
    (0.023, (L.L_ELBOW, L.L_WRIST)), (0.023, (L.R_ELBOW, L.R_WRIST)),
    (0.10, (L.L_HIP, L.L_KNEE)), (0.10, (L.R_HIP, L.R_KNEE)),
    (0.06, (L.L_KNEE, L.L_ANKLE)), (0.06, (L.R_KNEE, L.R_ANKLE)),
]
_TRUNK = 0.50


def com(kp) -> np.ndarray:
    """Estimated centre of mass, [..., 2]."""
    total = _TRUNK * 0.5 * (shoulder_mid(kp) + hip_mid(kp))
    for mass, (a, b) in _SEGMENTS:
        total = total + mass * 0.5 * (kp[..., a, :] + kp[..., b, :])
    return total


def com_offset(com_x: float, contact_xs, scale: float) -> float:
    """How far (torso lengths) the CoM sits sideways outside the span of the contacts.

    Negative = left of the span, positive = right of it, 0 = inside.
    """
    lo, hi = min(contact_xs), max(contact_xs)
    if com_x < lo:
        return (com_x - lo) / scale
    if com_x > hi:
        return (com_x - hi) / scale
    return 0.0


def poly_distance(polygon: np.ndarray, pt) -> float:
    """Distance from a point to a polygon outline; 0 when inside."""
    d = cv2.pointPolygonTest(polygon.astype(np.float32).reshape(-1, 1, 2),
                             (float(pt[0]), float(pt[1])), True)
    return float(max(0.0, -d))


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Runs of True in a boolean array as (start, end_exclusive), at least min_len long."""
    out = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def windowed_speed(p: np.ndarray, fps: float, window_s: float = 0.2) -> np.ndarray:
    """Speed of points p[T, ..., 2] measured over a short window (robust to jitter)."""
    T = len(p)
    w = max(1, int(round(window_s * fps / 2)))
    idx = np.arange(T)
    a, b = np.clip(idx - w, 0, T - 1), np.clip(idx + w, 0, T - 1)
    dt = np.maximum(b - a, 1) / fps
    dist = np.linalg.norm(p[b] - p[a], axis=-1)
    return dist / dt.reshape((-1,) + (1,) * (dist.ndim - 1))
