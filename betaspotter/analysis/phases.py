"""Split the climb into static frames (everything placed) and moves; find the floor."""
from __future__ import annotations

import numpy as np

from ..pose import landmarks as L
from .geometry import runs


def static_mask(limb_speed: np.ndarray, hip_speed: np.ndarray, valid: np.ndarray,
                fps: float, cfg: dict) -> np.ndarray:
    pc = cfg["phases"]
    still = valid & (limb_speed.max(axis=1) < pc["static_speed"]) & (hip_speed < pc["static_speed"])
    out = np.zeros(len(still), dtype=bool)
    for s, e in runs(still, max(1, int(round(pc["min_static_s"] * fps)))):
        out[s:e] = True
    return out


def on_wall_mask(kp: np.ndarray, valid: np.ndarray, scale: np.ndarray, cfg: dict) -> np.ndarray:
    """False while the climber stands on the floor (so starting poses are not judged).

    The floor is taken as the lowest the feet ever get. If the climber never gains
    real height in the clip (e.g. it starts on the wall) every frame counts.
    """
    pc = cfg["phases"]
    if not pc["ignore_ground"] or not valid.any():
        return valid.copy()
    ankle_y = kp[:, [L.L_ANKLE, L.R_ANKLE], 1]
    lowest = ankle_y.max(axis=1)  # the lower foot (y grows downwards)
    floor = np.percentile(lowest[valid], 98)
    med_scale = float(np.median(scale[valid]))
    if floor - np.percentile(lowest[valid], 2) < 1.0 * med_scale:
        return valid.copy()
    return valid & (lowest < floor - pc["ground_margin"] * med_scale)
