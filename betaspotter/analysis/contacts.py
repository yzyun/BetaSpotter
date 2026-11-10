"""Which limb is on which hold, per frame."""
from __future__ import annotations

import numpy as np

from ..types import NO_CONTACT, SMEAR, Hold
from .geometry import poly_distance, runs


def detect_contacts(limb_pts: np.ndarray, limb_speed: np.ndarray, valid: np.ndarray,
                    scale: np.ndarray, holds: list[Hold], fps: float, cfg: dict) -> np.ndarray:
    """Return contacts[T, 4]: hold id, SMEAR (still but on no known hold) or NO_CONTACT.

    A limb is placed when it stays still for `min_still_s`. It is assigned to the
    nearest hold whose outline is within `hold_tolerance` torso lengths.
    """
    cc = cfg["contacts"]
    T = len(limb_pts)
    contacts = np.full((T, 4), NO_CONTACT, dtype=int)
    min_len = max(1, int(round(cc["min_still_s"] * fps)))
    for limb in range(4):
        still = valid & (limb_speed[:, limb] < cc["still_speed"])
        for s, e in runs(still, min_len):
            pos = np.median(limb_pts[s:e, limb], axis=0)
            tol = cc["hold_tolerance"] * float(np.median(scale[s:e]))
            best, best_d = SMEAR, tol
            for hold in holds:
                d = poly_distance(hold.polygon, pos)
                if d <= best_d:
                    best, best_d = hold.id, d
            contacts[s:e, limb] = best
    return contacts


def episodes(contacts: np.ndarray, limb: int) -> list[tuple[int, int, int]]:
    """Contact episodes of one limb as (start, end_exclusive, hold id)."""
    out = []
    col = contacts[:, limb]
    start = None
    for i in range(len(col) + 1):
        cur = col[i] if i < len(col) else NO_CONTACT
        if start is not None and (cur != col[start]):
            out.append((start, i, int(col[start])))
            start = None
        if start is None and cur != NO_CONTACT:
            start = i
    return out
