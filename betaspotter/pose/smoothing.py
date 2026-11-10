"""Gap filling and One-Euro smoothing of keypoint tracks."""
from __future__ import annotations

import numpy as np

from .landmarks import MAIN_JOINTS


def fill_gaps(kp: np.ndarray, vis: np.ndarray, fps: float, min_vis: float, max_gap_s: float):
    """Interpolate short stretches where a joint was not seen.

    Returns (kp, valid) where valid[T] says whether the frame has a usable pose.
    """
    kp = kp.copy()
    T = len(kp)
    max_gap = max(1, int(round(max_gap_s * fps)))
    frame_ok = vis[:, MAIN_JOINTS].mean(axis=1) >= min_vis
    seen = (vis >= min_vis * 0.5) & frame_ok[:, None]  # [T, 33]
    filled_frame = frame_ok.copy()
    idx = np.arange(T)
    for j in range(kp.shape[1]):
        good = seen[:, j]
        if good.all() or not good.any():
            continue
        good_idx = idx[good]
        for d in range(2):
            interp = np.interp(idx, good_idx, kp[good, j, d])
            # only fill gaps that are short and have data on both sides
            prev_good = np.maximum.accumulate(np.where(good, idx, -10**9))
            next_good = np.minimum.accumulate(np.where(good, idx, 10**9)[::-1])[::-1]
            short = (~good) & (next_good - prev_good - 1 <= max_gap)
            kp[short, j, d] = interp[short]
    # a frame with no pose at all becomes valid if it sits in a short gap
    prev_ok = np.maximum.accumulate(np.where(frame_ok, idx, -10**9))
    next_ok = np.minimum.accumulate(np.where(frame_ok, idx, 10**9)[::-1])[::-1]
    filled_frame |= (~frame_ok) & (next_ok - prev_ok - 1 <= max_gap)
    return kp, filled_frame


def one_euro(x: np.ndarray, fps: float, min_cutoff: float = 1.7, beta: float = 0.01,
             d_cutoff: float = 1.0) -> np.ndarray:
    """One-Euro filter along axis 0, applied independently to every other element."""

    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    out = np.asarray(x, dtype=np.float64).copy()
    dx = np.zeros_like(out[0])
    a_d = alpha(d_cutoff)
    for i in range(1, len(out)):
        d = (out[i] - out[i - 1]) * fps
        dx = a_d * d + (1 - a_d) * dx
        a = alpha(min_cutoff + beta * np.abs(dx))
        out[i] = a * out[i] + (1 - a) * out[i - 1]
    return out


def smooth_track(kp: np.ndarray, vis: np.ndarray, fps: float, cfg: dict):
    """Full clean-up: fill gaps, then smooth each valid stretch separately."""
    pc = cfg["pose"]
    kp, valid = fill_gaps(kp, vis, fps, pc["min_visibility"], pc["max_gap_s"])
    out = kp.copy()
    T = len(kp)
    i = 0
    while i < T:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < T and valid[j]:
            j += 1
        out[i:j] = one_euro(kp[i:j], fps, pc["one_euro_min_cutoff"], pc["one_euro_beta"])
        i = j
    return out, valid
