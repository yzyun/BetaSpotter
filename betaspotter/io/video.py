"""Video reading helpers (OpenCV)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


def video_info(path: str | Path) -> tuple[float, int]:
    """Return (fps, approximate frame count)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not (1.0 <= fps <= 480.0):
        fps = 30.0
    return float(fps), count


def read_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index in the file, BGR frame) for every `stride`-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, frame
            i += 1
    finally:
        cap.release()


def grab_frames(path: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Read specific frames by file index (sequential read: seeking is unreliable)."""
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    for i, frame in read_frames(path):
        if i in wanted:
            out[i] = frame
        if i >= last:
            break
    return out
