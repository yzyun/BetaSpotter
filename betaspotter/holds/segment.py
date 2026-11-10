"""Find the holds of one problem by colour (HSV masking)."""
from __future__ import annotations

import cv2
import numpy as np

from ..types import Hold

# OpenCV HSV: H 0..179, S 0..255, V 0..255. Each colour is a list of (low, high) ranges.
HSV_PRESETS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 110, 70), (7, 255, 255)), ((172, 110, 70), (179, 255, 255))],
    "orange": [((8, 120, 90), (19, 255, 255))],
    "yellow": [((20, 90, 100), (34, 255, 255))],
    "green": [((38, 70, 50), (85, 255, 255))],
    "blue": [((95, 90, 50), (128, 255, 255))],
    "purple": [((129, 50, 40), (152, 255, 255))],
    "pink": [((153, 60, 90), (171, 255, 255))],
    "black": [((0, 0, 0), (179, 255, 45))],
    "white": [((0, 0, 205), (179, 40, 255))],
}

# Colour used to outline the holds in screenshots (BGR).
DISPLAY_BGR = {
    "red": (60, 60, 255), "orange": (0, 140, 255), "yellow": (0, 230, 255),
    "green": (80, 220, 80), "blue": (255, 140, 40), "purple": (220, 80, 180),
    "pink": (200, 130, 255), "black": (60, 60, 60), "white": (245, 245, 245),
}


def parse_hsv(text: str):
    """Parse 'h1,s1,v1,h2,s2,v2' into a single HSV range."""
    vals = [int(v) for v in text.split(",")]
    if len(vals) != 6:
        raise ValueError("--hsv needs six numbers: h_lo,s_lo,v_lo,h_hi,s_hi,v_hi")
    return [(tuple(vals[:3]), tuple(vals[3:]))]


def build_background(frames: list[np.ndarray], boxes: list[tuple[int, int, int, int] | None]) -> np.ndarray:
    """Median of sampled frames with the climber's bounding box left out of each.

    This gives a picture of the wall without the climber covering holds.
    """
    stack = np.stack(frames).astype(np.float32)
    for i, box in enumerate(boxes):
        if box is not None:
            x0, y0, x1, y1 = box
            stack[i, max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bg = np.nanmedian(stack, axis=0)
    hidden = np.isnan(bg)
    if hidden.any():  # pixels the climber covered in every sample
        bg[hidden] = np.median(np.stack(frames), axis=0)[hidden]
    return bg.astype(np.uint8)


def colour_mask(image_bgr: np.ndarray, ranges) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    return mask


def segment_holds(image_bgr: np.ndarray, ranges, min_area_frac: float = 0.00012,
                  max_area_frac: float = 0.03) -> list[Hold]:
    """Return holds matching the colour ranges, numbered top to bottom."""
    h, w = image_bgr.shape[:2]
    mask = colour_mask(image_bgr, ranges)
    k = max(3, int(round(min(h, w) / 200)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area_frac * h * w <= area <= max_area_frac * h * w):
            continue
        m = cv2.moments(c)
        centroid = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        found.append((centroid, c.reshape(-1, 2).astype(np.float64), area))
    found.sort(key=lambda f: (f[0][1], f[0][0]))
    return [Hold(id=i, polygon=poly, centroid=cen, area=float(area))
            for i, (cen, poly, area) in enumerate(found)]
