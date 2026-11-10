"""Guess each hold's usable pull direction from its outline; read/write holds.yaml.

This is a heuristic. A camera cannot see which side of a hold is positive, so the
guess is written to holds.yaml where the climber can correct it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from ..types import Hold

ELONGATED = 1.4  # long axis / short axis above which the shape tells us something


def estimate_orientations(holds: list[Hold]) -> None:
    """Fill in direction / confidence / kind on each hold in place.

    - Roundish hold: assume it is pulled straight down (low confidence).
    - Long hold lying flat or slanted: an edge/rail, pulled down, square to its long axis.
    - Long hold standing upright: a sidepull, pulled sideways towards the line of the route.
    """
    if not holds:
        return
    centre_x = float(np.median([h.centroid[0] for h in holds]))
    for hold in holds:
        m = cv2.moments(hold.polygon.astype(np.float32).reshape(-1, 1, 2))
        mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
        common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
        lam1, lam2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
        elong = float(np.sqrt(lam1 / lam2)) if lam2 > 1e-9 else 1.0
        if elong < ELONGATED:
            hold.direction, hold.confidence, hold.kind = np.array([0.0, 1.0]), 0.3, "round"
            continue
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)  # long axis angle
        normal = np.array([-np.sin(theta), np.cos(theta)])
        if abs(normal[1]) >= 0.5:  # long axis within 60 degrees of horizontal
            if normal[1] < 0:
                normal = -normal
            hold.kind = "edge"
        else:
            towards_centre = 1.0 if centre_x >= hold.centroid[0] else -1.0
            if normal[0] * towards_centre < 0:
                normal = -normal
            normal = normal + np.array([0.0, 0.3])  # sidepulls are loaded slightly downwards too
            hold.kind = "sidepull"
        hold.direction = normal / np.linalg.norm(normal)
        hold.confidence = float(min(0.9, 0.5 + 0.2 * (elong - ELONGATED)))


def direction_to_deg(direction: np.ndarray) -> float:
    """0 = pull right, 90 = pull down, 180 = pull left, 270 = pull up (as seen in the video)."""
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 360)


def deg_to_direction(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


def save_holds(path: str | Path, holds: list[Hold], colour: str) -> None:
    data = {
        "colour": colour,
        "help": "pull_direction_deg: 0=right, 90=down, 180=left, 270=up as seen in the video. "
                "Set confident: true after correcting a direction. Delete entries that are not holds.",
        "holds": [
            {
                "id": h.id,
                "kind": h.kind,
                "pull_direction_deg": round(direction_to_deg(h.direction), 1),
                "confident": bool(h.confidence >= 0.5),
                "centroid": [round(float(v), 1) for v in h.centroid],
                "polygon": [[int(round(x)), int(round(y))] for x, y in h.polygon],
            }
            for h in holds
        ],
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=10000))


def load_holds(path: str | Path) -> tuple[list[Hold], str]:
    data = yaml.safe_load(Path(path).read_text())
    holds = []
    for item in data.get("holds", []):
        poly = np.array(item["polygon"], dtype=np.float64)
        holds.append(Hold(
            id=int(item["id"]), polygon=poly,
            centroid=np.array(item.get("centroid", poly.mean(axis=0)), dtype=np.float64),
            area=float(cv2.contourArea(poly.astype(np.float32))),
            direction=deg_to_direction(float(item.get("pull_direction_deg", 90.0))),
            confidence=0.9 if item.get("confident", False) else 0.3,
            kind=str(item.get("kind", "unknown")),
        ))
    return holds, str(data.get("colour", "yellow"))
