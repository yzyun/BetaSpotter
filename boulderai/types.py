"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
HANDS = (0, 1)
FEET = (2, 3)

NO_CONTACT = -2  # limb is moving / in the air
SMEAR = -1  # limb is still, but not on a detected hold (smear, volume, other colour)


@dataclass
class Hold:
    id: int
    polygon: np.ndarray  # [N, 2] pixel coordinates
    centroid: np.ndarray  # [2]
    area: float
    # Unit vector (image coords, y down) of the direction the hold should be loaded in.
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    # 0..1: how much we trust `direction`. Below 0.5 it is only a default guess.
    confidence: float = 0.3
    kind: str = "unknown"


@dataclass
class Finding:
    rule: str  # slug, e.g. "bent-arms"
    title: str  # e.g. "BENT ARMS"
    t_start: float
    t_end: float
    frame: int  # index into the Climb arrays of the worst frame
    severity: str  # mild | moderate | severe
    limb: str | None  # "left", "right", "both" or None
    measured: str  # what was measured against the threshold
    tip: str  # one-line coaching tip
    evidence: dict = field(default_factory=dict)
    score: float = 0.0  # how bad, used to choose between merged findings


@dataclass
class Suggestion:
    kind: str  # "fix" (same holds) | "beta" (different hold)
    keypoints: np.ndarray  # [33, 2]
    tip: str
    cost: float = 0.0
    limb: int | None = None  # for beta: index into LIMBS of the limb that moves
    hold_id: int | None = None  # for beta: the hold it moves to
