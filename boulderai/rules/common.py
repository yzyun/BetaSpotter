from __future__ import annotations

import numpy as np

from ..types import NO_CONTACT


def severity(excess: float, moderate: float, severe: float) -> str:
    if excess >= severe:
        return "severe"
    if excess >= moderate:
        return "moderate"
    return "mild"


def in_contact(climb) -> np.ndarray:
    """[T, 4] bool: limb is placed on something."""
    return climb.contacts != NO_CONTACT
