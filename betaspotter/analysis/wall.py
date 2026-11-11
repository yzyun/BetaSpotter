"""The wall as seen from the side: a straight line in the image, and how steep it is."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .contacts import episodes


class WallError(ValueError):
    pass


@dataclass
class Wall:
    p0: np.ndarray  # a point on the wall surface (pixels)
    n: np.ndarray  # unit normal pointing from the wall towards the climber
    n_points: int = 0  # contact points the line was fitted through
    rms: float = 0.0  # scatter of those points about the line, torso lengths
    source: str = "estimated"  # or "given" when the angle came from the user

    @property
    def angle_deg(self) -> float:
        """Degrees past vertical: positive = overhang, 0 = vertical, negative = slab."""
        return float(np.degrees(np.arctan2(self.n[1], abs(self.n[0]))))

    @property
    def up(self) -> np.ndarray:
        """Unit vector along the wall surface, pointing up the wall."""
        u = np.array([self.n[1], -self.n[0]])
        return u if u[1] <= 0 else -u

    def distance(self, pts) -> np.ndarray:
        """Distance out from the wall surface (pixels); negative is inside the wall."""
        return (np.asarray(pts, dtype=float) - self.p0) @ self.n

    def foot(self, pt) -> np.ndarray:
        """The point on the wall closest to pt."""
        pt = np.asarray(pt, dtype=float)
        return pt - self.distance(pt) * self.n

    def describe(self) -> str:
        a = self.angle_deg
        if a > 3:
            return f"{a:.0f} deg overhang"
        if a < -3:
            return f"{-a:.0f} deg slab"
        return "vertical"


def normal_for(angle_deg: float, side: int) -> np.ndarray:
    """Wall normal for an overhang angle, with the climber on the +x (1) or -x (-1) side."""
    a = np.radians(angle_deg)
    return np.array([side * np.cos(a), np.sin(a)])


def contact_points(climb) -> np.ndarray:
    """One point per hand/foot placement made while on the wall."""
    pts = []
    for limb in range(4):
        for s, e, _ in episodes(climb.contacts, limb):
            keep = climb.on_wall[s:e]
            if keep.any():
                pts.append(np.median(climb.limb_pts[s:e, limb][keep], axis=0))
    return np.array(pts).reshape(-1, 2)


def estimate_wall(climb, angle_deg: float | None = None) -> Wall:
    """Fit the wall line through where the hands and feet were placed.

    With angle_deg the steepness is taken as given and only the wall's position (and
    which side the climber is on) comes from the footage.
    """
    cfg = climb.cfg["side"]
    pts = contact_points(climb)
    usable = climb.valid & climb.on_wall
    if len(pts) == 0 or not usable.any():
        raise WallError("Could not see any hand or foot placements, so the wall cannot be "
                        "located. Check the climber is filmed side-on with the whole body in frame.")
    scale = float(np.median(climb.scale[usable]))
    hips = G.hip_mid(climb.kp[usable])

    if angle_deg is not None:
        if not -45.0 <= angle_deg <= 90.0:
            raise WallError("--wall-angle must be between -45 (slab) and 90 (roof) degrees")
        c = np.median(pts, axis=0)
        n = max((normal_for(angle_deg, side) for side in (1, -1)),
                key=lambda n: float(np.median((hips - c) @ n)))
        r = (pts - c) @ n
        off = float(np.median(r))
        return Wall(p0=c + off * n, n=n, n_points=len(pts),
                    rms=float(np.sqrt(np.mean((r - off) ** 2))) / scale, source="given")

    if len(pts) < 3:
        raise WallError("Too few hand and foot placements to work out the wall angle. "
                        "Pass it with --wall-angle (degrees past vertical).")
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(4):  # refit, dropping placements that sit well off the line (volumes, flagging)
        c = pts[keep].mean(axis=0)
        d = np.linalg.svd(pts[keep] - c)[2][0]
        n = np.array([-d[1], d[0]])
        r = (pts - c) @ n
        med = float(np.median(r[keep]))
        mad = 1.4826 * float(np.median(np.abs(r[keep] - med)))
        new = np.abs(r - med) <= max(2.5 * mad, 0.15 * scale)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    if np.ptp((pts[keep] - c) @ d) < cfg["wall_min_span"] * scale:
        raise WallError("The hand and foot placements cover too little of the wall to work out "
                        "its angle. Pass it with --wall-angle (degrees past vertical).")
    if np.median((hips - c) @ n) < 0:
        n = -n
    r = (pts[keep] - c) @ n
    return Wall(p0=c + float(np.median(r)) * n, n=n, n_points=int(keep.sum()),
                rms=float(np.sqrt(np.mean((r - np.median(r)) ** 2))) / scale)
