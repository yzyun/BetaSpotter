"""The Climb: everything the rules need, derived once from the pose track and holds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from ..pose import landmarks as L
from ..types import Hold
from . import geometry as G
from .contacts import detect_contacts
from .phases import on_wall_mask, static_mask


@dataclass
class Climb:
    fps: float
    t: np.ndarray  # [T] seconds
    kp: np.ndarray  # [T, 33, 2] pixels
    valid: np.ndarray  # [T] usable pose
    holds: list[Hold]
    cfg: dict
    scale: np.ndarray  # [T] local torso length in pixels
    limb_pts: np.ndarray  # [T, 4, 2]
    limb_speed: np.ndarray  # [T, 4] torso lengths / s
    hip_speed: np.ndarray  # [T]
    contacts: np.ndarray  # [T, 4]
    static: np.ndarray  # [T]
    on_wall: np.ndarray  # [T]
    com: np.ndarray  # [T, 2]
    bone_ratio: dict  # bone -> typical length / torso length
    wall: object | None = None  # analysis.wall.Wall, set for side-view footage only

    @classmethod
    def build(cls, t, kp, valid, holds: list[Hold], cfg: dict, fps: float | None = None) -> "Climb":
        t = np.asarray(t, dtype=float)
        kp = np.asarray(kp, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        if fps is None:
            fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
        torso = G.torso_length(kp)
        fallback = float(np.median(torso[valid])) if valid.any() else 1.0
        torso = np.where(valid & (torso > 1e-3), torso, fallback)
        scale = median_filter(torso, size=max(1, int(round(fps)) | 1), mode="nearest")
        scale = np.maximum(scale, 1e-3)

        limb_pts = G.limb_points(kp)
        # Stillness is judged on the wrists and toes: the palm point swings with the
        # forearm even when the hand is firmly on a hold.
        anchors = kp[:, [L.L_WRIST, L.R_WRIST, L.L_TOE, L.R_TOE]]
        limb_speed = G.windowed_speed(anchors, fps) / scale[:, None]
        hip_speed = G.windowed_speed(G.hip_mid(kp), fps) / scale
        contacts = detect_contacts(limb_pts, limb_speed, valid, scale, holds, fps, cfg)
        static = static_mask(limb_speed, hip_speed, valid, fps, cfg)
        on_wall = on_wall_mask(kp, valid, scale, cfg)

        bone_ratio = {}
        for a, b in L.BONES:
            length = np.linalg.norm(kp[:, a] - kp[:, b], axis=-1) / scale
            bone_ratio[(a, b)] = float(np.percentile(length[valid], 80)) if valid.any() else 1.0

        return cls(fps=fps, t=t, kp=kp, valid=valid, holds=holds, cfg=cfg, scale=scale,
                   limb_pts=limb_pts, limb_speed=limb_speed, hip_speed=hip_speed,
                   contacts=contacts, static=static, on_wall=on_wall, com=G.com(kp),
                   bone_ratio=bone_ratio)

    def hold(self, hold_id: int) -> Hold | None:
        for h in self.holds:
            if h.id == hold_id:
                return h
        return None

    def frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def bone_targets(self, frame: int) -> dict:
        """Bone lengths the pose solver should keep at this frame.

        Limbs use the climber's typical length (a limb pointing at the camera looks
        short in 2D); the torso keeps what it measures right now.
        """
        kp, s = self.kp[frame], float(self.scale[frame])
        out = {}
        for a, b in L.LIMB_BONES:
            cur = float(np.linalg.norm(kp[a] - kp[b]))
            out[(a, b)] = max(cur, self.bone_ratio[(a, b)] * s)
        for a, b in L.TORSO_BONES:
            out[(a, b)] = float(np.linalg.norm(kp[a] - kp[b]))
        return out
