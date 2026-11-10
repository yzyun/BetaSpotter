"""MediaPipe Pose Landmarker wrapper (CPU)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .landmarks import N_LANDMARKS

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
CACHE_DIR = Path.home() / ".cache" / "boulderai"


def model_path(variant: str = "full") -> Path:
    """Return the local model file, downloading it on first use."""
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown pose model '{variant}', choose from {sorted(MODEL_URLS)}")
    path = CACHE_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.rename(path)
    return path


class PoseEstimator:
    """Estimates one person's pose per video frame.

    `estimate` returns (keypoints[33, 2] in pixels, visibility[33]); visibility is
    all zeros when nobody was found.
    """

    def __init__(self, variant: str = "full", model: str | Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model or model_path(variant))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def estimate(self, frame_bgr: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(round(t * 1000)), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        kp = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
        vis = np.zeros(N_LANDMARKS, dtype=np.float64)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks[0][:N_LANDMARKS]):
                kp[i] = (lm.x * w, lm.y * h)
                vis[i] = lm.visibility if lm.visibility is not None else 1.0
        return kp, vis

    def close(self) -> None:
        self._landmarker.close()
