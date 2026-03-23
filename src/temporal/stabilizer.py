"""Temporal landmark stabilization via velocity-adaptive EMA."""

from __future__ import annotations

from collections import deque

import numpy as np


class LandmarkStabilizer:
    """Velocity-adaptive exponential moving average for face landmarks.

    Uses heavier smoothing when landmarks move slowly (reduces jitter)
    and lighter smoothing during fast motion (reduces lag).

    Args:
        alpha_slow: EMA weight during slow/still motion (more smoothing).
        alpha_fast: EMA weight during fast motion (less smoothing).
        velocity_threshold: Mean pixel displacement threshold to switch modes.
        window: Frame history for the EMA.
    """

    def __init__(
        self,
        alpha_slow: float = 0.4,
        alpha_fast: float = 0.8,
        velocity_threshold: float = 3.0,
        window: int = 5,
    ):
        self._alpha_slow = alpha_slow
        self._alpha_fast = alpha_fast
        self._vel_thresh = velocity_threshold
        self._window = window
        self._smoothed: list[np.ndarray] = []

    def update(self, landmarks_list: list[np.ndarray]) -> list[np.ndarray]:
        """Smooth a list of landmark arrays (one per face).

        Returns smoothed landmark arrays as int32.
        """
        if not landmarks_list:
            return landmarks_list

        if len(self._smoothed) != len(landmarks_list):
            self._smoothed = [lm.astype(np.float32).copy() for lm in landmarks_list]
            return [lm.copy() for lm in landmarks_list]

        result = []
        for i, lm in enumerate(landmarks_list):
            lm_f = lm.astype(np.float32)
            velocity = np.mean(np.abs(lm_f - self._smoothed[i]))
            t = np.clip(velocity / self._vel_thresh, 0.0, 1.0)
            alpha = self._alpha_slow + t * (self._alpha_fast - self._alpha_slow)
            self._smoothed[i] = alpha * lm_f + (1.0 - alpha) * self._smoothed[i]
            result.append(self._smoothed[i].astype(np.int32))
        return result

    def stabilize_kps(self, kps: np.ndarray) -> np.ndarray:
        """Convenience: stabilize a single 5-point keypoint array."""
        smoothed = self.update([kps])
        return smoothed[0].astype(np.float32)

    def reset(self) -> None:
        self._smoothed = []


class BBoxStabilizer:
    """EMA smoothing for face bounding boxes."""

    def __init__(self, alpha: float = 0.7):
        self._alpha = alpha
        self._smoothed: np.ndarray | None = None

    def update(self, bbox: np.ndarray) -> np.ndarray:
        if self._smoothed is None or self._smoothed.shape != bbox.shape:
            self._smoothed = bbox.astype(np.float32).copy()
            return bbox
        self._smoothed = self._alpha * bbox + (1.0 - self._alpha) * self._smoothed
        return self._smoothed.astype(bbox.dtype)

    def reset(self) -> None:
        self._smoothed = None
