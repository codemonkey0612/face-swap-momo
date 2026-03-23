"""Temporal frame blending — reduces frame-to-frame flicker in the face region."""

from __future__ import annotations

import cv2
import numpy as np


class FrameBlender:
    """EMA blend of consecutive output frames to reduce flicker.

    Only blends within the face mask region — background stays sharp.

    Args:
        alpha: Weight for current frame (0.85 = strong stability, low ghosting).
    """

    def __init__(self, alpha: float = 0.85):
        self._alpha = alpha
        self._prev: np.ndarray | None = None

    def blend(self, current: np.ndarray, face_mask: np.ndarray | None = None) -> np.ndarray:
        """Blend current frame with previous output.

        Args:
            current: Current BGR frame.
            face_mask: Optional float32 [0,1] mask; if None, blend entire frame.

        Returns:
            Blended frame.
        """
        if self._prev is None or self._prev.shape != current.shape:
            self._prev = current.copy()
            return current

        blended = cv2.addWeighted(current, self._alpha, self._prev, 1.0 - self._alpha, 0)

        if face_mask is not None:
            # Only blend in face region; keep background from current frame
            alpha = np.clip(face_mask, 0.0, 1.0)[:, :, np.newaxis]
            result = (blended.astype(np.float32) * alpha
                      + current.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        else:
            result = blended

        self._prev = result.copy()
        return result

    def reset(self) -> None:
        """Clear buffer — call on scene cut or source face change."""
        self._prev = None

    @staticmethod
    def detect_scene_cut(current: np.ndarray, previous: np.ndarray,
                         threshold: float = 30.0) -> bool:
        """Return True if mean absolute difference exceeds threshold."""
        if previous is None or previous.shape != current.shape:
            return False
        diff = np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32)))
        return float(diff) > threshold
