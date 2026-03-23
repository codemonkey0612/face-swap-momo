"""Mask post-processing: morphological cleanup, feathering, temporal smoothing."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.image_utils import inward_feather


class MaskRefiner:
    """Post-processes raw segmentation masks for clean, stable blending.

    Pipeline per frame:
        1. Morphological closing  — fill small holes
        2. Morphological opening  — remove noise
        3. Erosion (optional)     — shrink mask inward to prevent boundary artifacts
        4. Inward-only feathering — soft alpha at edges without expansion
        5. Temporal EMA           — smooth mask flicker between frames
    """

    def __init__(
        self,
        erosion_px: int = 3,
        feather_kernel: int = 31,
        feather_sigma: float = 5.0,
        temporal_alpha: float = 0.8,
    ):
        self._erosion_px = erosion_px
        self._feather_kernel = feather_kernel
        self._feather_sigma = feather_sigma
        self._temporal_alpha = temporal_alpha
        self._prev_mask: np.ndarray | None = None

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """Full refinement pipeline. Input/output: float32 [0, 1]."""
        m = np.clip(mask, 0.0, 1.0)

        # Morphological closing: fill small holes
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m8 = (m * 255).astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k5)

        # Morphological opening: remove noise
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k3)

        m = m8.astype(np.float32) / 255.0

        # Optional erosion (shrink inward before feathering)
        if self._erosion_px > 0:
            e = self._erosion_px
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))
            m = cv2.erode((m * 255).astype(np.uint8), ke).astype(np.float32) / 255.0

        # Inward-only feathering
        m = inward_feather(m, self._feather_kernel, self._feather_sigma)

        return np.clip(m, 0.0, 1.0)

    def temporal_smooth(self, mask: np.ndarray) -> np.ndarray:
        """EMA blend with previous mask to reduce flicker."""
        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            self._prev_mask = mask.copy()
            return mask
        smoothed = self._temporal_alpha * mask + (1.0 - self._temporal_alpha) * self._prev_mask
        self._prev_mask = smoothed.copy()
        return smoothed

    def refine_and_smooth(self, mask: np.ndarray) -> np.ndarray:
        """Refine then temporally smooth — the standard call path."""
        return self.temporal_smooth(self.refine(mask))

    def reset(self) -> None:
        """Clear temporal buffer (call on scene cut or source face change)."""
        self._prev_mask = None
