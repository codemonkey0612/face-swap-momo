"""Final compositing: Poisson / alpha blending with occlusion support."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.image_utils import inward_feather


class FrameCompositor:
    """Composites the swapped face back into the original frame.

    Two blend modes:
        - Poisson (cv2.seamlessClone) — highest quality, ~15 ms per frame
        - Alpha (Gaussian feather)    — fast fallback, ~1 ms

    Occlusion: original pixels are restored wherever the occlusion mask > 0,
    so hands / microphones / cups sit naturally ON TOP of the swapped face.
    """

    def __init__(self, use_poisson: bool = True):
        self._use_poisson = use_poisson

    def composite(
        self,
        original: np.ndarray,
        pasted: np.ndarray,
        face_mask: np.ndarray,
        occlusion_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend swapped face into original frame.

        Args:
            original: Original BGR frame (pre-swap).
            pasted:   Swapped face warped back to frame space (BGR).
            face_mask: float32 [0,1] or uint8 [0,255] face blend mask.
            occlusion_mask: Optional float32 [0,1]; 1=occluder, 0=face.

        Returns:
            Final composited BGR frame.
        """
        # Normalize mask to uint8
        if face_mask.dtype != np.uint8:
            mask_u8 = (np.clip(face_mask, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            mask_u8 = face_mask.copy()

        # Subtract occlusion from mask
        if occlusion_mask is not None and occlusion_mask.max() > 0.05:
            mf = mask_u8.astype(np.float32) / 255.0
            mf = mf * (1.0 - np.clip(occlusion_mask, 0.0, 1.0))
            mask_u8 = (mf * 255).astype(np.uint8)

        if self._use_poisson:
            return self.poisson_blend(original, pasted, mask_u8)
        return self.alpha_blend(original, pasted, mask_u8)

    def poisson_blend(self, original: np.ndarray, pasted: np.ndarray,
                      mask_u8: np.ndarray) -> np.ndarray:
        """Poisson seamless clone. Falls back to alpha on cv2 error."""
        where = np.where(mask_u8 > 0)
        if len(where[0]) == 0:
            return original
        cy = int(np.mean(where[0]))
        cx = int(np.mean(where[1]))
        # Inward-only feathered mask for Poisson
        hard = (mask_u8 > 0).astype(np.uint8) * 255
        feathered = np.minimum(cv2.GaussianBlur(mask_u8, (21, 21), 0), hard)
        try:
            return cv2.seamlessClone(pasted, original, feathered, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            return self.alpha_blend(original, pasted, mask_u8)

    def alpha_blend(self, original: np.ndarray, pasted: np.ndarray,
                    mask_u8: np.ndarray) -> np.ndarray:
        """Fast alpha blend with inward-only Gaussian feathering."""
        where = np.where(mask_u8 > 0)
        if len(where[0]) == 0:
            return original
        sz = max(
            where[0].max() - where[0].min(),
            where[1].max() - where[1].min(),
        )
        bk = max(11, int(sz * 0.35)) | 1
        mf = mask_u8.astype(np.float32) / 255.0
        hard = (mf > 0.01).astype(np.float32)
        alpha = np.minimum(cv2.GaussianBlur(mf, (bk, bk), 0), hard)[:, :, np.newaxis]
        return (
            pasted.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
