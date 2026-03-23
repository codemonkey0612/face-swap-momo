"""Color transfer: Lab-space mean+std matching for seamless skin tone blending."""

from __future__ import annotations

import cv2
import numpy as np


def lab_color_transfer(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    blend: float = 0.7,
) -> np.ndarray:
    """Reinhard Lab color transfer within the masked region.

    Adjusts src's color distribution to match dst's in each Lab channel
    (mean + std). A blend factor < 1.0 softens the correction.

    Args:
        src: Source BGR image (face to color-correct).
        dst: Destination BGR image (target lighting/skin tone).
        mask: float32 [0,1] mask — only these pixels inform the statistics.
        blend: How much correction to apply (0=none, 1=full). Default 0.7.

    Returns:
        Color-corrected BGR image (same shape as src).
    """
    mb = mask > 0.3
    if not mb.any():
        return src

    sl = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float32)
    dl = cv2.cvtColor(dst, cv2.COLOR_BGR2Lab).astype(np.float32)
    out = sl.copy()

    for c in range(3):
        sv = sl[:, :, c][mb]
        dv = dl[:, :, c][mb]
        sm, ss = sv.mean(), max(sv.std(), 1e-6)
        dm, ds = dv.mean(), dv.std()
        corrected = (out[:, :, c] - sm) * (ds / ss) + dm
        out[:, :, c] = blend * corrected + (1.0 - blend) * out[:, :, c]

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
