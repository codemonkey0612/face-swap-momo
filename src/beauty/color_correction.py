"""Colour correction: Reinhard skin-tone matching, auto white balance, CLAHE.

Ensures the swapped face doesn't look colour-shifted or pasted-on by
matching skin tone statistics to the original frame's lighting conditions.
"""

from __future__ import annotations

import cv2
import numpy as np


class ColorCorrector:
    """Colour correction suite for swapped faces.

    Methods:
        match_skin_tone  — Reinhard Lab transfer from swap to original lighting
        auto_white_balance — illuminant estimation + cast correction
        enhance_contrast   — CLAHE on face region

    Args:
        config: dict with optional key:
            strength (float): 0.0-1.0 blend strength (default 1.0)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._strength = float(cfg.get("strength", 1.0))

    # ------------------------------------------------------------------
    # Reinhard colour transfer (Reinhard et al., 2001)
    # ------------------------------------------------------------------

    def match_skin_tone(
        self,
        swapped_frame: np.ndarray,
        original_frame: np.ndarray,
        face_mask: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """Match swapped face skin tone to the original frame's lighting.

        Computes mean/std of skin pixels in LAB for both frames,
        then applies per-channel Reinhard transfer within the face mask.

        Args:
            swapped_frame:  BGR frame after face swap.
            original_frame: BGR frame before swap (real webcam).
            face_mask:      float32 (H, W) — skin region to correct.
            strength:       Override blend strength if provided.

        Returns:
            Colour-corrected BGR frame.
        """
        s = strength if strength is not None else self._strength
        if s <= 0 or face_mask.max() < 0.01:
            return swapped_frame

        lab_swap = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_orig = cv2.cvtColor(original_frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        mask_bool = face_mask > 0.5

        if mask_bool.sum() < 100:
            return swapped_frame

        # Source statistics (swapped face)
        src_pixels = lab_swap[mask_bool]
        src_mean = src_pixels.mean(axis=0)
        src_std  = src_pixels.std(axis=0) + 1e-6

        # Target statistics (original frame — real lighting)
        tgt_pixels = lab_orig[mask_bool]
        tgt_mean = tgt_pixels.mean(axis=0)
        tgt_std  = tgt_pixels.std(axis=0) + 1e-6

        # Reinhard transfer
        lab_corrected = lab_swap.copy()
        for ch in range(3):
            lab_corrected[:, :, ch] = (
                (lab_swap[:, :, ch] - src_mean[ch]) * (tgt_std[ch] / src_std[ch])
                + tgt_mean[ch]
            )

        lab_out = cv2.merge([
            np.clip(lab_corrected[:, :, c], 0 if c > 0 else 0, 255 if c == 0 else 127).astype(np.uint8)
            for c in range(3)
        ])
        corrected = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

        # Blend with original swapped frame
        mask3 = face_mask[:, :, np.newaxis]
        result = (
            s * mask3 * corrected.astype(np.float32) +
            (1.0 - s * mask3) * swapped_frame.astype(np.float32)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Auto white balance
    # ------------------------------------------------------------------

    def auto_white_balance(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Estimate illuminant from face skin and apply mild correction.

        Args:
            frame:     Full BGR frame.
            face_mask: float32 (H, W) of skin region.

        Returns:
            White-balanced BGR frame.
        """
        mask_bool = face_mask > 0.5
        if mask_bool.sum() < 100:
            return frame

        # Grey-world assumption on skin pixels
        skin_pixels = frame[mask_bool].astype(np.float32)
        mean_b, mean_g, mean_r = skin_pixels[:, 0].mean(), skin_pixels[:, 1].mean(), skin_pixels[:, 2].mean()
        mean_all = (mean_b + mean_g + mean_r) / 3.0

        if mean_b < 1 or mean_g < 1 or mean_r < 1:
            return frame

        scale_b = mean_all / mean_b
        scale_g = mean_all / mean_g
        scale_r = mean_all / mean_r

        # Clamp scales to avoid over-correction
        scale_b = float(np.clip(scale_b, 0.7, 1.4))
        scale_g = float(np.clip(scale_g, 0.7, 1.4))
        scale_r = float(np.clip(scale_r, 0.7, 1.4))

        result = frame.astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Local contrast enhancement (CLAHE)
    # ------------------------------------------------------------------

    def enhance_contrast(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
        strength: float = 0.2,
    ) -> np.ndarray:
        """Adaptive CLAHE contrast enhancement on the face region only.

        Args:
            frame:     Full BGR frame.
            face_mask: float32 (H, W).
            strength:  Blend strength [0, 1].

        Returns:
            Contrast-enhanced BGR frame.
        """
        if strength <= 0 or face_mask.max() < 0.01:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_clahe = clahe.apply(L)

        # Blend CLAHE result within face mask only
        L_out = L.astype(np.float32)
        mask_s = face_mask * strength
        L_out = mask_s * L_clahe.astype(np.float32) + (1.0 - mask_s) * L_out
        L_out = np.clip(L_out, 0, 255).astype(np.uint8)

        lab_out = cv2.merge([L_out, A, B])
        return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    def update_strength(self, strength: float) -> None:
        self._strength = float(np.clip(strength, 0.0, 1.0))


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from insightface.app import FaceAnalysis
    from src.occlusion.face_parsing import FaceParser

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    parser = FaceParser({"model_path": "models/face_parser.onnx"})
    corrector = ColorCorrector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            face_mask = parser.get_face_region_mask(frame, bbox)
            enhanced = corrector.enhance_contrast(frame, face_mask, strength=0.3)
            vis = np.hstack([frame, enhanced])
        else:
            vis = frame
        cv2.imshow("ColorCorrector", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
