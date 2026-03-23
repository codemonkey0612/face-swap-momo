"""Skin smoothing: bilateral + guided filter in LAB space.

Applies smooth, magazine-quality skin retouching to the face region only,
preserving eyes, brows, lips, and non-skin features.
"""

from __future__ import annotations

import cv2
import numpy as np


class SkinSmoother:
    """Multi-step skin smoothing applied only within the face mask.

    Pipeline:
        1. Convert to LAB (process luminance only → preserves colour)
        2. Bilateral filter on L channel
        3. Guided filter (ximgproc) using original L as guide
        4. Blend result with original by strength parameter
        5. Apply only within the face mask (excluding eyes/lips)

    Args:
        config: dict with optional key:
            strength (float): 0.0-1.0 smoothing intensity (default 0.6)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._strength = float(cfg.get("strength", 0.6))
        # Check for ximgproc (opencv-contrib)
        try:
            _ = cv2.ximgproc
            self._has_ximgproc = True
        except AttributeError:
            self._has_ximgproc = False

    def smooth(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """Apply skin smoothing to face_mask region.

        Args:
            frame:     Full BGR frame (uint8).
            face_mask: float32 (H, W) in [0, 1] — where to apply smoothing.
            strength:  Override instance strength if provided.

        Returns:
            Smoothed BGR frame (uint8), background unchanged.
        """
        s = strength if strength is not None else self._strength
        if s <= 0 or face_mask.max() < 0.01:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab.astype(np.float32))

        # Step 1: Bilateral filter on L channel
        L_u8 = np.clip(L, 0, 255).astype(np.uint8)
        L_bilateral = cv2.bilateralFilter(L_u8, d=9, sigmaColor=75, sigmaSpace=75)
        L_smooth = L_bilateral.astype(np.float32)

        # Step 2: Guided filter (if available) — sharper edges, smoother skin
        if self._has_ximgproc:
            L_guided = cv2.ximgproc.guidedFilter(
                guide=L_u8,
                src=L_bilateral,
                radius=8,
                eps=100,
            ).astype(np.float32)
            # Blend bilateral and guided
            L_smooth = 0.5 * L_smooth + 0.5 * L_guided

        # Step 3: Further isolate actual skin pixels
        skin_sub_mask = self._detect_skin_regions(frame, face_mask)

        # Step 4: Blend using strength
        effective_mask = skin_sub_mask * s
        L_out = effective_mask * L_smooth + (1.0 - effective_mask) * L

        # Reconstruct
        result_lab = cv2.merge([
            np.clip(L_out, 0, 255).astype(np.uint8),
            cv2.split(lab)[1],
            cv2.split(lab)[2],
        ])
        return cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)

    def _detect_skin_regions(
        self, frame: np.ndarray, face_mask: np.ndarray
    ) -> np.ndarray:
        """Within face_mask, further restrict to actual skin pixels.

        Excludes eyes, nostrils, lip interior to preserve sharp details.

        Returns:
            float32 mask (H, W) of skin-only pixels.
        """
        # HSV skin range
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0,  20, 70], dtype=np.uint8)
        upper = np.array([30, 170, 255], dtype=np.uint8)
        skin_hsv = cv2.inRange(hsv, lower, upper).astype(np.float32) / 255.0

        # Dark pixels (eyes, nostrils, lip interior) — exclude
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark = (gray < 60).astype(np.float32)

        skin_mask = skin_hsv * face_mask * (1.0 - dark)

        # Smooth the mask to avoid hard edges
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
        return np.clip(skin_mask, 0.0, 1.0)

    def update_strength(self, strength: float) -> None:
        """Hot-update smoothing strength (e.g. from GUI slider)."""
        self._strength = float(np.clip(strength, 0.0, 1.0))


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from insightface.app import FaceAnalysis
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.occlusion.face_parsing import FaceParser

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    parser = FaceParser({"model_path": "models/face_parser.onnx"})
    smoother = SkinSmoother()
    strength = [0.6]

    def on_trackbar(val):
        strength[0] = val / 100.0

    cv2.namedWindow("SkinSmoother")
    cv2.createTrackbar("Strength %", "SkinSmoother", 60, 100, on_trackbar)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            face_mask = parser.get_face_region_mask(frame, bbox)
            smoothed = smoother.smooth(frame, face_mask, strength[0])
            vis = np.hstack([frame, smoothed])
        else:
            vis = frame
        cv2.imshow("SkinSmoother", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
