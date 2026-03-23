"""Eye enhancement: CLAHE contrast, sclera brightening, unsharp sharpening.

Applies subtle but effective eye enhancement using 5-point landmarks
(eyes, nose, mouth corners) from InsightFace / RetinaFace.
"""

from __future__ import annotations

import cv2
import numpy as np


class EyeEnhancer:
    """Enhances eye region: higher contrast iris, brighter sclera, sharpening.

    Args:
        config: dict with optional key:
            strength (float): 0.0-1.0 enhancement intensity (default 0.3)
    """

    # Padding around eye region relative to inter-eye distance
    _EYE_PAD_FRACTION = 0.35

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._strength = float(cfg.get("strength", 0.3))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def enhance(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        strength: float | None = None,
    ) -> np.ndarray:
        """Enhance both eyes in the frame.

        Args:
            frame:     Full BGR frame (uint8).
            landmarks: (5, 2) array — [left_eye, right_eye, nose, mouth_l, mouth_r].
            strength:  Override instance strength if provided.

        Returns:
            Enhanced BGR frame (same size as input).
        """
        s = strength if strength is not None else self._strength
        if s <= 0 or landmarks is None or len(landmarks) < 2:
            return frame

        result = frame.copy()
        lm = landmarks.astype(np.float32)

        for eye_idx in (0, 1):
            result = self._enhance_eye(result, lm, eye_idx, s)

        return result

    def _enhance_eye(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        eye_idx: int,
        strength: float,
    ) -> np.ndarray:
        """Enhance one eye identified by landmark index (0=left, 1=right)."""
        H, W = frame.shape[:2]
        cx, cy = landmarks[eye_idx]

        # Estimate eye radius from inter-eye distance
        inter_eye = np.linalg.norm(landmarks[0] - landmarks[1])
        rad = int(inter_eye * self._EYE_PAD_FRACTION)
        rad = max(rad, 12)

        x1 = int(max(0, cx - rad))
        y1 = int(max(0, cy - rad))
        x2 = int(min(W, cx + rad))
        y2 = int(min(H, cy + rad))
        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2].copy()

        # --- CLAHE on L channel for iris contrast ---
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        L_clahe = self._clahe.apply(L)
        # Subtle: blend CLAHE result with original L
        L_blend = cv2.addWeighted(L, 1.0 - strength, L_clahe, strength, 0)
        lab_clahe = cv2.merge([L_blend, A, B])
        roi_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_Lab2BGR)

        # --- Sclera brightening (bright pixels = whites) ---
        gray_roi = cv2.cvtColor(roi_clahe, cv2.COLOR_BGR2GRAY)
        sclera_mask = (gray_roi > 160).astype(np.float32)
        sclera_mask = cv2.GaussianBlur(sclera_mask, (5, 5), 0)
        brighten_amount = strength * 15
        roi_bright = np.clip(
            roi_clahe.astype(np.float32) + sclera_mask[:, :, np.newaxis] * brighten_amount,
            0, 255,
        ).astype(np.uint8)

        # --- Unsharp mask for sharpening ---
        blurred = cv2.GaussianBlur(roi_bright, (0, 0), sigmaX=1.5)
        roi_sharp = cv2.addWeighted(roi_bright, 1.0 + strength * 0.4, blurred, -strength * 0.4, 0)

        # --- Circular blend mask (soft edges) ---
        h_roi, w_roi = roi.shape[:2]
        blend = np.zeros((h_roi, w_roi), dtype=np.float32)
        cv2.circle(blend, (w_roi // 2, h_roi // 2), rad - 2, 1.0, -1)
        blend = cv2.GaussianBlur(blend, (int(rad * 0.5) | 1, int(rad * 0.5) | 1), 0)

        roi_out = (
            blend[:, :, np.newaxis] * roi_sharp.astype(np.float32) +
            (1.0 - blend[:, :, np.newaxis]) * roi.astype(np.float32)
        ).astype(np.uint8)

        result = frame.copy()
        result[y1:y2, x1:x2] = roi_out
        return result

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

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    enhancer = EyeEnhancer()
    strength = [0.3]

    def on_trackbar(val):
        strength[0] = val / 100.0

    cv2.namedWindow("EyeEnhancer")
    cv2.createTrackbar("Strength %", "EyeEnhancer", 30, 100, on_trackbar)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        if faces:
            lm = faces[0].kps  # (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r
            enhanced = enhancer.enhance(frame, lm, strength[0])
            vis = np.hstack([frame, enhanced])
        else:
            vis = frame
        cv2.imshow("EyeEnhancer", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
