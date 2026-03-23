"""Layer 4: Error-based anomaly detection (HEAR-Net style).

Instead of detecting WHAT is occluding, we detect WHERE the swap went wrong.
When the swap model tries to generate a face where there is actually a hand,
the output looks WRONG. By comparing the swapped result against expectations,
we find anomaly regions that should be restored from the original frame.

Reference: FaceShifter HEAR-Net (He et al., 2020)
"""

from __future__ import annotations

import cv2
import numpy as np


class ErrorDetector:
    """HEAR-Net-style swap anomaly detector — Layer 4 of the occlusion system.

    Compares the original frame against the swapped frame within the face
    bounding box and computes a per-pixel anomaly score from four signals:
        1. Structural error (LAB absolute difference)
        2. Colour anomaly (deviation from expected skin tone)
        3. Texture anomaly (Laplacian variance in 8×8 patches)
        4. Edge discontinuity (unexpected Sobel edges)

    Args:
        config: dict with optional keys:
            enabled (bool)
            sensitivity (float): 0-1, scales anomaly threshold (default 0.5)
    """

    # Weights for combining sub-signals
    _W_STRUCT  = 0.30
    _W_COLOUR  = 0.25
    _W_TEXTURE = 0.25
    _W_EDGE    = 0.20

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._sensitivity = float(cfg.get("sensitivity", 0.5))
        # Reference skin stats (set when source face is loaded)
        self._ref_skin_mean_lab: np.ndarray | None = None
        self._ref_skin_std_lab: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Source face reference
    # ------------------------------------------------------------------

    def set_source_face_reference(self, face_bgr: np.ndarray) -> None:
        """Compute reference skin colour stats from the source face image.

        Call this once after loading the source face image.

        Args:
            face_bgr: Cropped BGR image of the source (AI) face.
        """
        if face_bgr is None or face_bgr.size == 0:
            return
        lab = cv2.cvtColor(face_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        # Use centre 50% of face (most reliable skin area)
        h, w = lab.shape[:2]
        cy, cx = h // 2, w // 2
        roi = lab[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]
        self._ref_skin_mean_lab = roi.reshape(-1, 3).mean(axis=0)
        self._ref_skin_std_lab  = roi.reshape(-1, 3).std(axis=0) + 1e-6

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def compute_error_map(
        self,
        original_frame: np.ndarray,
        swapped_frame: np.ndarray,
        face_bbox: np.ndarray,
    ) -> np.ndarray:
        """Compute per-pixel anomaly score in [0, 1] within the face bbox.

        Args:
            original_frame: BGR frame before swapping.
            swapped_frame:  BGR frame after swapping.
            face_bbox:      [x1, y1, x2, y2].

        Returns:
            float32 anomaly map (H, W) in [0, 1]. 1.0 = high anomaly.
        """
        if not self._enabled:
            return np.zeros(original_frame.shape[:2], dtype=np.float32)

        H, W = original_frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((H, W), dtype=np.float32)

        orig_roi = original_frame[y1:y2, x1:x2].astype(np.float32)
        swap_roi = swapped_frame[y1:y2, x1:x2].astype(np.float32)
        rh, rw = orig_roi.shape[:2]

        # --- Signal 1: Structural error (LAB absolute difference) ---
        orig_lab = cv2.cvtColor(orig_roi.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        swap_lab = cv2.cvtColor(swap_roi.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        diff_lab = np.abs(orig_lab - swap_lab)
        struct_err = diff_lab.max(axis=2) / 128.0  # normalise to [0, 1]
        struct_err = np.clip(struct_err, 0, 1)

        # --- Signal 2: Colour anomaly vs expected skin tone ---
        if self._ref_skin_mean_lab is not None:
            pixel_dev = np.abs(swap_lab - self._ref_skin_mean_lab) / (
                self._ref_skin_std_lab * 3.0
            )
            colour_anom = np.clip(pixel_dev.max(axis=2), 0, 1)
        else:
            # Fallback: centre-of-face colour as reference
            cy, cx = rh // 2, rw // 2
            ref_lab = swap_lab[
                max(0, cy - rh // 8): cy + rh // 8,
                max(0, cx - rw // 8): cx + rw // 8,
            ]
            ref_mean = ref_lab.reshape(-1, 3).mean(axis=0)
            colour_anom = np.clip(
                np.abs(swap_lab - ref_mean).max(axis=2) / 80.0, 0, 1
            )

        # --- Signal 3: Texture anomaly (Laplacian variance in 8×8 patches) ---
        swap_gray = cv2.cvtColor(swap_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(swap_gray, cv2.CV_32F)
        # Compute local variance in 8×8 blocks
        lap_sq = laplacian ** 2
        local_var = cv2.blur(lap_sq, (8, 8))
        texture_anom = np.clip(local_var / 5000.0, 0, 1)  # normalise empirically

        # --- Signal 4: Edge discontinuity ---
        sobel_x = cv2.Sobel(swap_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(swap_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Dilate to get regions around unexpected edges
        edge_anom = cv2.dilate(
            np.clip(edge_mag / 200.0, 0, 1),
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        # --- Weighted combination ---
        anomaly_roi = (
            self._W_STRUCT  * struct_err +
            self._W_COLOUR  * colour_anom +
            self._W_TEXTURE * texture_anom +
            self._W_EDGE    * edge_anom
        )
        anomaly_roi = np.clip(anomaly_roi, 0, 1)

        # Embed back into full frame
        anomaly_map = np.zeros((H, W), dtype=np.float32)
        anomaly_map[y1:y2, x1:x2] = anomaly_roi
        return anomaly_map

    def threshold_anomaly(
        self,
        anomaly_map: np.ndarray,
        sensitivity: float | None = None,
    ) -> np.ndarray:
        """Convert the continuous anomaly map to a binary occlusion mask.

        Uses Otsu's method (adaptive) when the map has significant variance,
        else falls back to a fixed threshold scaled by sensitivity.

        Args:
            anomaly_map: float32 (H, W) in [0, 1].
            sensitivity: 0-1 (overrides instance default if given).

        Returns:
            float32 binary mask (H, W): 1.0 = anomalous / occluded.
        """
        if sensitivity is None:
            sensitivity = self._sensitivity

        anom_u8 = (anomaly_map * 255).astype(np.uint8)
        # Try Otsu
        _, otsu = cv2.threshold(anom_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fixed threshold: 0.5 sensitivity → 0.5 threshold, clamp to [0.2, 0.8]
        fixed_thresh = int(np.clip(0.8 - sensitivity * 0.6, 0.2, 0.8) * 255)

        # Blend: use whichever catches more anomaly
        _, fixed = cv2.threshold(anom_u8, fixed_thresh, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(otsu, fixed)

        # Mild dilation to cover boundary pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.dilate(combined, kernel, iterations=1)

        return combined.astype(np.float32) / 255.0

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # We need a real swap to demonstrate; if not available, synthesise.
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    detector = ErrorDetector({"sensitivity": 0.5})

    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = frame.copy()
        faces = app.get(frame)
        if faces and prev_frame is not None:
            bbox = faces[0].bbox
            # Use previous frame as fake "swapped" to demo anomaly detection
            amap = detector.compute_error_map(prev_frame, frame, bbox)
            thresh = detector.threshold_anomaly(amap)

            # Heatmap overlay (red = high anomaly)
            amap_u8 = (amap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.6, heatmap, 0.4, 0)

        prev_frame = frame.copy()
        cv2.imshow("ErrorDetector Layer 4 (red=anomaly)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
