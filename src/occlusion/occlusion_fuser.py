"""Occlusion fusion: combines all 5 layers into a single swap mask.

This is the brain of the occlusion system. It takes outputs from:
    Layer 1: Face parsing   (where is the face?)
    Layer 2: Hand detection  (hands over face)
    Layer 3: Body segmentation (arms, objects)
    Layer 4: Error detection  (swap anomalies)
    Layer 5: Depth estimation (z-order: things closer than face)

And produces a single float32 mask [0, 1]:
    1.0 = safe to show swapped face
    0.0 = show original frame (hand / arm / object in front)
    intermediate = feathered boundary

Temporal consistency: EMA with fast-react-to-occlusion / slow-fade-out.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


class OcclusionFuser:
    """Weighted voting fusion of all 5 occlusion layers.

    Args:
        config: dict with optional keys (see defaults in _DEFAULT_CFG).
    """

    _DEFAULT_CFG = {
        # Layer weights [0, 1]
        "hand_weight":   0.9,
        "depth_weight":  0.7,
        "error_weight":  0.6,
        "body_weight":   0.5,
        "parse_weight":  0.3,
        # Voting: a pixel is occluded if weighted score >= voting_threshold
        "voting_threshold": 0.65,
        # Temporal EMA
        "temporal_alpha": 0.80,
        # Fast-react alpha (occlusion appears)
        "appear_alpha": 0.95,
        # Slow-fade alpha (occlusion disappears)
        "disappear_alpha": 0.70,
    }

    def __init__(self, config: dict | None = None):
        cfg = {**self._DEFAULT_CFG, **(config or {})}
        self._w_hand   = cfg["hand_weight"]
        self._w_depth  = cfg["depth_weight"]
        self._w_error  = cfg["error_weight"]
        self._w_body   = cfg["body_weight"]
        self._w_parse  = cfg["parse_weight"]
        self._vote_thr = cfg["voting_threshold"]
        self._alpha    = cfg["temporal_alpha"]
        self._alpha_appear  = cfg["appear_alpha"]
        self._alpha_disappear = cfg["disappear_alpha"]
        self._prev_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Full 5-layer fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        face_parse_mask: np.ndarray,
        hand_mask:       np.ndarray,
        body_mask:       np.ndarray,
        error_mask:      np.ndarray,
        depth_mask:      np.ndarray,
    ) -> np.ndarray:
        """Produce a swap mask from all 5 layers.

        All inputs are float32 (H, W) in [0, 1].
        face_parse_mask: 1 = face pixel (from FaceParser).
        hand/body/error/depth masks: 1 = occluded pixel.

        Returns:
            float32 swap mask (H, W): 1.0 = show swap, 0.0 = show original.
        """
        # Step 1: Start from the face region as the maximum swap area
        swap_mask = face_parse_mask.copy()

        # Step 2: Compute weighted occlusion score per pixel
        # Higher score = more evidence of occlusion
        occluder_score = (
            self._w_hand  * hand_mask  +
            self._w_depth * depth_mask +
            self._w_error * error_mask +
            self._w_body  * body_mask
        )

        # Step 3: Voting — pixel is occluded if:
        #   a) hand_mask alone says so (high confidence), OR
        #   b) weighted score exceeds threshold
        #   c) depth AND one other layer agree
        depth_and_body  = np.minimum(depth_mask, body_mask)
        depth_and_hand  = np.minimum(depth_mask, hand_mask)
        depth_and_error = np.minimum(depth_mask, error_mask)
        depth_combo = np.maximum(np.maximum(depth_and_body, depth_and_hand), depth_and_error)

        # Occlusion: 1 = occluded
        occluded = np.clip(
            np.maximum(hand_mask, np.maximum(
                (occluder_score >= self._vote_thr).astype(np.float32),
                depth_combo,
            )),
            0.0, 1.0,
        )

        # Step 4: Subtract occluded regions from swap area
        swap_mask = swap_mask * (1.0 - occluded)
        swap_mask = np.clip(swap_mask, 0.0, 1.0)

        # Step 5: Temporal consistency
        return self._temporal_smooth(swap_mask)

    def fuse_fast(
        self,
        face_parse_mask: np.ndarray,
        hand_mask:       np.ndarray,
        body_mask:       np.ndarray,
    ) -> np.ndarray:
        """Lightweight 3-layer fusion (Layers 1-3 only) for high-FPS modes.

        Logic: face_parse_mask AND NOT (hand_mask OR body_mask)
        """
        occluded = np.clip(hand_mask + body_mask, 0.0, 1.0)
        swap_mask = face_parse_mask * (1.0 - occluded)
        return self._temporal_smooth(np.clip(swap_mask, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Temporal consistency
    # ------------------------------------------------------------------

    def _temporal_smooth(self, current: np.ndarray) -> np.ndarray:
        """EMA with asymmetric alphas: fast appear, slow disappear."""
        if self._prev_mask is None or self._prev_mask.shape != current.shape:
            self._prev_mask = current.copy()
            return current

        prev_area   = self._prev_mask.sum()
        curr_area   = current.sum()
        area_change = curr_area - prev_area

        # Occlusion APPEARING (mask shrinking = more occluded)
        if area_change < -100:
            alpha = self._alpha_appear
        # Occlusion DISAPPEARING (mask growing = face revealed)
        elif area_change > 100:
            alpha = self._alpha_disappear
        else:
            alpha = self._alpha

        smoothed = alpha * current + (1.0 - alpha) * self._prev_mask
        self._prev_mask = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        """Clear temporal buffer (call on scene cut or source change)."""
        self._prev_mask = None

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def debug_overlay(
        self,
        frame: np.ndarray,
        hand_mask:  np.ndarray,
        depth_mask: np.ndarray,
        body_mask:  np.ndarray,
        error_mask: np.ndarray,
        parse_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a colour-coded BGR frame showing which layer contributes what.

        Colours:
            Green  = hand detector
            Blue   = depth estimation
            Yellow = body segmentation
            Orange = error detection
            White  = face parsing boundary
        """
        vis = frame.copy().astype(np.float32)
        H, W = vis.shape[:2]

        def _add(mask: np.ndarray, bgr: tuple) -> None:
            for c, v in enumerate(bgr):
                vis[:, :, c] += mask * v * 0.4

        _add(hand_mask,  (0, 255, 0))     # Green
        _add(depth_mask, (255, 0, 0))     # Blue
        _add(body_mask,  (0, 255, 255))   # Yellow
        _add(error_mask, (0, 165, 255))   # Orange
        # Face parsing boundary
        parse_u8 = (parse_mask * 255).astype(np.uint8)
        boundary = cv2.Canny(parse_u8, 50, 150)
        for c in range(3):
            vis[:, :, c] += boundary.astype(np.float32) * 0.5

        return np.clip(vis, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from insightface.app import FaceAnalysis
    from src.occlusion.face_parsing  import FaceParser
    from src.occlusion.hand_detector import HandDetector
    from src.occlusion.body_segmenter import BodySegmenter
    from src.occlusion.error_detector import ErrorDetector
    from src.occlusion.depth_estimator import DepthEstimator
    from src.occlusion.mask_refiner import MaskRefiner

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    l1 = FaceParser({"model_path": "models/face_parser.onnx"})
    l2 = HandDetector()
    l3 = BodySegmenter()
    l4 = ErrorDetector()
    l5 = DepthEstimator({"model": "midas_small"})
    fuser = OcclusionFuser()
    refiner = MaskRefiner()

    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        vis = frame.copy()
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            parse_mask = l1.get_face_region_mask(frame, bbox)
            hand_mask  = l2.get_hand_mask(frame, bbox, parse_mask)
            body_mask  = l3.get_body_over_face_mask(frame, bbox, parse_mask)
            error_mask = l4.threshold_anomaly(
                l4.compute_error_map(prev_frame if prev_frame is not None else frame, frame, bbox)
            )
            depth_map  = l5.estimate_depth(frame)
            depth_mask = l5.get_foreground_mask(frame, bbox, depth_map)

            swap_mask = fuser.fuse(parse_mask, hand_mask, body_mask, error_mask, depth_mask)
            swap_mask = refiner.refine_and_smooth(swap_mask)

            debug = fuser.debug_overlay(frame, hand_mask, depth_mask, body_mask, error_mask, parse_mask)
            mask_vis = cv2.applyColorMap((swap_mask * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            vis = np.hstack([debug, mask_vis])

        prev_frame = frame.copy()
        cv2.imshow("OcclusionFuser — All 5 Layers | debug | swap_mask", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    l2.close()
    l3.close()
    cap.release()
    cv2.destroyAllWindows()
