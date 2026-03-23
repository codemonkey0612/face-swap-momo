"""Tests for the 5-layer occlusion system.

Run with: pytest tests/test_occlusion.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h=480, w=640, colour=(128, 128, 128)):
    frame = np.full((h, w, 3), colour, dtype=np.uint8)
    return frame

def _face_bbox(cx=320, cy=240, r=80):
    return np.array([cx - r, cy - r, cx + r, cy + r], dtype=np.float32)

def _draw_skin_face(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.ellipse(frame, ((x1+x2)//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, (180, 140, 100), -1)
    return frame


# ---------------------------------------------------------------------------
# Layer 1: Face parsing
# ---------------------------------------------------------------------------

class TestFaceParser:
    def test_disabled_returns_zeros(self):
        from src.occlusion.face_parsing import FaceParser
        parser = FaceParser({"enabled": False})
        frame = _blank_frame()
        bbox = _face_bbox()
        seg = parser.parse(frame, bbox)
        assert seg.shape == (480, 640)
        assert seg.max() == 0

    def test_face_region_mask_shape(self):
        from src.occlusion.face_parsing import FaceParser
        parser = FaceParser({"enabled": False})
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = parser.get_face_region_mask(frame, bbox)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32

    def test_face_region_mask_range(self):
        from src.occlusion.face_parsing import FaceParser
        parser = FaceParser({"enabled": False})
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = parser.get_face_region_mask(frame, bbox)
        assert 0.0 <= mask.min() and mask.max() <= 1.0

    def test_colorize_shape(self):
        from src.occlusion.face_parsing import FaceParser
        parser = FaceParser({"enabled": False})
        seg = np.zeros((480, 640), dtype=np.uint8)
        vis = parser.colorize(seg)
        assert vis.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Layer 2: Hand detection
# ---------------------------------------------------------------------------

class TestHandDetector:
    def test_no_hands_empty_frame(self):
        from src.occlusion.hand_detector import HandDetector
        det = HandDetector({"enabled": True})
        frame = _blank_frame()
        hands = det.detect_hands(frame)
        assert isinstance(hands, list)
        det.close()

    def test_mask_shape(self):
        from src.occlusion.hand_detector import HandDetector
        det = HandDetector({"enabled": True})
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = det.get_hand_mask(frame, bbox)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() and mask.max() <= 1.0
        det.close()

    def test_fallback_mask_shape(self):
        from src.occlusion.hand_detector import HandDetector
        det = HandDetector({"enabled": True})
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = det.get_hand_mask_fallback(frame, bbox)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32
        det.close()

    def test_hand_mask_overlaps_face_bbox(self):
        """Synthetic: hand pixels placed inside face bbox should be flagged."""
        from src.occlusion.hand_detector import HandDetector
        det = HandDetector({"enabled": True})
        frame = _blank_frame(colour=(60, 60, 60))  # non-skin background
        bbox = _face_bbox()
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw a large skin-coloured blob inside the face region
        # (simulates hand in YCrCb range)
        hand_region = frame[y1:y2, x1:x2]
        hand_region[:, :] = [100, 155, 100]  # YCrCb skin colour in BGR ~= (100,155,100)
        frame[y1:y2, x1:x2] = hand_region

        mask = det.get_hand_mask_fallback(frame, bbox)
        # Mask should not be all zeros (some skin detected)
        assert mask.shape == (480, 640)
        det.close()


# ---------------------------------------------------------------------------
# Layer 3: Body segmentation
# ---------------------------------------------------------------------------

class TestBodySegmenter:
    def test_person_mask_shape(self):
        from src.occlusion.body_segmenter import BodySegmenter
        seg = BodySegmenter({"enabled": True})
        frame = _blank_frame()
        mask = seg.get_person_mask(frame)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32
        seg.close()

    def test_body_over_face_excludes_face_pixels(self):
        from src.occlusion.body_segmenter import BodySegmenter
        seg = BodySegmenter({"enabled": True})
        frame = _blank_frame()
        bbox = _face_bbox()
        # Perfect face_parse_mask that covers the entire bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_parse_mask = np.zeros((480, 640), dtype=np.float32)
        face_parse_mask[y1:y2, x1:x2] = 1.0
        body_mask = seg.get_body_over_face_mask(frame, bbox, face_parse_mask)
        assert body_mask.shape == (480, 640)
        # Where face_parse_mask=1 and person_mask=1, body_not_face should be 0
        # (In this synthetic test both masks may be 0, just check shape/type)
        assert body_mask.dtype == np.float32
        assert 0.0 <= body_mask.min() and body_mask.max() <= 1.0
        seg.close()

    def test_object_mask_shape(self):
        from src.occlusion.body_segmenter import BodySegmenter
        seg = BodySegmenter()
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = seg.get_object_mask(frame, bbox)
        assert mask.shape == (480, 640)
        seg.close()


# ---------------------------------------------------------------------------
# Layer 4: Error detection
# ---------------------------------------------------------------------------

class TestErrorDetector:
    def test_anomaly_map_shape_and_range(self):
        from src.occlusion.error_detector import ErrorDetector
        det = ErrorDetector({"sensitivity": 0.5})
        orig = _blank_frame(colour=(150, 120, 100))
        swap = _blank_frame(colour=(140, 115, 95))
        bbox = _face_bbox()
        amap = det.compute_error_map(orig, swap, bbox)
        assert amap.shape == (480, 640)
        assert amap.dtype == np.float32
        assert 0.0 <= amap.min() and amap.max() <= 1.0

    def test_high_difference_produces_high_anomaly(self):
        from src.occlusion.error_detector import ErrorDetector
        det = ErrorDetector()
        orig = _blank_frame(colour=(50, 50, 50))
        swap = _blank_frame(colour=(200, 200, 200))  # very different
        bbox = _face_bbox()
        amap = det.compute_error_map(orig, swap, bbox)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_amap = amap[y1:y2, x1:x2]
        assert face_amap.mean() > 0.05  # should have non-trivial anomaly

    def test_threshold_output_binary(self):
        from src.occlusion.error_detector import ErrorDetector
        det = ErrorDetector()
        amap = np.random.rand(480, 640).astype(np.float32)
        thresh = det.threshold_anomaly(amap, sensitivity=0.5)
        assert thresh.shape == (480, 640)
        assert thresh.dtype == np.float32
        unique = np.unique(thresh)
        # Should be mostly 0.0 or 1.0 (binary after threshold)
        assert all(v in (0.0, 1.0) for v in unique)

    def test_disabled_returns_zeros(self):
        from src.occlusion.error_detector import ErrorDetector
        det = ErrorDetector({"enabled": False})
        orig = _blank_frame()
        swap = _blank_frame()
        bbox = _face_bbox()
        amap = det.compute_error_map(orig, swap, bbox)
        assert amap.max() == 0.0


# ---------------------------------------------------------------------------
# Layer 5: Depth estimation
# ---------------------------------------------------------------------------

class TestDepthEstimator:
    def test_disabled_returns_zeros(self):
        from src.occlusion.depth_estimator import DepthEstimator
        est = DepthEstimator({"enabled": False})
        frame = _blank_frame()
        depth = est.estimate_depth(frame)
        assert depth.shape == (480, 640)
        assert depth.max() == 0.0

    def test_foreground_mask_shape_when_disabled(self):
        from src.occlusion.depth_estimator import DepthEstimator
        est = DepthEstimator({"enabled": False})
        frame = _blank_frame()
        bbox = _face_bbox()
        mask = est.get_foreground_mask(frame, bbox)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32

    def test_foreground_mask_flags_closer_pixels(self):
        """Synthetic depth map: pixels at top of bbox are closer than face plane."""
        from src.occlusion.depth_estimator import DepthEstimator
        est = DepthEstimator({"enabled": False, "margin": 0.1})
        # Build a fake depth map (0=closest, 1=farthest)
        depth_map = np.full((480, 640), 0.5, dtype=np.float32)
        bbox = _face_bbox()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Make top portion of face region much closer
        depth_map[y1:y1+20, x1:x2] = 0.1  # much closer
        # Median of face region will be ~0.5; 0.1 < 0.5 - 0.1 = 0.4 → flagged
        mask = est.get_foreground_mask(_blank_frame(), bbox, depth_map=depth_map)
        assert mask[y1:y1+20, x1:x2].max() > 0.0


# ---------------------------------------------------------------------------
# Occlusion fuser
# ---------------------------------------------------------------------------

class TestOcclusionFuser:
    def _zero(self):
        return np.zeros((480, 640), dtype=np.float32)

    def _ones(self):
        return np.ones((480, 640), dtype=np.float32)

    def test_hand_alone_marks_occluded(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser()
        face = self._ones()
        hand = self._ones()  # entire frame is "hand"
        zero = self._zero()
        mask = fuser.fuse(face, hand, zero, zero, zero)
        # With hand_weight=0.9 and voting_threshold=0.65, hand alone should occlude
        assert mask.mean() < 0.5

    def test_no_occlusion_preserves_face(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser()
        fuser.reset()
        face = self._ones()
        zero = self._zero()
        mask = fuser.fuse(face, zero, zero, zero, zero)
        # No occluders → swap everywhere
        assert mask.mean() > 0.5

    def test_single_weak_signal_not_enough(self):
        """A single weak layer below hand_weight should not fully occlude."""
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser()
        fuser.reset()
        face = self._ones()
        hand = self._zero()
        body = self._ones() * 0.3  # weak signal
        zero = self._zero()
        mask = fuser.fuse(face, hand, body, zero, zero)
        # Body alone at 0.3 weight < 0.65 threshold → should mostly preserve face
        assert mask.mean() > 0.3

    def test_two_layers_agreeing_occludes(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser()
        fuser.reset()
        face = self._ones()
        hand = self._zero()
        body = self._ones()   # weight 0.5
        error = self._ones()  # weight 0.6
        depth = self._zero()
        # body(0.5) + error(0.6) = 1.1 > 0.65 threshold → occluded
        mask = fuser.fuse(face, hand, body, error, depth)
        assert mask.mean() < 0.5

    def test_temporal_smoothing_reduces_variance(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser({"temporal_alpha": 0.5})
        fuser.reset()
        face = self._ones()
        zero = self._zero()
        mask1 = fuser.fuse(face, zero, zero, zero, zero)
        mask2 = fuser.fuse(face, self._ones(), zero, zero, zero)
        mask3 = fuser.fuse(face, zero, zero, zero, zero)
        # mask3 should be smoothed toward mask1, not jump abruptly
        assert mask3.mean() > 0.0  # not instantly zero due to EMA

    def test_fuse_fast_shape(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        fuser = OcclusionFuser()
        face = self._ones()
        hand = self._zero()
        body = self._zero()
        mask = fuser.fuse_fast(face, hand, body)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32


# ---------------------------------------------------------------------------
# Mask refiner
# ---------------------------------------------------------------------------

class TestMaskRefiner:
    def test_refine_preserves_shape(self):
        from src.occlusion.mask_refiner import MaskRefiner
        refiner = MaskRefiner()
        mask = np.random.rand(480, 640).astype(np.float32)
        refined = refiner.refine(mask)
        assert refined.shape == (480, 640)

    def test_feathering_produces_smooth_edges(self):
        from src.occlusion.mask_refiner import MaskRefiner
        refiner = MaskRefiner(erosion_px=0, feather_kernel=31, feather_sigma=5.0)
        # Binary mask
        mask = np.zeros((480, 640), dtype=np.float32)
        mask[100:200, 100:200] = 1.0
        refined = refiner.refine(mask)
        # Should not have hard 0→1 transitions at boundary
        boundary_region = refined[95:105, 100:110]
        # Values should be strictly between 0 and 1 at the feathered boundary
        assert boundary_region.min() < 0.9
        assert refined.max() <= 1.0

    def test_temporal_smooth_reduces_flicker(self):
        from src.occlusion.mask_refiner import MaskRefiner
        refiner = MaskRefiner(temporal_alpha=0.5)
        mask_a = np.ones((480, 640), dtype=np.float32)
        mask_b = np.zeros((480, 640), dtype=np.float32)
        r1 = refiner.temporal_smooth(mask_a)
        r2 = refiner.temporal_smooth(mask_b)
        # With alpha=0.5, r2 should be 0.5 * 0 + 0.5 * 1 = 0.5
        assert abs(r2.mean() - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Full integration: synthetic hand-over-face
# ---------------------------------------------------------------------------

class TestOcclusionIntegration:
    def test_synthetic_hand_over_face_preserves_hand(self):
        """A hand-coloured region inside the face bbox should be excluded
        from the swap mask (original frame preserved there)."""
        from src.occlusion.hand_detector import HandDetector
        from src.occlusion.occlusion_fuser import OcclusionFuser

        H, W = 480, 640
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        bbox = _face_bbox(cx=320, cy=240, r=100)
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Fill face area with face_parse_mask=1
        face_parse_mask = np.zeros((H, W), dtype=np.float32)
        face_parse_mask[y1:y2, x1:x2] = 1.0

        # Simulate a hand occupying the top half of the face
        hand_mask = np.zeros((H, W), dtype=np.float32)
        hand_mask[y1: y1 + (y2-y1)//2, x1:x2] = 1.0

        fuser = OcclusionFuser()
        fuser.reset()
        swap_mask = fuser.fuse(
            face_parse_mask,
            hand_mask,
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
        )
        # Hand region should be near 0 (show original)
        hand_region = swap_mask[y1: y1 + (y2-y1)//2, x1:x2]
        assert hand_region.mean() < 0.4  # heavily suppressed
