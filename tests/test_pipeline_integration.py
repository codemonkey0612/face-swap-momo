"""Integration tests for the full pipeline.

These tests use synthetic frames (no real webcam/models needed) to verify
pipeline structure and graceful error handling.

Run with: pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2


def _blank_frame(h=480, w=640, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestOcclusionFuserIntegration:
    """Full occlusion pipeline integration."""

    def test_all_zeros_no_occlusion(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        from src.occlusion.mask_refiner import MaskRefiner
        fuser = OcclusionFuser()
        fuser.reset()
        refiner = MaskRefiner(erosion_px=0, feather_kernel=7)
        H, W = 480, 640
        face_mask = np.ones((H, W), dtype=np.float32)
        zeros = np.zeros((H, W), dtype=np.float32)
        swap_mask = fuser.fuse(face_mask, zeros, zeros, zeros, zeros)
        refined = refiner.refine_and_smooth(swap_mask)
        assert refined.shape == (H, W)
        assert refined.dtype == np.float32
        assert refined.mean() > 0.3  # face should be mostly swappable

    def test_full_hand_occlusion(self):
        from src.occlusion.occlusion_fuser import OcclusionFuser
        from src.occlusion.mask_refiner import MaskRefiner
        fuser = OcclusionFuser()
        fuser.reset()
        refiner = MaskRefiner(erosion_px=0, feather_kernel=7)
        H, W = 480, 640
        face_mask = np.ones((H, W), dtype=np.float32)
        hand_mask = np.ones((H, W), dtype=np.float32)  # entire frame is "hand"
        zeros = np.zeros((H, W), dtype=np.float32)
        swap_mask = fuser.fuse(face_mask, hand_mask, zeros, zeros, zeros)
        refined = refiner.refine_and_smooth(swap_mask)
        assert refined.mean() < 0.4  # heavily suppressed by hand


class TestCompositing:
    def test_alpha_blend_output_shape(self):
        from src.compositing.blender import FrameCompositor
        comp = FrameCompositor(use_poisson=False)
        H, W = 480, 640
        orig    = _blank_frame(H, W, val=50)
        swapped = _blank_frame(H, W, val=200)
        mask_u8 = np.full((H, W), 128, dtype=np.uint8)
        result  = comp.alpha_blend(orig, swapped, mask_u8)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_mask_zero_shows_original(self):
        from src.compositing.blender import FrameCompositor
        comp = FrameCompositor(use_poisson=False)
        H, W = 480, 640
        orig    = _blank_frame(H, W, val=50)
        swapped = _blank_frame(H, W, val=200)
        mask_u8 = np.zeros((H, W), dtype=np.uint8)
        result  = comp.alpha_blend(orig, swapped, mask_u8)
        np.testing.assert_array_equal(result, orig)

    def test_output_dimensions_match_input(self):
        from src.compositing.blender import FrameCompositor
        comp = FrameCompositor(use_poisson=False)
        for h, w in [(360, 480), (480, 640), (720, 1280)]:
            orig    = _blank_frame(h, w)
            swapped = _blank_frame(h, w, val=200)
            # mask with active region in centre
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2
            mask_u8[cy-50:cy+50, cx-50:cx+50] = 255
            result  = comp.alpha_blend(orig, swapped, mask_u8)
            assert result.shape == (h, w, 3)


class TestTemporalModules:
    def test_landmark_stabilizer(self):
        from src.temporal.stabilizer import LandmarkStabilizer
        stab = LandmarkStabilizer()
        lm = np.array([[100., 100.], [200., 100.], [150., 150.], [120., 200.], [180., 200.]])
        s1 = stab.update([lm])
        assert len(s1) == 1
        assert s1[0].shape == lm.shape
        # Second call returns smoothed version
        lm2 = lm + 20.0
        s2 = stab.update([lm2])
        assert len(s2) == 1

    def test_frame_blender(self):
        from src.temporal.frame_blender import FrameBlender
        blender = FrameBlender(alpha=0.85)
        H, W = 480, 640
        frame = _blank_frame(H, W, 100)
        mask = np.ones((H, W), dtype=np.float32)
        r1 = blender.blend(frame, mask)
        assert r1.shape == frame.shape
        r2 = blender.blend(_blank_frame(H, W, 150), mask)
        assert r2.shape == frame.shape

    def test_scene_cut_detection(self):
        from src.temporal.frame_blender import FrameBlender
        blender = FrameBlender()
        a = _blank_frame(val=10)
        b = _blank_frame(val=220)
        assert blender.detect_scene_cut(a, b, threshold=30)
        c = _blank_frame(val=12)
        assert not blender.detect_scene_cut(a, c, threshold=30)
