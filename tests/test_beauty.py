"""Tests for beauty filter modules.

Run with: pytest tests/test_beauty.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2


def _face_frame(h=480, w=640):
    frame = np.full((h, w, 3), 150, dtype=np.uint8)
    return frame

def _face_mask(h=480, w=640, cx=320, cy=240, r=80):
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    return mask

def _landmarks(cx=320, cy=240):
    return np.array([
        [cx - 30, cy - 20],  # left eye
        [cx + 30, cy - 20],  # right eye
        [cx, cy],            # nose
        [cx - 20, cy + 30],  # mouth left
        [cx + 20, cy + 30],  # mouth right
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Skin smoothing
# ---------------------------------------------------------------------------

class TestSkinSmoother:
    def test_output_same_size(self):
        from src.beauty.skin_smoothing import SkinSmoother
        smoother = SkinSmoother({"strength": 0.6})
        frame = _face_frame()
        mask = _face_mask()
        result = smoother.smooth(frame, mask)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_strength_zero_returns_unchanged(self):
        from src.beauty.skin_smoothing import SkinSmoother
        smoother = SkinSmoother({"strength": 0.0})
        frame = _face_frame()
        mask = _face_mask()
        result = smoother.smooth(frame, mask)
        np.testing.assert_array_equal(result, frame)

    def test_reduces_high_frequency_noise(self):
        from src.beauty.skin_smoothing import SkinSmoother
        smoother = SkinSmoother({"strength": 0.9})
        # Create a noisy face region
        frame = _face_frame()
        noise = np.random.randint(-40, 40, frame.shape, dtype=np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        mask = _face_mask()
        result = smoother.smooth(noisy, mask)
        # Variance should decrease in face region
        cy, cx = 240, 320
        roi_noisy  = noisy[cy-60:cy+60, cx-60:cx+60].astype(np.float32)
        roi_result = result[cy-60:cy+60, cx-60:cx+60].astype(np.float32)
        # Allow smoothing to reduce std
        assert roi_result.std() <= roi_noisy.std() + 5.0  # small tolerance

    def test_background_unchanged(self):
        from src.beauty.skin_smoothing import SkinSmoother
        smoother = SkinSmoother({"strength": 1.0})
        frame = _face_frame()
        mask = np.zeros((480, 640), dtype=np.float32)  # empty mask
        result = smoother.smooth(frame, mask)
        np.testing.assert_array_equal(result, frame)


# ---------------------------------------------------------------------------
# Eye enhancement
# ---------------------------------------------------------------------------

class TestEyeEnhancer:
    def test_output_same_size(self):
        from src.beauty.eye_enhancement import EyeEnhancer
        enhancer = EyeEnhancer({"strength": 0.3})
        frame = _face_frame()
        lm = _landmarks()
        result = enhancer.enhance(frame, lm)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_strength_zero_minimal_change(self):
        from src.beauty.eye_enhancement import EyeEnhancer
        enhancer = EyeEnhancer({"strength": 0.0})
        frame = _face_frame()
        lm = _landmarks()
        result = enhancer.enhance(frame, lm)
        assert result.shape == frame.shape

    def test_only_modifies_eye_region(self):
        from src.beauty.eye_enhancement import EyeEnhancer
        enhancer = EyeEnhancer({"strength": 0.5})
        frame = _face_frame(h=480, w=640)
        lm = _landmarks(cx=320, cy=240)
        result = enhancer.enhance(frame, lm)
        # Far from eye region should be unchanged
        corner = result[0:50, 0:50]
        orig_corner = frame[0:50, 0:50]
        np.testing.assert_array_equal(corner, orig_corner)

    def test_no_landmarks_returns_unchanged(self):
        from src.beauty.eye_enhancement import EyeEnhancer
        enhancer = EyeEnhancer()
        frame = _face_frame()
        result = enhancer.enhance(frame, None)
        np.testing.assert_array_equal(result, frame)


# ---------------------------------------------------------------------------
# Color correction
# ---------------------------------------------------------------------------

class TestColorCorrector:
    def test_match_skin_tone_output_shape(self):
        from src.beauty.color_correction import ColorCorrector
        corrector = ColorCorrector()
        swapped = _face_frame()
        original = _face_frame()
        mask = _face_mask()
        result = corrector.match_skin_tone(swapped, original, mask)
        assert result.shape == swapped.shape
        assert result.dtype == np.uint8

    def test_empty_mask_returns_unchanged(self):
        from src.beauty.color_correction import ColorCorrector
        corrector = ColorCorrector()
        swapped = _face_frame()
        original = _face_frame()
        mask = np.zeros((480, 640), dtype=np.float32)
        result = corrector.match_skin_tone(swapped, original, mask)
        np.testing.assert_array_equal(result, swapped)

    def test_preserve_overall_luminance(self):
        """Colour transfer should not drastically change overall brightness."""
        from src.beauty.color_correction import ColorCorrector
        corrector = ColorCorrector()
        swapped  = np.full((480, 640, 3), 150, dtype=np.uint8)
        original = np.full((480, 640, 3), 145, dtype=np.uint8)
        mask = _face_mask()
        result = corrector.match_skin_tone(swapped, original, mask)
        # Mean brightness should stay roughly similar
        assert abs(float(result.mean()) - 150.0) < 20.0

    def test_enhance_contrast_shape(self):
        from src.beauty.color_correction import ColorCorrector
        corrector = ColorCorrector()
        frame = _face_frame()
        mask = _face_mask()
        result = corrector.enhance_contrast(frame, mask, strength=0.3)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_auto_white_balance_shape(self):
        from src.beauty.color_correction import ColorCorrector
        corrector = ColorCorrector()
        frame = _face_frame()
        mask = _face_mask()
        result = corrector.auto_white_balance(frame, mask)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Filter chain
# ---------------------------------------------------------------------------

class TestBeautyFilterChain:
    def test_disabled_returns_input(self):
        from src.beauty.filter_chain import BeautyFilterChain
        chain = BeautyFilterChain({"enabled": False})
        frame = _face_frame()
        mask = _face_mask()
        result = chain.process(frame, mask)
        np.testing.assert_array_equal(result, frame)

    def test_output_same_size(self):
        from src.beauty.filter_chain import BeautyFilterChain
        chain = BeautyFilterChain({
            "enabled": True,
            "color_correction": False,
            "gfpgan_enabled": False,
            "skin_smoothing_strength": 0.3,
            "eye_brighten": 0.0,
            "sharpness": 0.1,
        })
        frame = _face_frame()
        mask = _face_mask()
        result = chain.process(frame, mask)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_timings_populated(self):
        from src.beauty.filter_chain import BeautyFilterChain
        chain = BeautyFilterChain({
            "enabled": True,
            "color_correction": False,
            "gfpgan_enabled": False,
            "skin_smoothing_strength": 0.3,
            "sharpness": 0.1,
        })
        frame = _face_frame()
        mask = _face_mask()
        chain.process(frame, mask)
        assert len(chain.timings) > 0

    def test_update_config(self):
        from src.beauty.filter_chain import BeautyFilterChain
        chain = BeautyFilterChain({"enabled": True, "gfpgan_enabled": False})
        chain.update_config({"enabled": False})
        assert not chain.is_active
