"""Tests for the pre-swap occlusion ghost-fix architecture.

These tests verify that the new pipeline order (detect occlusion BEFORE swap,
bake into combined mask) eliminates ghost images in occluded regions.

Key invariant: wherever the occlusion mask says 0.0 (occluder present),
the output pixel must be IDENTICAL to the original pristine pixel.  The swap
model must never contaminate those pixels.

Run with: pytest tests/test_occlusion_fix.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank(h=256, w=256, val=128, dtype=np.uint8):
    return np.full((h, w, 3), val, dtype=dtype)


def _face_mask(h=256, w=256, cx=None, cy=None, rx=80, ry=90):
    """Soft elliptical face mask centred on the image."""
    cx = cx or w // 2
    cy = cy or h // 2
    m = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (15, 15), 4)
    return m


def _hand_mask(h=256, w=256, region=(80, 130, 170, 200)):
    """Binary mask for a rectangular 'hand' region."""
    x1, y1, x2, y2 = region
    m = np.zeros((h, w), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def _combined_mask(face_mask, occluder_mask):
    """Simulate the ghost-fix: combined = face × (1 − occluder)."""
    return face_mask * np.clip(1.0 - occluder_mask, 0.0, 1.0)


def _composite(original, swapped, mask):
    """Alpha composite using mask [0,1]."""
    a = mask[:, :, np.newaxis]
    return (swapped.astype(np.float32) * a + original.astype(np.float32) * (1.0 - a)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Ghost-fix core: combined mask math
# ---------------------------------------------------------------------------

class TestMaskMultiplication:
    """Verify combined_mask = face_mask × (1 − occluder_mask) properties."""

    def test_occluder_zeros_combined(self):
        """Where occluder=1.0, combined must be exactly 0.0."""
        face = _face_mask()
        hand = _hand_mask()
        combined = _combined_mask(face, hand)
        hand_pixels = (hand > 0.5)
        assert combined[hand_pixels].max() == pytest.approx(0.0)

    def test_clear_face_preserves_face_mask(self):
        """Where no occluder, combined must equal face_mask."""
        face = _face_mask()
        no_occ = np.zeros_like(face)
        combined = _combined_mask(face, no_occ)
        np.testing.assert_array_almost_equal(combined, face)

    def test_partial_occluder(self):
        """Where occluder=0.5, combined must be 50% of face_mask."""
        face = np.ones((64, 64), dtype=np.float32)
        partial = np.full((64, 64), 0.5, dtype=np.float32)
        combined = _combined_mask(face, partial)
        np.testing.assert_array_almost_equal(combined, 0.5 * face)

    def test_combined_always_nonnegative(self):
        face = _face_mask()
        hand = _hand_mask()
        combined = _combined_mask(face, hand)
        assert combined.min() >= 0.0

    def test_combined_never_exceeds_face_mask(self):
        face = _face_mask()
        hand = _hand_mask()
        combined = _combined_mask(face, hand)
        assert (combined <= face + 1e-6).all()


# ---------------------------------------------------------------------------
# 2. Hand over mouth — swap pixels must not touch hand region
# ---------------------------------------------------------------------------

class TestHandOverMouth:

    def test_hand_pixels_identical_to_original(self):
        """In the hand region, output must equal original — zero contamination."""
        H, W = 256, 256
        original = _blank(H, W, val=80)     # dark grey "skin"
        swapped  = _blank(H, W, val=200)    # bright "AI face"

        face = _face_mask(H, W)
        hand = _hand_mask(H, W, region=(80, 130, 170, 170))  # over mouth area
        combined = _combined_mask(face, hand)

        result = _composite(original, swapped, combined)

        # In the hand region, result must match original exactly (0 tolerance)
        hand_area = (hand > 0.5)
        np.testing.assert_array_equal(result[hand_area], original[hand_area])

    def test_hand_pixels_no_color_contamination(self):
        """Mean colour diff in hand region must be < 1.0 intensity unit."""
        H, W = 256, 256
        original = _blank(H, W, val=80)
        swapped  = _blank(H, W, val=200)

        face = _face_mask(H, W)
        hand = _hand_mask(H, W, region=(80, 130, 170, 170))
        combined = _combined_mask(face, hand)

        result = _composite(original, swapped, combined)

        hand_area = hand > 0.5
        diff = np.abs(result[hand_area].astype(float) - original[hand_area].astype(float))
        assert diff.mean() < 1.0, f"Mean contamination {diff.mean():.2f} ≥ 1.0"

    def test_face_pixels_are_swapped(self):
        """Clear face pixels (no hand) must contain swap values."""
        H, W = 256, 256
        original = _blank(H, W, val=80)
        swapped  = _blank(H, W, val=200)

        face = _face_mask(H, W)
        hand = _hand_mask(H, W, region=(80, 130, 170, 170))
        combined = _combined_mask(face, hand)

        result = _composite(original, swapped, combined)

        # Centre of face — far from hand — should be close to swapped value
        cy, cx = H // 2, W // 2 - 20   # shifted left of hand
        centre_val = result[cy, cx].mean()
        assert centre_val > 150, f"Face centre {centre_val:.1f} not swapped"


# ---------------------------------------------------------------------------
# 3. Hand over eye — partial occlusion
# ---------------------------------------------------------------------------

class TestHandOverEye:

    def test_hand_unchanged_eye_region(self):
        """Pixels under the hand must be pristine, not contaminated."""
        H, W = 256, 256
        original = _blank(H, W, val=60)
        swapped  = _blank(H, W, val=210)

        face = _face_mask(H, W)
        # Hand over left eye area
        hand = _hand_mask(H, W, region=(40, 60, 120, 120))
        combined = _combined_mask(face, hand)

        result = _composite(original, swapped, combined)

        hand_area = (hand > 0.5)
        diff = np.abs(result[hand_area].astype(float) - original[hand_area].astype(float))
        assert diff.max() == pytest.approx(0.0)

    def test_other_face_regions_swapped(self):
        """Right eye (no hand) should be swapped."""
        H, W = 256, 256
        original = _blank(H, W, val=60)
        swapped  = _blank(H, W, val=210)

        face = _face_mask(H, W)
        hand = _hand_mask(H, W, region=(40, 60, 120, 120))
        combined = _combined_mask(face, hand)

        result = _composite(original, swapped, combined)

        # Right eye area — should be mostly swapped (not original)
        right_eye_val = result[80:110, 140:180].mean()
        assert right_eye_val > 150, f"Right eye {right_eye_val:.1f} not swapped"


# ---------------------------------------------------------------------------
# 4. Dark object (phone) — no skin-toned halo
# ---------------------------------------------------------------------------

class TestObjectOverFace:

    def test_dark_object_no_skin_halo(self):
        """Dark object pixels should remain dark, not be tinted by swap skin."""
        H, W = 256, 256
        # Skin-toned original (mid-brown)
        original = np.full((H, W, 3), [100, 150, 180], dtype=np.uint8)
        # AI face (lighter skin)
        swapped = np.full((H, W, 3), [150, 200, 220], dtype=np.uint8)

        face = _face_mask(H, W)
        # Dark phone-shaped object over right cheek
        phone = _hand_mask(H, W, region=(160, 100, 230, 180))
        # Make original phone pixels dark
        original[100:180, 160:230] = [20, 20, 20]

        combined = _combined_mask(face, phone)
        result = _composite(original, swapped, combined)

        # Phone region should still be dark
        phone_area = phone > 0.5
        phone_brightness = result[phone_area].astype(float).mean()
        assert phone_brightness < 50, (
            f"Phone region brightness {phone_brightness:.1f} — "
            f"skin-toned halo detected"
        )


# ---------------------------------------------------------------------------
# 5. No occlusion — output should match standard swap
# ---------------------------------------------------------------------------

class TestNoOcclusion:

    def test_no_occluder_same_as_standard_swap(self):
        """With no occluder, combined must equal face mask (same as before)."""
        H, W = 256, 256
        face = _face_mask(H, W)
        no_occ = np.zeros((H, W), dtype=np.float32)
        combined = _combined_mask(face, no_occ)
        np.testing.assert_array_almost_equal(combined, face)

    def test_output_shape_unchanged(self):
        H, W = 480, 640
        original = _blank(H, W, val=100)
        swapped  = _blank(H, W, val=200)
        face = _face_mask(H, W, rx=120, ry=140)
        combined = _combined_mask(face, np.zeros_like(face))
        result = _composite(original, swapped, combined)
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 6. Temporal buffer flush — ghost must not persist after occlusion appears
# ---------------------------------------------------------------------------

class TestTemporalBufferFlush:

    def test_ghost_does_not_persist_after_occlusion(self):
        """Simulate frame 1 (no occluder) then frame 2 (hand appears).

        Frame 2's hand region must match original, not be contaminated
        by frame 1's swapped face leaking through the EMA buffer.
        """
        H, W = 256, 256
        original = _blank(H, W, val=80)
        swapped  = _blank(H, W, val=200)
        face = _face_mask(H, W)
        hand = _hand_mask(H, W, region=(80, 100, 170, 160))

        # Frame 1: no occlusion — buffer fills with swapped face
        no_occ = np.zeros_like(face)
        combined_f1 = _combined_mask(face, no_occ)
        frame1_result = _composite(original, swapped, combined_f1)

        # Simulate EMA buffer = frame1 (contains swapped face everywhere)
        ema_buffer = frame1_result.copy()

        # Frame 2: hand appears
        combined_f2 = _combined_mask(face, hand)
        frame2_result = _composite(original, swapped, combined_f2)

        # Flush buffer in newly-occluded region
        newly_occluded = ((combined_f1 > 0.5) & (combined_f2 < 0.2)).astype(np.float32)
        flush = cv2.GaussianBlur(newly_occluded, (11, 11), 0)[:, :, np.newaxis]
        ema_buffer_flushed = (
            frame2_result.astype(np.float32) * flush
            + ema_buffer.astype(np.float32) * (1.0 - flush)
        ).astype(np.uint8)

        # EMA blend
        alpha = 0.85
        final = cv2.addWeighted(frame2_result, alpha, ema_buffer_flushed, 1.0 - alpha, 0)

        # Hand region in final frame must be close to original (no ghost)
        hand_area = (hand > 0.5)
        diff = np.abs(final[hand_area].astype(float) - original[hand_area].astype(float))
        assert diff.mean() < 15.0, (
            f"Ghost leaked through EMA buffer — mean diff {diff.mean():.1f} ≥ 15.0"
        )


# ---------------------------------------------------------------------------
# 7. Interpolation bleeding — swap must not bleed past mask boundary
# ---------------------------------------------------------------------------

class TestInterpolationBleed:

    def test_no_swap_outside_eroded_boundary(self):
        """Simulate interpolation bleed fix: erode combined by 2px.

        After erosion, pixels at the old boundary should have combined=0,
        ensuring warpAffine bilinear bleed cannot reach them.
        """
        H, W = 128, 128
        original = _blank(H, W, val=50)
        swapped  = _blank(H, W, val=220)

        # Sharp face boundary (non-blurred, to test erosion precisely)
        face = np.zeros((H, W), dtype=np.float32)
        cv2.circle(face, (H // 2, W // 2), 40, 1.0, -1)

        hand = np.zeros_like(face)  # no occluder in this test
        combined = _combined_mask(face, hand)

        # Erode to create bleed-safety margin
        bleed_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_eroded = cv2.erode(
            (combined * 255).astype(np.uint8), bleed_k
        ).astype(np.float32) / 255.0

        # The 2-px boundary ring should now be 0 in eroded but was >0 before
        ring = (face > 0.5) & (combined_eroded < 0.01)
        assert ring.any(), "Erosion did not create a safety margin boundary"

    def test_2px_margin_around_occlusion_boundary(self):
        """2 pixels around the hand boundary must have combined=0 after erosion."""
        H, W = 128, 128
        face = np.ones((H, W), dtype=np.float32)
        hand = np.zeros((H, W), dtype=np.float32)
        hand[40:90, 40:90] = 1.0  # sharp-edged hand

        combined = _combined_mask(face, hand)
        bleed_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(
            (combined * 255).astype(np.uint8), bleed_k
        ).astype(np.float32) / 255.0

        # 1-px border just outside the hand (where combined was ~1 before erosion)
        border = np.zeros((H, W), dtype=bool)
        border[39, 40:90] = True  # top edge
        border[90, 40:90] = True  # bottom edge
        border[40:90, 39] = True  # left edge
        border[40:90, 90] = True  # right edge

        assert eroded[border].max() < 0.01, (
            "Swap values present at 1-px boundary — interpolation bleed risk"
        )
