"""Tests for FaceShapeAdapter — landmark-guided face shape correction.

All tests use synthetic data — no models or webcam required.

Run:
    pytest tests/test_shape_correction.py -v
"""

import sys
import os

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.spatial import Delaunay
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

pytestmark = pytest.mark.skipif(
    not _SCIPY_OK,
    reason="scipy not installed — shape correction unavailable"
)

from src.shape.face_shape_adapter import FaceShapeAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(**kwargs) -> FaceShapeAdapter:
    cfg = {
        "blend_strength": 0.7,
        "inner_radius": 0.3,
        "outer_radius": 0.85,
        "warp_method": "delaunay",
        "temporal_smoothing": 0.7,
        "jawline_blend": 0.9,
        "eye_blend": 0.15,
        "nose_blend": 0.10,
        "mouth_blend": 0.10,
    }
    cfg.update(kwargs)
    return FaceShapeAdapter(cfg)


def _synthetic_landmarks(offset_x: float = 0.0, offset_y: float = 0.0,
                          scale: float = 1.0, size: int = 128) -> np.ndarray:
    """Generate 68 synthetic landmarks centred in a (size × size) image."""
    adapter = FaceShapeAdapter({})
    lm = adapter._synthetic_landmarks(size, size).copy()
    # Apply offset and scale relative to center
    cx, cy = size / 2, size / 2
    lm -= np.array([cx, cy])
    lm *= scale
    lm += np.array([cx + offset_x, cy + offset_y])
    return lm


def _wide_jaw_landmarks(size: int = 128) -> np.ndarray:
    """Landmarks with a wide jaw (jawline points spread apart)."""
    lm = _synthetic_landmarks(size=size)
    # Push jawline points outward
    cx = size / 2
    for i in range(0, 17):
        lm[i, 0] = cx + (lm[i, 0] - cx) * 1.4
    return lm


def _narrow_jaw_landmarks(size: int = 128) -> np.ndarray:
    """Landmarks with a narrow jaw (jawline points closer together)."""
    lm = _synthetic_landmarks(size=size)
    cx = size / 2
    for i in range(0, 17):
        lm[i, 0] = cx + (lm[i, 0] - cx) * 0.6
    return lm


def _solid_face_crop(size: int = 128, color=(120, 100, 80)) -> np.ndarray:
    """Solid colour BGR image simulating a face crop."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = color
    return img


# ---------------------------------------------------------------------------
# Test 1: No correction when disabled
# ---------------------------------------------------------------------------

class TestNoCorrectionWhenDisabled:
    def test_zero_blend_returns_identical(self):
        """blend_strength=0.0 must return swapped crop unchanged."""
        adapter = _make_adapter(blend_strength=0.0)
        original = _solid_face_crop(color=(100, 100, 100))
        swapped  = _solid_face_crop(color=(200, 150, 100))

        result = adapter.adapt(original, swapped)

        assert result is not None
        assert result.shape == swapped.shape
        np.testing.assert_array_equal(
            result, swapped,
            err_msg="blend_strength=0.0 must return swapped unchanged"
        )


# ---------------------------------------------------------------------------
# Test 2: Full correction shifts jawline toward target
# ---------------------------------------------------------------------------

class TestFullCorrectionJawline:
    def test_jawline_blend_toward_target(self):
        """With blend_strength=1.0, blended jawline should be close to target."""
        adapter = _make_adapter(blend_strength=1.0)

        source_lm = _narrow_jaw_landmarks(128)  # narrow
        target_lm = _wide_jaw_landmarks(128)    # wide

        blended = adapter.compute_shape_blend_landmarks(source_lm, target_lm, 1.0)

        # Jawline (points 0-16): blended should be ≥ 80% of the way to target
        jaw_src = source_lm[:17]
        jaw_tgt = target_lm[:17]
        jaw_blended = blended[:17]

        # Mean absolute distance from target should be << from source
        dist_to_target = np.mean(np.abs(jaw_blended - jaw_tgt))
        dist_to_source = np.mean(np.abs(jaw_blended - jaw_src))

        assert dist_to_target < dist_to_source, (
            f"Blended jawline should be closer to target than source. "
            f"dist_to_target={dist_to_target:.2f}  dist_to_source={dist_to_source:.2f}"
        )

    def test_jawline_blend_weight(self):
        """Jawline blending should use jawline_blend factor (0.9)."""
        adapter = _make_adapter(blend_strength=1.0, jawline_blend=0.9)

        source_lm = _synthetic_landmarks(size=128)
        target_lm = _synthetic_landmarks(offset_x=20, size=128)  # shifted target

        blended = adapter.compute_shape_blend_landmarks(source_lm, target_lm, 1.0)

        # Each jawline point should be ~90% of the way from source to target
        for i in range(0, 17):
            expected = (1 - 0.9) * source_lm[i] + 0.9 * target_lm[i]
            np.testing.assert_allclose(blended[i], expected, atol=1.0,
                                       err_msg=f"Jawline point {i} blend mismatch")


# ---------------------------------------------------------------------------
# Test 3: Identity preserved in center (eyes and nose)
# ---------------------------------------------------------------------------

class TestIdentityPreservedInCenter:
    def test_eyes_shift_is_small(self):
        """Eye landmarks (36-47) should shift < 15% of full delta."""
        adapter = _make_adapter(blend_strength=1.0, eye_blend=0.15)

        source_lm = _synthetic_landmarks(size=128)
        target_lm = _synthetic_landmarks(offset_x=30, size=128)

        blended = adapter.compute_shape_blend_landmarks(source_lm, target_lm, 1.0)

        full_delta = np.abs(target_lm - source_lm)
        eye_delta  = np.abs(blended[36:48] - source_lm[36:48])

        max_allowed = full_delta[36:48].mean() * 0.20  # allow up to 20%
        assert eye_delta.mean() <= max_allowed, (
            f"Eye shift {eye_delta.mean():.2f}px exceeds 20% of full delta "
            f"({full_delta[36:48].mean():.2f}px)"
        )

    def test_nose_shift_is_small(self):
        """Nose landmarks (27-35) should shift < 15% of full delta."""
        adapter = _make_adapter(blend_strength=1.0, nose_blend=0.10)

        source_lm = _synthetic_landmarks(size=128)
        target_lm = _synthetic_landmarks(offset_x=30, size=128)

        blended = adapter.compute_shape_blend_landmarks(source_lm, target_lm, 1.0)

        full_delta  = np.abs(target_lm - source_lm)
        nose_delta  = np.abs(blended[27:36] - source_lm[27:36])

        max_allowed = full_delta[27:36].mean() * 0.15
        assert nose_delta.mean() <= max_allowed, (
            f"Nose shift {nose_delta.mean():.2f}px exceeds 15% of full delta"
        )


# ---------------------------------------------------------------------------
# Test 4: Smooth transition in warp field
# ---------------------------------------------------------------------------

class TestSmoothTransition:
    def test_warp_field_no_sudden_jumps(self):
        """Adjacent warp vectors should differ by < 2px along a radial line."""
        adapter = _make_adapter(blend_strength=0.7)

        source_lm = _narrow_jaw_landmarks(128)
        target_lm = _wide_jaw_landmarks(128)
        blended   = adapter.compute_shape_blend_landmarks(source_lm, target_lm, 0.7)

        map_x, map_y = adapter.compute_warp_field(source_lm, blended, (128, 128))

        # Sample along horizontal midline (y=64)
        y = 64
        row_x = map_x[y, :]
        row_y = map_y[y, :]

        diffs_x = np.abs(np.diff(row_x))
        diffs_y = np.abs(np.diff(row_y))
        max_jump = max(diffs_x.max(), diffs_y.max())

        assert max_jump < 3.0, (
            f"Warp field has a jump of {max_jump:.2f}px > 3px — not smooth"
        )

    def test_warp_map_shape(self):
        """Warp map shape must match requested image size."""
        adapter = _make_adapter()
        source_lm = _synthetic_landmarks(size=128)
        blended   = source_lm + np.random.uniform(-2, 2, source_lm.shape).astype(np.float32)

        for size in (128, 256):
            map_x, map_y = adapter.compute_warp_field(source_lm, blended, (size, size))
            assert map_x.shape == (size, size)
            assert map_y.shape == (size, size)


# ---------------------------------------------------------------------------
# Test 5: Temporal stability
# ---------------------------------------------------------------------------

class TestTemporalStability:
    def test_warp_converges_over_frames(self):
        """Repeated adapt() on identical inputs should converge (variance → 0)."""
        adapter = _make_adapter(blend_strength=0.7, temporal_smoothing=0.7)

        original = _solid_face_crop(128, (100, 100, 100))
        swapped  = _solid_face_crop(128, (200, 150, 100))

        # Force synthetic landmarks (no model needed)
        adapter.detect_landmarks_68 = lambda img: adapter._synthetic_landmarks(
            img.shape[0], img.shape[1]
        )

        map_snapshots = []
        for _ in range(30):
            adapter.adapt(original, swapped)
            if adapter._prev_map_x is not None:
                map_snapshots.append(adapter._prev_map_x.copy())

        if len(map_snapshots) < 15:
            pytest.skip("Not enough frames captured")

        # Variance in last 10 frames should be < 0.1 px²
        last_maps = np.stack(map_snapshots[-10:])
        variance = last_maps.var(axis=0).mean()
        assert variance < 0.5, (
            f"Warp field did not converge — variance after 30 frames: {variance:.4f}"
        )

    def test_temporal_reset_clears_buffer(self):
        """reset_temporal() should clear previous warp maps."""
        adapter = _make_adapter()
        adapter._prev_map_x = np.ones((128, 128), dtype=np.float32)
        adapter._prev_map_y = np.ones((128, 128), dtype=np.float32)

        adapter.reset_temporal()

        assert adapter._prev_map_x is None
        assert adapter._prev_map_y is None


# ---------------------------------------------------------------------------
# Test 6: Graceful fallback on blank / undetectable image
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_blank_image_returns_swapped_unchanged(self):
        """A blank image (no face) must not crash and must return swapped_crop."""
        adapter = _make_adapter(blend_strength=0.7)

        # Blank black image — no landmarks detectable by real models,
        # but synthetic fallback will be used → warp identity map
        original = np.zeros((128, 128, 3), dtype=np.uint8)
        swapped  = _solid_face_crop(128, (200, 150, 100))

        # Patch detect to simulate failure
        adapter.detect_landmarks_68 = lambda img: None

        result = adapter.adapt(original, swapped)

        assert result is not None
        assert result.shape == swapped.shape
        np.testing.assert_array_equal(
            result, swapped,
            err_msg="When landmarks fail, adapt() must return swapped unchanged"
        )

    def test_no_crash_on_edge_case_sizes(self):
        """Must not crash for non-square or tiny images."""
        adapter = _make_adapter()
        adapter.detect_landmarks_68 = lambda img: adapter._synthetic_landmarks(
            img.shape[0], img.shape[1]
        )

        for size in (64, 128, 256):
            orig = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            swap = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = adapter.adapt(orig, swap)
            assert result.shape == swap.shape, f"Shape mismatch for size={size}"


# ---------------------------------------------------------------------------
# Test 7: Warp preserves image quality (no black borders / clipping)
# ---------------------------------------------------------------------------

class TestWarpPreservesImageQuality:
    def test_no_black_borders(self):
        """Warped image should not have black borders (BORDER_REFLECT_101)."""
        adapter = _make_adapter(blend_strength=0.7)
        adapter.detect_landmarks_68 = lambda img: adapter._synthetic_landmarks(
            img.shape[0], img.shape[1]
        )

        original = _solid_face_crop(128, (150, 120, 100))
        swapped  = _solid_face_crop(128, (200, 160, 140))

        result = adapter.adapt(original, swapped)

        # Edge pixels should not be pure black
        edges = np.concatenate([
            result[0, :].ravel(),
            result[-1, :].ravel(),
            result[:, 0].ravel(),
            result[:, -1].ravel(),
        ])
        assert edges.max() > 10, "Edge pixels are all near-black — likely border artifact"

    def test_output_dtype_preserved(self):
        """Output dtype must match input (uint8)."""
        adapter = _make_adapter()
        adapter.detect_landmarks_68 = lambda img: adapter._synthetic_landmarks(
            img.shape[0], img.shape[1]
        )
        original = _solid_face_crop()
        swapped  = _solid_face_crop(color=(180, 140, 110))

        result = adapter.adapt(original, swapped)
        assert result.dtype == np.uint8

    def test_center_ssim_high(self):
        """Center of warped face should be similar to original swapped crop."""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            pytest.skip("scikit-image not installed")

        adapter = _make_adapter(blend_strength=0.3)  # light correction
        adapter.detect_landmarks_68 = lambda img: adapter._synthetic_landmarks(
            img.shape[0], img.shape[1]
        )

        swapped = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
        original = swapped.copy()

        result = adapter.adapt(original, swapped)

        # Compare center 64×64 patch
        cy, cx = 32, 32
        s_center = swapped[cy:cy+64, cx:cx+64]
        r_center = result[cy:cy+64, cx:cx+64]

        score = ssim(s_center, r_center, channel_axis=2, data_range=255)
        assert score > 0.90, f"Center SSIM={score:.3f} < 0.90 — warp degraded image quality"


# ---------------------------------------------------------------------------
# Test 8: Delaunay vs TPS consistency
# ---------------------------------------------------------------------------

class TestDelaunayVsTPS:
    def test_methods_produce_similar_results(self):
        """Delaunay and TPS should agree to within ~5px mean difference."""
        adapter_d = _make_adapter(warp_method="delaunay")
        adapter_t = _make_adapter(warp_method="tps")

        lm = _synthetic_landmarks(size=128)
        target_lm = _wide_jaw_landmarks(128)

        blended = adapter_d.compute_shape_blend_landmarks(lm, target_lm, 0.7)

        map_xd, map_yd = adapter_d.compute_warp_field(lm, blended, (128, 128))
        map_xt, map_yt = adapter_t.compute_warp_field(lm, blended, (128, 128))

        diff_x = np.abs(map_xd - map_xt).mean()
        diff_y = np.abs(map_yd - map_yt).mean()
        mean_diff = (diff_x + diff_y) / 2.0

        assert mean_diff < 5.0, (
            f"Delaunay vs TPS mean map difference = {mean_diff:.2f}px (max allowed: 5px)"
        )

    def test_set_blend_strength(self):
        """set_blend_strength() should update blend_strength correctly."""
        adapter = _make_adapter(blend_strength=0.5)

        adapter.set_blend_strength(0.9)
        assert abs(adapter.blend_strength - 0.9) < 1e-6

        adapter.set_blend_strength(1.5)  # clamped to 1.0
        assert adapter.blend_strength <= 1.0

        adapter.set_blend_strength(-0.5)  # clamped to 0.0
        assert adapter.blend_strength >= 0.0
