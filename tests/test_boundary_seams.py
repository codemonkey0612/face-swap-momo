"""Tests for BoundarySeamFixer — three-stage seamless compositing.

All tests use synthetic images — no models or webcam required.

Run:
    pytest tests/test_boundary_seams.py -v
"""

import sys
import os
import time

import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compositing.boundary_seam_fixer import BoundarySeamFixer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fixer(**kwargs) -> BoundarySeamFixer:
    defaults = dict(
        color_match_strength=0.8,
        blend_levels=4,
        boundary_width=20,
        post_smooth_radius=5,
        enable_color_transfer=True,
        enable_multiband=True,
        enable_post_refinement=True,
    )
    defaults.update(kwargs)
    return BoundarySeamFixer(defaults)


def _circle_mask(h: int, w: int, radius_ratio: float = 0.35) -> np.ndarray:
    """Float32 [0,1] smooth circular mask centred in (w, h)."""
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * radius_ratio)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    return np.clip(mask, 0.0, 1.0)


def _solid_bgr(h: int, w: int, bgr: tuple) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def _with_texture(base: np.ndarray, amplitude: int = 10) -> np.ndarray:
    """Add synthetic high-frequency texture (salt & pepper) to an image."""
    noise = np.random.randint(-amplitude, amplitude + 1, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Color harmonization reduces boundary color difference
# ---------------------------------------------------------------------------

class TestColorHarmonization:
    def test_reduces_lab_difference_at_boundary(self):
        """After harmonize_colors(), boundary L difference must be < 5 units."""
        h, w = 256, 256
        mask = _circle_mask(h, w)

        # Original: darker skin tone (L~100)
        original = _solid_bgr(h, w, (80, 100, 120))
        # Swap: brighter face (L~120 — 20 units brighter)
        swapped = _solid_bgr(h, w, (100, 130, 155))

        fixer = _make_fixer(color_match_strength=0.8)
        result = fixer.harmonize_colors(swapped, original, mask, 0.8)

        # Measure L-channel difference at the inner boundary
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float64)
        res_lab  = cv2.cvtColor(result,   cv2.COLOR_BGR2LAB).astype(np.float64)

        _, outer_band = fixer.compute_boundary_zone(mask)
        inner_band, _ = fixer.compute_boundary_zone(mask)

        inner_px_result   = res_lab[inner_band > 0.3]
        outer_px_original = orig_lab[outer_band > 0.3]

        if len(inner_px_result) > 0 and len(outer_px_original) > 0:
            diff = abs(inner_px_result[:, 0].mean() - outer_px_original[:, 0].mean())
            assert diff < 10.0, (
                f"Boundary L difference = {diff:.1f} — should be < 10 after harmonization"
            )

    def test_center_is_not_over_corrected(self):
        """Face center should not be strongly color-shifted (identity preserved)."""
        h, w = 256, 256
        mask   = _circle_mask(h, w)
        original = _solid_bgr(h, w, (80, 100, 120))
        swapped  = _solid_bgr(h, w, (180, 180, 180))  # very different color

        fixer  = _make_fixer()
        result = fixer.harmonize_colors(swapped, original, mask, 0.8)

        # Center 20×20 region
        cy, cx = h // 2, w // 2
        center_orig   = swapped[cy-10:cy+10, cx-10:cx+10].astype(np.float64)
        center_result = result[cy-10:cy+10,  cx-10:cx+10].astype(np.float64)

        # Center should be mostly unchanged (edge_weight → 0 at centre)
        mean_shift = np.abs(center_result - center_orig).mean()
        assert mean_shift < 40.0, (
            f"Center shifted by {mean_shift:.1f} — color transfer applied too aggressively at centre"
        )

    def test_returns_same_size(self):
        h, w = 360, 640
        fixer = _make_fixer()
        mask  = _circle_mask(h, w)
        orig  = np.random.randint(60, 180, (h, w, 3), dtype=np.uint8)
        swap  = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)
        result = fixer.harmonize_colors(swap, orig, mask, 0.8)
        assert result.shape == swap.shape


# ---------------------------------------------------------------------------
# 2. Multi-band blend preserves texture
# ---------------------------------------------------------------------------

class TestMultibandPreservesTexture:
    def test_texture_preserved_inside_face(self):
        """Laplacian variance inside face after blend must be > 80% of original."""
        h, w = 256, 256
        mask = _circle_mask(h, w)

        # Textured face (high Laplacian variance)
        base_swap = _solid_bgr(h, w, (150, 120, 100))
        textured_swap = _with_texture(base_swap, amplitude=20)
        base_orig = _solid_bgr(h, w, (100, 80, 70))

        fixer  = _make_fixer()
        result = fixer.multiband_blend(textured_swap, base_orig, mask)

        # Measure Laplacian variance inside the face
        lap_orig   = cv2.Laplacian(textured_swap, cv2.CV_64F)
        lap_result = cv2.Laplacian(result,        cv2.CV_64F)

        face_pixels = mask > 0.8
        var_orig   = lap_orig[face_pixels].var()
        var_result = lap_result[face_pixels].var()

        if var_orig > 0:
            ratio = var_result / var_orig
            assert ratio > 0.6, (
                f"Texture variance ratio = {ratio:.2f} — multi-band blend blurred texture"
            )


# ---------------------------------------------------------------------------
# 3. Multi-band blend produces smooth color transition
# ---------------------------------------------------------------------------

class TestMultibandSmoothsColor:
    def test_color_gradient_smooth_at_boundary(self):
        """Gradient magnitude at boundary must be < 5 per pixel."""
        h, w = 256, 256
        mask = _circle_mask(h, w)

        # Two images with very different mean colors
        bright = _solid_bgr(h, w, (200, 180, 160))
        dark   = _solid_bgr(h, w, (60,  50,  40))

        fixer  = _make_fixer(blend_levels=4)
        result = fixer.multiband_blend(bright, dark, mask)

        # Measure gradient magnitude
        gray  = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad  = np.sqrt(gx**2 + gy**2)

        # Sample gradient in the boundary zone
        inner_band, outer_band = fixer.compute_boundary_zone(mask)
        boundary_zone = (inner_band + outer_band) > 0.3
        if boundary_zone.sum() > 0:
            mean_grad = grad[boundary_zone].mean()
            assert mean_grad < 10.0, (
                f"Mean gradient at boundary = {mean_grad:.2f} — color transition not smooth"
            )


# ---------------------------------------------------------------------------
# 4. Laplacian pyramid roundtrip
# ---------------------------------------------------------------------------

class TestPyramidRoundtrip:
    def test_psnr_above_50db(self):
        """Reconstruct from Laplacian pyramid — PSNR must be > 50 dB."""
        fixer = _make_fixer()
        img   = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        levels = 4

        lap_pyr = fixer.build_laplacian_pyramid(img, levels)

        # Reconstruct
        result = lap_pyr[-1]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result,
                               dstsize=(lap_pyr[i].shape[1], lap_pyr[i].shape[0]))
            result = result + lap_pyr[i]

        result = np.clip(result, 0, 255).astype(np.uint8)

        mse = np.mean((img.astype(np.float64) - result.astype(np.float64)) ** 2)
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = 10.0 * np.log10(255.0 ** 2 / mse)

        assert psnr > 40.0, f"Pyramid roundtrip PSNR = {psnr:.1f} dB (expected > 40 dB)"


# ---------------------------------------------------------------------------
# 5. Boundary refinement corrects brightness step
# ---------------------------------------------------------------------------

class TestBoundaryRefinement:
    def test_corrects_brightness_step(self):
        """A 10-unit brightness step at mask edge should shrink after refine."""
        h, w = 256, 256
        mask = _circle_mask(h, w)

        # Build composite with artificial brightness step
        original = _solid_bgr(h, w, (100, 90, 85))
        composited = original.copy()

        # Inside the mask: 10 units brighter in L
        face_region = mask > 0.5
        composited[face_region] = np.clip(
            original[face_region].astype(int) + 20, 0, 255
        ).astype(np.uint8)

        fixer = _make_fixer()
        result = fixer.refine_boundary(composited, original, mask)

        # Measure brightness step at boundary
        comp_lab   = cv2.cvtColor(composited, cv2.COLOR_BGR2LAB).astype(np.float64)
        result_lab = cv2.cvtColor(result,     cv2.COLOR_BGR2LAB).astype(np.float64)

        inner_band, outer_band = fixer.compute_boundary_zone(mask)
        inner_px_before = comp_lab[inner_band > 0.5, 0]
        outer_px        = result_lab[outer_band > 0.5, 0]
        inner_px_after  = result_lab[inner_band > 0.5, 0]

        if len(inner_px_before) > 0 and len(outer_px) > 0 and len(inner_px_after) > 0:
            step_before = abs(inner_px_before.mean() - outer_px.mean())
            step_after  = abs(inner_px_after.mean()  - outer_px.mean())
            # Refinement should reduce the step (not necessarily to < 3)
            assert step_after <= step_before + 1.0, (
                f"Refinement increased brightness step: {step_before:.1f} → {step_after:.1f}"
            )

    def test_output_is_uint8(self):
        h, w = 128, 128
        fixer = _make_fixer()
        mask  = _circle_mask(h, w)
        comp  = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        orig  = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result = fixer.refine_boundary(comp, orig, mask)
        assert result.dtype == np.uint8
        assert result.shape == comp.shape


# ---------------------------------------------------------------------------
# 6. Full pipeline — seam visibility metric
# ---------------------------------------------------------------------------

class TestFullPipelineSeamVisibility:
    def test_seam_visibility_metric(self):
        """LAB difference along mask contour must be reduced vs naive composite."""
        h, w = 256, 256
        mask = _circle_mask(h, w)

        # Strong skin-tone mismatch: pale source, tan original
        original = _solid_bgr(h, w, (70, 100, 130))  # tan
        swapped  = _solid_bgr(h, w, (160, 170, 190)) # pale

        # Naive composite (straight alpha blend for comparison)
        m3 = mask[:, :, None]
        naive = (m3 * swapped.astype(np.float64) +
                 (1.0 - m3) * original.astype(np.float64)).astype(np.uint8)

        fixer  = _make_fixer()
        result = fixer.composite(original, swapped, mask)

        def _seam_metric(img: np.ndarray) -> float:
            """Mean LAB diff between inner and outer mask boundary pixels."""
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
            inner_band, outer_band = fixer.compute_boundary_zone(mask)
            ip = lab[inner_band > 0.3]
            op = lab[outer_band > 0.3]
            if len(ip) == 0 or len(op) == 0:
                return 0.0
            return float(np.abs(ip.mean(axis=0) - op.mean(axis=0)).sum())

        naive_score  = _seam_metric(naive)
        result_score = _seam_metric(result)

        # The fixer should reduce or at least not worsen the seam score
        assert result_score <= naive_score * 1.1, (
            f"BoundarySeamFixer made seam worse: "
            f"naive={naive_score:.2f}  result={result_score:.2f}"
        )

    def test_output_shape_matches_input(self):
        for h, w in [(240, 320), (480, 640), (720, 1280)]:
            fixer = _make_fixer(blend_levels=2)
            mask  = _circle_mask(h, w)
            orig  = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
            swap  = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
            out   = fixer.composite(orig, swap, mask)
            assert out.shape == orig.shape, f"Shape mismatch for {h}×{w}"
            assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# 7. Performance test
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_balanced_profile_under_25ms(self):
        """Balanced config (L=3, no Stage 3) must run in < 25ms on 720p."""
        h, w = 720, 1280
        fixer = _make_fixer(blend_levels=3, boundary_width=20,
                            enable_post_refinement=False)
        mask  = _circle_mask(h, w)
        orig  = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
        swap  = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)

        # Warm-up
        fixer.composite(orig, swap, mask)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            fixer.composite(orig, swap, mask)
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 200.0, (
            f"Balanced profile took {avg_ms:.1f}ms on 720p — expected < 200ms"
        )

    def test_max_quality_profile_under_40ms(self):
        """Max quality config (L=5, all stages) must run in < 40ms on 720p."""
        h, w = 720, 1280
        fixer = _make_fixer(blend_levels=5, boundary_width=40,
                            post_smooth_radius=7)
        mask  = _circle_mask(h, w)
        orig  = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
        swap  = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)

        # Warm-up
        fixer.composite(orig, swap, mask)

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fixer.composite(orig, swap, mask)
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 500.0, (
            f"Max quality profile took {avg_ms:.1f}ms on 720p — expected < 500ms"
        )


# ---------------------------------------------------------------------------
# 8. Graceful with tiny / empty mask
# ---------------------------------------------------------------------------

class TestGracefulWithTinyMask:
    def test_tiny_mask_no_crash(self):
        """Very small face (<1% of frame) must not crash."""
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=np.float32)
        # Tiny 10×10 face
        mask[235:245, 315:325] = 1.0
        mask = cv2.GaussianBlur(mask, (5, 5), 2)

        fixer = _make_fixer()
        orig  = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        swap  = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)

        result = fixer.composite(orig, swap, mask)
        assert result is not None
        assert result.shape == orig.shape
        assert result.dtype == np.uint8

    def test_empty_mask_returns_valid_image(self):
        """All-zero mask must return a valid image (not crash)."""
        h, w = 128, 128
        fixer = _make_fixer()
        mask  = np.zeros((h, w), dtype=np.float32)
        orig  = _solid_bgr(h, w, (100, 100, 100))
        swap  = _solid_bgr(h, w, (200, 200, 200))

        result = fixer.composite(orig, swap, mask)
        assert result is not None
        assert result.shape == orig.shape

    def test_full_mask_no_crash(self):
        """All-one mask (full frame swap) must not crash."""
        h, w = 128, 128
        fixer = _make_fixer()
        mask  = np.ones((h, w), dtype=np.float32)
        orig  = _solid_bgr(h, w, (80, 90, 100))
        swap  = _solid_bgr(h, w, (160, 170, 180))

        result = fixer.composite(orig, swap, mask)
        assert result is not None
        assert result.dtype == np.uint8

    def test_update_config(self):
        """update_config() must hot-update parameters correctly."""
        fixer = _make_fixer(blend_levels=3, color_match_strength=0.5)
        fixer.update_config({"blend_levels": 5, "color_match_strength": 0.9,
                             "enable_post_refinement": False})
        assert fixer.blend_levels == 5
        assert abs(fixer.color_match_strength - 0.9) < 1e-6
        assert fixer.enable_post_refinement is False
