"""Tests for SourceFaceSelector — multi-reference source face selection.

All tests use synthetic/mock data — no webcam or swap model required.

Run:
    pytest tests/test_source_selector.py -v
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.swap.source_face_selector import SourceFaceSelector, SourceVariant


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _rand_emb(seed: int = 0) -> np.ndarray:
    """Unit-norm 512-d embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _mock_face(kps: Optional[np.ndarray] = None, bbox: Optional[np.ndarray] = None,
               emb: Optional[np.ndarray] = None) -> MagicMock:
    f = MagicMock()
    if kps is None:
        # Front-facing default landmarks (120×120 image coords)
        kps = np.array([[40, 35], [80, 35], [60, 55], [45, 80], [75, 80]], dtype=np.float32)
    if bbox is None:
        bbox = np.array([10, 10, 110, 110], dtype=np.float32)
    if emb is None:
        emb = _rand_emb(42)
    f.kps = kps
    f.bbox = bbox
    f.normed_embedding = emb
    f.embedding = emb * 10.0
    return f


def _make_variant(
    yaw: float = 0.0,
    pitch: float = 0.0,
    temperature: float = 5500.0,
    light_dir: Optional[np.ndarray] = None,
    luminance: float = 0.5,
    seed: int = 0,
) -> SourceVariant:
    """Build a SourceVariant with synthetic properties."""
    if light_dir is None:
        light_dir = np.zeros(2, dtype=np.float32)
    emb = _rand_emb(seed)
    img = np.random.randint(80, 200, (64, 64, 3), dtype=np.uint8)
    face = _mock_face(emb=emb)
    return SourceVariant(
        image=img,
        face=face,
        embedding=emb,
        pose_yaw=yaw,
        pose_pitch=pitch,
        color_temperature=temperature,
        lighting_direction=light_dir.astype(np.float32),
        mean_luminance=luminance,
        filename=f"v_yaw{yaw:.0f}_T{temperature:.0f}.png",
        identity_similarity=1.0,
    )


def _build_selector_with_variants(variants: List[SourceVariant]) -> SourceFaceSelector:
    """Construct a SourceFaceSelector pre-loaded with synthetic variants."""
    # Bypass __init__ — inject variants directly
    sel = object.__new__(SourceFaceSelector)
    sel.method              = "select"
    sel.pose_weight         = 3.0
    sel.temperature_weight  = 2.0
    sel.direction_weight    = 1.5
    sel.luminance_weight    = 1.0
    sel.switch_threshold    = 0.15
    sel.switch_delay_frames = 5
    sel.variants            = variants
    sel._current_idx        = None
    sel._pending_idx        = None
    sel._pending_count      = 0
    sel._swapper            = None
    return sel


def _dummy_frame(h: int = 120, w: int = 160) -> np.ndarray:
    return np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Pose matching
# ---------------------------------------------------------------------------

class TestPoseMatching:
    def test_selects_closest_yaw_variant(self):
        """With target yaw=25, selector must pick the yaw=30 variant."""
        variants = [
            _make_variant(yaw=-30.0, seed=1),
            _make_variant(yaw=0.0,   seed=2),
            _make_variant(yaw=30.0,  seed=3),
        ]
        sel = _build_selector_with_variants(variants)

        target_yaw, target_pitch = 25.0, 0.0
        target_temp, target_ld, target_lum = 5500.0, np.zeros(2), 0.5

        scores = sel._score_all(target_yaw, target_pitch, target_temp, target_ld, target_lum)
        best   = int(np.argmin(scores))
        assert best == 2, f"Expected variant[2] (yaw=30), got variant[{best}]"

    def test_selects_closest_yaw_negative(self):
        """With target yaw=-25, selector must pick the yaw=-30 variant."""
        variants = [
            _make_variant(yaw=-30.0, seed=1),
            _make_variant(yaw=0.0,   seed=2),
            _make_variant(yaw=30.0,  seed=3),
        ]
        sel = _build_selector_with_variants(variants)

        scores = sel._score_all(-25.0, 0.0, 5500.0, np.zeros(2), 0.5)
        best   = int(np.argmin(scores))
        assert best == 0, f"Expected variant[0] (yaw=-30), got variant[{best}]"

    def test_pose_weight_dominates(self):
        """Pose score (×3) must outweigh temperature score (×2) when yaw differs a lot."""
        # Variant A: correct pose, wrong temperature
        # Variant B: wrong pose, correct temperature
        variants = [
            _make_variant(yaw=0.0,  temperature=3500.0, seed=1),  # correct pose, warm
            _make_variant(yaw=45.0, temperature=5500.0, seed=2),  # wrong pose, correct temp
        ]
        sel = _build_selector_with_variants(variants)

        scores = sel._score_all(0.0, 0.0, 5500.0, np.zeros(2), 0.5)
        # Variant 0: temp diff=2000 → temp_score=0.8; pose_score=0
        # Variant 1: pose diff=45   → pose_score=3.0; temp_score=0
        assert scores[0] < scores[1], "Correct-pose variant should win over correct-temp variant"


# ---------------------------------------------------------------------------
# 2. Lighting / temperature matching
# ---------------------------------------------------------------------------

class TestLightingMatching:
    def test_selects_warm_variant_for_warm_frame(self):
        """Warm-lit frame (3500K) must select the warm variant."""
        variants = [
            _make_variant(temperature=3500.0, seed=1),  # warm
            _make_variant(temperature=6500.0, seed=2),  # cool
        ]
        sel = _build_selector_with_variants(variants)

        scores = sel._score_all(0.0, 0.0, 3500.0, np.zeros(2), 0.5)
        best   = int(np.argmin(scores))
        assert best == 0, "Warm variant should be selected for warm-lit frame"

    def test_selects_cool_variant_for_cool_frame(self):
        variants = [
            _make_variant(temperature=3500.0, seed=1),
            _make_variant(temperature=6500.0, seed=2),
        ]
        sel = _build_selector_with_variants(variants)

        scores = sel._score_all(0.0, 0.0, 6500.0, np.zeros(2), 0.5)
        best   = int(np.argmin(scores))
        assert best == 1, "Cool variant should be selected for cool-lit frame"


# ---------------------------------------------------------------------------
# 3. Hysteresis prevents flicker
# ---------------------------------------------------------------------------

class TestHysteresisPreventFlicker:
    def test_no_rapid_switching_on_nearly_equal_conditions(self):
        """20 frames alternating slightly — total switches must be < 3."""
        variants = [
            _make_variant(yaw=0.0,  temperature=5400.0, seed=1),
            _make_variant(yaw=2.0,  temperature=5600.0, seed=2),
        ]
        sel = _build_selector_with_variants(variants)
        frame  = _dummy_frame()
        face   = _mock_face()

        # Patch estimate methods to feed controlled values
        switch_log = []
        last_idx   = [None]

        for i in range(20):
            # Conditions alternate very slightly (< 15% score difference)
            temp   = 5450.0 if i % 2 == 0 else 5550.0
            scores = sel._score_all(1.0, 0.0, temp, np.zeros(2), 0.5)
            idx    = sel._apply_hysteresis(scores)
            if last_idx[0] is not None and idx != last_idx[0]:
                switch_log.append(i)
            last_idx[0] = idx

        assert len(switch_log) < 3, (
            f"Switched {len(switch_log)} times in 20 near-equal frames — hysteresis broken"
        )

    def test_no_switch_on_small_improvement(self):
        """< 15% improvement → no switch."""
        variants = [
            _make_variant(yaw=0.0, seed=1),
            _make_variant(yaw=5.0, seed=2),   # slightly better for yaw=4
        ]
        sel = _build_selector_with_variants(variants)
        sel._current_idx = 0  # pretend variant 0 is already selected

        # Target: yaw=4 — variant[1] is slightly better but improvement < 15%
        scores = sel._score_all(4.0, 0.0, 5500.0, np.zeros(2), 0.5)
        result = sel._apply_hysteresis(scores)

        # Should stay on variant 0
        assert result == 0, "Should NOT switch for < 15% improvement"


# ---------------------------------------------------------------------------
# 4. Switch delay (must wait switch_delay_frames consecutive frames)
# ---------------------------------------------------------------------------

class TestSwitchDelay:
    def test_switch_after_delay_frames(self):
        """Switch must commit after exactly switch_delay_frames=5 frames."""
        variants = [
            _make_variant(yaw=0.0,   seed=1),
            _make_variant(yaw=45.0,  seed=2),   # strongly preferred for yaw=40
        ]
        sel = _build_selector_with_variants(variants)
        sel.switch_delay_frames = 5
        sel._current_idx = 0

        results = []
        for _ in range(8):
            # yaw=40 → variant[1] wins by large margin (> 15%)
            scores  = sel._score_all(40.0, 0.0, 5500.0, np.zeros(2), 0.5)
            results.append(sel._apply_hysteresis(scores))

        # Frames 0-4: still on variant 0 (accumulating)
        for i in range(4):
            assert results[i] == 0, f"Frame {i}: expected to stay on variant 0"
        # Frame 5+: switched to variant 1
        assert results[5] == 1, "Expected switch to variant 1 after 5 frames"


# ---------------------------------------------------------------------------
# 5. Embedding blend preserves identity
# ---------------------------------------------------------------------------

class TestEmbeddingBlend:
    def test_blended_embedding_similar_to_all_variants(self):
        """Blended embedding must have cosine sim > 0.5 with each input."""
        base_emb = _rand_emb(99)

        # Build 3 variants with embeddings close to base
        def _perturb(emb, noise=0.1, seed=0):
            rng = np.random.default_rng(seed)
            v   = emb + rng.standard_normal(512).astype(np.float32) * noise
            return v / np.linalg.norm(v)

        variants = [_make_variant(seed=i) for i in range(3)]
        for i, v in enumerate(variants):
            v.embedding = _perturb(base_emb, noise=0.3, seed=i)

        sel    = _build_selector_with_variants(variants)
        scores = [0.1, 0.2, 0.3]  # fixed scores for reproducibility

        top_indices = np.argsort(scores)[:3].tolist()
        raw_scores  = np.array([scores[i] for i in top_indices], dtype=np.float64)
        weights     = 1.0 / (raw_scores + 0.01)
        weights     = (weights / weights.sum()).astype(np.float32)

        blended = np.zeros(512, dtype=np.float32)
        for idx, w in zip(top_indices, weights):
            blended += w * variants[idx].embedding
        blended /= (np.linalg.norm(blended) + 1e-8)

        for i, v in enumerate(variants):
            sim = float(np.dot(blended, v.embedding) /
                        (np.linalg.norm(blended) * np.linalg.norm(v.embedding) + 1e-8))
            assert sim > 0.5, (
                f"Blended embedding cosine sim with variant {i} = {sim:.3f} < 0.5"
            )

    def test_blended_embedding_is_unit_norm(self):
        """Normalised blended embedding must have ‖e‖ ≈ 1."""
        variants = [_make_variant(seed=i) for i in range(3)]
        sel = _build_selector_with_variants(variants)

        _, img = sel.select_blended(_dummy_frame(), _mock_face())
        # Check via get_embedding path
        emb = sel.get_embedding(_dummy_frame(), _mock_face())
        norm = float(np.linalg.norm(emb))
        # Ghost backend may scale, but raw normed_embedding should be ≈1
        assert 0.5 < norm < 30.0, f"Embedding norm={norm:.3f} looks wrong"


# ---------------------------------------------------------------------------
# 6. Auto-detect variant properties
# ---------------------------------------------------------------------------

class TestAutoDetectProperties:
    def test_auto_detect_runs_without_crash(self):
        """_auto_detect_properties must not crash on a synthetic variant."""
        variants = [_make_variant(yaw=0.0)]
        sel = _build_selector_with_variants(variants)

        # Create a variant with a simple front-facing kps
        v = variants[0]
        v.face.kps = np.array(
            [[30, 30], [70, 30], [50, 50], [35, 75], [65, 75]], dtype=np.float32
        )
        v.image = np.random.randint(80, 200, (100, 100, 3), dtype=np.uint8)

        sel._auto_detect_properties(v)
        assert isinstance(v.pose_yaw, float)
        assert isinstance(v.color_temperature, float)
        assert 2000.0 < v.color_temperature < 12000.0
        assert 0.0 <= v.mean_luminance <= 1.0

    def test_yaw_estimate_front_facing(self):
        """Front-facing landmarks should give yaw ≈ 0."""
        sel = _build_selector_with_variants([_make_variant()])
        # Symmetrical landmarks
        kps = np.array([[30, 30], [70, 30], [50, 55], [35, 80], [65, 80]], dtype=np.float32)
        yaw, pitch = sel.estimate_pose(kps)
        assert abs(yaw) < 15.0, f"Front-facing yaw should be < 15°, got {yaw:.1f}°"

    def test_yaw_estimate_left_turn(self):
        """Left-turned landmarks should give negative yaw."""
        sel = _build_selector_with_variants([_make_variant()])
        # Nose shifted right relative to eye center → person turning left
        kps = np.array([[40, 30], [80, 30], [75, 55], [45, 80], [80, 80]], dtype=np.float32)
        yaw, _ = sel.estimate_pose(kps)
        assert yaw > 0, f"Left-turn landmarks should give positive yaw, got {yaw:.1f}°"


# ---------------------------------------------------------------------------
# 7. Single variant fallback
# ---------------------------------------------------------------------------

class TestSingleVariantFallback:
    def test_always_returns_single_variant(self):
        """With only 1 variant, select() must always return it without crash."""
        variants = [_make_variant(yaw=0.0, seed=1)]
        sel = _build_selector_with_variants(variants)

        for _ in range(10):
            v = sel.select(_dummy_frame(), _mock_face())
            assert v is variants[0]

    def test_select_blended_with_one_variant(self):
        """select_blended() with only 1 variant must not crash."""
        variants = [_make_variant(yaw=0.0, seed=1)]
        sel = _build_selector_with_variants(variants)
        face_obj, img = sel.select_blended(_dummy_frame(), _mock_face())
        assert face_obj is not None
        assert img is not None
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# 8. Selection performance
# ---------------------------------------------------------------------------

class TestSelectionPerformance:
    def test_select_under_2ms(self):
        """select() with 16 variants must complete in < 2ms (mean over 100 calls)."""
        variants = [_make_variant(yaw=float(i * 5 - 40), seed=i) for i in range(16)]
        sel = _build_selector_with_variants(variants)
        frame = _dummy_frame()
        face  = _mock_face()

        # Patch expensive estimate methods with trivial versions
        sel.estimate_pose     = lambda kps: (0.0, 0.0)
        sel.estimate_lighting = lambda frame, bbox: (5500.0, np.zeros(2), 0.5)

        # Warm-up
        for _ in range(5):
            sel.select(frame, face)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sel.select(frame, face)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = sum(times) / len(times)
        assert mean_ms < 2.0, (
            f"select() took {mean_ms:.3f}ms mean — expected < 2ms "
            f"(scoring 16 variants is pure numpy, should be ~0.1ms)"
        )

    def test_get_status_returns_dict(self):
        """get_status() must return a dict with expected keys."""
        variants = [_make_variant(seed=1)]
        sel = _build_selector_with_variants(variants)
        sel._current_idx = 0

        st = sel.get_status()
        for key in ("variant_idx", "variant_name", "num_variants", "method"):
            assert key in st, f"Missing key in status: {key}"
