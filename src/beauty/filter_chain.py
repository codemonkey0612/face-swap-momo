"""Beauty filter chain: composable pipeline applied to the swapped face region.

Order: color_correction → skin_smoothing → eye_enhancement → sharpening → GFPGAN

Each filter can be individually enabled/disabled via config or at runtime.
Operates ONLY on face pixels (using the occlusion mask), leaving background
and body completely untouched.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.beauty.skin_smoothing  import SkinSmoother
from src.beauty.eye_enhancement import EyeEnhancer
from src.beauty.color_correction import ColorCorrector


class BeautyFilterChain:
    """Composable beauty pipeline.

    Args:
        config: dict with optional keys:
            enabled (bool): master switch (default True)
            color_correction (bool): enable Reinhard colour match (default True)
            skin_smoothing_strength (float): 0-1 (default 0.6)
            eye_brighten (float): 0-1 eye enhancement strength (default 0.3)
            sharpness (float): 0-1 final unsharp mask amount (default 0.2)
            gfpgan_enabled (bool): run GFPGAN restoration (default True)
            model_dir (str): directory containing GFPGANv1.4.pth
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._timings: dict[str, float] = {}

        # Sub-modules
        self._color_corrector = ColorCorrector({
            "strength": 1.0,
        })
        self._color_enabled = cfg.get("color_correction", True)

        self._smoother = SkinSmoother({
            "strength": cfg.get("skin_smoothing_strength", 0.6),
        })

        self._eye_enhancer = EyeEnhancer({
            "strength": cfg.get("eye_brighten", 0.3),
        })
        self._eye_enabled = cfg.get("eye_brighten", 0.3) > 0

        self._sharpness = float(cfg.get("sharpness", 0.2))

        # Optional GFPGAN
        self._gfpgan: Optional[object] = None
        if self._enabled and cfg.get("gfpgan_enabled", True):
            import os
            model_path = os.path.join(cfg.get("model_dir", "models"), "GFPGANv1.4.pth")
            if os.path.exists(model_path):
                try:
                    from enhancer import FaceEnhancer
                    self._gfpgan = FaceEnhancer(model_path=model_path)
                except Exception as e:
                    print(f"[Beauty] GFPGAN unavailable: {e}")
            else:
                print(f"[Beauty] GFPGANv1.4.pth not found — GFPGAN skipped")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        face_mask: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        original_frame: Optional[np.ndarray] = None,
        faces: Optional[list] = None,
    ) -> np.ndarray:
        """Run all enabled beauty filters in sequence.

        Args:
            frame:          Post-swap BGR frame.
            face_mask:      float32 (H, W) mask from OcclusionFuser (swap region).
            landmarks:      (5, 2) array of facial landmarks (for eye enhancement).
            original_frame: Pre-swap BGR frame (for colour matching).
            faces:          InsightFace Face objects (for GFPGAN).

        Returns:
            Enhanced BGR frame.
        """
        if not self._enabled:
            return frame

        result = frame

        # 1. Colour correction (match swapped face to original lighting)
        if self._color_enabled and original_frame is not None:
            t0 = time.perf_counter()
            result = self._color_corrector.match_skin_tone(
                result, original_frame, face_mask
            )
            self._timings["color_correction"] = (time.perf_counter() - t0) * 1000

        # 2. Skin smoothing
        t0 = time.perf_counter()
        result = self._smoother.smooth(result, face_mask)
        self._timings["skin_smoothing"] = (time.perf_counter() - t0) * 1000

        # 3. Eye enhancement
        if self._eye_enabled and landmarks is not None:
            t0 = time.perf_counter()
            result = self._eye_enhancer.enhance(result, landmarks)
            self._timings["eye_enhancement"] = (time.perf_counter() - t0) * 1000

        # 4. Final sharpening (unsharp mask on face region)
        if self._sharpness > 0:
            t0 = time.perf_counter()
            result = self._apply_sharpening(result, face_mask)
            self._timings["sharpening"] = (time.perf_counter() - t0) * 1000

        # 5. Optional GFPGAN
        if self._gfpgan is not None and faces:
            t0 = time.perf_counter()
            result = self._gfpgan.enhance_frame(result, faces)
            self._timings["gfpgan"] = (time.perf_counter() - t0) * 1000

        return result

    def _apply_sharpening(self, frame: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Unsharp mask (amount=sharpness, radius=1.5) on face region only."""
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
        a = 1.0 + self._sharpness * 0.5
        b = -self._sharpness * 0.5
        sharpened = cv2.addWeighted(frame, a, blurred, b, 0)
        mask3 = face_mask[:, :, np.newaxis]
        result = mask3 * sharpened.astype(np.float32) + (1.0 - mask3) * frame.astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    def update_config(self, config: dict) -> None:
        """Hot-reload parameters (e.g. from GUI sliders)."""
        if "enabled" in config:
            self._enabled = bool(config["enabled"])
        if "skin_smoothing_strength" in config:
            self._smoother.update_strength(config["skin_smoothing_strength"])
        if "eye_brighten" in config:
            self._eye_enhancer.update_strength(config["eye_brighten"])
            self._eye_enabled = config["eye_brighten"] > 0
        if "sharpness" in config:
            self._sharpness = float(config["sharpness"])
        if "color_correction" in config:
            self._color_enabled = bool(config["color_correction"])

    @property
    def timings(self) -> dict[str, float]:
        """Per-filter timing in ms from the last process() call."""
        return dict(self._timings)

    @property
    def is_active(self) -> bool:
        return self._enabled
