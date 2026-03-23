"""Boundary seam fixer — three-stage seamless face/frame compositing.

WHY SIMPLE ALPHA BLENDING CREATES SEAMS:
  1. Color mismatch: swapped face has different skin tone than performer's
     neck/ears/hairline → visible color step at the boundary
  2. Alpha blending uses the SAME transition width for ALL frequencies:
     sharp textures need a narrow blend zone, color shifts need a wide one.
     Using one width makes either the color look pasted-on or the texture blurry.
  3. Residual brightness differences survive even a wide feather, showing up
     as a visible halo or shadow ring around the face.

THE THREE-STAGE FIX:
  Stage 1 — Color harmonization (pre-composite):
    Match the swap face's border colors to surrounding original skin using
    Reinhard color transfer in LAB space.  Full strength at the edge,
    fading toward zero at face centre (preserving identity there).

  Stage 2 — Laplacian pyramid multi-band blending:
    Blend low frequencies (overall color, brightness) with a WIDE transition
    zone and high frequencies (skin texture, pores) with a NARROW zone.
    This eliminates "double exposure" artefacts while keeping sharp details.

  Stage 3 — Post-composite boundary refinement:
    Bilateral filter micro-correction along the exact seam contour.
    Also corrects any residual brightness step (> 8 L-channel units)
    that survived the first two stages.

INTEGRATION (pipeline.py step 15):
  Replace the _gauss_blend / _poisson_composite call with:
      result = self._boundary_fixer.composite(pristine, warped_comp, mask_f32)
  where mask_f32 = warped_ellipse (float32 [0,1]).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class BoundarySeamFixer:
    """Three-stage seamless compositing to eliminate face/frame seams.

    Args:
        config: dict with optional keys:
            color_match_strength  (float, default 0.8)
            blend_levels          (int,   default 4)
            boundary_width        (int,   default 30)
            post_smooth_radius    (int,   default 5)
            enable_color_transfer (bool,  default True)
            enable_multiband      (bool,  default True)
            enable_post_refinement(bool,  default True)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.color_match_strength   = float(cfg.get("color_match_strength",   0.8))
        self.blend_levels           = int(cfg.get("blend_levels",             4))
        self.boundary_width         = int(cfg.get("boundary_width",           30))
        self.post_smooth_radius     = int(cfg.get("post_smooth_radius",       5))
        self.enable_color_transfer  = bool(cfg.get("enable_color_transfer",   True))
        self.enable_multiband       = bool(cfg.get("enable_multiband",        True))
        self.enable_post_refinement = bool(cfg.get("enable_post_refinement",  True))

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def compute_boundary_zone(
        self,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute inner and outer boundary bands around the mask edge.

        Args:
            mask: float32 [0,1] face mask

        Returns:
            (inner_band, outer_band) — both float32, high near the edge
            inner_band: inside the mask, near its boundary
            outer_band: outside the mask, near its boundary
        """
        bw = self.boundary_width
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bw * 2 + 1, bw * 2 + 1)
        )
        mask_u8 = (mask > 0.5).astype(np.uint8)

        outer = cv2.dilate(mask_u8, kernel).astype(np.float32)
        inner = cv2.erode(mask_u8, kernel).astype(np.float32)

        inner_band = mask * (1.0 - inner)   # inside mask, near edge
        outer_band = outer * (1.0 - mask)   # outside mask, near edge

        return inner_band, outer_band

    def harmonize_colors(
        self,
        swapped_frame: np.ndarray,
        original_frame: np.ndarray,
        mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Match swap border colors to surrounding original skin (Stage 1).

        Applies Reinhard color transfer in LAB space, with full strength at
        the mask boundary fading to zero at the face centre.

        Args:
            swapped_frame:  BGR frame containing the swapped face
            original_frame: pristine BGR frame
            mask:           float32 [0,1] face mask
            strength:       blend strength (0=off, 1=full)

        Returns:
            Color-harmonized BGR frame (same size as input).
        """
        inner_band, outer_band = self.compute_boundary_zone(mask)

        swap_lab = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2LAB).astype(np.float64)
        orig_lab = cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB).astype(np.float64)

        # Sample boundary pixels
        inner_px = swap_lab[inner_band > 0.3]
        outer_px = orig_lab[outer_band > 0.3]

        if len(inner_px) < 50 or len(outer_px) < 50:
            return swapped_frame  # not enough pixels — skip

        inner_mean = np.mean(inner_px, axis=0)
        inner_std  = np.std(inner_px,  axis=0) + 1e-6
        outer_mean = np.mean(outer_px, axis=0)
        outer_std  = np.std(outer_px,  axis=0) + 1e-6

        # Edge-proximity weight: high near boundary, low at face centre
        ero_k = np.ones((15, 15), np.uint8)
        edge_weight = 1.0 - cv2.erode(mask, ero_k, iterations=2)
        edge_weight = cv2.GaussianBlur(edge_weight, (31, 31), 10)
        edge_weight = np.clip(edge_weight * 2.0, 0.0, 1.0)

        face_mask = mask > 0.01
        result_lab = swap_lab.copy()

        for ch in range(3):
            channel    = swap_lab[:, :, ch]
            normalized = (channel - inner_mean[ch]) / inner_std[ch]
            transferred = normalized * outer_std[ch] + outer_mean[ch]

            blend_w = strength * edge_weight
            corrected = (1.0 - blend_w) * channel + blend_w * transferred
            result_lab[:, :, ch] = np.where(face_mask, corrected, channel)

        result_lab = np.clip(result_lab, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Stage 2 — Laplacian pyramid multi-band blending
    # ------------------------------------------------------------------

    def build_laplacian_pyramid(
        self,
        image: np.ndarray,
        levels: int,
    ) -> List[np.ndarray]:
        """Build a Laplacian pyramid for multi-band blending.

        Returns list of length (levels + 1):
          indices 0..levels-1 = Laplacian detail bands (fine → coarse)
          index   levels      = coarsest Gaussian residual
        """
        gaussian: List[np.ndarray] = [image.astype(np.float64)]
        for _ in range(levels):
            gaussian.append(cv2.pyrDown(gaussian[-1]))

        laplacian: List[np.ndarray] = []
        for i in range(levels):
            up = cv2.pyrUp(
                gaussian[i + 1],
                dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]),
            )
            laplacian.append(gaussian[i] - up)
        laplacian.append(gaussian[-1])  # residual
        return laplacian

    def build_mask_pyramid(
        self,
        mask: np.ndarray,
        levels: int,
    ) -> List[np.ndarray]:
        """Build a Gaussian pyramid for the blending mask.

        Progressively blurring the mask at each level is the key to
        multi-band blending: low-frequency bands see a wide transition.
        """
        mf = mask.astype(np.float64)
        if mf.ndim == 2:
            mf = mf[:, :, np.newaxis]

        pyramid: List[np.ndarray] = [mf]
        for _ in range(levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    def multiband_blend(
        self,
        swapped: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Laplacian pyramid multi-band blend (Stage 2).

        Args:
            swapped:  color-harmonized BGR frame with swapped face
            original: pristine BGR frame
            mask:     float32 [0,1] face mask

        Returns:
            Blended BGR frame (uint8).
        """
        levels = self.blend_levels

        lap_swap = self.build_laplacian_pyramid(swapped,  levels)
        lap_orig = self.build_laplacian_pyramid(original, levels)
        mask_pyr = self.build_mask_pyramid(mask, levels)

        blended_pyr: List[np.ndarray] = []
        for i in range(levels + 1):
            m = mask_pyr[i]
            target_h, target_w = lap_swap[i].shape[:2]
            if m.shape[:2] != (target_h, target_w):
                m = cv2.resize(m, (target_w, target_h))
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
            blended_pyr.append(m * lap_swap[i] + (1.0 - m) * lap_orig[i])

        # Reconstruct from coarsest level upward
        result = blended_pyr[-1]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(
                result,
                dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]),
            )
            result = result + blended_pyr[i]

        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Stage 3 — Post-composite boundary refinement
    # ------------------------------------------------------------------

    def refine_boundary(
        self,
        composited: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Micro-corrections along the exact seam line (Stage 3).

        Applies bilateral filter smoothing in the boundary zone, then
        corrects any residual brightness step > 8 L-channel units.

        Args:
            composited: multi-band blended BGR frame
            original:   pristine BGR frame
            mask:       float32 [0,1] face mask

        Returns:
            Refined BGR frame (uint8).
        """
        r = self.post_smooth_radius

        # Build seam mask from Canny on the alpha mask
        boundary_u8 = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        boundary_u8 = cv2.dilate(boundary_u8, np.ones((3, 3), np.uint8), iterations=1)
        seam = boundary_u8.astype(np.float32) / 255.0
        seam = cv2.GaussianBlur(seam, (r * 2 + 1, r * 2 + 1), r * 0.4)

        # Bilateral smoothing in the seam zone (preserves hair/fine edges)
        smoothed = cv2.bilateralFilter(composited, d=9, sigmaColor=25, sigmaSpace=25)

        result = composited.astype(np.float64)
        result = ((1.0 - seam[:, :, None]) * result
                  + seam[:, :, None] * smoothed.astype(np.float64))

        # Residual brightness correction
        boundary_u8_thin = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        inner_sample = cv2.erode(boundary_u8_thin, np.ones((3, 3), np.uint8))
        outer_sample = (
            cv2.dilate(boundary_u8_thin, np.ones((3, 3), np.uint8))
            - boundary_u8_thin
        )

        comp_u8  = np.clip(result, 0.0, 255.0).astype(np.uint8)
        comp_lab = cv2.cvtColor(comp_u8, cv2.COLOR_BGR2LAB).astype(np.float64)

        inner_color: Optional[np.ndarray] = (
            np.mean(comp_lab[inner_sample > 0], axis=0)
            if np.any(inner_sample) and inner_sample.sum() > 0
            else None
        )
        outer_color: Optional[np.ndarray] = (
            np.mean(comp_lab[outer_sample > 0], axis=0)
            if np.any(outer_sample) and outer_sample.sum() > 0
            else None
        )

        if inner_color is not None and outer_color is not None:
            l_diff = abs(float(inner_color[0]) - float(outer_color[0]))
            if l_diff > 8.0:
                correction = (outer_color[0] - inner_color[0]) * 0.3
                comp_lab[:, :, 0] += correction * seam
                comp_lab = np.clip(comp_lab, 0.0, 255.0)
                result = cv2.cvtColor(comp_lab.astype(np.uint8),
                                      cv2.COLOR_LAB2BGR).astype(np.float64)

        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main composite method
    # ------------------------------------------------------------------

    def composite(
        self,
        original_frame: np.ndarray,
        swapped_frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Three-stage seamless composite — replaces alpha/Poisson blend.

        Args:
            original_frame: pristine BGR webcam frame
            swapped_frame:  BGR frame with swapped face already composited
                            in the face region (occluder pixels already
                            baked in from the aligned-space composite step)
            mask:           float32 [0,1] face ellipse mask (full frame size)

        Returns:
            Seamlessly composited BGR frame (uint8).
        """
        # Ensure float32 mask with correct spatial dims
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.float32)

        # Ensure images are uint8
        if original_frame.dtype != np.uint8:
            original_frame = np.clip(original_frame, 0, 255).astype(np.uint8)
        if swapped_frame.dtype != np.uint8:
            swapped_frame = np.clip(swapped_frame, 0, 255).astype(np.uint8)

        result = swapped_frame.copy()

        # Stage 1: Color harmonization
        if self.enable_color_transfer:
            result = self.harmonize_colors(
                result, original_frame, mask, self.color_match_strength
            )

        # Stage 2: Multi-band blending
        if self.enable_multiband:
            result = self.multiband_blend(result, original_frame, mask)
        else:
            m = mask[:, :, np.newaxis]
            result = (
                m * result.astype(np.float64)
                + (1.0 - m) * original_frame.astype(np.float64)
            ).astype(np.uint8)

        # Stage 3: Boundary refinement
        if self.enable_post_refinement:
            result = self.refine_boundary(result, original_frame, mask)

        return result

    # ------------------------------------------------------------------
    # Runtime tuning
    # ------------------------------------------------------------------

    def update_config(self, config: dict) -> None:
        """Hot-update parameters (e.g., from GUI sliders)."""
        if "color_match_strength" in config:
            self.color_match_strength = float(config["color_match_strength"])
        if "blend_levels" in config:
            self.blend_levels = max(1, int(config["blend_levels"]))
        if "boundary_width" in config:
            self.boundary_width = max(1, int(config["boundary_width"]))
        if "post_smooth_radius" in config:
            self.post_smooth_radius = max(1, int(config["post_smooth_radius"]))
        if "enable_color_transfer" in config:
            self.enable_color_transfer = bool(config["enable_color_transfer"])
        if "enable_multiband" in config:
            self.enable_multiband = bool(config["enable_multiband"])
        if "enable_post_refinement" in config:
            self.enable_post_refinement = bool(config["enable_post_refinement"])


# ---------------------------------------------------------------------------
# __main__ — 4-panel webcam demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import time

    print("BoundarySeamFixer — 4-panel webcam demo")
    print("  Top-left:     simple alpha blend (old — shows seams)")
    print("  Top-right:    Stage 1 only (color harmonization)")
    print("  Bottom-left:  Stage 2 only (multi-band blend)")
    print("  Bottom-right: All 3 stages (final result)")
    print("  Press 'q' to quit, 'c' to cycle modes")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    try:
        from neural_swap import NeuralFaceSwapper, _align_face, _best_providers, _ARCFACE_112
    except ImportError as e:
        print(f"[ERROR] Cannot import neural_swap: {e}")
        sys.exit(1)

    swap_path   = "models/ghost_3_256.onnx"
    source_path = "source_faces/source.png"

    if not os.path.exists(swap_path):
        print(f"[ERROR] Swap model not found: {swap_path}")
        sys.exit(1)

    swapper = NeuralFaceSwapper(model_path=swap_path, occlusion=False)

    src_emb = None
    if os.path.exists(source_path):
        src_img   = cv2.imread(source_path)
        src_faces = swapper.detect(src_img)
        if src_faces:
            src_emb = src_faces[0].normed_embedding

    # Fixer instances for each panel
    fixer_none    = BoundarySeamFixer({"enable_color_transfer": False,
                                       "enable_multiband": False,
                                       "enable_post_refinement": False})
    fixer_color   = BoundarySeamFixer({"enable_color_transfer": True,
                                       "enable_multiband": False,
                                       "enable_post_refinement": False})
    fixer_multi   = BoundarySeamFixer({"enable_color_transfer": False,
                                       "enable_multiband": True,
                                       "enable_post_refinement": False})
    fixer_full    = BoundarySeamFixer()

    cv2.namedWindow("BoundarySeamFixer Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BoundarySeamFixer Demo", 960, 720)

    strength_val = [80]
    levels_val   = [4]
    width_val    = [30]

    def _on_strength(v): fixer_full.color_match_strength = v / 100.0
    def _on_levels(v):   fixer_full.blend_levels = max(1, v)
    def _on_width(v):    fixer_full.boundary_width = max(1, v)

    cv2.createTrackbar("color_strength×100", "BoundarySeamFixer Demo", 80, 100, _on_strength)
    cv2.createTrackbar("blend_levels",       "BoundarySeamFixer Demo", 4,  6,   _on_levels)
    cv2.createTrackbar("boundary_width",     "BoundarySeamFixer Demo", 30, 50,  _on_width)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    stage_times: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        faces = swapper.detect(frame)
        if not faces:
            cv2.imshow("BoundarySeamFixer Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face = faces[0]
        if src_emb is None:
            src_emb = face.normed_embedding

        size = getattr(swapper._backend, "_size", 256)
        aligned, M = _align_face(frame, face.kps, size)

        try:
            swapped_crop = swapper.swap_aligned(aligned, src_emb)
        except Exception:
            swapped_crop = aligned

        if swapped_crop is None:
            swapped_crop = aligned

        # Build a simple ellipse mask in aligned space then warp to frame
        mask_aligned = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(mask_aligned, (size//2, size//2),
                    (int(size*0.38), int(size*0.45)),
                    0, 0, 360, 1.0, -1)
        mask_aligned = cv2.GaussianBlur(mask_aligned,
                                         (size//8*2+1, size//8*2+1),
                                         size//16)

        Mi = cv2.invertAffineTransform(M)
        wc = cv2.warpAffine(swapped_crop, Mi, (w, h))
        wm = cv2.warpAffine(mask_aligned,  Mi, (w, h))

        t0 = time.perf_counter()
        panel_none  = fixer_none.composite(frame, wc, wm)
        panel_color = fixer_color.composite(frame, wc, wm)
        panel_multi = fixer_multi.composite(frame, wc, wm)
        panel_full  = fixer_full.composite(frame, wc, wm)
        dt = (time.perf_counter() - t0) * 1000.0
        stage_times.append(dt)
        if len(stage_times) > 30:
            stage_times.pop(0)

        ds = 300
        def _rs(img):
            return cv2.resize(img, (ds, ds))

        top = np.hstack([_rs(panel_none),  _rs(panel_color)])
        bot = np.hstack([_rs(panel_multi), _rs(panel_full)])
        panel = np.vstack([top, bot])

        avg = sum(stage_times) / len(stage_times)
        cv2.putText(panel, f"full pipeline: {avg:.1f}ms",
                    (5, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        for i, label in enumerate(["alpha only", "color harm.", "multiband", "all stages"]):
            row, col = divmod(i, 2)
            cv2.putText(panel, label,
                        (col * ds + 5, row * ds + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("BoundarySeamFixer Demo", panel)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
