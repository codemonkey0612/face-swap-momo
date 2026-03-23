"""Landmark-guided face shape adapter.

WHY THIS IS NEEDED:
  InSwapper/Ghost transfers the source face's IDENTITY (eyes, nose, mouth
  features) but also partially transfers the source face's SHAPE (jawline,
  face width, forehead proportions).  When the performer has a round wide
  face but the source AI face is narrow and long, the swapped result looks
  warped at the edges — the jawline doesn't match the performer's head,
  creating a visible "mask" floating on the wrong-shaped head.

THE FIX — gradient shape blending:
  Keep the source identity in the CENTER of the face (eyes, nose, mouth)
  but morph the OUTER contour (jawline, forehead, cheeks) to match the
  performer's actual face shape.

  Inner face (r < inner_radius): 100% source shape (identity preserved)
  Outer face (r > outer_radius): blends toward performer's shape
  Transition zone: smooth cubic (smoothstep) gradient

  Additionally, per-group blending weights override the radial gradient for
  landmark groups where identity vs shape tradeoff is well-defined:
    jawline  (0-16):  0.9 × blend_strength  — most visible shape indicator
    eyebrows (17-26): 0.4 × blend_strength  — moderate (position matters)
    eyes     (36-47): 0.15 × blend_strength — keep source (identity critical)
    nose     (27-35): 0.10 × blend_strength — keep source
    mouth    (48-67): 0.10 × blend_strength — keep source

INTEGRATION:
  This module runs on the ALIGNED crop (128×128 or 256×256) produced by
  norm_crop2 / _align_face, AFTER the swap model and BEFORE the
  aligned-space composite step.

  In pipeline.py process_frame():
    # After step 9 (swap):
    swapped = self._shape_adapter.adapt(aligned, swapped)
    # Before step 10 (aligned composite)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import landmark detection backends
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False

# InsightFace's 68-landmark model (part of buffalo_l / buffalo_s packs)
try:
    import insightface
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False

try:
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# iBUG 68-point landmark groups (standard convention)
# ---------------------------------------------------------------------------
_JAWLINE    = list(range(0, 17))   # 17 pts — most visible shape indicator
_L_EYEBROW  = list(range(17, 22))  # 5 pts
_R_EYEBROW  = list(range(22, 27))  # 5 pts
_EYEBROWS   = _L_EYEBROW + _R_EYEBROW
_NOSE_BRIDGE = list(range(27, 31)) # 4 pts
_NOSE_LOWER  = list(range(31, 36)) # 5 pts
_NOSE       = _NOSE_BRIDGE + _NOSE_LOWER
_L_EYE      = list(range(36, 42))  # 6 pts
_R_EYE      = list(range(42, 48))  # 6 pts
_EYES       = _L_EYE + _R_EYE
_OUTER_LIP  = list(range(48, 60))  # 12 pts
_INNER_LIP  = list(range(60, 68))  # 8 pts
_MOUTH      = _OUTER_LIP + _INNER_LIP


class FaceShapeAdapter:
    """Corrects face shape mismatch between source and performer.

    Args:
        config: dict with optional keys:
            blend_strength     (float, 0-1, default 0.7)
            inner_radius_ratio (float, 0-1, default 0.3)
            outer_radius_ratio (float, 0-1, default 0.85)
            warp_method        ('delaunay' | 'tps', default 'delaunay')
            temporal_smoothing (float, 0-1, default 0.7)
            jawline_blend      (float, default 0.9)
            eye_blend          (float, default 0.15)
            nose_blend         (float, default 0.10)
            mouth_blend        (float, default 0.10)
            landmark_model     (str path, optional onnx 68-pt model)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.blend_strength     = float(cfg.get("blend_strength",     0.7))
        self.inner_radius_ratio = float(cfg.get("inner_radius",       0.3))
        self.outer_radius_ratio = float(cfg.get("outer_radius",       0.85))
        self.warp_method        = str(cfg.get("warp_method",          "delaunay"))
        self.temporal_alpha     = float(cfg.get("temporal_smoothing",  0.7))
        self.jawline_blend      = float(cfg.get("jawline_blend",       0.9))
        self.eye_blend          = float(cfg.get("eye_blend",           0.15))
        self.nose_blend         = float(cfg.get("nose_blend",          0.10))
        self.mouth_blend        = float(cfg.get("mouth_blend",         0.10))

        if not _SCIPY_OK:
            log.warning("[FaceShapeAdapter] scipy not installed — warp disabled. "
                        "pip install scipy")

        # Temporal warp field smoothing
        self._prev_map_x: Optional[np.ndarray] = None
        self._prev_map_y: Optional[np.ndarray] = None

        # 68-point landmark detector (ONNX)
        self._lm_sess = None
        self._lm_input = None
        self._lm_output = None
        self._lm_size = 192  # default input size for face landmark models

        lm_path = cfg.get("landmark_model", "")
        if lm_path and _ORT_OK:
            self._load_landmark_model(lm_path)
        else:
            self._try_load_default_landmark_model()

    # ------------------------------------------------------------------
    # Landmark model loading
    # ------------------------------------------------------------------

    def _load_landmark_model(self, path: str) -> bool:
        """Load an ONNX 68-point landmark model."""
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(path, providers=providers)
            inp = sess.get_inputs()[0]
            self._lm_sess   = sess
            self._lm_input  = inp.name
            self._lm_output = sess.get_outputs()[0].name
            shape = inp.shape
            self._lm_size = int(shape[-1]) if len(shape) >= 4 else 192
            log.info(f"[FaceShapeAdapter] Loaded landmark model: {path}")
            return True
        except Exception as e:
            log.warning(f"[FaceShapeAdapter] Could not load landmark model {path}: {e}")
            return False

    def _try_load_default_landmark_model(self) -> None:
        """Auto-discover a 68-pt landmark model in models/."""
        candidates = [
            "models/landmark_68.onnx",
            "models/2d106det.onnx",   # InsightFace 106-pt (we use first 68)
            "models/lm68.onnx",
        ]
        for path in candidates:
            import os
            if os.path.exists(path) and _ORT_OK:
                if self._load_landmark_model(path):
                    return
        log.info("[FaceShapeAdapter] No ONNX landmark model found — "
                 "using InsightFace 5-pt → 68-pt fallback")

    # ------------------------------------------------------------------
    # 68-point landmark detection
    # ------------------------------------------------------------------

    def detect_landmarks_68(self, aligned_crop: np.ndarray) -> Optional[np.ndarray]:
        """Detect 68 iBUG landmarks on an aligned face crop.

        Returns:
            np.ndarray of shape (68, 2) with (x, y) pixel positions,
            or None if detection fails.
        """
        h, w = aligned_crop.shape[:2]

        # --- ONNX landmark model ---
        if self._lm_sess is not None:
            try:
                return self._detect_onnx(aligned_crop, h, w)
            except Exception as e:
                log.debug(f"[FaceShapeAdapter] ONNX landmark failed: {e}")

        # --- InsightFace FaceAnalysis fallback ---
        if _INSIGHTFACE_OK:
            try:
                return self._detect_insightface(aligned_crop, h, w)
            except Exception as e:
                log.debug(f"[FaceShapeAdapter] InsightFace landmark failed: {e}")

        # --- Synthetic 68 pts from face geometry (last resort) ---
        return self._synthetic_landmarks(h, w)

    def _detect_onnx(self, crop: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        """Run ONNX 68-pt landmark model on crop."""
        size = self._lm_size
        resized = cv2.resize(crop, (size, size))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
        raw = self._lm_sess.run([self._lm_output], {self._lm_input: blob})[0]
        pts = raw.squeeze()  # (68, 2) or (136,)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if len(pts) < 68:
            return None
        pts = pts[:68]
        # Scale from model input size back to crop size
        pts[:, 0] *= w / size
        pts[:, 1] *= h / size
        return pts.astype(np.float32)

    def _detect_insightface(self, crop: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        """Use InsightFace's built-in landmark model (106-pt → first 68)."""
        import insightface
        # Run retinaface detection on the crop itself
        app = getattr(self, "_if_app", None)
        if app is None:
            app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "landmark_2d_106"],
            )
            app.prepare(ctx_id=0, det_size=(crop.shape[1], crop.shape[0]))
            self._if_app = app  # cache
        faces = app.get(crop)
        if not faces:
            return None
        # InsightFace 106-pt landmarks: take the first 68
        lm = faces[0].landmark_2d_106
        if lm is None or len(lm) < 68:
            return None
        return lm[:68].astype(np.float32)

    def _synthetic_landmarks(self, h: int, w: int) -> np.ndarray:
        """Generate a synthetic 68-pt landmark set from face geometry.

        This is used ONLY when no real model is available.  The synthetic
        points follow the average face proportions scaled to (w, h).
        Shape blending will still work, just less accurate.
        """
        # Normalised 68-pt template (average face, coords in [0, 1])
        # Generated from the 300-W average face shape
        template = np.array([
            # Jawline (0-16)
            [0.14, 0.72], [0.17, 0.80], [0.21, 0.87], [0.27, 0.92],
            [0.33, 0.96], [0.41, 0.98], [0.50, 0.99], [0.59, 0.98],
            [0.67, 0.96], [0.73, 0.92], [0.79, 0.87], [0.83, 0.80],
            [0.86, 0.72], [0.86, 0.64], [0.85, 0.56], [0.84, 0.48],
            [0.83, 0.40],
            # Left eyebrow (17-21)
            [0.23, 0.37], [0.30, 0.33], [0.37, 0.32], [0.43, 0.33], [0.49, 0.36],
            # Right eyebrow (22-26)
            [0.51, 0.36], [0.57, 0.33], [0.63, 0.32], [0.70, 0.33], [0.77, 0.37],
            # Nose bridge (27-30)
            [0.50, 0.41], [0.50, 0.47], [0.50, 0.52], [0.50, 0.57],
            # Nose tip/lower (31-35)
            [0.42, 0.62], [0.46, 0.64], [0.50, 0.65], [0.54, 0.64], [0.58, 0.62],
            # Left eye (36-41)
            [0.27, 0.45], [0.32, 0.42], [0.38, 0.42], [0.42, 0.45],
            [0.38, 0.48], [0.32, 0.48],
            # Right eye (42-47)
            [0.58, 0.45], [0.62, 0.42], [0.68, 0.42], [0.73, 0.45],
            [0.68, 0.48], [0.62, 0.48],
            # Outer lip (48-59)
            [0.38, 0.76], [0.42, 0.73], [0.46, 0.72], [0.50, 0.73],
            [0.54, 0.72], [0.58, 0.73], [0.62, 0.76], [0.58, 0.80],
            [0.54, 0.82], [0.50, 0.83], [0.46, 0.82], [0.42, 0.80],
            # Inner lip (60-67)
            [0.38, 0.76], [0.43, 0.74], [0.50, 0.74], [0.57, 0.74],
            [0.62, 0.76], [0.57, 0.80], [0.50, 0.81], [0.43, 0.80],
        ], dtype=np.float32)

        pts = template.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts

    # ------------------------------------------------------------------
    # Core blending algorithm
    # ------------------------------------------------------------------

    def compute_shape_blend_landmarks(
        self,
        source_lm: np.ndarray,
        target_lm: np.ndarray,
        blend_strength: float,
    ) -> np.ndarray:
        """Blend source and target landmarks with a radial gradient.

        Inner zone (r < inner_radius):  keep source shape (identity)
        Outer zone (r > outer_radius):  match target shape (performer)
        Transition: smoothstep gradient

        Then override with per-group weights for critical groups.

        Args:
            source_lm:      (68, 2) landmarks from swapped crop
            target_lm:      (68, 2) landmarks from original performer crop
            blend_strength: global strength multiplier (0=off, 1=full)

        Returns:
            blended (68, 2) landmark array
        """
        blended = source_lm.copy()

        # Use nose landmarks (27-35) as the stable face center
        center = np.mean(target_lm[27:36], axis=0)

        # Radial weights
        dists = np.linalg.norm(source_lm - center, axis=1)  # (68,)
        max_dist = dists.max()
        if max_dist < 1e-6:
            return blended

        norm_dists = dists / max_dist

        for i in range(68):
            nd = norm_dists[i]
            if nd < self.inner_radius_ratio:
                weight = 0.0
            elif nd > self.outer_radius_ratio:
                weight = blend_strength
            else:
                t = (nd - self.inner_radius_ratio) / (
                    self.outer_radius_ratio - self.inner_radius_ratio
                )
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                weight = t * blend_strength
            blended[i] = (1.0 - weight) * source_lm[i] + weight * target_lm[i]

        # Per-group overrides (applied on top of radial blend)
        def _lerp(src, tgt, w):
            return (1.0 - w) * src + w * tgt

        # Jawline: highest blend — defines face shape most visibly
        jaw_w = blend_strength * self.jawline_blend
        for i in _JAWLINE:
            blended[i] = _lerp(source_lm[i], target_lm[i], jaw_w)

        # Eyebrows: medium blend
        brow_w = blend_strength * 0.4
        for i in _EYEBROWS:
            blended[i] = _lerp(source_lm[i], target_lm[i], brow_w)

        # Eyes: low blend — eye shape is critical for identity
        eye_w = blend_strength * self.eye_blend
        for i in _EYES:
            blended[i] = _lerp(source_lm[i], target_lm[i], eye_w)

        # Nose: very low blend
        nose_w = blend_strength * self.nose_blend
        for i in _NOSE:
            blended[i] = _lerp(source_lm[i], target_lm[i], nose_w)

        # Mouth: very low blend
        mouth_w = blend_strength * self.mouth_blend
        for i in _MOUTH:
            blended[i] = _lerp(source_lm[i], target_lm[i], mouth_w)

        return blended

    # ------------------------------------------------------------------
    # Warp field computation
    # ------------------------------------------------------------------

    def compute_warp_field(
        self,
        source_lm: np.ndarray,
        blended_lm: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a dense pixel-wise warp map (map_x, map_y).

        Args:
            source_lm:   (68, 2) current landmark positions
            blended_lm:  (68, 2) desired landmark positions after blending
            image_size:  (h, w) of the target image

        Returns:
            (map_x, map_y) float32 arrays for cv2.remap()
            map_x[y, x] = source X coordinate to sample
            map_y[y, x] = source Y coordinate to sample
        """
        h, w = image_size

        # Add boundary control points so the warp is stable at edges
        boundary = np.array([
            [0,    0   ], [w//2, 0   ], [w-1, 0   ],
            [0,    h//2], [w-1, h//2],
            [0,    h-1 ], [w//2, h-1 ], [w-1, h-1 ],
        ], dtype=np.float32)

        src_pts = np.vstack([source_lm, boundary]).astype(np.float32)
        dst_pts = np.vstack([blended_lm, boundary]).astype(np.float32)

        if self.warp_method == "tps" and _SCIPY_OK:
            return self._warp_tps(src_pts, dst_pts, h, w)
        else:
            return self._warp_delaunay(src_pts, dst_pts, h, w)

    def _warp_delaunay(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Delaunay triangulation warp (~3ms for 128×128).

        For each triangle in the destination mesh, compute the affine
        transform from dst→src and fill the corresponding source coordinates.
        """
        if not _SCIPY_OK:
            return self._identity_map(h, w)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        # Build a pixel-coordinate identity map as default (no warp)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x[:] = grid_x.astype(np.float32)
        map_y[:] = grid_y.astype(np.float32)

        tri = Delaunay(dst_pts)

        for simplex in tri.simplices:
            # dst triangle vertices
            d0, d1, d2 = dst_pts[simplex[0]], dst_pts[simplex[1]], dst_pts[simplex[2]]
            # src triangle vertices
            s0, s1, s2 = src_pts[simplex[0]], src_pts[simplex[1]], src_pts[simplex[2]]

            # Build integer polygon for filling
            poly = np.array([d0, d1, d2], dtype=np.int32)

            # Create triangle mask in dst space
            tmask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(tmask, poly, 255)

            pixel_ys, pixel_xs = np.where(tmask > 0)
            if len(pixel_xs) == 0:
                continue

            # Affine transform from dst triangle → src triangle
            dst_tri = np.float32([[d0], [d1], [d2]])
            src_tri = np.float32([[s0], [s1], [s2]])
            M = cv2.getAffineTransform(dst_tri, src_tri)
            if M is None:
                continue

            # Apply M to all pixels inside this dst triangle
            pts_h = np.ones((3, len(pixel_xs)), dtype=np.float32)
            pts_h[0] = pixel_xs
            pts_h[1] = pixel_ys
            src_coords = M @ pts_h  # (2, N)

            map_x[pixel_ys, pixel_xs] = src_coords[0].astype(np.float32)
            map_y[pixel_ys, pixel_xs] = src_coords[1].astype(np.float32)

        return map_x, map_y

    def _warp_tps(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Thin Plate Spline warp via SciPy RBFInterpolator (~20ms).

        Smoother than Delaunay but slower — recommended for quality mode.
        """
        rbf_x = RBFInterpolator(dst_pts, src_pts[:, 0], kernel="thin_plate_spline")
        rbf_y = RBFInterpolator(dst_pts, src_pts[:, 1], kernel="thin_plate_spline")

        gy, gx = np.mgrid[0:h, 0:w]
        grid = np.column_stack([gx.ravel().astype(np.float32),
                                gy.ravel().astype(np.float32)])

        map_x = rbf_x(grid).reshape(h, w).astype(np.float32)
        map_y = rbf_y(grid).reshape(h, w).astype(np.float32)
        return map_x, map_y

    def _identity_map(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a no-op identity warp map."""
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                              np.arange(h, dtype=np.float32))
        return gx, gy

    # ------------------------------------------------------------------
    # Warp application
    # ------------------------------------------------------------------

    def warp_face(
        self,
        swapped_crop: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
    ) -> np.ndarray:
        """Apply warp map to reshape the swapped face crop.

        Args:
            swapped_crop: uint8 BGR/RGB aligned face crop
            map_x, map_y: float32 source coordinate maps

        Returns:
            Warped crop of same size
        """
        return cv2.remap(
            swapped_crop, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
    ) -> np.ndarray:
        """Shape-correct the swapped face to match the performer's face shape.

        Call this AFTER the swap model and BEFORE the aligned-space composite.

        Args:
            original_crop: aligned crop of the performer's real face (pre-swap)
            swapped_crop:  aligned crop output from the swap model

        Returns:
            Shape-corrected swapped crop (same dimensions as swapped_crop).
            Returns swapped_crop unchanged if detection fails or scipy is missing.
        """
        if not _SCIPY_OK:
            return swapped_crop

        if self.blend_strength <= 0.0:
            return swapped_crop

        h, w = swapped_crop.shape[:2]

        # Detect 68 landmarks on both crops
        target_lm = self.detect_landmarks_68(original_crop)
        source_lm = self.detect_landmarks_68(swapped_crop)

        # Graceful fallback if detection failed on either
        if target_lm is None or source_lm is None:
            return swapped_crop

        # Compute blended landmark positions
        blended_lm = self.compute_shape_blend_landmarks(
            source_lm, target_lm, self.blend_strength
        )

        # Compute dense warp field
        map_x, map_y = self.compute_warp_field(source_lm, blended_lm, (h, w))

        # Temporal warp field smoothing — prevents flicker between frames
        if self._prev_map_x is not None and self._prev_map_x.shape == map_x.shape:
            a = self.temporal_alpha
            map_x = a * map_x + (1.0 - a) * self._prev_map_x
            map_y = a * map_y + (1.0 - a) * self._prev_map_y

        self._prev_map_x = map_x.copy()
        self._prev_map_y = map_y.copy()

        # Apply warp
        return self.warp_face(swapped_crop, map_x, map_y)

    def set_blend_strength(self, strength: float) -> None:
        """Hot-update blend strength for real-time tuning (e.g., GUI slider)."""
        self.blend_strength = float(np.clip(strength, 0.0, 1.0))

    def reset_temporal(self) -> None:
        """Reset temporal warp buffer (call on scene change / source switch)."""
        self._prev_map_x = None
        self._prev_map_y = None


# ---------------------------------------------------------------------------
# __main__ — webcam demo with 4-panel view and blend trackbar
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("FaceShapeAdapter — webcam demo")
    print("  Top-left:    original face")
    print("  Top-right:   swapped face (NO shape correction)")
    print("  Bottom-left: swapped face (WITH shape correction)")
    print("  Bottom-right: warp field magnitude visualization")
    print("  Trackbar: blend_strength")
    print("  Press 'q' to quit, 't' to toggle Delaunay/TPS")

    if not _SCIPY_OK:
        print("[ERROR] scipy is required: pip install scipy")
        sys.exit(1)

    # --- Load swap model ---
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from neural_swap import NeuralFaceSwapper, _align_face, _best_providers
    except ImportError as e:
        print(f"[ERROR] Cannot import neural_swap: {e}")
        sys.exit(1)

    swapper_path = "models/ghost_3_256.onnx"
    source_path  = "source_faces/source.png"

    if not os.path.exists(swapper_path):
        print(f"[ERROR] Swap model not found: {swapper_path}")
        sys.exit(1)
    if not os.path.exists(source_path):
        print(f"[WARN]  Source face not found: {source_path}")
        print(f"        Using first detected face as source")
        source_path = None  # type: ignore

    swapper = NeuralFaceSwapper(model_path=swapper_path, occlusion=False)

    if source_path:
        src_img = cv2.imread(source_path)
        src_faces = swapper.detect(src_img)
        if not src_faces:
            print(f"[ERROR] No face in {source_path}")
            sys.exit(1)
        src_emb = src_faces[0].normed_embedding
    else:
        src_emb = None  # will grab from first frame

    adapter = FaceShapeAdapter({
        "blend_strength": 0.7,
        "warp_method": "delaunay",
    })

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    cv2.namedWindow("FaceShapeAdapter Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FaceShapeAdapter Demo", 800, 600)

    blend_val = [70]  # trackbar value × 0.01 → float

    def _on_trackbar(val):
        blend_val[0] = val
        adapter.set_blend_strength(val / 100.0)

    cv2.createTrackbar("blend_strength ×100", "FaceShapeAdapter Demo", 70, 100, _on_trackbar)

    use_tps = [False]
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        faces = swapper.detect(frame)
        if not faces:
            cv2.imshow("FaceShapeAdapter Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face = faces[0]

        # Auto-source from first frame
        if src_emb is None:
            src_emb = face.normed_embedding

        # Align
        size = swapper._backend._size if hasattr(swapper._backend, "_size") else 256
        aligned, M = _align_face(frame, face.kps, size)

        # Swap
        swapped = swapper.swap_aligned(aligned, src_emb)
        if swapped is None:
            swapped = aligned.copy()

        # Shape adapt
        t0 = time.perf_counter()
        adapted = adapter.adapt(aligned, swapped)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        frame_times.append(dt_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)

        # Warp field magnitude visualization
        if adapter._prev_map_x is not None:
            gx, gy = np.meshgrid(np.arange(size, dtype=np.float32),
                                  np.arange(size, dtype=np.float32))
            dx = adapter._prev_map_x - gx
            dy = adapter._prev_map_y - gy
            mag = np.sqrt(dx**2 + dy**2)
            mag_vis = np.clip(mag / max(mag.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
            mag_vis_bgr = cv2.applyColorMap(mag_vis, cv2.COLORMAP_JET)
        else:
            mag_vis_bgr = np.zeros((size, size, 3), dtype=np.uint8)

        # Build 4-panel display
        disp_size = min(300, size)
        def _rs(img):
            return cv2.resize(img, (disp_size, disp_size))

        top = np.hstack([_rs(aligned), _rs(swapped)])
        bot = np.hstack([_rs(adapted), _rs(mag_vis_bgr)])
        panel = np.vstack([top, bot])

        # Stats overlay
        avg_ms = sum(frame_times) / len(frame_times)
        cv2.putText(panel, f"adapt: {avg_ms:.1f}ms  blend={blend_val[0]/100:.2f}",
                    (5, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        method = "TPS" if use_tps[0] else "Delaunay"
        cv2.putText(panel, f"method: {method}  [t]=toggle",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("FaceShapeAdapter Demo", panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            use_tps[0] = not use_tps[0]
            adapter.warp_method = "tps" if use_tps[0] else "delaunay"
            adapter.reset_temporal()
            print(f"Switched to {adapter.warp_method}")

    cap.release()
    cv2.destroyAllWindows()
