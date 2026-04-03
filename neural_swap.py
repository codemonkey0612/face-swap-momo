"""Neural face swap backend — supports two model families:

  • Ghost (ghost_3_256.onnx) — BEST quality, 256 px output  ← recommended
  • InsightFace inswapper (inswapper_128.onnx) — legacy, 128 px

Model auto-selection is based on filename:
  "ghost" / "simswap" / "blendswap" / "uniface" → Ghost backend
  anything else                                  → InsightFace backend

Ghost model requires a companion crossface_ghost.onnx embedding converter
(auto-discovered next to the ghost model).

Setup:
    pip install insightface onnxruntime-gpu   # (or onnxruntime for CPU)

Models (place in models/ directory):
    ghost_3_256.onnx       (~816 MB) — best quality
    crossface_ghost.onnx   (~22 MB)  — required companion for ghost
    inswapper_128.onnx     (~529 MB) — legacy alternative

Occlusion / face parsing models (optional, auto-discovered in models/):
    face_parser.onnx       (~90 MB)  — BiSeNet-FP 19-class face parsing (BEST)
    face_occluder.onnx     (~3 MB)   — binary face/occluder segmentation
    Download BiSeNet: https://github.com/facefusion/facefusion-assets/
                      releases/download/models-3.0.0/bisenet_resnet_34.onnx
                      → save as models/face_parser.onnx
    Priority: face_parser > face_occluder > MediaPipe Hands (CPU fallback).
"""

import os

import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

# 5-point ArcFace landmark template in 112×112 space
_ARCFACE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _best_providers():
    """Return best available ONNX execution providers, validated by a live DLL check.

    NOTE: cuDNN 9.x requires Turing+ GPU (SM 7.5+). Pascal GPUs (GTX 1050 = SM 6.1)
    are incompatible — force CPU on those. RTX 20xx+ will use GPU automatically.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        # Check GPU compute capability — skip CUDA on Pascal (SM < 7.0)
        if "CUDAExecutionProvider" in available:
            try:
                import subprocess, re
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                caps = result.stdout.strip().split("\n")
                max_cap = max(float(c.strip()) for c in caps if c.strip())
                if max_cap < 7.0:
                    print(f"[NeuralSwap] GPU compute capability {max_cap} < 7.0 "
                          f"(cuDNN 9.x requires Turing+) — using CPU")
                    return ["CPUExecutionProvider"]
            except Exception:
                pass
        for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider"]:
            if p in available:
                # Validate the provider DLL actually loads before committing to it.
                # On Windows, CUDA/cuDNN DLLs may be missing even when the provider
                # appears in get_available_providers().
                try:
                    import ctypes, os
                    ort_capi = os.path.join(os.path.dirname(ort.__file__), "capi")
                    if "CUDA" in p:
                        dll = os.path.join(ort_capi, "onnxruntime_providers_cuda.dll")
                        if os.path.exists(dll):
                            ctypes.CDLL(dll)  # raises OSError if deps missing
                except OSError:
                    print(f"[NeuralSwap] {p} DLL failed to load — falling back to CPU.")
                    continue
                return [p, "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _ctx_id(providers):
    for p in providers:
        if "CUDA" in p or "CoreML" in p:
            return 0
    return -1


def _align_face(frame, kps, size):
    """Affine-warp face to canonical size×size template using 5 keypoints."""
    template = _ARCFACE_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(
        kps, template, method=cv2.RANSAC, ransacReprojThreshold=100
    )
    if M is None:
        # Fallback if RANSAC fails
        M = cv2.getAffineTransform(kps[:3], template[:3])
    aligned = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
    return aligned, M


def _landmark_face_mask(frame_shape, target_face, erode_ratio=0.03):
    """Build a precise face mask from InsightFace landmark_2d_106 contour.

    Uses the 106-point landmark's face contour (indices 0-32) to create
    a convex hull that follows the actual jawline, chin, and face boundary.
    Falls back to bbox-based ellipse if landmarks are unavailable.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Try to use precise face contour from 106-point landmarks
    lm106 = getattr(target_face, 'landmark_2d_106', None)
    if lm106 is not None and len(lm106) >= 33:
        # Indices 0-32: face contour (jawline ear-to-ear + forehead boundary)
        contour_pts = lm106[:33].astype(np.int32)
        hull = cv2.convexHull(contour_pts)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Fallback: ellipse from bbox
        x1, y1, x2, y2 = (int(v) for v in target_face.bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = int((x2 - x1) * 0.45)
        ry = int((y2 - y1) * 0.50)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # Erode slightly to avoid warp-edge artifacts
    face_w = int(target_face.bbox[2] - target_face.bbox[0])
    erode_px = max(2, int(face_w * erode_ratio))
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    mask = cv2.erode(mask, kern, iterations=1)
    return mask


def _soft_blend(frame, pasted, mask_bin, blur_ratio=0.35):
    """Alpha-blend pasted face into frame using inward-only Gaussian feathering.

    The blur creates a gradual transition at the mask boundary but is
    **clamped** so it never expands beyond the original mask footprint.
    This prevents swapped-face pixels from leaking over occluding objects
    (hands, cups, microphones).
    """
    # Compute blur kernel size from mask extent
    where = np.where(mask_bin > 0)
    if len(where[0]) == 0:
        return frame
    face_h = where[0].max() - where[0].min()
    face_w = where[1].max() - where[1].min()
    face_size = max(face_h, face_w)

    # Gaussian blur for soft edges
    blur_px = max(11, int(face_size * blur_ratio))
    blur_px = blur_px | 1  # must be odd
    mask_f = mask_bin.astype(np.float32) / 255.0
    mask_soft = cv2.GaussianBlur(mask_f, (blur_px, blur_px), 0)

    # INWARD-ONLY: clamp so blurred mask never exceeds original boundary.
    # This prevents the swap from bleeding past occluders.
    mask_hard = (mask_f > 0.01).astype(np.float32)
    mask_soft = np.minimum(mask_soft, mask_hard)
    alpha = mask_soft[:, :, np.newaxis]

    # Blend
    return (
        pasted.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def _match_histogram_skin(src, dst, mask):
    """Per-channel histogram matching in the masked skin region.

    Adjusts src's color distribution to match dst's within the mask zone.
    Uses a gentle blend (70% corrected, 30% original) to avoid over-correction.
    """
    mb = mask > 0
    if not mb.any():
        return src
    out = src.copy().astype(np.float32)
    dst_f = dst.astype(np.float32)
    for c in range(3):
        s_vals = out[:, :, c][mb]
        d_vals = dst_f[:, :, c][mb]
        s_mean, s_std = s_vals.mean(), s_vals.std()
        d_mean, d_std = d_vals.mean(), d_vals.std()
        if s_std < 1e-6:
            continue
        # Normalize and re-scale to match destination
        corrected = (out[:, :, c] - s_mean) * (d_std / s_std) + d_mean
        # Gentle blend: 70% corrected + 30% original to avoid over-correction
        out[:, :, c] = 0.7 * corrected + 0.3 * out[:, :, c]
    return np.clip(out, 0, 255).astype(np.uint8)


def _detect_frame_occluders(frame, bbox, face_parser):
    """Detect occluding objects in frame space using BiSeNet (Face Dragon technique).

    Runs BiSeNet on the original camera frame crop (not the affine-aligned face)
    so hands, cups, mics appear as they really are — not distorted by the warp.

    Returns a float32 occluder mask [0,1] in full frame space:
      1.0 = occluder present (hand, cup, etc.)
      0.0 = clear face or outside analysis region

    IMPORTANT: parse_crop() returns zeros both outside the crop AND for
    background.  We compute the crop region explicitly so we only mark
    "not-face inside crop" as an occluder — zeros outside the crop are
    ignored (left as 0.0 = no occluder signal).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    fw, fh = x2 - x1, y2 - y1
    pad_x, pad_y = int(fw * 0.3), int(fh * 0.3)
    cy1, cx1 = max(0, y1 - pad_y), max(0, x1 - pad_x)
    cy2, cx2 = min(h, y2 + pad_y), min(w, x2 + pad_x)

    # Frame-space face probability from BiSeNet
    frame_face_prob = face_parser.parse_crop(frame, bbox)

    # Build a validity mask for the crop region
    crop_valid = np.zeros((h, w), dtype=np.float32)
    crop_valid[cy1:cy2, cx1:cx2] = 1.0

    # Occluder = inside crop region AND BiSeNet says "not face"
    not_face = np.clip(1.0 - frame_face_prob, 0.0, 1.0)
    occluder = crop_valid * not_face

    # Smooth first, then threshold — blurring before thresholding preserves
    # more of the occluder signal near the boundary.
    occluder = cv2.GaussianBlur(occluder, (5, 5), 0)

    # Threshold: suppress weak noise but keep genuine non-face signals.
    # 0.3 (was 0.6) — more sensitive to partially-occluded regions.
    occluder = np.where(occluder > 0.3, occluder, 0.0).astype(np.float32)

    return occluder


def _paste_back(frame, aligned_swap, M, size, target_face,
                occlusion_mask=None, parsed_face_mask=None,
                face_parser=None):
    """Paste swapped face back with occlusion-aware compositing.

    Face Dragon technique: foreground objects (hands, cups, mics) sit
    naturally ON TOP of the transformed face.

    Two-layer occlusion approach:
      1. **Aligned-space BiSeNet mask** — pixel-accurate face segmentation
         from the canonical-aligned face.
      2. **Frame-space occluder detection** — runs BiSeNet on the original
         camera crop to find objects that the aligned-space view may have
         missed (hands get distorted by affine warp and can fool BiSeNet).

    The frame-space occluder is SUBTRACTED from the aligned-space mask,
    but only within the analyzed crop region.  Areas outside the crop
    are untouched.
    """
    M_inv = cv2.invertAffineTransform(M)
    h, w = frame.shape[:2]

    # Warp swapped face back to original frame coordinates
    pasted = cv2.warpAffine(
        aligned_swap, M_inv, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    # --- Choose face mask ---
    if parsed_face_mask is not None:
        # BiSeNet-FP path: warp the parsed mask from aligned space to frame space.
        mask_f = cv2.warpAffine(
            parsed_face_mask, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderValue=0.0,
        )
        mask_f = np.clip(mask_f, 0.0, 1.0)

        # Frame-space occluder subtraction (Face Dragon technique)
        if face_parser is not None:
            occluder = _detect_frame_occluders(frame, target_face.bbox, face_parser)
            mask_f = mask_f * (1.0 - occluder)

        mask_bin = (mask_f * 255).astype(np.uint8)
    else:
        # Landmark-based fallback
        mask_bin = _landmark_face_mask(frame.shape, target_face)
        # Subtract separate occlusion mask if provided
        if occlusion_mask is not None:
            mask_f = mask_bin.astype(np.float32) / 255.0
            mask_f = mask_f * (1.0 - occlusion_mask)
            mask_bin = (mask_f * 255.0).astype(np.uint8)

    # Match skin color in the face region before blending
    pasted = _match_histogram_skin(pasted, frame, mask_bin)

    # Soft Gaussian blend — inward-only feathering
    return _soft_blend(frame, pasted, mask_bin)


def _sharpen(img, sigma=2.0, amount=0.4):
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


# ---------------------------------------------------------------------------
# Occlusion detection  (hands + non-skin objects)
# ---------------------------------------------------------------------------

class OcclusionDetector:
    """GPU-accelerated occlusion detection with ONNX model + temporal smoothing.

    Detection priority (first available wins):

      1. **ONNX face_occluder model on GPU** (~1-2 ms on RTX) — a lightweight
         segmentation model that directly classifies face vs non-face pixels.
         Runs via ONNX Runtime on the same CUDA device as the swap model.
         Handles hands, microphones, cups — any occluding object.

      2. **ONNX face_parser model on GPU** — multi-class face segmentation
         (BiSeNet-based).  Non-face classes within the face region = occluders.

      3. **MediaPipe Hands on CPU** (fallback, ~5-10 ms) — convex hull of
         hand landmarks.  Used when no ONNX occlusion model is available.

    All modes supplement with **dual-space adaptive skin-color analysis**
    (YCrCb + HSV with IQR tolerance) to catch non-skin objects that might
    be missed by the primary detector.

    **Temporal EMA smoothing** eliminates frame-to-frame flicker.
    Output is a **soft float32 mask** in [0, 1].
    """

    _OCCLUDER_URL = (
        "https://github.com/facefusion/facefusion-assets/"
        "releases/download/models-3.0.0/face_occluder.onnx"
    )

    def __init__(self, model_dir="models", providers=None, smooth_alpha=0.55):
        """
        Args:
            model_dir: Directory to search for face_occluder.onnx / face_parser.onnx.
            providers: ONNX Runtime execution providers (e.g. ['CUDAExecutionProvider']).
            smooth_alpha: EMA weight for the current frame (0.55 = good for 25-30 FPS).
        """
        import onnxruntime as ort

        self._providers = providers or _best_providers()
        self._smooth_alpha = smooth_alpha
        self._prev_mask = None
        self._sess = None
        self._hands = None
        self._is_parser = False

        # --- Priority 1 & 2: ONNX face occlusion model on GPU ---
        for name, is_parser in [("face_occluder.onnx", False), ("face_parser.onnx", True)]:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                self._sess = ort.InferenceSession(path, providers=self._providers)
                inp = self._sess.get_inputs()[0]
                self._input_name = inp.name
                self._input_h = inp.shape[2]
                self._input_w = inp.shape[3]
                self._output_name = self._sess.get_outputs()[0].name
                self._is_parser = is_parser
                gpu = any("CUDA" in p or "CoreML" in p for p in self._providers)
                print(f"[Occlusion] {'GPU' if gpu else 'CPU'} model: {name} "
                      f"({self._input_w}x{self._input_h}) + EMA α={smooth_alpha}")
                break

        # --- Priority 3: MediaPipe Hands fallback (CPU) ---
        if self._sess is None:
            if MEDIAPIPE_AVAILABLE:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print(f"[Occlusion] CPU fallback: MediaPipe Hands + skin analysis + EMA α={smooth_alpha}")
                print(f"[Occlusion] For GPU acceleration, download face_occluder.onnx (~3 MB):")
                print(f"[Occlusion]   {self._OCCLUDER_URL}")
                print(f"[Occlusion]   Place at: {os.path.abspath(os.path.join(model_dir, 'face_occluder.onnx'))}")
            else:
                print("[Occlusion] WARNING: No ONNX model and no mediapipe — skin-color only")

    # ------------------------------------------------------------------
    # GPU path: ONNX face occlusion / parsing model
    # ------------------------------------------------------------------

    def _detect_model(self, frame, face_mask, target_face):
        """Run ONNX face occluder/parser on a padded face crop (GPU)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in target_face.bbox)

        # Pad bbox by 30% for surrounding context
        fw, fh = x2 - x1, y2 - y1
        pad_x, pad_y = int(fw * 0.3), int(fh * 0.3)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        # Prep: BGR→RGB, resize to model input, normalize [0,1], NCHW
        rgb = crop[:, :, ::-1].copy()
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # GPU inference
        output = self._sess.run([self._output_name], {self._input_name: blob})[0]

        if self._is_parser:
            # Multi-class face parser: face classes → NOT occluded
            classes = output[0].argmax(axis=0)  # (H, W)
            # BiSeNet face classes: 1=skin, 2-3=brows, 4-5=eyes, 6=glasses,
            # 7-8=ears, 9=earring, 10=nose, 11=mouth, 12-13=lips
            face_classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
            occ_prob = (~np.isin(classes, list(face_classes))).astype(np.float32)
        else:
            # Binary occluder: output is face probability → invert for occlusion
            occ_prob = 1.0 - np.clip(output[0, 0], 0.0, 1.0)

        # Resize back to crop dimensions
        occ_crop = cv2.resize(occ_prob, (cx2 - cx1, cy2 - cy1))

        # Place into full-frame mask
        full = np.zeros((h, w), dtype=np.float32)
        full[cy1:cy2, cx1:cx2] = occ_crop

        # Restrict to face region
        return full * (face_mask.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------
    # CPU fallback: MediaPipe Hands
    # ------------------------------------------------------------------

    def _detect_hands(self, rgb, h, w, face_mask):
        """Return float32 [0,1] hand mask, dilated proportional to face size."""
        mask = np.zeros((h, w), dtype=np.float32)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return mask

        for hand_lm in results.multi_hand_landmarks:
            pts = np.array(
                [[int(lm.x * w), int(lm.y * h)] for lm in hand_lm.landmark],
                dtype=np.int32,
            )
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 1.0)

        # Adaptive dilation proportional to the face radius
        face_area = cv2.countNonZero(face_mask)
        if face_area > 0:
            face_radius = int(np.sqrt(face_area / np.pi))
            d = max(8, int(face_radius * 0.08))
        else:
            d = max(8, int(min(h, w) * 0.02))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
        mask = cv2.dilate(mask, kern)
        return mask

    # ------------------------------------------------------------------
    # Supplementary: adaptive skin-color analysis (fast CPU, ~1-2 ms)
    # ------------------------------------------------------------------

    def _detect_nonskin(self, frame, face_mask):
        """Adaptive dual-space (YCrCb + HSV) non-skin detection in the face region.

        Skin tolerance is derived from the IQR of the face's own pixel
        distribution — adapts to any skin tone and lighting.
        A pixel is considered skin if *either* colour space votes yes (union),
        minimising false-positive occlusion.
        """
        face_bool = face_mask > 0

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        fp_ycc = ycrcb[face_bool]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fp_hsv = hsv[face_bool]

        if len(fp_ycc) < 200:
            skin = cv2.inRange(ycrcb, (0, 120, 65), (255, 185, 140))
        else:
            # YCrCb adaptive range: median ± 2.5× IQR
            med_cr = np.median(fp_ycc[:, 1])
            med_cb = np.median(fp_ycc[:, 2])
            iqr_cr = float(np.percentile(fp_ycc[:, 1], 75) - np.percentile(fp_ycc[:, 1], 25))
            iqr_cb = float(np.percentile(fp_ycc[:, 2], 75) - np.percentile(fp_ycc[:, 2], 25))
            cr_tol = max(22, int(iqr_cr * 2.5))
            cb_tol = max(22, int(iqr_cb * 2.5))
            skin_ycc = cv2.inRange(
                ycrcb,
                np.array([0, max(0, med_cr - cr_tol), max(0, med_cb - cb_tol)], dtype=np.uint8),
                np.array([255, min(255, med_cr + cr_tol), min(255, med_cb + cb_tol)], dtype=np.uint8),
            )

            # HSV adaptive range: median ± 2.5× IQR
            med_h = np.median(fp_hsv[:, 0])
            med_s = np.median(fp_hsv[:, 1])
            iqr_h = float(np.percentile(fp_hsv[:, 0], 75) - np.percentile(fp_hsv[:, 0], 25))
            iqr_s = float(np.percentile(fp_hsv[:, 1], 75) - np.percentile(fp_hsv[:, 1], 25))
            h_tol = max(15, int(iqr_h * 2.5))
            s_tol = max(30, int(iqr_s * 2.5))
            skin_hsv = cv2.inRange(
                hsv,
                np.array([max(0, med_h - h_tol), max(0, med_s - s_tol), 30], dtype=np.uint8),
                np.array([min(179, med_h + h_tol), min(255, med_s + s_tol), 255], dtype=np.uint8),
            )

            skin = cv2.bitwise_or(skin_ycc, skin_hsv)

        non_skin = cv2.bitwise_and(cv2.bitwise_not(skin), face_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_OPEN, k)
        non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_CLOSE, k)
        return non_skin.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, face_mask, target_face=None):
        """Build a soft occlusion mask — float32 in [0, 1], temporally smoothed.

        Args:
            frame: Original BGR frame (before swap).
            face_mask: Binary uint8 mask of the expected face region.
            target_face: Face object with bbox (required for ONNX model path).

        Returns:
            float32 mask in [0, 1] — 1.0 = fully occluded, 0.0 = no occlusion.
        """
        h, w = frame.shape[:2]

        # Primary detection layer
        if self._sess is not None and target_face is not None:
            primary = self._detect_model(frame, face_mask, target_face)
        elif self._hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            primary = self._detect_hands(rgb, h, w, face_mask)
        else:
            primary = np.zeros((h, w), dtype=np.float32)

        # Supplementary: skin-color analysis (catches non-skin objects the
        # primary detector might miss — fast CPU, ~1-2 ms)
        nonskin = self._detect_nonskin(frame, face_mask)

        # Fuse: either signal triggers occlusion
        raw = np.maximum(primary, nonskin)

        # Restrict to face region
        raw = raw * (face_mask.astype(np.float32) / 255.0)

        # Temporal EMA smoothing — eliminates flicker, creates smooth fade
        if self._prev_mask is not None and self._prev_mask.shape == raw.shape:
            smoothed = self._smooth_alpha * raw + (1.0 - self._smooth_alpha) * self._prev_mask
        else:
            smoothed = raw
        self._prev_mask = smoothed.copy()

        return smoothed

    def close(self):
        if self._hands is not None:
            self._hands.close()
        self._prev_mask = None


# ---------------------------------------------------------------------------
# BiSeNet-FP face parser  (face_parser.onnx — best-quality masking)
# ---------------------------------------------------------------------------

class FaceParser:
    """BiSeNet-FP 19-class face parsing on GPU.

    Produces a per-pixel soft face mask from semantic segmentation.  When
    used, it **replaces** both the landmark-based face mask and the separate
    occlusion detector — a single GPU call (~2-3 ms on RTX) handles
    everything.

    The key insight: BiSeNet classifies every pixel as one of 19 face/non-face
    classes.  Anything within the face bbox that is NOT a face class (skin,
    brows, eyes, nose, lips) is automatically excluded — hands, microphones,
    cups, glasses, hair all get their own classes and are naturally masked out.

    BiSeNet 19-class labels:
        0=background  1=skin      2=l_brow   3=r_brow   4=l_eye    5=r_eye
        6=glasses     7=l_ear     8=r_ear    9=earring  10=nose    11=mouth
        12=upper_lip  13=lower_lip 14=neck   15=necklace 16=cloth  17=hair  18=hat

    Download face_parser.onnx (~15 MB):
        https://github.com/facefusion/facefusion-assets/
        releases/download/models-3.0.0/face_parser.onnx
    """

    # Classes that constitute "swap target" — only these get the swapped face
    FACE_CLASSES = frozenset({1, 2, 3, 4, 5, 10, 11, 12, 13})  # skin, brows, eyes, nose, lips
    EAR_CLASSES = frozenset({7, 8})  # include ears in blend region

    _PARSER_URL = (
        "https://github.com/facefusion/facefusion-assets/"
        "releases/download/models-3.0.0/bisenet_resnet_34.onnx"
    )

    def __init__(self, model_path, providers):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(model_path, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self._size = inp.shape[2]  # typically 512
        self._output_name = self._sess.get_outputs()[0].name
        # Inner face only — exclude EAR_CLASSES (7, 8).
        # Ears sit at the widest part of the face boundary, which is exactly
        # where source/target face-width mismatch is most visible.  Including
        # ear pixels forces the swap mask out to the edge of the bounding box,
        # creating a seam whenever the two faces have different widths.
        # Limiting to FACE_CLASSES (skin, brows, eyes, nose, lips) keeps the
        # mask well inside the outer face contour where swap quality is highest.
        self._include = self.FACE_CLASSES
        self._last_mask = None  # for debug overlay

        gpu = any("CUDA" in p or "CoreML" in p for p in providers)
        print(f"[FaceParser] BiSeNet-FP {self._size}x{self._size} on "
              f"{'GPU' if gpu else 'CPU'} — replaces landmark mask + occlusion")

    def _infer(self, bgr_image):
        """Run BiSeNet inference and return softmax class probabilities."""
        resized = cv2.resize(bgr_image, (self._size, self._size))
        # BGR → RGB, normalize to [0, 1], NCHW
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])

        output = self._sess.run([self._output_name], {self._input_name: blob})[0]
        logits = output[0]  # (C, H, W)

        # Numerically stable softmax
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=0, keepdims=True)  # (C, H, W)

    def _face_prob(self, probs, out_h, out_w):
        """Sum included class probabilities → soft face mask [0, 1]."""
        face_mask = sum(probs[c] for c in self._include)
        face_mask = np.clip(face_mask, 0.0, 1.0)
        if face_mask.shape != (out_h, out_w):
            face_mask = cv2.resize(face_mask, (out_w, out_h))
        return face_mask

    def parse_aligned(self, aligned_bgr):
        """Parse an aligned face — returns soft face mask in aligned space.

        This is the high-quality path for the Ghost backend: the parser runs
        on the same aligned coordinate system as the swap model, so the mask
        is pixel-accurate.

        Args:
            aligned_bgr: BGR uint8 aligned face (e.g. 256×256).

        Returns:
            float32 mask [0, 1] — face=high, occluder=low.
        """
        h, w = aligned_bgr.shape[:2]
        probs = self._infer(aligned_bgr)
        mask = self._face_prob(probs, h, w)
        self._last_mask = mask
        return mask

    def parse_crop(self, frame, bbox):
        """Parse a padded bbox crop — returns soft face mask in frame space.

        Used for InsightFace backend (post-processing) where we can't access
        the aligned face directly.

        Args:
            frame: Full BGR frame.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            float32 mask [0, 1] in full frame coordinate space.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        fw, fh = x2 - x1, y2 - y1
        pad_x, pad_y = int(fw * 0.3), int(fh * 0.3)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        probs = self._infer(crop)
        face_crop = self._face_prob(probs, cy2 - cy1, cx2 - cx1)

        full = np.zeros((h, w), dtype=np.float32)
        full[cy1:cy2, cx1:cx2] = face_crop
        self._last_mask = full
        return full

    @property
    def last_mask(self):
        """Last parsed face mask — for debug overlay."""
        return self._last_mask


# ---------------------------------------------------------------------------
# Ghost backend  (ghost_3_256.onnx + crossface_ghost.onnx)
# ---------------------------------------------------------------------------

class _GhostBackend:
    """
    Pipeline:
      1. Detect face → 5 kps + normed_embedding via buffalo_l
      2. Convert source embedding through crossface_ghost.onnx
      3. Align target to 256×256 canonical template
      4. ONNX inference: [target (1,3,256,256), source_emb (1,512)] → swapped (1,3,256,256)
      5. Color-correct swapped to match target lighting in aligned space
      6. Mild sharpening (256 px is reasonably crisp)
      7. Paste back via inverse affine + feathered mask
    """

    def __init__(self, model_path, crossface_path, providers):
        import onnxruntime as ort

        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=_ctx_id(providers), det_size=(640, 640), det_thresh=0.3)

        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._crossface = ort.InferenceSession(crossface_path, providers=providers)

        # Ghost model I/O:  target (1,3,256,256) + source (1,512) → output (1,3,256,256)
        inputs = self._sess.get_inputs()
        self._frame_input = None
        self._embed_input = None
        self._size = 256

        for inp in inputs:
            if len(inp.shape) == 4:          # (1, 3, H, W)
                self._frame_input = inp.name
                self._size = inp.shape[-1]
            elif len(inp.shape) == 2:        # (1, 512)
                self._embed_input = inp.name

        self._output_name = self._sess.get_outputs()[0].name

        # Crossface I/O:  input (1,512) → output (1,512)
        self._cf_input = self._crossface.get_inputs()[0].name
        self._cf_output = self._crossface.get_outputs()[0].name

        print(f"[Ghost] size={self._size}  inputs=({self._frame_input}, {self._embed_input})")

    # -- helpers --

    def _convert_embedding(self, normed_embedding):
        """Run ArcFace embedding through crossface converter for ghost compatibility."""
        emb = normed_embedding[np.newaxis].astype(np.float32) if normed_embedding.ndim == 1 else normed_embedding.astype(np.float32)
        return self._crossface.run([self._cf_output], {self._cf_input: emb})[0]

    def _prep(self, aligned_bgr):
        """BGR uint8 → NCHW float32 in [-1, 1]."""
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        return np.expand_dims(rgb.transpose(2, 0, 1), 0)

    def _restore(self, nchw):
        """NCHW float32 in [-1, 1] → BGR uint8."""
        rgb = np.clip(nchw[0], -1, 1)
        rgb = ((rgb + 1) / 2 * 255).round().astype(np.uint8)
        return rgb.transpose(1, 2, 0)[:, :, ::-1]

    def _run(self, aligned_bgr, converted_emb):
        target_in = self._prep(aligned_bgr)
        feed = {
            self._frame_input: target_in,
            self._embed_input: converted_emb,
        }
        out = self._sess.run([self._output_name], feed)[0]
        return self._restore(out)

    def _swap_one(self, frame, target_face, converted_emb,
                  occlusion_mask=None, face_parser=None):
        aligned, M = _align_face(frame, target_face.kps, self._size)
        swapped_aligned = self._run(aligned, converted_emb)
        swapped_aligned = _sharpen(swapped_aligned, sigma=1.5, amount=0.25)

        # BiSeNet-FP: parse the aligned face for a pixel-accurate mask
        parsed_mask = None
        if face_parser is not None:
            parsed_mask = face_parser.parse_aligned(aligned)

        # Pass face_parser for frame-space cross-validation (Face Dragon technique)
        return _paste_back(frame, swapped_aligned, M, self._size, target_face,
                           occlusion_mask=occlusion_mask,
                           parsed_face_mask=parsed_mask,
                           face_parser=face_parser)

    # -- public --

    def detect(self, img):
        faces = self._app.get(img)
        return sorted(faces, key=lambda f: f.bbox[0]) if faces else []

    def swap(self, frame, target_face, source_face,
             occlusion_mask=None, face_parser=None):
        converted_emb = self._convert_embedding(source_face.normed_embedding)
        return self._swap_one(frame, target_face, converted_emb,
                              occlusion_mask, face_parser)



# ---------------------------------------------------------------------------
# InsightFace inswapper backend  (inswapper_128.onnx — legacy)
# ---------------------------------------------------------------------------

class _InsightFaceBackend:
    """Wraps InsightFace inswapper.get() with sharpening post-process."""

    def __init__(self, model_path, providers):
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=_ctx_id(providers), det_size=(640, 640), det_thresh=0.3)

        self._swapper = insightface.model_zoo.get_model(
            model_path, download=False, download_zip=False,
        )
        self._swapper.prepare(ctx_id=_ctx_id(providers))

    def detect(self, img):
        faces = self._app.get(img)
        return sorted(faces, key=lambda f: f.bbox[0]) if faces else []

    def _sharpen_bbox(self, img, face):
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        region = img[y1:y2, x1:x2]
        if region.size == 0:
            return img
        out = img.copy()
        out[y1:y2, x1:x2] = _sharpen(region, sigma=3.0, amount=0.5)
        return out

    def swap(self, frame, target_face, source_face,
             occlusion_mask=None, face_parser=None):
        result = frame.copy()
        result = self._swapper.get(result, target_face, source_face, paste_back=True)
        result = self._sharpen_bbox(result, target_face)

        # InsightFace has its own internal paste_back — apply occlusion as
        # post-processing by restoring original pixels in occluded areas.
        if face_parser is not None:
            # BiSeNet-FP: parse the face crop → get face probability map
            # Occlusion = where parser says "not face" within the face region
            face_prob = face_parser.parse_crop(frame, target_face.bbox)
            lm_mask = _landmark_face_mask(frame.shape, target_face)
            lm_f = lm_mask.astype(np.float32) / 255.0
            # Occluded = inside landmark region but NOT face according to parser
            occ = np.clip(lm_f - face_prob, 0.0, 1.0)
            if occ.max() > 0.05:
                blur_px = max(15, int(min(frame.shape[:2]) * 0.025)) | 1
                occ_hard = (occ > 0.01).astype(np.float32)
                alpha_raw = cv2.GaussianBlur(occ, (blur_px, blur_px), 0)
                # Inward-only: don't expand occlusion restoration into face
                alpha = np.minimum(alpha_raw, occ_hard)[:, :, np.newaxis]
                result = (
                    frame.astype(np.float32) * alpha
                    + result.astype(np.float32) * (1.0 - alpha)
                ).astype(np.uint8)
        elif occlusion_mask is not None and occlusion_mask.any():
            blur_px = max(15, int(min(frame.shape[:2]) * 0.025)) | 1
            occ_hard = (occlusion_mask > 0.01).astype(np.float32)
            alpha_raw = cv2.GaussianBlur(occlusion_mask, (blur_px, blur_px), 0)
            alpha = np.minimum(alpha_raw, occ_hard)[:, :, np.newaxis]
            result = (
                frame.astype(np.float32) * alpha
                + result.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        return result


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def _is_ghost_model(path):
    name = os.path.basename(path).lower()
    return any(k in name for k in ("ghost", "simswap", "blendswap", "uniface"))


def _find_crossface(model_path):
    """Look for crossface_ghost.onnx next to the ghost model."""
    model_dir = os.path.dirname(model_path) or "."
    crossface = os.path.join(model_dir, "crossface_ghost.onnx")
    if os.path.exists(crossface):
        return crossface
    raise FileNotFoundError(
        f"crossface_ghost.onnx not found in {os.path.abspath(model_dir)}\n"
        "Download from: https://github.com/facefusion/facefusion-assets/"
        "releases/download/models-3.4.0/crossface_ghost.onnx\n"
        f"Place at: {os.path.abspath(crossface)}"
    )


def build_averaged_face(swapper, image_paths):
    """Build a synthetic Face object with an averaged normed_embedding.

    Detects faces in each image, collects their ArcFace embeddings,
    averages them, and L2-normalizes. This produces a more robust
    identity representation than any single photo — less sensitive to
    angle, expression, and lighting variation.

    Args:
        swapper: NeuralFaceSwapper instance (already initialized).
        image_paths: list of image file paths.

    Returns:
        A Face object whose normed_embedding is the averaged identity.
        Also carries kps/bbox from the best-quality detection (largest face).
    """
    embeddings = []
    best_face = None
    best_area = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] Cannot read: {path}")
            continue
        faces = swapper.detect(img)
        if not faces:
            print(f"  [skip] No face found: {path}")
            continue
        face = faces[0]  # largest / leftmost
        embeddings.append(face.normed_embedding)
        # Track the largest face for kps/bbox reference
        x1, y1, x2, y2 = face.bbox
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_face = face
        print(f"  [OK] {os.path.basename(path)}")

    if not embeddings:
        raise ValueError("No faces detected in any of the provided images")

    # Average and L2-normalize
    stacked = np.stack(embeddings, axis=0)
    avg_emb = stacked.mean(axis=0).astype(np.float32)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

    # normed_embedding is a read-only property computed from embedding,
    # so we must set embedding directly (preserving the original norm).
    orig_norm = np.linalg.norm(best_face.embedding) if best_face.embedding is not None else 1.0
    best_face.embedding = avg_emb * orig_norm

    print(f"[MultiSource] Averaged {len(embeddings)} embeddings from {len(image_paths)} images")
    return best_face


class NeuralFaceSwapper:
    """Auto-selects Ghost or InsightFace backend from the model filename.

    Occlusion handling priority (auto-selected by model availability):
      1. BiSeNet-FP face_parser.onnx — best quality, single GPU call replaces
         both face mask and occlusion detection.
      2. OcclusionDetector (face_occluder.onnx → MediaPipe → skin-color) —
         fallback when no face parser available.
    """

    def __init__(self, model_path="models/ghost_3_256.onnx", occlusion=True):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface is not installed.\n"
                "Run: pip install insightface onnxruntime-gpu"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Place at: {os.path.abspath(model_path)}"
            )

        providers = _best_providers()
        model_dir = os.path.dirname(model_path) or "models"
        print(f"[NeuralSwap] providers={providers}  model={os.path.basename(model_path)}")

        if _is_ghost_model(model_path):
            crossface_path = _find_crossface(model_path)
            self._backend = _GhostBackend(model_path, crossface_path, providers)
            print("[NeuralSwap] Backend: Ghost (256 px) — best quality")
        else:
            self._backend = _InsightFaceBackend(model_path, providers)
            print("[NeuralSwap] Backend: InsightFace inswapper (128 px) — legacy")

        # --- Occlusion strategy: BiSeNet-FP > OcclusionDetector > none ---
        self._face_parser = None
        self._occlusion = None
        self._last_occlusion_mask = None   # exposed for debug overlay

        if occlusion:
            # Priority 1: BiSeNet-FP face parser (best — replaces everything)
            for name in ("face_parser.onnx", "bisenet_resnet_34.onnx", "bisenet_fp.onnx"):
                parser_path = os.path.join(model_dir, name)
                if os.path.exists(parser_path):
                    self._face_parser = FaceParser(parser_path, providers)
                    break

            # Priority 2: OcclusionDetector fallback
            if self._face_parser is None:
                self._occlusion = OcclusionDetector(
                    model_dir=model_dir, providers=providers,
                )
                print(f"[NeuralSwap] For best occlusion quality, download face_parser.onnx:")
                print(f"[NeuralSwap]   {FaceParser._PARSER_URL}")
                print(f"[NeuralSwap]   Place at: {os.path.abspath(os.path.join(model_dir, 'face_parser.onnx'))}")

    def detect(self, img):
        return self._backend.detect(img)

    def swap(self, img, target_face, source_face):
        """Swap target face with source identity, with occlusion-aware blending.

        Routing:
          • face_parser available → pass to backend (Ghost: aligned-space parsing,
            InsightFace: bbox-crop post-processing).  No separate occlusion step.
          • Otherwise → run OcclusionDetector, pass occlusion mask to backend.
        """
        if self._face_parser is not None:
            # BiSeNet-FP path — parser handles both mask and occlusion natively
            result = self._backend.swap(
                img, target_face, source_face, face_parser=self._face_parser,
            )
            self._last_occlusion_mask = None
            return result

        # OcclusionDetector fallback path
        occ_mask = None
        if self._occlusion is not None:
            face_mask = _landmark_face_mask(img.shape, target_face)
            occ_mask = self._occlusion.detect(img, face_mask, target_face)
            if occ_mask.max() < 0.05:
                occ_mask = None

        self._last_occlusion_mask = occ_mask
        return self._backend.swap(img, target_face, source_face,
                                  occlusion_mask=occ_mask)

    @property
    def last_occlusion_mask(self):
        """Last occlusion mask (float32 [0,1] or None).  For debug overlay."""
        return self._last_occlusion_mask

    @property
    def face_parser(self):
        """Active FaceParser instance, or None.  For debug overlay."""
        return self._face_parser
