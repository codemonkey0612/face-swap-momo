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
    """Return best available ONNX execution providers."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider"]:
            if p in available:
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
    """Alpha-blend pasted face into frame using a heavily Gaussian-blurred mask.

    The wide blur (35% of face size) creates a very gradual transition that
    hides any color/texture discontinuity at the boundary without the
    artifacts that seamlessClone introduces on neural swap output.
    """
    # Compute blur kernel size from mask extent
    where = np.where(mask_bin > 0)
    if len(where[0]) == 0:
        return frame
    face_h = where[0].max() - where[0].min()
    face_w = where[1].max() - where[1].min()
    face_size = max(face_h, face_w)

    # Wide Gaussian blur for very soft feathering
    blur_px = max(11, int(face_size * blur_ratio))
    blur_px = blur_px | 1  # must be odd
    mask_f = mask_bin.astype(np.float32) / 255.0
    mask_soft = cv2.GaussianBlur(mask_f, (blur_px, blur_px), 0)
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


def _paste_back(frame, aligned_swap, M, size, target_face):
    """Paste swapped face back using landmark-based mask + soft Gaussian blend.

    This is the approach used by top face swap tools (FaceFusion, etc.):
      1. Build precise face mask from landmark_2d_106 contour (or bbox fallback)
      2. Match skin color of swapped face to original in the face region
      3. Wide Gaussian blur on mask for very soft, invisible transition
      4. Simple alpha blend — NO seamlessClone (it causes artifacts on neural output)
    """
    M_inv = cv2.invertAffineTransform(M)
    h, w = frame.shape[:2]

    # Warp swapped face back to original frame coordinates
    pasted = cv2.warpAffine(
        aligned_swap, M_inv, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    # Build precise face mask from landmarks in frame space
    mask_bin = _landmark_face_mask(frame.shape, target_face)

    # Match skin color in the face region before blending
    pasted = _match_histogram_skin(pasted, frame, mask_bin)

    # Soft Gaussian blend — wide blur hides edge discontinuity
    return _soft_blend(frame, pasted, mask_bin)


def _sharpen(img, sigma=2.0, amount=0.4):
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


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

    def _swap_one(self, frame, target_face, converted_emb):
        aligned, M = _align_face(frame, target_face.kps, self._size)
        swapped_aligned = self._run(aligned, converted_emb)
        swapped_aligned = _sharpen(swapped_aligned, sigma=1.5, amount=0.25)
        return _paste_back(frame, swapped_aligned, M, self._size, target_face)

    # -- public --

    def detect(self, img):
        faces = self._app.get(img)
        return sorted(faces, key=lambda f: f.bbox[0]) if faces else []

    def swap(self, frame, target_face, source_face):
        converted_emb = self._convert_embedding(source_face.normed_embedding)
        return self._swap_one(frame, target_face, converted_emb)

    def swap_bidirectional(self, frame, faces):
        face_a, face_b = faces[0], faces[1]
        emb_a = self._convert_embedding(face_a.normed_embedding)
        emb_b = self._convert_embedding(face_b.normed_embedding)
        result = self._swap_one(frame, face_b, emb_a)
        result = self._swap_one(result, face_a, emb_b)
        return result


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
        self._swapper.prepare(ctx_id=0)

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

    def swap(self, frame, target_face, source_face):
        result = frame.copy()
        result = self._swapper.get(result, target_face, source_face, paste_back=True)
        return self._sharpen_bbox(result, target_face)

    def swap_bidirectional(self, frame, faces):
        face_a, face_b = faces[0], faces[1]
        result = self._swapper.get(frame.copy(), face_b, face_a, paste_back=True)
        result = self._sharpen_bbox(result, face_b)
        result = self._swapper.get(result, face_a, face_b, paste_back=True)
        result = self._sharpen_bbox(result, face_a)
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
    """Auto-selects Ghost or InsightFace backend from the model filename."""

    def __init__(self, model_path="models/ghost_3_256.onnx"):
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
        print(f"[NeuralSwap] providers={providers}  model={os.path.basename(model_path)}")

        if _is_ghost_model(model_path):
            crossface_path = _find_crossface(model_path)
            self._backend = _GhostBackend(model_path, crossface_path, providers)
            print("[NeuralSwap] Backend: Ghost (256 px) — best quality")
        else:
            self._backend = _InsightFaceBackend(model_path, providers)
            print("[NeuralSwap] Backend: InsightFace inswapper (128 px) — legacy")

    def detect(self, img):
        return self._backend.detect(img)

    def swap(self, img, target_face, source_face):
        return self._backend.swap(img, target_face, source_face)

    def swap_bidirectional(self, frame, faces):
        return self._backend.swap_bidirectional(frame, faces)
