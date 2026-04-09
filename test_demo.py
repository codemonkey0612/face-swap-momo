"""Demo v5 — clean pipeline, no over-processing."""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_swap import NeuralFaceSwapper, FaceParser, _align_face, _best_providers, _sharpen
from src.shape.face_shape_adapter import FaceShapeAdapter


def main():
    source_dir = "source_faces"
    input_image = "original_face.png"
    if not os.path.exists(input_image) or os.path.getsize(input_image) < 1000:
        input_image = ".venv/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"
        print(f"[Demo] Using test image: {input_image}")

    frame = cv2.imread(input_image)
    if frame is None:
        print(f"ERROR: Cannot read {input_image}")
        return
    pristine = frame.copy()
    h, w = frame.shape[:2]

    # --- Load ---
    providers = _best_providers()
    swapper = NeuralFaceSwapper(model_path="models/ghost_3_256.onnx", occlusion=False)
    backend = swapper._backend
    sz = backend._size

    parser = None
    p = os.path.join("models", "face_parser.onnx")
    if os.path.exists(p):
        parser = FaceParser(p, providers)

    cf_sess = None
    p = os.path.join("models", "codeformer.onnx")
    if os.path.exists(p):
        import onnxruntime as ort
        cf_sess = ort.InferenceSession(p, providers=providers)

    shape_adapter = None
    try:
        shape_adapter = FaceShapeAdapter({
            "blend_strength": 0.85, "inner_radius": 0.20, "outer_radius": 0.75,
            "warp_method": "delaunay", "jawline_blend": 0.95,
            "eye_blend": 0.12, "nose_blend": 0.08, "mouth_blend": 0.08,
        })
    except Exception:
        pass

    # --- Detect ---
    target_face = swapper.detect(frame)
    if not target_face:
        print("ERROR: No face"); return
    target_face = target_face[0]

    source_img = cv2.imread(os.path.join(source_dir, "front_neutral.png"))
    source_face = swapper.detect(source_img)
    if not source_face:
        print("ERROR: No source face"); return
    source_face = source_face[0]
    src_emb = backend._convert_embedding(source_face.normed_embedding)
    src_aligned, _ = _align_face(source_img, source_face.kps, sz)

    # --- ALIGNED SPACE ---
    aligned, M = _align_face(frame, target_face.kps, sz)

    # 1. Ghost swap
    print("[1] Ghost swap")
    swapped = backend._run(aligned, src_emb)

    # 2. Shape match to original face contours
    if shape_adapter:
        print("[2] Shape adaptation")
        swapped = shape_adapter.adapt(aligned, swapped)

    # 3. HISTOGRAM MATCH — force exact color distribution from AI face
    print("[3] Histogram color match to AI face")
    swapped = _histogram_match_lab(swapped, src_aligned)

    # 4. Per-region brightness correction to match AI face exactly
    print("[4] Regional brightness matching")
    swapped = _regional_brightness_match(swapped, src_aligned)

    # 5. Skin smoothing
    print("[5] Skin smoothing")
    swapped = _bilateral_smooth(swapped, strength=0.35)

    # 6. CodeFormer for detail
    if cf_sess:
        print("[6] CodeFormer")
        swapped = _codeformer(cf_sess, swapped, fidelity=0.35)

    # 7. Re-apply color match + regional correction AFTER CodeFormer
    print("[7] Re-match color after CodeFormer")
    swapped = _histogram_match_lab(swapped, src_aligned)
    swapped = _regional_brightness_match(swapped, src_aligned)

    # 8. FINAL — pixel-level color match using dense flow from AI source
    print("[8] Pixel-level color match")
    swapped = _pixelwise_color_match(swapped, src_aligned)

    # 9. Pupil sharpening
    print("[9] Pupil sharpening")
    swapped = _sharpen_pupils(swapped)

    # 10. Final sharpen
    swapped = _sharpen(swapped, sigma=0.8, amount=0.25)

    # Debug: save aligned comparison
    cv2.imwrite("debug_aligned_compare.png", np.hstack([aligned, swapped, src_aligned]))

    # --- COMPOSITE ---
    # Face ellipse mask (large enough to cover full face)
    ell = np.zeros((sz, sz), dtype=np.float32)
    cv2.ellipse(ell, (sz // 2, sz // 2),
                (int(sz * 0.48), int(sz * 0.55)), 0, 0, 360, 1.0, -1)

    # BiSeNet inner face + dilate for coverage
    if parser:
        inner = parser.parse_aligned(aligned)
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        inner = cv2.dilate((inner * 255).astype(np.uint8), dk).astype(np.float32) / 255.0
        mask = ell * inner
    else:
        mask = ell

    # Feather inward
    mask = cv2.GaussianBlur(mask, (41, 41), 7.0)
    mask = np.minimum(mask, (mask > 0.005).astype(np.float32))

    # Composite in aligned space
    c3 = mask[:, :, np.newaxis]
    comp = (c3 * swapped.astype(np.float32)
            + (1.0 - c3) * aligned.astype(np.float32)).astype(np.uint8)

    # Warp back to frame
    M_inv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(comp, M_inv, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wmask = cv2.warpAffine(ell, M_inv, (w, h),
                            flags=cv2.INTER_LINEAR, borderValue=0.0)
    wmask = np.clip(wmask, 0, 1)
    wmask = cv2.GaussianBlur(wmask, (31, 31), 6.0)
    wmask = np.minimum(wmask, (wmask > 0.005).astype(np.float32))

    m3 = wmask[:, :, np.newaxis]
    result = (m3 * warped.astype(np.float32)
              + (1.0 - m3) * pristine.astype(np.float32))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # --- Save ---
    cv2.imwrite("demo_swapped.png", result)

    th = min(h, 720)
    sc = th / h
    i1 = cv2.resize(pristine, (int(w * sc), th))
    i2 = cv2.resize(result, (int(w * sc), th))
    cv2.putText(i1, "BEFORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(i2, "AFTER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite("demo_result.png",
                np.hstack([i1, np.full((th, 4, 3), 255, np.uint8), i2]))
    print("Done — demo_result.png, debug_aligned_compare.png")


# ---------------------------------------------------------------------------

def _histogram_match_lab(src, ref):
    """Skin-only Reinhard transfer — uses an elliptical skin mask to sample
    only face pixels from the reference, ignoring hair/background.

    Full mean+std match (strength=1.0) on all 3 Lab channels.
    """
    h, w = src.shape[:2]
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float64)
    if lab_r.shape[:2] != (h, w):
        lab_r = cv2.resize(lab_r, (w, h)).astype(np.float64)

    # Skin region mask — center ellipse, excludes hair/background/neck
    skin = np.zeros((h, w), dtype=bool)
    cv2.ellipse(skin.view(np.uint8), (w // 2, int(h * 0.43)),
                (int(w * 0.33), int(h * 0.35)), 0, 0, 360, 1, -1)

    result = lab_s.copy()
    for c in range(3):
        s_pixels = lab_s[:, :, c][skin]
        r_pixels = lab_r[:, :, c][skin]
        if len(s_pixels) < 100 or len(r_pixels) < 100:
            continue
        sm, ss = s_pixels.mean(), max(s_pixels.std(), 1e-6)
        rm, rs = r_pixels.mean(), max(r_pixels.std(), 1e-6)
        # Full transfer on entire image (not just masked region)
        result[:, :, c] = (lab_s[:, :, c] - sm) * (rs / ss) + rm

    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _pixelwise_color_match(src, ref):
    """Pixel-level color matching using heavily blurred difference map.

    Instead of computing region stats, directly compute the per-pixel
    difference between src and ref in Lab space, blur it heavily to get
    a smooth correction field, and apply it. This naturally handles every
    zone — nose, eyes, cheeks — without explicit region definitions.
    """
    h, w = src.shape[:2]
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float64)
    if lab_r.shape[:2] != (h, w):
        lab_r = cv2.resize(lab_r, (w, h)).astype(np.float64)

    # Face-only mask to avoid correcting hair/background
    face_mask = np.zeros((h, w), dtype=np.float64)
    cv2.ellipse(face_mask, (w // 2, int(h * 0.43)),
                (int(w * 0.38), int(h * 0.40)), 0, 0, 360, 1.0, -1)
    face_mask = cv2.GaussianBlur(face_mask, (31, 31), 8.0)

    for c in range(3):
        # Per-pixel difference
        diff = lab_r[:, :, c] - lab_s[:, :, c]
        # Heavy blur → smooth correction field (no pixel-level noise)
        diff_smooth = cv2.GaussianBlur(diff, (51, 51), 15.0)
        # Apply only within face region
        lab_s[:, :, c] = np.clip(
            lab_s[:, :, c] + diff_smooth * face_mask, 0, 255
        )

    return cv2.cvtColor(lab_s.astype(np.uint8), cv2.COLOR_Lab2BGR)


def _regional_brightness_match(src, ref):
    """Match brightness per-region so every face zone matches the AI source.

    Divides the face into overlapping soft regions (forehead, cheeks, eyes,
    nose, chin) and shifts each region's L channel to match the reference.
    Uses large Gaussian blending so there are no hard boundaries.
    """
    h, w = src.shape[:2]
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float64)
    if lab_r.shape[:2] != (h, w):
        lab_r = cv2.resize(lab_r, (w, h)).astype(np.float64)

    # Dense grid of overlapping regions covering the entire face
    regions = [
        (0.20, 0.50, 0.12, 0.30),  # forehead
        (0.30, 0.35, 0.10, 0.15),  # left brow/eye
        (0.30, 0.65, 0.10, 0.15),  # right brow/eye
        (0.38, 0.30, 0.10, 0.12),  # left eye
        (0.38, 0.70, 0.10, 0.12),  # right eye
        (0.45, 0.50, 0.10, 0.12),  # nose upper
        (0.55, 0.50, 0.10, 0.12),  # nose lower
        (0.50, 0.25, 0.12, 0.15),  # left cheek
        (0.50, 0.75, 0.12, 0.15),  # right cheek
        (0.60, 0.30, 0.10, 0.15),  # left lower cheek
        (0.60, 0.70, 0.10, 0.15),  # right lower cheek
        (0.70, 0.50, 0.10, 0.20),  # chin
    ]

    # Correct ALL 3 channels (L, A, B) per region
    corrections = [np.zeros((h, w), dtype=np.float64) for _ in range(3)]
    weight_sum = np.zeros((h, w), dtype=np.float64) + 1e-10

    for cy_f, cx_f, ry_f, rx_f in regions:
        cy, cx = int(h * cy_f), int(w * cx_f)
        ry, rx = max(int(h * ry_f), 8), max(int(w * rx_f), 8)

        y1, y2 = max(0, cy - ry), min(h, cy + ry)
        x1, x2 = max(0, cx - rx), min(w, cx + rx)

        # Soft Gaussian mask — large blur for seamless blending
        mask = np.zeros((h, w), dtype=np.float64)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        k = max(ry, rx) * 4 + 1
        k = k | 1
        mask = cv2.GaussianBlur(mask, (k, k), max(ry, rx))

        for c in range(3):
            s_mean = lab_s[y1:y2, x1:x2, c].mean()
            r_mean = lab_r[y1:y2, x1:x2, c].mean()
            corrections[c] += mask * (r_mean - s_mean)

        weight_sum += mask

    for c in range(3):
        lab_s[:, :, c] = np.clip(
            lab_s[:, :, c] + corrections[c] / weight_sum, 0, 255
        )

    return cv2.cvtColor(lab_s.astype(np.uint8), cv2.COLOR_Lab2BGR)


def _equalize_face_tone(face_bgr):
    """Remove brightness variation across the face.

    Computes a local brightness map via heavy blur, then divides it out
    so eye sockets, cheeks, and forehead have the same luminance.
    Then scales back to the original global mean.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    L = lab[:, :, 0]

    # Heavy blur captures the regional brightness pattern
    L_local = cv2.GaussianBlur(L, (65, 65), 20.0)
    global_mean = L.mean()

    # Normalize: remove local variation, keep global brightness
    # Where L_local is bright (eye area), this darkens it
    # Where L_local is dark (cheeks), this brightens it
    L_local = np.maximum(L_local, 1.0)
    L_eq = L / L_local * global_mean

    # Blend: 50% equalized + 50% original to keep some natural depth
    lab[:, :, 0] = np.clip(0.5 * L_eq + 0.5 * L, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def _brighten_face_region(frame, mask, gamma=0.75):
    """Apply gamma correction to brighten only the face region.

    gamma < 1.0 = brighter. Applied only where mask > 0.
    """
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype=np.uint8)
    bright = cv2.LUT(frame, lut)
    m3 = mask[:, :, np.newaxis]
    return np.clip(m3 * bright.astype(np.float32)
                   + (1.0 - m3) * frame.astype(np.float32),
                   0, 255).astype(np.uint8)


def _force_brightness(src, ref):
    """Force L channel to exactly match the AI source face distribution.

    Full histogram match on L — no partial blending, no strength param.
    This is the definitive brightness fix applied after all other processing.
    """
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float64)
    if lab_r.shape[:2] != lab_s.shape[:2]:
        lab_r = cv2.resize(lab_r, (lab_s.shape[1], lab_s.shape[0])).astype(np.float64)

    # Full mean+std match on L channel
    sm, ss = lab_s[:, :, 0].mean(), max(lab_s[:, :, 0].std(), 1e-6)
    rm, rs = lab_r[:, :, 0].mean(), max(lab_r[:, :, 0].std(), 1e-6)
    lab_s[:, :, 0] = (lab_s[:, :, 0] - sm) * (rs / ss) + rm

    # Also nudge A and B channels to match (skin hue)
    for c in [1, 2]:
        scm, scs = lab_s[:, :, c].mean(), max(lab_s[:, :, c].std(), 1e-6)
        rcm, rcs = lab_r[:, :, c].mean(), max(lab_r[:, :, c].std(), 1e-6)
        lab_s[:, :, c] = (lab_s[:, :, c] - scm) * (rcs / scs) + rcm

    return cv2.cvtColor(np.clip(lab_s, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _reinhard_transfer(src, ref, strength=0.5):
    """Global Reinhard Lab color transfer — clean, no artifacts."""
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float64)
    if lab_r.shape[:2] != lab_s.shape[:2]:
        lab_r = cv2.resize(lab_r, (lab_s.shape[1], lab_s.shape[0])).astype(np.float64)

    out = lab_s.copy()
    for c in range(3):
        sm, ss = lab_s[:, :, c].mean(), max(lab_s[:, :, c].std(), 1e-6)
        rm, rs = lab_r[:, :, c].mean(), max(lab_r[:, :, c].std(), 1e-6)
        t = (lab_s[:, :, c] - sm) * (rs / ss) + rm
        out[:, :, c] = strength * t + (1.0 - strength) * lab_s[:, :, c]
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _bilateral_smooth(face, strength=0.45):
    """Uniform bilateral smoothing — full face, no exclusions."""
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    L_s = cv2.bilateralFilter(L.astype(np.uint8), d=12, sigmaColor=75, sigmaSpace=75).astype(np.float32)
    try:
        L_g = cv2.ximgproc.guidedFilter(
            guide=L.astype(np.uint8), src=L_s.astype(np.uint8),
            radius=10, eps=80).astype(np.float32)
        L_s = 0.5 * L_s + 0.5 * L_g
    except AttributeError:
        pass
    lab[:, :, 0] = np.clip(strength * L_s + (1.0 - strength) * L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _sharpen_pupils(face):
    """CLAHE + unsharp on iris region only. No sclera/catchlight changes."""
    h, w = face.shape[:2]
    result = face.copy()
    for cx_f, cy_f in [(0.35, 0.38), (0.65, 0.38)]:
        cx, cy = int(w * cx_f), int(h * cy_f)
        r = int(w * 0.09)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = result[y1:y2, x1:x2].copy()
        rh, rw = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < 15:
            continue

        (ix, iy), ir = cv2.minEnclosingCircle(best)
        ix, iy, ir = int(ix), int(iy), max(int(ir), 3)

        # Iris mask with soft edge
        imask = np.zeros((rh, rw), dtype=np.float32)
        cv2.circle(imask, (ix, iy), ir + 2, 1.0, -1)
        imask = cv2.GaussianBlur(imask, (5, 5), 1.5)

        # CLAHE on L
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        L_e = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = np.clip(
            imask * L_e.astype(np.float32) + (1.0 - imask) * lab[:, :, 0].astype(np.float32),
            0, 255).astype(np.uint8)
        roi = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

        # Unsharp on iris
        blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=0.7)
        sharp = cv2.addWeighted(roi, 1.35, blur, -0.35, 0)
        im3 = imask[:, :, np.newaxis]
        roi = np.clip(im3 * sharp.astype(np.float32)
                      + (1.0 - im3) * roi.astype(np.float32), 0, 255).astype(np.uint8)

        result[y1:y2, x1:x2] = roi
    return result


def _codeformer(sess, face, fidelity=0.4):
    """CodeFormer restoration — sharpens everything including eyes."""
    h, w = face.shape[:2]
    inp = cv2.resize(face, (512, 512))
    inp = inp[:, :, ::-1].astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    inp = np.expand_dims(inp.transpose(2, 0, 1), 0)

    feeds = {}
    for n in sess.get_inputs():
        if n.shape and len(n.shape) == 4:
            feeds[n.name] = inp
        else:
            dt = np.float64 if "double" in n.type else np.float32
            feeds[n.name] = np.array([[fidelity]], dtype=dt)
    try:
        out = sess.run(None, feeds)[0]
    except Exception as e:
        print(f"  CodeFormer failed: {e}")
        return face

    out = np.clip(out[0], -1, 1)
    out = ((out + 1) / 2 * 255).round().astype(np.uint8)
    out = out.transpose(1, 2, 0)[:, :, ::-1]
    out = cv2.resize(out, (w, h))
    # Blend: mostly CodeFormer, slight original preservation
    return cv2.addWeighted(out, 0.8, face, 0.2, 0)


if __name__ == "__main__":
    main()
