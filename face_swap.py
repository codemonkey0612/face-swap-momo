"""Core face swap pipeline: triangulation, warping, blending with mouth exclusion."""

import cv2
import numpy as np

from landmarks import (
    get_face_polygon,
    get_lip_polygon,
    get_eye_regions,
)

# Eye corner landmark indices used for rotation alignment
_LEFT_EYE = 33
_RIGHT_EYE = 263


def compute_triangles(landmarks, frame_shape):
    """Compute Delaunay triangulation and return list of index triples."""
    h, w = frame_shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Insert all 478 landmarks that fall inside the frame
    valid_indices = []
    for i, (x, y) in enumerate(landmarks):
        if 0 <= x < w and 0 <= y < h:
            subdiv.insert((int(x), int(y)))
            valid_indices.append(i)

    # Build a lookup from (x,y) -> index for fast triangle mapping
    pt_to_idx = {}
    for i in valid_indices:
        key = (int(landmarks[i][0]), int(landmarks[i][1]))
        pt_to_idx[key] = i

    triangles = subdiv.getTriangleList()
    index_triples = []

    for t in triangles:
        pts = [
            (int(t[0]), int(t[1])),
            (int(t[2]), int(t[3])),
            (int(t[4]), int(t[5])),
        ]
        indices = []
        for p in pts:
            if p in pt_to_idx:
                indices.append(pt_to_idx[p])
            else:
                break
        if len(indices) == 3:
            index_triples.append(tuple(indices))

    return index_triples


def _warp_triangle(src_frame, dst_frame, src_tri, dst_tri):
    """Warp a single triangle from src onto dst using affine transform."""
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    src_cropped = []
    dst_cropped = []
    for i in range(3):
        src_cropped.append((src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]))
        dst_cropped.append((dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]))

    src_cropped = np.float32(src_cropped)
    dst_cropped = np.float32(dst_cropped)

    # Crop source region
    sx, sy, sw, sh = sr
    src_rect = src_frame[sy : sy + sh, sx : sx + sw]
    if src_rect.size == 0:
        return

    # Affine warp
    warp_mat = cv2.getAffineTransform(src_cropped, dst_cropped)
    dx, dy, dw, dh = dr
    warped = cv2.warpAffine(
        src_rect, warp_mat, (dw, dh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Mask for this triangle
    mask = np.zeros((dh, dw, 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_cropped), (255, 255, 255))

    # Blend into destination
    region = dst_frame[dy : dy + dh, dx : dx + dw]
    if region.shape != mask.shape:
        return
    dst_frame[dy : dy + dh, dx : dx + dw] = (
        region * (1 - mask / 255.0) + warped * (mask / 255.0)
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Quality improvement helpers
# ---------------------------------------------------------------------------

def _color_correct(warped, dst, mask):
    """Reinhard color transfer in LAB space for perceptually accurate skin-tone matching."""
    mask_bool = mask > 0
    if not mask_bool.any():
        return warped

    src_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = src_lab.copy()

    for c in range(3):
        src_ch = src_lab[:, :, c][mask_bool]
        dst_ch = dst_lab[:, :, c][mask_bool]
        src_std = src_ch.std()
        if src_std < 1e-6:
            continue
        result_lab[:, :, c][mask_bool] = (
            (src_ch - src_ch.mean()) * (dst_ch.std() / src_std) + dst_ch.mean()
        )

    return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _sharpen_region(img, mask, amount=0.4):
    """Apply subtle unsharp-mask sharpening only inside the face mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    alpha = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    return (sharpened * alpha + img * (1 - alpha)).astype(np.uint8)


def _enhance_eyes(result, dst_landmarks, sharpen_amount=0.8, clarity_amount=0.3):
    """Apply targeted sharpening and contrast enhancement to eye regions.

    Eyes lose detail during triangle warping. This applies:
    1. Strong unsharp mask (sharpen_amount) to recover fine edge detail
    2. CLAHE local contrast enhancement (clarity_amount blend) for iris clarity
    Both are feathered at the eye region boundary to avoid hard edges.
    """
    h, w = result.shape[:2]
    eye_boxes = get_eye_regions(dst_landmarks, pad_scale=1.4)

    for (x1, y1, x2, y2) in eye_boxes:
        # Clip to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        region = result[y1:y2, x1:x2].copy()
        rh, rw = region.shape[:2]

        # 1) Unsharp mask — stronger than the general face pass
        blur = cv2.GaussianBlur(region, (0, 0), 1.5)
        sharp = cv2.addWeighted(region, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

        # 2) CLAHE on L channel for iris/pupil contrast
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        # Blend CLAHE result with sharpened (clarity_amount controls strength)
        combined = cv2.addWeighted(enhanced, clarity_amount, sharp, 1.0 - clarity_amount, 0)

        # 3) Feathered elliptical mask so enhancement fades at edges
        eye_mask = np.zeros((rh, rw), dtype=np.uint8)
        center = (rw // 2, rh // 2)
        axes = (rw // 2 - 2, rh // 2 - 2)
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(eye_mask, center, axes, 0, 0, 360, 255, -1)
            eye_mask = cv2.GaussianBlur(eye_mask, (0, 0), max(3, min(rw, rh) * 0.15))
        else:
            eye_mask[:] = 255

        alpha = (eye_mask / 255.0)[:, :, np.newaxis]
        result[y1:y2, x1:x2] = (combined * alpha + result[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)

    return result


def _get_eye_angle(landmarks):
    """Compute the tilt angle (degrees) of the eye line."""
    left = landmarks[_LEFT_EYE].astype(float)
    right = landmarks[_RIGHT_EYE].astype(float)
    return np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))


def _rotate_landmarks(landmarks, M):
    pts = landmarks.astype(np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return ((M @ np.hstack([pts, ones]).T).T).astype(np.int32)


def _align_source_rotation(src_frame, src_landmarks, dst_landmarks):
    """Rotate source face to match destination face tilt angle."""
    diff = _get_eye_angle(dst_landmarks) - _get_eye_angle(src_landmarks)
    if abs(diff) < 1.0:
        return src_frame, src_landmarks
    cx, cy = src_landmarks[[_LEFT_EYE, _RIGHT_EYE]].mean(axis=0)
    center = (float(cx), float(cy))
    h, w = src_frame.shape[:2]
    M = cv2.getRotationMatrix2D(center, diff, 1.0)
    rotated = cv2.warpAffine(src_frame, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return rotated, _rotate_landmarks(src_landmarks, M)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def swap_face(src_frame, src_landmarks, dst_frame, dst_landmarks,
              triangles=None, use_seamless=True, blend_mode="auto"):
    """Swap source face onto destination frame, excluding mouth region.

    Args:
        src_frame: Source BGR image.
        src_landmarks: (478, 2) landmarks of source face.
        dst_frame: Destination BGR image (will not be modified).
        dst_landmarks: (478, 2) landmarks of destination face.
        triangles: Precomputed triangle index triples (or None to compute).
        use_seamless: Use cv2.seamlessClone (better quality, slower).
        blend_mode: "auto" (seamless→multiband fallback), "multiband", or "alpha".

    Returns:
        Composited frame with face swapped, mouth preserved.
    """
    result = dst_frame.copy()

    # Step 1: Align source face rotation to destination tilt
    src_frame, src_landmarks = _align_source_rotation(src_frame, src_landmarks, dst_landmarks)

    # Step 2: Compute triangulation on destination face
    if triangles is None:
        triangles = compute_triangles(dst_landmarks, dst_frame.shape)

    # Step 3: Create a blank canvas for the warped face
    warped_face = np.zeros_like(dst_frame)

    # Step 4: Warp each triangle from source to destination
    for idx_triple in triangles:
        i, j, k = idx_triple
        src_tri = [src_landmarks[i].tolist(), src_landmarks[j].tolist(), src_landmarks[k].tolist()]
        dst_tri = [dst_landmarks[i].tolist(), dst_landmarks[j].tolist(), dst_landmarks[k].tolist()]
        _warp_triangle(src_frame, warped_face, src_tri, dst_tri)

    # Step 5: Build the face mask from convex hull, then erode inward so the
    # boundary sits well inside the warped region (avoids edge artifacts at
    # forehead/jawline where warped data thins out).
    face_hull = cv2.convexHull(get_face_polygon(dst_landmarks))
    mask = np.zeros(dst_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, face_hull, 255)
    # Erode to pull mask away from the sparse outer edges of the warp
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, erode_k, iterations=1)

    # Step 6: Carve out mouth region
    lip_poly = get_lip_polygon(dst_landmarks)
    cv2.fillPoly(mask, [lip_poly], 0)

    # Step 7: Color correct warped face to match destination lighting/tone
    warped_face = _color_correct(warped_face, dst_frame, mask)

    # Step 8: Blend
    if blend_mode == "alpha" or (not use_seamless and blend_mode != "multiband"):
        result = _alpha_blend(warped_face, result, mask)
    elif blend_mode == "multiband":
        result = _multiband_blend(warped_face, result, mask)
    else:
        # "auto": try seamlessClone, fall back to multiband.
        # seamlessClone needs a mask whose boundary is fully inside the
        # valid warped region.  The face-hull mask was already eroded in
        # step 5; we only shrink it a little more here so the Poisson
        # solver never touches pixels outside the warped content.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clone = cv2.erode(mask, kernel, iterations=1)
        ys, xs = np.where(mask_clone > 0)
        if len(xs) == 0:
            return _multiband_blend(warped_face, result, mask)
        center = (int(xs.mean()), int(ys.mean()))
        mask_3ch = cv2.merge([mask_clone, mask_clone, mask_clone])
        try:
            result = cv2.seamlessClone(
                warped_face, result, mask_3ch, center, cv2.NORMAL_CLONE
            )
        except cv2.error:
            result = _multiband_blend(warped_face, result, mask)

    # Step 9: Subtle sharpening inside the swapped region
    result = _sharpen_region(result, mask, amount=0.3)

    # Step 10: Targeted eye enhancement — stronger sharpening + local contrast
    result = _enhance_eyes(result, dst_landmarks)

    return result


def _feather_mask(mask, edge_width=None):
    """Create a feathered mask using distance transform for smooth edges.

    If edge_width is None, it is set to ~15 % of the mask's bounding height,
    giving a wide, natural transition that hides colour / texture seams.
    """
    if edge_width is None:
        ys = np.where(mask.any(axis=1))[0]
        face_h = (ys[-1] - ys[0]) if len(ys) > 1 else 100
        edge_width = max(12, int(face_h * 0.15))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = np.clip(dist / edge_width, 0, 1)
    return dist.astype(np.float32)


def _alpha_blend(warped, dst, mask):
    """Fast alpha blending with distance-transform feathered mask."""
    alpha = _feather_mask(mask)
    alpha = np.stack([alpha] * 3, axis=-1)
    blended = (warped * alpha + dst * (1 - alpha)).astype(np.uint8)
    return blended


def _multiband_blend(warped, dst, mask, levels=4):
    """Laplacian pyramid blending for seamless compositing.

    Much better quality than simple alpha blend, comparable to seamlessClone
    but faster and more controllable.
    """
    alpha = _feather_mask(mask)
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    # Build Gaussian pyramids for the mask
    gp_mask = [alpha_3ch.astype(np.float32)]
    for _ in range(levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # Build Laplacian pyramids for both images
    src_f = warped.astype(np.float32)
    dst_f = dst.astype(np.float32)

    gp_src = [src_f]
    gp_dst = [dst_f]
    for _ in range(levels):
        gp_src.append(cv2.pyrDown(gp_src[-1]))
        gp_dst.append(cv2.pyrDown(gp_dst[-1]))

    lp_src = []
    lp_dst = []
    for i in range(levels):
        up_src = cv2.pyrUp(gp_src[i + 1], dstsize=(gp_src[i].shape[1], gp_src[i].shape[0]))
        up_dst = cv2.pyrUp(gp_dst[i + 1], dstsize=(gp_dst[i].shape[1], gp_dst[i].shape[0]))
        lp_src.append(gp_src[i] - up_src)
        lp_dst.append(gp_dst[i] - up_dst)
    lp_src.append(gp_src[levels])
    lp_dst.append(gp_dst[levels])

    # Blend at each level
    blended_pyr = []
    for i in range(levels + 1):
        m = gp_mask[i] if i < len(gp_mask) else gp_mask[-1]
        if m.shape[:2] != lp_src[i].shape[:2]:
            m = cv2.resize(m, (lp_src[i].shape[1], lp_src[i].shape[0]))
        blended_pyr.append(lp_src[i] * m + lp_dst[i] * (1 - m))

    # Reconstruct
    result = blended_pyr[levels]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result += blended_pyr[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def swap_faces_bidirectional(frame, landmarks_list, use_seamless=True, blend_mode="auto"):
    """Swap two faces in a single frame with each other."""
    lm_a, lm_b = landmarks_list[0], landmarks_list[1]

    # Compute triangulations
    tri_a = compute_triangles(lm_a, frame.shape)
    tri_b = compute_triangles(lm_b, frame.shape)

    # Swap A->B then B->A on the original frame
    result = swap_face(frame, lm_a, frame, lm_b, tri_b, use_seamless, blend_mode)
    result = swap_face(frame, lm_b, result, lm_a, tri_a, use_seamless, blend_mode)
    return result
