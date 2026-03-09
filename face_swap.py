"""Core face swap pipeline: triangulation, warping, blending with mouth exclusion."""

import cv2
import numpy as np

from landmarks import (
    FACE_OVAL_INDICES,
    get_face_polygon,
    get_lip_polygon,
)


def _rect_contains(rect, point):
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def compute_triangles(landmarks, frame_shape):
    """Compute Delaunay triangulation and return list of index triples."""
    h, w = frame_shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Use face oval + a grid of interior points for good coverage
    # Insert all 468 landmarks that fall inside the frame
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
        # Map each vertex back to a landmark index
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


def swap_face(src_frame, src_landmarks, dst_frame, dst_landmarks,
              triangles=None, use_seamless=True):
    """Swap source face onto destination frame, excluding mouth region.

    Args:
        src_frame: Source BGR image.
        src_landmarks: (468, 2) landmarks of source face.
        dst_frame: Destination BGR image (will not be modified).
        dst_landmarks: (468, 2) landmarks of destination face.
        triangles: Precomputed triangle index triples (or None to compute).
        use_seamless: Use cv2.seamlessClone (better quality, slower).

    Returns:
        Composited frame with face swapped, mouth preserved.
    """
    result = dst_frame.copy()

    # Step 1: Compute triangulation on destination face
    if triangles is None:
        triangles = compute_triangles(dst_landmarks, dst_frame.shape)

    # Step 2: Create a blank canvas for the warped face
    warped_face = np.zeros_like(dst_frame)

    # Step 3: Warp each triangle from source to destination
    for idx_triple in triangles:
        i, j, k = idx_triple
        src_tri = [src_landmarks[i].tolist(), src_landmarks[j].tolist(), src_landmarks[k].tolist()]
        dst_tri = [dst_landmarks[i].tolist(), dst_landmarks[j].tolist(), dst_landmarks[k].tolist()]
        _warp_triangle(src_frame, warped_face, src_tri, dst_tri)

    # Step 4: Build the face mask from convex hull of face oval
    face_hull = cv2.convexHull(get_face_polygon(dst_landmarks))
    mask = np.zeros(dst_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, face_hull, 255)

    # Step 5: Carve out mouth region
    lip_poly = get_lip_polygon(dst_landmarks)
    cv2.fillPoly(mask, [lip_poly], 0)

    # Step 6: Blend
    if use_seamless:
        center = tuple(face_hull.mean(axis=0).astype(int).flatten())
        mask_3ch = cv2.merge([mask, mask, mask])
        try:
            result = cv2.seamlessClone(
                warped_face, result, mask_3ch, center, cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback to alpha blending if seamlessClone fails
            result = _alpha_blend(warped_face, result, mask)
    else:
        result = _alpha_blend(warped_face, result, mask)

    return result


def _alpha_blend(warped, dst, mask):
    """Fast alpha blending with feathered mask."""
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 10)
    alpha = blurred_mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    blended = (warped * alpha + dst * (1 - alpha)).astype(np.uint8)
    return blended


def swap_faces_bidirectional(frame, landmarks_list, use_seamless=True):
    """Swap two faces in a single frame with each other."""
    lm_a, lm_b = landmarks_list[0], landmarks_list[1]

    # Compute triangulations
    tri_a = compute_triangles(lm_a, frame.shape)
    tri_b = compute_triangles(lm_b, frame.shape)

    # Swap A->B then B->A on the original frame
    result = swap_face(frame, lm_a, frame, lm_b, tri_b, use_seamless)
    result = swap_face(frame, lm_b, result, lm_a, tri_a, use_seamless)
    return result
