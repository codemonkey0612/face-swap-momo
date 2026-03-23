"""Shared image processing utilities: alignment, blending, color matching."""

import cv2
import numpy as np

# 5-point ArcFace landmark template in 112×112 space
ARCFACE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame: np.ndarray, kps: np.ndarray, size: int):
    """Affine-warp face to canonical size×size template using 5 keypoints.

    Returns:
        (aligned_bgr, affine_matrix M)
    """
    template = ARCFACE_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(
        kps, template, method=cv2.RANSAC, ransacReprojThreshold=100
    )
    if M is None:
        M = cv2.getAffineTransform(kps[:3], template[:3])
    aligned = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
    return aligned, M


def sharpen(img: np.ndarray, sigma: float = 2.0, amount: float = 0.4) -> np.ndarray:
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def inward_feather(mask: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Gaussian blur clamped to never expand past the original mask boundary.

    Standard blur expands masks outward, which leaks swapped pixels onto
    occluding objects. Inward-only feathering creates soft edges INSIDE the
    mask boundary only.
    """
    boundary = (mask > 0.01).astype(np.float32)
    k = ksize | 1  # ensure odd
    blurred = cv2.GaussianBlur(mask, (k, k), sigma)
    return np.minimum(blurred, boundary)


def soft_blend(frame: np.ndarray, pasted: np.ndarray,
               mask_bin: np.ndarray, blur_ratio: float = 0.35) -> np.ndarray:
    """Alpha-blend pasted face into frame with inward-only Gaussian feathering."""
    where = np.where(mask_bin > 0)
    if len(where[0]) == 0:
        return frame
    face_h = where[0].max() - where[0].min()
    face_w = where[1].max() - where[1].min()
    face_size = max(face_h, face_w)
    blur_px = max(11, int(face_size * blur_ratio))
    blur_px = blur_px | 1
    mask_f = mask_bin.astype(np.float32) / 255.0
    mask_soft = cv2.GaussianBlur(mask_f, (blur_px, blur_px), 0)
    mask_hard = (mask_f > 0.01).astype(np.float32)
    mask_soft = np.minimum(mask_soft, mask_hard)
    alpha = mask_soft[:, :, np.newaxis]
    return (
        pasted.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def match_histogram_skin(src: np.ndarray, dst: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """Per-channel histogram matching in the masked skin region (70% blend)."""
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
        corrected = (out[:, :, c] - s_mean) * (d_std / s_std) + d_mean
        out[:, :, c] = 0.7 * corrected + 0.3 * out[:, :, c]
    return np.clip(out, 0, 255).astype(np.uint8)
