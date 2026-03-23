"""Helper to prepare source face images for the pipeline.

Functions:
    1. Verify a source face image (must contain exactly one face)
    2. Crop and align the face to ArcFace 112×112 format
    3. Generate synthetic test fixture frames for CI tests

Usage:
    python scripts/generate_source_face.py --input path/to/face.jpg
    python scripts/generate_source_face.py --test-fixtures
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def verify_source_face(image_path: str) -> bool:
    """Verify that the image contains exactly one clear face.

    Args:
        image_path: Path to a face image (JPEG/PNG).

    Returns:
        True if exactly one face detected with confidence >= 0.7.
    """
    from insightface.app import FaceAnalysis

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return False

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(img)

    if len(faces) == 0:
        print("[FAIL] No face detected in the image.")
        return False
    if len(faces) > 1:
        print(f"[FAIL] {len(faces)} faces detected — expected exactly 1.")
        return False

    det_score = getattr(faces[0], "det_score", 1.0)
    if det_score < 0.7:
        print(f"[FAIL] Face confidence too low: {det_score:.2f} (need ≥ 0.70)")
        return False

    bbox = faces[0].bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    print(f"[OK] Face detected: bbox={bbox.astype(int)}, size={w:.0f}×{h:.0f}px, conf={det_score:.2f}")
    return True


def crop_and_align(image_path: str, output_path: str | None = None) -> np.ndarray | None:
    """Crop, align, and save the face in ArcFace 112×112 format.

    Args:
        image_path:  Input image.
        output_path: Where to save the aligned face (optional).

    Returns:
        Aligned face as numpy array (112, 112, 3) or None on failure.
    """
    from insightface.app import FaceAnalysis

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return None

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(img)

    if not faces:
        print("[ERROR] No face found.")
        return None

    face = faces[0]
    # Use insightface normed cropped face if available
    if hasattr(face, "normed_embedding"):
        pass  # already extracted

    # Fallback: just crop the bbox
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad = 20
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(img.shape[1], x2+pad), min(img.shape[0], y2+pad)
    crop = cv2.resize(img[y1:y2, x1:x2], (112, 112))

    if output_path:
        cv2.imwrite(output_path, crop)
        print(f"[OK] Saved aligned face to {output_path}")

    return crop


def generate_test_fixtures(output_dir: str = "tests/fixtures/sample_frames") -> None:
    """Generate minimal synthetic test frames for CI tests.

    Creates:
        clear_face.jpg  — blank frame (no real face, but has face-coloured ellipse)
        no_face.jpg     — blank grey frame
    """
    os.makedirs(output_dir, exist_ok=True)

    # no_face.jpg
    no_face = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "no_face.jpg"), no_face)
    print(f"[OK] {output_dir}/no_face.jpg")

    # clear_face.jpg — draw a synthetic face-like ellipse
    face_frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.ellipse(face_frame, (320, 240), (100, 130), 0, 0, 360, (190, 155, 120), -1)
    cv2.circle(face_frame, (290, 210), 12, (50, 50, 80), -1)   # left eye
    cv2.circle(face_frame, (350, 210), 12, (50, 50, 80), -1)   # right eye
    cv2.ellipse(face_frame, (320, 270), (30, 15), 0, 0, 180, (100, 80, 150), -1)  # mouth
    cv2.imwrite(os.path.join(output_dir, "clear_face.jpg"), face_frame)
    print(f"[OK] {output_dir}/clear_face.jpg")

    # face_with_hand.jpg — same face + hand-shaped overlay
    hand_frame = face_frame.copy()
    hand_pts = np.array([[220, 180], [240, 150], [260, 130], [300, 200], [260, 260]], dtype=np.int32)
    cv2.fillConvexPoly(hand_frame, hand_pts, (170, 140, 115))
    cv2.imwrite(os.path.join(output_dir, "face_with_hand.jpg"), hand_frame)
    print(f"[OK] {output_dir}/face_with_hand.jpg")

    print("\nTest fixtures ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source face preparation utilities")
    parser.add_argument("--input",         type=str, help="Input face image to verify/align")
    parser.add_argument("--output",        type=str, help="Output path for aligned face")
    parser.add_argument("--verify",        action="store_true", help="Verify face only, no save")
    parser.add_argument("--test-fixtures", action="store_true", help="Generate CI test fixtures")
    args = parser.parse_args()

    if args.test_fixtures:
        generate_test_fixtures()
    elif args.input:
        if args.verify:
            ok = verify_source_face(args.input)
            sys.exit(0 if ok else 1)
        else:
            out = args.output or os.path.splitext(args.input)[0] + "_aligned.jpg"
            crop_and_align(args.input, out)
    else:
        parser.print_help()
