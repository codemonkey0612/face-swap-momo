"""Real-time face swap pipeline using MediaPipe and OpenCV."""

import argparse

import cv2

from landmarks import create_face_mesh, extract_landmarks
from face_swap import swap_face, swap_faces_bidirectional, compute_triangles
from utils import FPSCounter, FreezeFrameManager, draw_debug_overlay, draw_fps


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face swap")
    p.add_argument("--source", type=str, default=None,
                   help="Path to source face image. If omitted, swaps two webcam faces.")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-seamless", action="store_true",
                   help="Use fast alpha blending instead of seamless clone")
    p.add_argument("--debug", action="store_true", help="Show landmarks/triangles")
    return p.parse_args()


def load_source(path, face_mesh):
    """Load a source face image and extract its landmarks."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read source image: {path}")
    landmarks = extract_landmarks(img, face_mesh)
    if not landmarks:
        raise ValueError(f"No face detected in source image: {path}")
    return img, landmarks[0]


def main():
    args = parse_args()
    use_seamless = not args.no_seamless

    face_mesh = create_face_mesh(max_faces=2)
    fps_counter = FPSCounter()
    freeze = FreezeFrameManager()

    # Load source image if provided
    src_img, src_landmarks = None, None
    if args.source:
        src_img, src_landmarks = load_source(args.source, face_mesh)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit, 'd' to toggle debug overlay")
    show_debug = args.debug

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        fps_counter.tick()

        landmarks_list = extract_landmarks(frame, face_mesh)
        result = frame

        if args.source:
            # Source image -> webcam face swap
            if landmarks_list:
                result = swap_face(
                    src_img, src_landmarks, frame, landmarks_list[0],
                    use_seamless=use_seamless,
                )
                freeze.update(result)
            else:
                freeze.miss()
                frozen = freeze.get_frozen()
                if frozen is not None:
                    result = frozen
        else:
            # Two-face webcam swap
            if len(landmarks_list) >= 2:
                result = swap_faces_bidirectional(
                    frame, landmarks_list, use_seamless=use_seamless,
                )
                freeze.update(result)
            else:
                freeze.miss()
                frozen = freeze.get_frozen()
                if frozen is not None:
                    result = frozen

        if show_debug and landmarks_list:
            for lm in landmarks_list:
                draw_debug_overlay(result, lm)

        draw_fps(result, fps_counter)
        cv2.imshow("Face Swap", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
