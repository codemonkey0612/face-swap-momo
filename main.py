"""Real-time face swap pipeline — replaces webcam face with AI source identity."""

import argparse
import glob
import os
import threading

import cv2

from utils import FPSCounter, FreezeFrameManager, draw_fps


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face swap")
    p.add_argument("--source", type=str, default=None,
                   help="Path to a single source face image.")
    p.add_argument("--source-dir", type=str, default="images/1",
                   help="Folder of source face images — averages all embeddings "
                        "for a more robust identity.")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=str, default="models/ghost_2_256.onnx",
                   help="Path to swap model (ghost_2_256.onnx recommended)")
    p.add_argument("--enhance", action="store_true",
                   help="Run GFPGAN face restoration after each swap (better quality, slower)")
    p.add_argument("--debug", action="store_true", help="Show detection bbox")
    return p.parse_args()


def _load_source(swapper, args):
    """Load source identity from --source-dir or --source."""
    from neural_swap import build_averaged_face

    if args.source_dir:
        img_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
                    "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP")
        image_paths = []
        for ext in img_exts:
            image_paths.extend(glob.glob(os.path.join(args.source_dir, ext)))
        image_paths = sorted(set(image_paths))
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {args.source_dir}")
        print(f"[Source] Found {len(image_paths)} images in {args.source_dir}")
        return build_averaged_face(swapper, image_paths)

    if args.source:
        src_img = cv2.imread(args.source)
        if src_img is None:
            raise FileNotFoundError(f"Cannot read source image: {args.source}")
        faces = swapper.detect(src_img)
        if not faces:
            raise ValueError(f"No face detected in source image: {args.source}")
        print(f"[Source] Loaded from {args.source}")
        return faces[0]

    raise ValueError("Provide --source-dir or --source for the AI face identity.")


class _AsyncWorker:
    """Single-slot async worker: submits frames, reads latest result without blocking."""

    def __init__(self, fn):
        self._fn = fn
        self._result = None
        self._pending = None
        self._lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, *args):
        with self._pending_lock:
            self._pending = args

    @property
    def result(self):
        with self._lock:
            return self._result

    def _worker(self):
        while True:
            args = None
            with self._pending_lock:
                if self._pending is not None:
                    args = self._pending
                    self._pending = None
            if args is not None:
                out = self._fn(*args)
                with self._lock:
                    self._result = out
            else:
                threading.Event().wait(0.001)


class AsyncEnhancer:
    """Runs GFPGAN in a background thread; main loop never blocks on it."""

    def __init__(self, enhancer):
        self._worker = _AsyncWorker(enhancer.enhance_frame)
        self._latest_enhanced = None
        self._lock = threading.Lock()

    def submit(self, frame, faces):
        self._worker.submit(frame.copy(), list(faces))

    def get(self, fallback):
        result = self._worker.result
        if result is not None:
            with self._lock:
                self._latest_enhanced = result
        with self._lock:
            return self._latest_enhanced if self._latest_enhanced is not None else fallback


def main():
    args = parse_args()

    fps_counter = FPSCounter()
    freeze = FreezeFrameManager()

    # --- Enhancer setup ---
    async_enhancer = None
    if args.enhance:
        from enhancer import FaceEnhancer
        async_enhancer = AsyncEnhancer(FaceEnhancer())

    # --- Neural swap setup ---
    from neural_swap import NeuralFaceSwapper
    swapper = NeuralFaceSwapper(model_path=args.model)
    src_face = _load_source(swapper, args)

    # --- Camera ---
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

        frame = cv2.flip(frame, 1)
        fps_counter.tick()
        result = frame

        # Detect the first face in the webcam frame
        detected = swapper.detect(frame)

        if detected:
            target_face = detected[0]
            result = swapper.swap(frame, target_face, src_face)

            if async_enhancer:
                async_enhancer.submit(result, [target_face])
                result = async_enhancer.get(result)

            freeze.update(result)

            if show_debug:
                x1, y1, x2, y2 = (int(v) for v in target_face.bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            freeze.miss()
            frozen = freeze.get_frozen()
            result = frozen if frozen is not None else result
            cv2.putText(result, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
