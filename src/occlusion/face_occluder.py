"""XSeg face occluder — purpose-trained aligned-space occlusion detection.

WHY XSEG INSTEAD OF 5 GENERAL-PURPOSE LAYERS:
  All general-purpose approaches (BiSeNet, MediaPipe Hands, depth estimation,
  body segmentation) fail on the hardest case: skin-coloured occluders touching
  the face (palm flat on cheek).

  - BiSeNet: no "hand" class — labels hand skin as class 1 "face skin"
  - MediaPipe Hands: loses hand/face boundary when contact is flush
  - Depth (MiDaS): 0-3cm hand-face gap is within the ~5-10cm noise floor
  - Body segmentation: face_parse already wrongly includes the hand

  XSeg was trained specifically on faces WITH occluders (hands, glasses,
  microphones, phones, cups, masks) in canonical ALIGNED 256×256 space.
  It learned the contextual difference between face-skin and hand-skin:
  "there should be a nose here but there's a flat textured surface = occluder."

COORDINATE SPACE (critical):
  XSeg input is the ALIGNED CROP (same 256×256 space as the swap model).
  The output mask is in that same space → zero alignment mismatch with the
  swap result.  There is no warping of a frame-space mask and hoping it aligns.

Compatible models (place in models/):
  xseg_2.onnx        (~70 MB)  ← recommended
  xseg_1.onnx        (~70 MB)
  xseg_3.onnx        (~70 MB)
  face_occluder.onnx (~3 MB)   — compact facefusion model (also works)

Download from FaceFusion assets:
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/

Output convention:
  1.0 = face pixel   → apply swapped face
  0.0 = occluder     → preserve original pixel (hand, cup, phone, etc.)
"""

import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False


class FaceOccluder:
    """XSeg-based face occluder running in aligned 256×256 crop space.

    A single inference call replaces the entire 5-layer occlusion system
    and produces dramatically better results on skin-coloured occluders.
    """

    # Model discovery order — best quality first
    MODEL_NAMES = [
        "xseg_2.onnx",
        "xseg_1.onnx",
        "xseg_3.onnx",
        "face_occluder.onnx",
    ]

    def __init__(self, model_path: str, providers=None):
        if not _ORT_OK:
            raise ImportError("onnxruntime is required for FaceOccluder")

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._sess = ort.InferenceSession(model_path, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        # Detect layout: NCHW has shape (1,3,H,W), NHWC has (1,H,W,3)
        shape = inp.shape
        if len(shape) >= 4 and shape[-1] in (1, 3):
            # NHWC layout — size is shape[-2]
            self._nhwc = True
            self._size = int(shape[-2]) if shape[-2] not in (None, 'None') else 256
        else:
            # NCHW layout — size is shape[-1]
            self._nhwc = False
            self._size = int(shape[-1]) if len(shape) >= 4 else 256
        self._output_name = self._sess.get_outputs()[0].name

        gpu = any("CUDA" in p or "CoreML" in p for p in providers)
        layout = "NHWC" if self._nhwc else "NCHW"
        print(
            f"[FaceOccluder] XSeg {self._size}×{self._size} ({layout}) on "
            f"{'GPU' if gpu else 'CPU'} — {os.path.basename(model_path)}"
        )

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------

    @classmethod
    def find_model(cls, model_dir: str = "models") -> "str | None":
        """Return path of the first XSeg model found in model_dir."""
        for name in cls.MODEL_NAMES:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                return path
        return None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def create_occlusion_mask(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Generate face/occluder mask from an aligned face crop.

        The input MUST be the aligned crop in the same coordinate system as
        the face swap model (e.g. the 256×256 output of _align_face).  Running
        on the aligned crop — not the full frame — ensures the mask is
        pixel-perfect with the swapped output: no warping, no alignment error.

        Args:
            aligned_bgr: uint8 BGR aligned face crop (any square size).

        Returns:
            float32 [0, 1] mask of the same H×W as aligned_bgr.
              1.0 → face pixel   (apply swap here)
              0.0 → occluder     (preserve original pixel)
        """
        h, w = aligned_bgr.shape[:2]
        size = self._size

        # Resize to model input size
        crop = cv2.resize(aligned_bgr, (size, size), interpolation=cv2.INTER_LINEAR) \
            if (h != size or w != size) else aligned_bgr

        # BGR → RGB, normalise to [0, 1]
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        if self._nhwc:
            # NHWC layout: (1, H, W, C)
            blob = np.ascontiguousarray(rgb[np.newaxis])
        else:
            # NCHW layout: (1, C, H, W)
            blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])

        # Inference
        raw = self._sess.run([self._output_name], {self._input_name: blob})[0]
        logits = raw.squeeze()  # collapse batch + channel dims
        if logits.ndim == 3:
            logits = logits[0]

        # Apply sigmoid when model outputs raw logits (min < 0)
        if logits.min() < 0:
            prob = 1.0 / (1.0 + np.exp(-logits))
        else:
            prob = logits
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

        # Resize back to input dimensions
        if prob.shape[0] != h or prob.shape[1] != w:
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        # Soft edges (sigma=5 as specified) — blurs the hard threshold boundary
        prob = cv2.GaussianBlur(prob, (0, 0), 5)
        return np.clip(prob, 0.0, 1.0)


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model_dir = "models"
    path = FaceOccluder.find_model(model_dir)
    if path is None:
        print("[ERROR] No XSeg model found in models/")
        print("  Download xseg_2.onnx from:")
        print("  https://github.com/facefusion/facefusion-assets/releases")
        sys.exit(1)

    try:
        from neural_swap import _align_face, NeuralFaceSwapper
        swapper = NeuralFaceSwapper(model_path="models/ghost_3_256.onnx", occlusion=False)
        occluder = FaceOccluder(path)
    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    print("Running — put your hand on your face to test occlusion. Press q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        faces = swapper.detect(frame)
        if faces:
            face = faces[0]
            from neural_swap import _align_face, _best_providers
            from neural_swap import _ARCFACE_112
            import numpy as np
            kps = face.kps
            size = 256
            template = _ARCFACE_112 * (size / 112.0)
            M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC, ransacReprojThreshold=100)
            if M is not None:
                aligned = cv2.warpAffine(frame, M, (size, size))
                mask = occluder.create_occlusion_mask(aligned)
                # Overlay: green = face (safe), red = occluder
                overlay = aligned.copy()
                overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(float) + mask * 60, 0, 255).astype(np.uint8)
                overlay[:, :, 2] = np.clip(overlay[:, :, 2].astype(float) + (1 - mask) * 60, 0, 255).astype(np.uint8)
                cv2.imshow("Aligned + XSeg mask (green=face, red=occluder)", overlay)
                cv2.imshow("XSeg mask", mask)
        else:
            cv2.imshow("Aligned + XSeg mask (green=face, red=occluder)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
