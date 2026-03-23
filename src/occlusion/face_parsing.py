"""Layer 1: Face parsing via BiSeNet ONNX model.

Identifies which pixels belong to the face (skin, brows, eyes, nose, lips).
This defines the MAXIMUM area where face swapping is valid.

Note: This layer tells us WHERE the face IS, but not whether something
is in FRONT of it. A hand over the mouth makes those pixels appear as
'not face' — which is correct, but Layer 2-5 tell us WHY.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional

# BiSeNet 19-class labels
FACE_INNER_CLASSES = {1, 2, 3, 4, 5, 10, 11, 12, 13}  # skin, brows, eyes, nose, lips
FACE_ALL_CLASSES   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}  # + ears, earring


class FaceParser:
    """BiSeNet face parsing — Layer 1 of the occlusion system.

    Wraps the ONNX face_parser.onnx (BiSeNet ResNet-34) already used
    by neural_swap.py, exposing the CLAUDE.md-specified interface.

    Args:
        config: dict with optional keys:
            model_path (str): path to face_parser.onnx
            enabled (bool): enable/disable this layer
    """

    # 19-class colour palette for visualisation
    _PALETTE = np.array([
        [0,   0,   0  ],  # 0  background
        [255, 220, 177],  # 1  skin
        [139, 119,  86],  # 2  l_brow
        [139, 119,  86],  # 3  r_brow
        [0,   162, 232],  # 4  l_eye
        [0,   162, 232],  # 5  r_eye
        [200, 200, 200],  # 6  eye_glasses
        [255, 140,   0],  # 7  l_ear
        [255, 140,   0],  # 8  r_ear
        [218, 165,  32],  # 9  earring
        [255, 100, 100],  # 10 nose
        [220,  20,  60],  # 11 mouth_inside
        [255, 105, 180],  # 12 u_lip
        [255, 105, 180],  # 13 l_lip
        [255, 228, 196],  # 14 neck
        [210, 180, 140],  # 15 necklace
        [128, 128, 128],  # 16 cloth
        [101,  67,  33],  # 17 hair
        [169, 169, 169],  # 18 hat
    ], dtype=np.uint8)

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._model_path = cfg.get("model_path", "models/face_parser.onnx")
        self._session = None

        if self._enabled:
            self._load_model()

    def _load_model(self) -> None:
        import os
        if not os.path.exists(self._model_path):
            print(f"[FaceParser] Model not found: {self._model_path} — layer disabled")
            self._enabled = False
            return
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            print(f"[FaceParser] Loaded {self._model_path}")
        except Exception as e:
            print(f"[FaceParser] Load failed: {e} — layer disabled")
            self._enabled = False

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def parse(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Run BiSeNet on the face region and return a 19-class map at frame resolution.

        Args:
            frame: Full BGR frame (H, W, 3).
            bbox:  Face bounding box [x1, y1, x2, y2].

        Returns:
            seg_map: uint8 array (H, W) with class indices 0-18,
                     or zeros if disabled/failed.
        """
        if not self._enabled or self._session is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 40% padding around bbox
        pw = int((x2 - x1) * 0.4)
        ph = int((y2 - y1) * 0.4)
        rx1 = max(0, x1 - pw)
        ry1 = max(0, y1 - ph)
        rx2 = min(W, x2 + pw)
        ry2 = min(H, y2 + ph)

        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # Prepare 512×512 input
        inp = cv2.resize(crop, (512, 512)).astype(np.float32)
        inp = (inp / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        try:
            out = self._session.run(None, {self._session.get_inputs()[0].name: inp})[0]
        except Exception as e:
            print(f"[FaceParser] Inference error: {e}")
            return np.zeros((H, W), dtype=np.uint8)

        # Argmax → class map in crop space
        seg_crop = np.argmax(out[0], axis=0).astype(np.uint8)  # (512, 512)
        seg_crop = cv2.resize(seg_crop, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST)

        # Embed back into full frame
        seg_map = np.zeros((H, W), dtype=np.uint8)
        seg_map[ry1:ry2, rx1:rx2] = seg_crop
        return seg_map

    def get_face_region_mask(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Return a float32 mask [0, 1] of inner face pixels at frame resolution.

        Inner face classes: skin(1), l_brow(2), r_brow(3), l_eye(4), r_eye(5),
                            nose(10), mouth_inside(11), u_lip(12), l_lip(13).

        Args:
            frame: Full BGR frame.
            bbox:  Face bounding box [x1, y1, x2, y2].

        Returns:
            float32 mask (H, W): 1.0 = face pixel, 0.0 = not face.
        """
        seg = self.parse(frame, bbox)
        mask = np.zeros(seg.shape, dtype=np.float32)
        for cls in FACE_INNER_CLASSES:
            mask[seg == cls] = 1.0
        return mask

    def get_face_all_mask(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Broader face mask including ears and accessories."""
        seg = self.parse(frame, bbox)
        mask = np.zeros(seg.shape, dtype=np.float32)
        for cls in FACE_ALL_CLASSES:
            mask[seg == cls] = 1.0
        return mask

    def colorize(self, seg_map: np.ndarray) -> np.ndarray:
        """Return a colour-coded BGR visualisation of the segmentation map."""
        vis = self._PALETTE[np.clip(seg_map, 0, 18)]
        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import insightface
    from insightface.app import FaceAnalysis

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    parser = FaceParser({"model_path": "models/face_parser.onnx"})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            seg = parser.parse(frame, bbox)
            vis = parser.colorize(seg)
            overlay = cv2.addWeighted(frame, 0.5, vis, 0.5, 0)
        else:
            overlay = frame
        cv2.imshow("FaceParser Layer 1", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
