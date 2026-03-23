"""Layer 3: Body and object segmentation.

Beyond hands, arms, shoulders, clothing, and held objects (cups, phones,
microphones) can occlude the face. MediaPipe Selfie Segmentation gives a
person-vs-background mask; combined with face parsing we isolate
'person but not face' = potential occluders in the face bounding box.
"""

from __future__ import annotations

import cv2
import numpy as np


class BodySegmenter:
    """MediaPipe Selfie Segmentation wrapper — Layer 3 of the occlusion system.

    Strategy:
        person_mask  = selfie segmentation (everything that is a person)
        face_mask    = face parsing (just the face pixels)
        body_not_face = person_mask AND NOT face_mask
        body_over_face = body_not_face AND face_bbox_region
        → anything that is 'person but not face' inside the face area
          is almost certainly in FRONT of the face (arm, hand, shoulder…)

    Args:
        config: dict with optional keys:
            enabled (bool)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._selfie = None

        if self._enabled:
            self._load_mediapipe()

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            self._selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # landscape (faster)
            )
            print("[BodySegmenter] MediaPipe SelfieSegmentation loaded")
        except Exception as e:
            print(f"[BodySegmenter] MediaPipe not available: {e} — layer disabled")
            self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_person_mask(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return float32 (H, W) mask: 1.0 = person pixel, 0.0 = background."""
        if not self._enabled or self._selfie is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._selfie.process(rgb)
        if results.segmentation_mask is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        mask = results.segmentation_mask  # float32 in [0, 1]
        return (mask > threshold).astype(np.float32)

    def get_body_over_face_mask(
        self,
        frame: np.ndarray,
        face_bbox: np.ndarray,
        face_parse_mask: np.ndarray,
        person_threshold: float = 0.5,
    ) -> np.ndarray:
        """Return float32 mask of body/arm pixels overlapping the face bbox.

        Args:
            frame:           Full BGR frame.
            face_bbox:       [x1, y1, x2, y2].
            face_parse_mask: Float32 (H, W) mask from FaceParser (inner face).
            person_threshold: Selfie segmentation confidence threshold.

        Returns:
            float32 mask (H, W): 1.0 = body-but-not-face pixel over face area.
        """
        H, W = frame.shape[:2]
        person_mask = self.get_person_mask(frame, person_threshold)

        # body_not_face: person pixels that are NOT face pixels
        body_not_face = person_mask * (1.0 - face_parse_mask)

        # Restrict to face bounding box region
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        face_bbox_region = np.zeros((H, W), dtype=np.float32)
        face_bbox_region[y1:y2, x1:x2] = 1.0

        body_over_face = body_not_face * face_bbox_region
        return np.clip(body_over_face, 0.0, 1.0)

    def get_object_mask(
        self, frame: np.ndarray, face_bbox: np.ndarray
    ) -> np.ndarray:
        """Heuristic non-body object detection within the face bounding box.

        Uses Canny edge detection to find strong edges inconsistent with
        smooth face skin — likely microphones, cups, phones, etc.

        Returns float32 mask (H, W).
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((H, W), dtype=np.float32)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Strong edges only (high threshold)
        edges = cv2.Canny(gray, 80, 200)

        # Dilate edges to create regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_d = cv2.dilate(edges, kernel, iterations=2)

        # Find connected regions
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edges_d, connectivity=8)

        object_roi = np.zeros_like(edges_d, dtype=np.float32)
        roi_gray = gray.astype(np.float32)

        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 200:  # too small, skip
                continue
            region_pixels = roi_gray[labels == lbl]
            # High local texture variance suggests a non-face object
            if region_pixels.std() > 25:
                object_roi[labels == lbl] = 1.0

        mask = np.zeros((H, W), dtype=np.float32)
        mask[y1:y2, x1:x2] = object_roi
        return mask

    def close(self) -> None:
        if self._selfie is not None:
            self._selfie.close()
            self._selfie = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from insightface.app import FaceAnalysis
    from src.occlusion.face_parsing import FaceParser

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    parser = FaceParser({"model_path": "models/face_parser.onnx"})
    segmenter = BodySegmenter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = frame.copy()
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            face_mask = parser.get_face_region_mask(frame, bbox)
            body_mask = segmenter.get_body_over_face_mask(frame, bbox, face_mask)
            yellow = np.zeros_like(frame)
            yellow[:, :, 0] = (body_mask * 255).astype(np.uint8)
            yellow[:, :, 2] = (body_mask * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 0.6, yellow, 0.4, 0)
        cv2.imshow("BodySegmenter Layer 3 (yellow=body-over-face)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    segmenter.close()
    cap.release()
    cv2.destroyAllWindows()
