"""Face detection and landmark extraction.

Wraps InsightFace (RetinaFace / buffalo_l) for face detection and
MediaPipe FaceMesh for dense 478-point landmark extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.utils.gpu_utils import best_providers, ctx_id

# ---------------------------------------------------------------------------
# MediaPipe landmark index constants
# ---------------------------------------------------------------------------

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

ALL_LIP_INDICES = set(OUTER_LIP_INDICES + INNER_LIP_INDICES)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    """Unified face detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    kps: np.ndarray           # 5×2 ArcFace keypoints
    confidence: float
    embedding: np.ndarray     # 512-d ArcFace (normed)
    mesh: Optional[np.ndarray] = field(default=None)  # 478×2 MediaPipe landmarks
    face_object: object = field(default=None)          # raw InsightFace face


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class FaceDetector:
    """InsightFace RetinaFace wrapper with optional MediaPipe mesh extraction.

    Args:
        det_size: Detection input resolution (larger = slower but more accurate).
        det_thresh: Minimum detection confidence (InsightFace default 0.3).
        mesh: If True, also run MediaPipe FaceMesh on each detected face.
        model_pack: InsightFace model pack name (default 'buffalo_l').
    """

    def __init__(
        self,
        det_size: int = 640,
        det_thresh: float = 0.3,
        mesh: bool = False,
        model_pack: str = "buffalo_l",
    ):
        from insightface.app import FaceAnalysis

        self._providers = best_providers()
        self._app = FaceAnalysis(name=model_pack, providers=self._providers)
        self._app.prepare(
            ctx_id=ctx_id(self._providers),
            det_size=(det_size, det_size),
            det_thresh=det_thresh,
        )
        self._mesh_inst = None
        if mesh:
            self._mesh_inst = self._create_mesh()

    @staticmethod
    def _create_mesh(max_faces: int = 2):
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in frame, sorted by bbox area (largest first)."""
        raw = self._app.get(frame)
        if not raw:
            return []
        results = []
        for f in raw:
            mesh = self._extract_mesh(frame) if self._mesh_inst else None
            results.append(DetectedFace(
                bbox=f.bbox,
                kps=f.kps,
                confidence=float(f.det_score),
                embedding=f.normed_embedding,
                mesh=mesh,
                face_object=f,
            ))
        # Sort by area descending
        results.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return results

    def detect_primary(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Return the single largest face, or None."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    def _extract_mesh(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe FaceMesh. Returns (478, 2) int32 or None."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mesh_inst.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0]
        return np.array([(int(p.x * w), int(p.y * h)) for p in lm.landmark], dtype=np.int32)

    def get_lip_polygon(self, mesh: np.ndarray, scale: float = 1.12) -> np.ndarray:
        """Return outer lip contour polygon, expanded by scale factor."""
        pts = mesh[OUTER_LIP_INDICES].astype(np.float64)
        centroid = pts.mean(axis=0)
        expanded = centroid + scale * (pts - centroid)
        return expanded.astype(np.int32)

    def get_eye_regions(self, mesh: np.ndarray, pad_scale: float = 1.4) -> list[tuple]:
        """Return (x1, y1, x2, y2) bboxes for left and right eye+brow regions."""
        regions = []
        for eye_idx, brow_idx in [(LEFT_EYE_INDICES, LEFT_EYEBROW_INDICES),
                                   (RIGHT_EYE_INDICES, RIGHT_EYEBROW_INDICES)]:
            pts = mesh[eye_idx + brow_idx].astype(np.float64)
            centroid = pts.mean(axis=0)
            expanded = centroid + pad_scale * (pts - centroid)
            x1, y1 = expanded.min(axis=0).astype(int)
            x2, y2 = expanded.max(axis=0).astype(int)
            regions.append((x1, y1, x2, y2))
        return regions

    def close(self) -> None:
        if self._mesh_inst:
            self._mesh_inst.close()
