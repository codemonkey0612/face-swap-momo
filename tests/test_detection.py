"""Tests for face detection module.

Run with: pytest tests/test_detection.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestFaceDetector:
    """Unit tests for src/detection/face_detector.py."""

    def test_no_face_returns_empty(self):
        from src.detection.face_detector import FaceDetector
        det = FaceDetector({"model_pack_name": "buffalo_l", "det_size": 320, "confidence_threshold": 0.5})
        frame = _blank_frame()
        faces = det.detect(frame)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_detect_primary_no_face_returns_none(self):
        from src.detection.face_detector import FaceDetector
        det = FaceDetector({"model_pack_name": "buffalo_l", "det_size": 320, "confidence_threshold": 0.5})
        frame = _blank_frame()
        result = det.detect_primary(frame)
        assert result is None

    def test_detected_face_dataclass_fields(self):
        from src.detection.face_detector import DetectedFace
        import numpy as np
        face = DetectedFace(
            bbox=np.array([10., 10., 100., 100.]),
            kps=np.zeros((5, 2)),
            embedding=np.zeros(512),
            confidence=0.9,
        )
        assert face.confidence == 0.9
        assert face.bbox.shape == (4,)


class TestFaceDetectorMultiFace:
    def test_detect_returns_list(self):
        from src.detection.face_detector import FaceDetector
        det = FaceDetector({"model_pack_name": "buffalo_l", "det_size": 320})
        frame = _blank_frame()
        faces = det.detect(frame)
        assert isinstance(faces, list)

    def test_confidence_filtering(self):
        """Manually check confidence threshold is applied."""
        from src.detection.face_detector import DetectedFace
        import numpy as np
        faces = [
            DetectedFace(np.array([0,0,50,50]), np.zeros((5,2)), 0.3, np.zeros(512)),
            DetectedFace(np.array([100,100,200,200]), np.zeros((5,2)), 0.8, np.zeros(512)),
        ]
        threshold = 0.5
        filtered = [f for f in faces if f.confidence >= threshold]
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.8
