"""MediaPipe FaceMesh landmark extraction and region index constants."""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Face oval indices (ordered contour from MediaPipe's FACEMESH_FACE_OVAL)
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Outer lip contour indices
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Inner lip contour indices
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

ALL_LIP_INDICES = set(OUTER_LIP_INDICES + INNER_LIP_INDICES)


def create_face_mesh(max_faces=2):
    """Create a MediaPipe FaceMesh instance."""
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(frame, face_mesh):
    """Extract face landmarks from a BGR frame.

    Returns a list of (468, 2) int32 arrays, one per detected face.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return []

    faces = []
    for face_lm in results.multi_face_landmarks:
        pts = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark],
            dtype=np.int32,
        )
        faces.append(pts)
    return faces


def get_face_polygon(landmarks):
    """Return the face oval contour as an ordered polygon."""
    return landmarks[FACE_OVAL_INDICES]


def get_lip_polygon(landmarks, scale=1.12):
    """Return the outer lip contour, expanded outward by `scale` factor."""
    pts = landmarks[OUTER_LIP_INDICES].astype(np.float64)
    centroid = pts.mean(axis=0)
    expanded = centroid + scale * (pts - centroid)
    return expanded.astype(np.int32)
