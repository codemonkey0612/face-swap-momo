"""Face shape correction — landmark-guided shape blending.

FaceShapeAdapter corrects the shape mismatch between the AI source face
(narrow/wide) and the performer's actual face shape.  It runs AFTER the
swap model and BEFORE the aligned-space composite.

Usage:
    from src.shape import FaceShapeAdapter
"""

from src.shape.face_shape_adapter import FaceShapeAdapter

__all__ = ["FaceShapeAdapter"]
