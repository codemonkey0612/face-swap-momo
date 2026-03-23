"""Occlusion handling — XSeg aligned-space system (v2).

Primary:
  face_occluder  — XSeg purpose-trained model in aligned 256×256 space.
                   Single model replaces the entire 5-layer general-purpose system.
                   Accurate on skin-coloured occluders (hands flat on face) where
                   all general-purpose layers fail.

Supporting:
  face_parsing   — BiSeNet: still used for REGION identification in beauty filters
                   (skin, eye, lip detection), NOT for occlusion detection.
  mask_refiner   — Morphological cleanup + feathering (unchanged).

Legacy (kept for reference, no longer called from main pipeline):
  hand_detector, body_segmenter, error_detector, depth_estimator, occlusion_fuser
"""

from src.occlusion.face_occluder  import FaceOccluder
from src.occlusion.face_parsing   import FaceParser
from src.occlusion.mask_refiner   import MaskRefiner

# Legacy imports (still importable for tests / backward compat)
try:
    from src.occlusion.hand_detector   import HandDetector
    from src.occlusion.body_segmenter  import BodySegmenter
    from src.occlusion.error_detector  import ErrorDetector
    from src.occlusion.depth_estimator import DepthEstimator
    from src.occlusion.occlusion_fuser import OcclusionFuser
except ImportError:
    pass  # optional heavy deps (mediapipe, timm) may not be installed

__all__ = [
    "FaceOccluder",   # primary — XSeg
    "FaceParser",     # region masking for beauty filters
    "MaskRefiner",    # mask post-processing
]
