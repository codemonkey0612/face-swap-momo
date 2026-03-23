"""Face swap engine — thin re-export of NeuralFaceSwapper from neural_swap.py.

Supports two backends auto-selected by model filename:
  • Ghost (ghost_*.onnx)       — best quality, 256 px, recommended
  • InsightFace (inswapper_128.onnx) — legacy, 128 px
"""

from neural_swap import (
    NeuralFaceSwapper,
    _GhostBackend as GhostBackend,
    _InsightFaceBackend as InsightFaceBackend,
    _is_ghost_model as is_ghost_model,
    _find_crossface as find_crossface,
)

__all__ = [
    "NeuralFaceSwapper",
    "GhostBackend",
    "InsightFaceBackend",
    "is_ghost_model",
    "find_crossface",
]
