"""GPU/ONNX Runtime provider utilities."""

import onnxruntime as ort


def best_providers() -> list[str]:
    """Return best available ONNX Runtime execution providers."""
    available = ort.get_available_providers()
    for p in ["CUDAExecutionProvider", "CoreMLExecutionProvider"]:
        if p in available:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def ctx_id(providers: list[str]) -> int:
    """Return InsightFace ctx_id: 0 for GPU, -1 for CPU."""
    for p in providers:
        if "CUDA" in p or "CoreML" in p:
            return 0
    return -1


def is_gpu(providers: list[str]) -> bool:
    return any("CUDA" in p or "CoreML" in p for p in providers)
