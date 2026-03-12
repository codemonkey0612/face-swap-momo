"""GFPGAN face restoration — optional quality post-processor for neural swap."""

import os
import urllib.request

import cv2
import numpy as np

_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
_MODEL_PATH = "models/GFPGANv1.4.pth"


def _download_model():
    os.makedirs("models", exist_ok=True)
    print(f"[GFPGAN] Downloading model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[GFPGAN] Download complete.")


class FaceEnhancer:
    """Wraps GFPGANer for per-face restoration after swap."""

    def __init__(self, model_path=_MODEL_PATH, upscale=1):
        if not os.path.exists(model_path):
            _download_model()

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from gfpgan import GFPGANer
        self._enhancer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(device),
        )
        print(f"[GFPGAN] Loaded model: {model_path}")

    def enhance_frame(self, img, faces):
        """Run GFPGAN on each swapped face region and paste back.

        Args:
            img: BGR frame after swap.
            faces: InsightFace Face objects (provides bbox for region).

        Returns:
            Enhanced BGR frame.
        """
        if not faces:
            return img

        result = img.copy()
        for face in faces:
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            pad = 30
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(img.shape[1], x2 + pad)
            y2c = min(img.shape[0], y2 + pad)

            crop = result[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            try:
                _, _, enhanced = self._enhancer.enhance(
                    crop,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                )
                if enhanced is not None:
                    result[y1c:y2c, x1c:x2c] = enhanced
            except Exception:
                pass  # fall back to unenhanced on error

        return result
