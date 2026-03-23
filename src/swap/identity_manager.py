"""Source face identity management — loads, caches, and averages embeddings.

Usage:
    mgr = IdentityManager(swapper)
    mgr.load("source_faces/ai_face.jpg")
    src_face = mgr.get_source_face()
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import cv2
import numpy as np

from neural_swap import NeuralFaceSwapper, build_averaged_face


class IdentityManager:
    """Manages one or more source face identities.

    Supports loading a single image, multiple images (averaged embedding),
    and switching between multiple loaded identities at runtime.
    """

    def __init__(self, swapper: NeuralFaceSwapper):
        self._swapper = swapper
        self._faces: list = []         # raw InsightFace Face objects
        self._names: list[str] = []    # display names / paths
        self._active: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> bool:
        """Load a single source face image. Returns True on success."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        faces = self._swapper.detect(img)
        if not faces:
            raise ValueError(f"No face detected in: {path}")
        if len(faces) > 1:
            print(f"[Identity] {len(faces)} faces found in {os.path.basename(path)} — using largest")
        self._faces.append(faces[0])
        self._names.append(os.path.basename(path))
        self._active = len(self._faces) - 1
        print(f"[Identity] Loaded: {self._names[-1]}")
        return True

    def load_directory(self, directory: str) -> int:
        """Load all images in directory, build averaged embedding. Returns count."""
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP")
        paths: list[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(directory, ext)))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No images found in: {directory}")
        avg_face = build_averaged_face(self._swapper, paths)
        self._faces.append(avg_face)
        self._names.append(f"avg({os.path.basename(directory)},{len(paths)})")
        self._active = len(self._faces) - 1
        return len(paths)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_source_face(self):
        """Return the active source Face object."""
        if not self._faces:
            raise RuntimeError("No source face loaded — call load() or load_directory() first")
        return self._faces[self._active]

    def get_embedding(self) -> np.ndarray:
        """Return the active 512-d normed_embedding."""
        return self.get_source_face().normed_embedding

    def set_active(self, index: int) -> None:
        if index < 0 or index >= len(self._faces):
            raise IndexError(f"Index {index} out of range (loaded: {len(self._faces)})")
        self._active = index
        print(f"[Identity] Active: {self._names[index]}")

    def list_identities(self) -> list[str]:
        return list(self._names)

    def similarity(self, other_embedding: np.ndarray) -> float:
        """Cosine similarity between active source and given embedding."""
        src = self.get_embedding()
        n1 = np.linalg.norm(src)
        n2 = np.linalg.norm(other_embedding)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(src, other_embedding) / (n1 * n2))
