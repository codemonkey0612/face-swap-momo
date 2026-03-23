"""FPS counter and freeze-frame manager."""

import time
from collections import deque

import cv2
import numpy as np


class FPSCounter:
    """Rolling-average FPS tracker."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class FreezeFrameManager:
    """Holds the last good frame when face detection drops out."""

    def __init__(self, max_frozen_frames: int = 90):
        self._frozen: np.ndarray | None = None
        self._miss_count: int = 0
        self._max = max_frozen_frames

    def update(self, frame: np.ndarray) -> None:
        self._frozen = frame
        self._miss_count = 0

    def miss(self) -> None:
        self._miss_count += 1

    def get_frozen(self) -> np.ndarray | None:
        if self._frozen is not None and self._miss_count < self._max:
            return self._frozen
        return None
