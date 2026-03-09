"""Helpers: FPS counter, freeze-frame manager, debug overlay."""

import time
from collections import deque

import cv2
import numpy as np


class FPSCounter:
    """Rolling-average FPS tracker."""

    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.monotonic())

    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed == 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


class FreezeFrameManager:
    """Holds the last good frame when face detection drops out."""

    def __init__(self, max_frozen_frames=90):
        self._frozen = None
        self._miss_count = 0
        self._max = max_frozen_frames

    def update(self, frame):
        self._frozen = frame
        self._miss_count = 0

    def miss(self):
        self._miss_count += 1

    def get_frozen(self):
        if self._frozen is not None and self._miss_count < self._max:
            return self._frozen
        return None


def draw_debug_overlay(frame, landmarks, triangles=None):
    """Draw landmarks and optional Delaunay triangles on frame."""
    for pt in landmarks:
        cv2.circle(frame, tuple(pt), 1, (0, 255, 0), -1)

    if triangles is not None:
        for i, j, k in triangles:
            pts = np.array([landmarks[i], landmarks[j], landmarks[k]], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_fps(frame, fps_counter):
    """Draw FPS counter on frame."""
    fps = fps_counter.fps()
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    return frame
