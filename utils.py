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


class LandmarkSmoother:
    """Velocity-adaptive exponential moving average smoother.

    Uses heavier smoothing when landmarks move slowly (reduces jitter)
    and lighter smoothing during fast motion (reduces lag).
    alpha_slow: smoothing factor when face is mostly still.
    alpha_fast: smoothing factor during rapid movement.
    velocity_threshold: pixel displacement per landmark that switches to fast mode.
    """

    def __init__(self, alpha_slow=0.4, alpha_fast=0.8, velocity_threshold=3.0):
        self._alpha_slow = alpha_slow
        self._alpha_fast = alpha_fast
        self._vel_thresh = velocity_threshold
        self._smoothed = []

    def update(self, landmarks_list):
        if not landmarks_list:
            return landmarks_list
        # Reset if number of detected faces changed
        if len(self._smoothed) != len(landmarks_list):
            self._smoothed = [lm.astype(np.float32).copy() for lm in landmarks_list]
            return [lm.copy() for lm in landmarks_list]
        result = []
        for i, lm in enumerate(landmarks_list):
            lm_f = lm.astype(np.float32)
            # Compute per-landmark velocity (mean pixel displacement)
            velocity = np.mean(np.abs(lm_f - self._smoothed[i]))
            # Interpolate alpha based on velocity
            t = np.clip(velocity / self._vel_thresh, 0, 1)
            alpha = self._alpha_slow + t * (self._alpha_fast - self._alpha_slow)
            self._smoothed[i] = alpha * lm_f + (1 - alpha) * self._smoothed[i]
            result.append(self._smoothed[i].astype(np.int32))
        return result


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
