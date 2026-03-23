"""Thread-safe webcam capture — always delivers the latest frame (no queuing)."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class WebcamCapture:
    """Opens a webcam in a background thread for low-latency frame delivery.

    Always returns the MOST RECENT frame — old frames are discarded.
    This prevents frame-queue buildup that would cause latency creep.

    Args:
        device_id: Camera index (0 = default).
        width: Requested capture width.
        height: Requested capture height.
        fps: Requested capture framerate.
        warmup_frames: Frames to discard on startup (let auto-exposure settle).
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 10,
    ):
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup = warmup_frames

        self._frame: np.ndarray | None = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self._open()

    def _open(self) -> None:
        # DirectShow backend on Windows for best compatibility
        cap = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {self._device_id}")

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait for first frame
        for _ in range(50):
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.02)

    def _capture_loop(self) -> None:
        warmup = self._warmup
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if warmup > 0:
                warmup -= 1
                continue
            with self._lock:
                self._frame = frame
                self._timestamp = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> tuple[np.ndarray | None, float]:
        """Return (latest BGR frame, timestamp). Thread-safe."""
        with self._lock:
            return (self._frame.copy() if self._frame is not None else None,
                    self._timestamp)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """cv2-compatible read() interface."""
        frame, _ = self.get_frame()
        return (frame is not None), frame

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        if not self._cap:
            return 0, 0
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self._cap = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
