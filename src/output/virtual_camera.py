"""Virtual camera output via pyvirtualcam (OBS Virtual Camera backend).

Windows setup:
  Option A — OBS Virtual Camera (recommended):
    1. Install OBS Studio (https://obsproject.com) — v26+ includes virtual camera.
    2. Start faceswap-stream FIRST (it claims the device), then open OBS.
    3. In OBS add "Video Capture Device" → select "OBS Virtual Camera".

  Option B — Unity Capture (alternative, avoids OBS conflicts):
    1. Download UnityCapture: https://github.com/schellingb/UnityCapture
    2. Run Install.bat, then set backend='unitycapture' in config.
    3. In OBS add "Video Capture Device" → select "Unity Video Capture".
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class VirtualCameraOutput:
    """Sends frames to a virtual camera device via pyvirtualcam.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        fps: Target frames per second.
        backend: 'obs' or 'unitycapture'.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        backend: str = "obs",
    ):
        import pyvirtualcam

        kwargs = dict(width=width, height=height, fps=fps,
                      fmt=pyvirtualcam.PixelFormat.BGR)
        if backend == "unitycapture":
            kwargs["backend"] = "unitycapture"

        self._cam = pyvirtualcam.Camera(**kwargs)
        self._fps = fps
        self._frame_dur = 1.0 / fps
        self._last_send = 0.0
        print(f"[VirtualCamera] {self._cam.device}  {width}x{height}@{fps}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, frame: np.ndarray) -> None:
        """Send a BGR frame. Handles BGR format and rate limiting."""
        now = time.monotonic()
        elapsed = now - self._last_send
        if elapsed < self._frame_dur:
            time.sleep(self._frame_dur - elapsed)
        self._cam.send(frame)
        self._cam.sleep_until_next_frame()
        self._last_send = time.monotonic()

    def close(self) -> None:
        try:
            self._cam.close()
        except Exception:
            pass

    @property
    def device(self) -> str:
        return self._cam.device
