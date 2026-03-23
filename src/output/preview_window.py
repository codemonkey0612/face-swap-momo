"""Local preview window with stats overlay and keyboard shortcuts."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.utils.metrics import FPSCounter


class PreviewWindow:
    """OpenCV preview window with HUD overlay and keyboard handling.

    Keyboard shortcuts:
        q       Quit
        d       Toggle debug / diagnostics overlay
        b       Toggle beauty filters
        o       Toggle occlusion overlay
        s       Save screenshot
        +/-     Adjust beauty strength
        c       Recalibrate (forwarded to caller via .recalibrate flag)
    """

    def __init__(self, title: str = "Face Swap", show: bool = True):
        self._title = title
        self._show = show
        self._fps = FPSCounter()
        self._screenshot_dir = "."

        # State flags readable by the main loop
        self.quit = False
        self.debug = False
        self.beauty_enabled = True
        self.occlusion_overlay = False
        self.recalibrate = False

    def show(self, frame: np.ndarray, metrics: dict | None = None) -> None:
        """Display frame with optional HUD and handle keyboard."""
        if not self._show:
            return

        self._fps.tick()
        display = frame.copy()

        if metrics or self.debug:
            display = self._draw_hud(display, metrics or {})

        cv2.imshow(self._title, display)
        self._handle_keys(frame)

    def _draw_hud(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw FPS, latency, failsafe state and other metrics."""
        fps = self._fps.fps()
        y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28

        conf = metrics.get("detection_confidence", 0.0)
        color = (0, 255, 0) if conf >= 0.85 else (0, 165, 255)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 22

        state = metrics.get("failsafe_state", "")
        if state and state != "INACTIVE":
            cv2.putText(frame, f"Failsafe: {state}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 22

        latencies = metrics.get("stage_latencies", {})
        total = sum(latencies.values())
        if total > 0:
            cv2.putText(frame, f"Total: {total:.1f}ms", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 18
            for name, ms in latencies.items():
                cv2.putText(frame, f"  {name}: {ms:.1f}ms", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
                y += 14

        return frame

    def _handle_keys(self, original_frame: np.ndarray) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.quit = True
        elif key == ord("d"):
            self.debug = not self.debug
        elif key == ord("b"):
            self.beauty_enabled = not self.beauty_enabled
            print(f"[Preview] Beauty: {'ON' if self.beauty_enabled else 'OFF'}")
        elif key == ord("o"):
            self.occlusion_overlay = not self.occlusion_overlay
        elif key == ord("s"):
            path = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(path, original_frame)
            print(f"[Preview] Saved: {path}")
        elif key == ord("c"):
            self.recalibrate = True

    def close(self) -> None:
        if self._show:
            cv2.destroyWindow(self._title)


def draw_fps(frame: np.ndarray, fps_counter: FPSCounter) -> np.ndarray:
    """Draw FPS counter on frame (utility for quick scripts)."""
    fps = fps_counter.fps()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
