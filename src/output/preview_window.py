"""Local preview window with a clean, friendly HUD overlay and keyboard shortcuts."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.utils.metrics import FPSCounter

# ---------------------------------------------------------------------------
# Friendly status messages (plain English, no jargon)
# ---------------------------------------------------------------------------
_STATUS_LABELS: dict[str, str] = {
    "INACTIVE":        "Starting up...",
    "ACTIVE":          "Face detected",
    "FROZEN":          "Searching for your face...",
    "FADING_OUT":      "Face lost...",
    "FADING_IN":       "Found you!",
    "BLACK":           "No face in frame",
    "MANUAL_OVERRIDE": "Manual mode",
    "":                "Running",
}

# ---------------------------------------------------------------------------
# Color palette  (BGR)
# ---------------------------------------------------------------------------
_PINK     = (180, 105, 255)   # hot-pink accent
_LAVENDER = (210, 180, 220)   # soft lavender
_MINT     = (160, 220, 160)   # soft mint — positive state
_PEACH    = (130, 180, 255)   # peach — warning state
_WHITE    = (245, 245, 245)
_LGRAY    = (190, 190, 190)
_PANEL    = (20,  18,  22)    # near-black panel background


class PreviewWindow:
    """OpenCV preview window with a soft, friendly HUD overlay.

    Keyboard shortcuts
    ------------------
    q       Quit
    d       Toggle detailed diagnostics overlay
    b       Toggle beauty filters
    o       Toggle occlusion overlay
    s       Save screenshot
    +/-     Adjust beauty strength
    c       Recalibrate (flag read by main loop)
    """

    def __init__(self, title: str = "Face Transform Studio", show: bool = True):
        self._title = title
        self._show = show
        self._fps = FPSCounter()
        self._screenshot_dir = "."

        # State flags readable by the main loop
        self.quit             = False
        self.debug            = False
        self.beauty_enabled   = True
        self.occlusion_overlay = False
        self.recalibrate      = False

    # ------------------------------------------------------------------
    def show(self, frame: np.ndarray, metrics: dict | None = None) -> None:
        """Display frame with optional HUD and handle keyboard."""
        if not self._show:
            return

        self._fps.tick()
        display = frame.copy()
        display = self._draw_status_bar(display, metrics or {})
        if self.debug:
            display = self._draw_debug_panel(display, metrics or {})

        cv2.imshow(self._title, display)
        self._handle_keys(frame)

    # ------------------------------------------------------------------
    # Status bar — semi-transparent panel at the bottom of the frame
    # ------------------------------------------------------------------
    def _draw_status_bar(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h = 52

        # Semi-transparent dark strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), _PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Thin accent line at top of bar
        cv2.line(frame, (0, h - bar_h), (w, h - bar_h), _LAVENDER, 1)

        fps  = self._fps.fps()
        conf = metrics.get("detection_confidence", 0.0)
        state = metrics.get("failsafe_state", "")
        status_text = _STATUS_LABELS.get(state, state or "Running")

        # --- FPS badge (left side) ---
        fps_color = _MINT if fps >= 20 else _PEACH
        cv2.putText(frame, f"{fps:.0f} FPS",
                    (14, h - bar_h + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_color, 2, cv2.LINE_AA)

        # --- Status text (center) ---
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cx = (w - tw) // 2
        status_color = _MINT if state in ("ACTIVE", "FADING_IN", "") else _PEACH
        cv2.putText(frame, status_text,
                    (cx, h - bar_h + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 1, cv2.LINE_AA)

        # --- Confidence bar (right side) ---
        bar_w_max = 120
        bar_w     = int(bar_w_max * conf)
        bx = w - bar_w_max - 14
        by = h - bar_h + 16
        # Background track
        cv2.rectangle(frame, (bx, by), (bx + bar_w_max, by + 10), (60, 55, 65), -1)
        # Filled portion
        fill_color = _MINT if conf >= 0.85 else _PEACH
        if bar_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 10), fill_color, -1)
        cv2.putText(frame, "Confidence",
                    (bx, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _LGRAY, 1, cv2.LINE_AA)

        # --- Keyboard hint (very small, bottom-right corner) ---
        hints = "q quit  |  s screenshot  |  d debug  |  b beauty"
        cv2.putText(frame, hints,
                    (bx - 80, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 95, 110), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Debug panel — detailed latency breakdown (toggled with 'd')
    # ------------------------------------------------------------------
    def _draw_debug_panel(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        latencies = metrics.get("stage_latencies", {})
        if not latencies:
            return frame

        total = sum(latencies.values())
        lines = [f"Total: {total:.1f} ms"] + [
            f"  {name}: {ms:.1f} ms" for name, ms in latencies.items()
        ]

        panel_w  = 230
        line_h   = 16
        panel_h  = len(lines) * line_h + 18
        px, py   = 10, 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), _PANEL, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), _LAVENDER, 1)

        for i, line in enumerate(lines):
            color = _PINK if i == 0 else _LGRAY
            cv2.putText(frame, line,
                        (px + 8, py + 14 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    def _handle_keys(self, original_frame: np.ndarray) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.quit = True
        elif key == ord("d"):
            self.debug = not self.debug
        elif key == ord("b"):
            self.beauty_enabled = not self.beauty_enabled
            label = "ON" if self.beauty_enabled else "OFF"
            print(f"[Preview] Beauty filters: {label}")
        elif key == ord("o"):
            self.occlusion_overlay = not self.occlusion_overlay
        elif key == ord("s"):
            path = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(path, original_frame)
            print(f"[Preview] Screenshot saved: {path}")
        elif key == ord("c"):
            self.recalibrate = True

    def close(self) -> None:
        if self._show:
            try:
                cv2.destroyWindow(self._title)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def draw_fps(frame: np.ndarray, fps_counter: FPSCounter) -> np.ndarray:
    """Draw a small FPS badge on the frame (utility for quick scripts)."""
    fps = fps_counter.fps()
    cv2.putText(frame, f"{fps:.0f} FPS", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _MINT, 2, cv2.LINE_AA)
    return frame
