"""Main pipeline orchestrator — wires all src/ modules together.

This module re-exports the production-grade UltimateSwapPipeline and its
config from pipeline.py, and provides SwapPipeline as a higher-level
wrapper that owns the webcam, virtual camera, and preview window.

Usage (simple):
    from src.pipeline import SwapPipeline, PipelineConfig, load_config
    cfg = load_config("config/default.yaml")
    with SwapPipeline(cfg) as pipe:
        pipe.run()

Usage (advanced — direct control):
    from src.pipeline import UltimateSwapPipeline, PipelineConfig
    cfg = PipelineConfig(swap_model="models/ghost_3_256.onnx", ...)
    pipe = UltimateSwapPipeline(source_image_path="source_faces/ai.jpg", config=cfg)
    result = pipe.process_frame(frame)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

# Re-export production pipeline and config
from pipeline import UltimateSwapPipeline, PipelineConfig, load_config_yaml

from src.capture.webcam import WebcamCapture
from src.output.virtual_camera import VirtualCameraOutput
from src.output.preview_window import PreviewWindow
from src.utils.metrics import FPSCounter


def load_config(path: str = "config/default.yaml") -> PipelineConfig:
    """Load a YAML config file into a PipelineConfig.

    Searches config/default.yaml first, falls back to config.yaml.
    """
    import os
    for candidate in (path, "config/default.yaml", "config.yaml"):
        if os.path.exists(candidate):
            return load_config_yaml(candidate)
    return PipelineConfig()


class SwapPipeline:
    """Full pipeline: webcam → swap → virtual camera + preview.

    Owns all hardware resources (webcam, virtual camera) and runs the
    main loop. UltimateSwapPipeline handles the per-frame processing.

    Args:
        config: PipelineConfig (use load_config() to build from YAML).
        source: Path to source face image (overrides config.source_image_path).
        camera: Webcam device index (overrides default).
        show_preview: Show local OpenCV preview window.
    """

    def __init__(
        self,
        config: PipelineConfig,
        source: str | None = None,
        camera: int = 0,
        show_preview: bool = True,
    ):
        self._cfg = config
        if source:
            config.source_image_path = source

        # --- Webcam ---
        self._cap = WebcamCapture(
            device_id=camera,
            width=config.output_width,
            height=config.output_height,
            fps=30,
        )

        # --- Core swap pipeline ---
        self._pipe = UltimateSwapPipeline(
            source_image_path=config.source_image_path,
            config=config,
        )

        # --- Virtual camera (optional) ---
        self._vcam: Optional[VirtualCameraOutput] = None
        if config.obs_virtual_camera:
            try:
                self._vcam = VirtualCameraOutput(
                    width=config.output_width,
                    height=config.output_height,
                    fps=30,
                )
            except Exception as e:
                print(f"[SwapPipeline] Virtual camera unavailable: {e}")

        # --- Preview window ---
        self._preview = PreviewWindow(show=show_preview)
        self._fps_counter = FPSCounter()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Capture → process → output loop. Runs until 'q' or KeyboardInterrupt."""
        print("[SwapPipeline] Running. Press 'q' to quit.")
        stats_timer = time.monotonic()

        try:
            while not self._preview.quit:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                self._fps_counter.tick()

                # Resize to configured output resolution if needed
                h, w = frame.shape[:2]
                if w != self._cfg.output_width or h != self._cfg.output_height:
                    frame = cv2.resize(
                        frame,
                        (self._cfg.output_width, self._cfg.output_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                result = self._pipe.process_frame(frame)

                if self._vcam:
                    self._vcam.send(result)

                diag = self._pipe.get_diagnostics()
                diag["fps"] = self._fps_counter.fps()
                self._preview.show(result, diag)

                # Periodic stats print
                now = time.monotonic()
                if now - stats_timer >= 5.0:
                    self._print_stats(diag)
                    stats_timer = now

        except KeyboardInterrupt:
            print("\n[SwapPipeline] Interrupted.")
        finally:
            self.shutdown()

    def _print_stats(self, diag: dict) -> None:
        fps = diag.get("fps", 0.0)
        lat = diag.get("stage_latencies", {})
        total = sum(lat.values())
        print(f"\n[Stats] FPS={fps:.1f}  total={total:.1f}ms")
        for k, v in lat.items():
            print(f"   {k:20s} {v:.1f}ms")

    def shutdown(self) -> None:
        """Release all resources in correct order."""
        print("[SwapPipeline] Shutting down...")
        self._pipe.shutdown()
        self._cap.release()
        if self._vcam:
            self._vcam.close()
        self._preview.close()
        cv2.destroyAllWindows()


__all__ = [
    "SwapPipeline",
    "UltimateSwapPipeline",
    "PipelineConfig",
    "load_config",
    "load_config_yaml",
]
