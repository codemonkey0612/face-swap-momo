"""Pipeline benchmark: measures per-stage latency and overall FPS.

Usage:
    python scripts/benchmark.py [--frames 300] [--source source_faces/ai.jpg]
                                [--device 0] [--no-display]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _mean_ms(times: list[float]) -> str:
    if not times:
        return "  N/A"
    return f"{np.mean(times)*1000:6.1f} ms  (σ={np.std(times)*1000:.1f})"


def run_benchmark(args):
    print("\n=== faceswap-stream benchmark ===\n")

    # --- Load pipeline ---
    from src.pipeline import load_config, SwapPipeline
    from pipeline import UltimateSwapPipeline, PipelineConfig, load_config_yaml

    cfg_path = "config/default.yaml"
    if os.path.exists(cfg_path):
        cfg = load_config_yaml(cfg_path)
    else:
        cfg = PipelineConfig()
    if args.source:
        cfg.source_image_path = args.source

    print("Loading UltimateSwapPipeline...")
    t_load = time.perf_counter()
    try:
        pipe = UltimateSwapPipeline(
            source_image_path=cfg.source_image_path,
            config=cfg,
        )
    except Exception as e:
        print(f"[ERROR] Pipeline init failed: {e}")
        return
    print(f"  Loaded in {(time.perf_counter() - t_load)*1000:.0f} ms\n")

    # --- Open webcam ---
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_times: list[float] = []
    stage_times: dict[str, list[float]] = defaultdict(list)

    print(f"Running {args.frames} frames...\n")
    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, using blank frame")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        result = pipe.process_frame(frame)
        dt = time.perf_counter() - t0
        frame_times.append(dt)

        for stage, ms in pipe.timings.items():
            stage_times[stage].append(ms / 1000.0)  # convert ms → s

        if not args.no_display and result is not None:
            cv2.imshow("Benchmark", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if (i + 1) % 50 == 0:
            fps = 1.0 / np.mean(frame_times[-50:]) if frame_times else 0
            print(f"  Frame {i+1}/{args.frames}  {fps:.1f} FPS")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    pipe.shutdown()

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    total_ms = [t * 1000 for t in frame_times]
    print(f"Total frames processed : {len(frame_times)}")
    print(f"Mean frame time        : {np.mean(total_ms):6.1f} ms")
    print(f"Median frame time      : {np.median(total_ms):6.1f} ms")
    print(f"P95 frame time         : {np.percentile(total_ms, 95):6.1f} ms")
    print(f"Mean FPS               : {1000/np.mean(total_ms):6.1f}")
    print()
    print("Per-stage breakdown:")
    for stage, times in sorted(stage_times.items()):
        ms_list = [t * 1000 for t in times]
        print(f"  {stage:<30s}  mean={np.mean(ms_list):6.1f} ms  p95={np.percentile(ms_list,95):6.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faceswap-stream pipeline benchmark")
    parser.add_argument("--frames",     type=int, default=300, help="Number of frames to process")
    parser.add_argument("--source",     type=str, default=None, help="Source face image path")
    parser.add_argument("--device",     type=int, default=0,   help="Webcam device index")
    parser.add_argument("--no-display", action="store_true",   help="Disable preview window")
    run_benchmark(parser.parse_args())
