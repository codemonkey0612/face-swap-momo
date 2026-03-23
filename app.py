"""faceswap-stream — application entry point.

Run:
    python app.py --source source_faces/ai.jpg
    python app.py --source-dir source_faces/
    python app.py --source source_faces/ai.jpg --obs
    python app.py --source source_faces/ai.jpg --enhance

Keyboard shortcuts (preview window):
    q   Quit
    d   Toggle debug overlay
    b   Toggle beauty filters
    o   Toggle occlusion overlay
    s   Screenshot
"""

import argparse
import glob
import os
import sys


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time face swap pipeline")

    # Source face
    p.add_argument("--source", type=str, default=None,
                   help="Single source face image path")
    p.add_argument("--source-dir", type=str, default="source_faces",
                   help="Directory of source face images (embeddings averaged)")

    # Config
    p.add_argument("--config", type=str, default="config/default.yaml",
                   help="YAML config file")
    p.add_argument("--profile", type=str, default=None,
                   help="Profile name from config/profiles/ (e.g. max_quality, balanced)")

    # Hardware
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--width", type=int, default=None, help="Override capture width")
    p.add_argument("--height", type=int, default=None, help="Override capture height")
    p.add_argument("--model", type=str, default=None, help="Override swap model path")

    # Output
    p.add_argument("--obs", action="store_true", help="Stream to OBS virtual camera")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window")
    p.add_argument("--output-width", type=int, default=None)
    p.add_argument("--output-height", type=int, default=None)

    # Feature flags
    p.add_argument("--no-poisson", action="store_true", default=False,
                   help="Use fast alpha blend instead of Poisson seamless clone")
    p.add_argument("--no-lab-color", action="store_true")
    p.add_argument("--no-mouth-exclusion", action="store_true")
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("--enhance", action="store_true", help="Enable GFPGAN/CodeFormer")
    p.add_argument("--erosion", type=int, default=3)
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Source resolution helpers
# ---------------------------------------------------------------------------

def _find_source_image(args) -> str | None:
    if args.source and os.path.exists(args.source):
        return args.source
    source_dir = args.source_dir or "source_faces"
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
            "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(source_dir, ext)))
    return sorted(set(paths))[0] if paths else None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _health_check(config) -> bool:
    ok = True

    # GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu = any("CUDA" in p for p in providers)
        print(f"[Health] GPU: {'YES (CUDA)' if gpu else 'NO — CPU only'}")
        if not gpu:
            print("[Health]  → Install onnxruntime-gpu and matching CUDA toolkit")
    except ImportError:
        print("[Health] CRITICAL: onnxruntime not installed")
        ok = False

    # Swap model
    model = config.swap_model
    if os.path.exists(model):
        size_mb = os.path.getsize(model) / 1024 / 1024
        print(f"[Health] Model: {model} ({size_mb:.0f} MB) ✓")
    else:
        print(f"[Health] MISSING model: {model}")
        ok = False

    # Face parser
    for name in ("face_parser.onnx", "bisenet_resnet_34.onnx"):
        path = os.path.join("models", name)
        if os.path.exists(path):
            print(f"[Health] Parser: {name} ✓")
            break
    else:
        print("[Health] face_parser.onnx not found — run: python models/download_models.py")

    return ok


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run(args, cap):
    from src.pipeline import load_config, UltimateSwapPipeline
    from src.output.preview_window import PreviewWindow
    from src.utils.metrics import FPSCounter
    import cv2, time

    cfg = load_config(args.config)

    # Apply profile overrides
    if args.profile:
        profile_path = f"config/profiles/{args.profile}.yaml"
        if os.path.exists(profile_path):
            from pipeline import load_config_yaml
            profile_cfg = load_config_yaml(profile_path)
            for field in vars(profile_cfg):
                val = getattr(profile_cfg, field)
                if val is not None:
                    setattr(cfg, field, val)
            print(f"[App] Profile: {args.profile}")

    # CLI overrides
    if args.model:
        cfg.swap_model = args.model
    if args.output_width:
        cfg.output_width = args.output_width
    if args.output_height:
        cfg.output_height = args.output_height
    cfg.obs_virtual_camera = args.obs
    if args.no_lab_color:
        cfg.enable_lab_color = False
    if args.no_mouth_exclusion:
        cfg.enable_mouth_exclusion = False
    if args.no_occlusion:
        cfg.enable_occlusion = False
    if args.no_poisson:
        cfg.enable_poisson = False
    cfg.enable_codeformer = args.enhance
    cfg.enable_esrgan = args.enhance
    cfg.mask_erosion_px = args.erosion

    # Source
    source = _find_source_image(args)
    if not source:
        print("[App] ERROR: No source face found. Use --source or put images in source_faces/")
        sys.exit(1)
    cfg.source_image_path = source

    # Health check
    if not _health_check(cfg):
        print("[App] Health check failed — aborting")
        sys.exit(1)

    pipe = UltimateSwapPipeline(source_image_path=source, config=cfg)

    vcam = None
    if args.obs:
        try:
            from src.output.virtual_camera import VirtualCameraOutput
            vcam = VirtualCameraOutput(cfg.output_width, cfg.output_height)
        except Exception as e:
            print(f"[App] Virtual camera: {e}")

    preview = PreviewWindow(title="Face Swap", show=not args.no_preview)
    fps_counter = FPSCounter()

    print("[App] Running — q=quit  d=debug  s=screenshot")
    stats_t = time.monotonic()

    _PROC_W, _PROC_H = min(cfg.output_width, 1280), min(cfg.output_height, 720)

    try:
        while not preview.quit:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            if frame.shape[1] != _PROC_W or frame.shape[0] != _PROC_H:
                proc_frame = cv2.resize(frame, (_PROC_W, _PROC_H))
            else:
                proc_frame = frame

            fps_counter.tick()
            result = pipe.process_frame(proc_frame)

            if result is not None and (result.shape[1] != cfg.output_width or
                                       result.shape[0] != cfg.output_height):
                result = cv2.resize(result, (cfg.output_width, cfg.output_height))

            if vcam:
                vcam.send(result)

            diag = pipe.get_diagnostics()
            diag["fps"] = fps_counter.fps()
            if args.debug:
                preview.debug = True
            preview.show(result, diag)

            now = time.monotonic()
            if now - stats_t >= 5.0:
                lat = diag.get("stage_latencies", {})
                total = sum(lat.values())
                top = sorted(lat.items(), key=lambda x: x[1], reverse=True)[:4]
                top_str = "  ".join(f"{k}={v:.0f}ms" for k, v in top)
                print(f"[Stats] FPS={diag['fps']:.1f}  total={total:.1f}ms  "
                      f"failsafe={diag.get('failsafe_state','')}  |  {top_str}")
                stats_t = now

    except KeyboardInterrupt:
        print("\n[App] Interrupted")
    finally:
        pipe.shutdown()
        preview.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  faceswap-stream")
    print("=" * 60)

    width = args.width or 1920
    height = args.height or 1080

    import cv2
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("[App] ERROR: Cannot open webcam")
        sys.exit(1)

    try:
        _run(args, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
