"""Face Transform Studio — Gradio web UI.

A user-friendly browser-based interface for the faceswap-stream pipeline.

Run:
    python app_gui.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Inject CUDA DLLs before any ONNX imports (Windows)
# ---------------------------------------------------------------------------
def _inject_cuda_dll_dirs() -> None:
    for _root in glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12*"):
        _d = os.path.join(_root, "bin")
        if os.path.isdir(_d):
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

_inject_cuda_dll_dirs()

# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------
_lock = threading.Lock()

_state: dict = {
    "running":         False,
    "pipeline":        None,
    "cap":             None,
    "latest_frame":    None,   # numpy RGB array — latest processed frame
    "status":          "Ready",
    "fps":             0.0,
    "face_detected":   False,
    # Beauty knobs (written by sliders, read by pipeline thread)
    "skin_smooth":     0.6,
    "eye_brightness":  0.3,
    "enhance_on":      False,
    "beauty_on":       True,
    "obs_on":          False,
}

# ---------------------------------------------------------------------------
# Source face helpers
# ---------------------------------------------------------------------------

def _list_source_faces() -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp",
            "*.PNG", "*.JPG", "*.JPEG")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join("source_faces", ext)))
    return sorted(set(paths))


def _placeholder_frame(msg: str = "Press  ▶ Start  to begin") -> np.ndarray:
    """Return a dark placeholder image shown when the pipeline is idle."""
    img = np.full((480, 854, 3), 22, dtype=np.uint8)
    # Gradient overlay
    for y in range(img.shape[0]):
        alpha = y / img.shape[0]
        img[y, :] = (int(22 + 8 * alpha),) * 3
    # Centered text
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cx = (img.shape[1] - tw) // 2
    cy = img.shape[0] // 2 + th // 2
    cv2.putText(img, msg, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 140, 180), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Background pipeline thread
# ---------------------------------------------------------------------------

def _pipeline_thread(source_path: str, camera_idx: int) -> None:
    """Capture + process frames in a background thread. Writes to _state."""
    try:
        from src.pipeline import load_config, UltimateSwapPipeline

        _state["status"] = "Loading models — please wait..."
        cfg = load_config("config/default.yaml")
        cfg.source_image_path = source_path
        cfg.obs_virtual_camera = _state["obs_on"]
        cfg.enable_codeformer  = _state["enhance_on"]
        cfg.enable_esrgan      = _state["enhance_on"]

        pipe = UltimateSwapPipeline(source_image_path=source_path, config=cfg)

        vcam = None
        if _state["obs_on"]:
            try:
                from src.output.virtual_camera import VirtualCameraOutput
                vcam = VirtualCameraOutput(cfg.output_width, cfg.output_height)
            except Exception as e:
                print(f"[GUI] Virtual camera unavailable: {e}")

        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        with _lock:
            _state["pipeline"] = pipe
            _state["cap"]      = cap
            _state["status"]   = "Running"

        fps_window: list[float] = []

        while _state["running"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            t0     = time.monotonic()
            result = pipe.process_frame(frame)
            elapsed = time.monotonic() - t0

            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)
            avg_t = sum(fps_window) / len(fps_window)

            diag = pipe.get_diagnostics()

            with _lock:
                if result is not None:
                    _state["latest_frame"] = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                _state["fps"]          = 1.0 / avg_t if avg_t > 0 else 0.0
                _state["face_detected"] = diag.get("detection_confidence", 0.0) > 0.5

            if vcam and result is not None:
                try:
                    vcam.send(result)
                except Exception:
                    pass

        # ---- Shutdown ----
        cap.release()
        pipe.shutdown()
        if vcam:
            try:
                vcam.close()
            except Exception:
                pass

        with _lock:
            _state["pipeline"]     = None
            _state["cap"]          = None
            _state["latest_frame"] = None
            _state["status"]       = "Stopped"
            _state["fps"]          = 0.0

    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _lock:
            _state["status"]  = f"Error: {exc}"
            _state["running"] = False


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def _start(selected_face: str | None, camera_idx: int,
           obs_on: bool, enhance_on: bool,
           skin_smooth: float, eye_brightness: float) -> tuple[str, np.ndarray]:
    """Start the pipeline. Called when user clicks ▶ Start."""
    if _state["running"]:
        return _status_line(), _get_frame()

    sources = _list_source_faces()
    if not sources:
        return "No source faces found — add images to source_faces/", _placeholder_frame()

    # Pick the source face
    if selected_face and os.path.exists(selected_face):
        src = selected_face
    else:
        src = sources[0]

    with _lock:
        _state["running"]      = True
        _state["obs_on"]       = obs_on
        _state["enhance_on"]   = enhance_on
        _state["skin_smooth"]  = skin_smooth
        _state["eye_brightness"] = eye_brightness
        _state["status"]       = "Starting..."

    t = threading.Thread(target=_pipeline_thread,
                         args=(src, int(camera_idx)),
                         daemon=True)
    t.start()
    return _status_line(), _placeholder_frame("Loading models...")


def _stop() -> tuple[str, np.ndarray]:
    """Stop the pipeline. Called when user clicks ⏹ Stop."""
    _state["running"] = False
    return "Stopping...", _placeholder_frame("Stopped")


def _screenshot() -> str:
    """Save the current frame. Called by the 📸 Screenshot button."""
    with _lock:
        frame = _state.get("latest_frame")
    if frame is None:
        return "Nothing to save — start the pipeline first."
    path = f"screenshot_{int(time.time())}.png"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    return f"Saved: {path}"


def _get_frame() -> np.ndarray:
    """Return the latest processed frame (or a placeholder)."""
    with _lock:
        frame = _state.get("latest_frame")
    return frame if frame is not None else _placeholder_frame()


def _status_line() -> str:
    status = _state.get("status", "Ready")
    fps    = _state.get("fps", 0.0)
    if _state.get("running") and fps > 0:
        detected = "Face detected" if _state.get("face_detected") else "Searching..."
        return f"{status}  |  {fps:.1f} FPS  |  {detected}"
    return status


def _update_skin(val: float) -> None:
    _state["skin_smooth"] = val
    pipe = _state.get("pipeline")
    if pipe:
        try:
            pipe.config.beauty_smoothing_strength = val
        except Exception:
            pass


def _update_eye(val: float) -> None:
    _state["eye_brightness"] = val
    pipe = _state.get("pipeline")
    if pipe:
        try:
            pipe.config.eye_brighten_strength = val
        except Exception:
            pass


def _toggle_beauty(enabled: bool) -> None:
    _state["beauty_on"] = enabled
    pipe = _state.get("pipeline")
    if pipe:
        try:
            pipe.config.enable_beauty = enabled
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr

    sources = _list_source_faces()
    source_thumbnails = []
    for p in sources:
        try:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.cvtColor(
                    cv2.resize(img, (160, 160)), cv2.COLOR_BGR2RGB)
                source_thumbnails.append((img, os.path.basename(p)))
        except Exception:
            pass

    # ---- CSS ----
    css = """
    #title   { text-align: center; padding: 18px 0 4px; }
    #subtitle{ text-align: center; color: #9e8fab; margin-bottom: 12px; font-size: 14px; }
    #status  { font-size: 14px; padding: 6px 12px; border-radius: 8px;
               background: #1e1a22; border: 1px solid #3d3545 !important; color: #ddd; }
    #preview { border-radius: 12px; overflow: hidden; }
    .section-label { font-size: 13px; font-weight: 600; color: #c8a8d0;
                     margin-bottom: 6px; }
    footer { display: none !important; }
    """

    theme = gr.themes.Soft(
        primary_hue   = gr.themes.colors.pink,
        secondary_hue = gr.themes.colors.purple,
        neutral_hue   = gr.themes.colors.slate,
        font          = [gr.themes.GoogleFont("Inter"), "sans-serif"],
    ).set(
        body_background_fill        = "#16131b",
        body_background_fill_dark   = "#16131b",
        block_background_fill       = "#201c28",
        block_background_fill_dark  = "#201c28",
        block_border_color          = "#3a3245",
        block_border_color_dark     = "#3a3245",
        input_background_fill       = "#2a2535",
        input_background_fill_dark  = "#2a2535",
        button_primary_background_fill        = "linear-gradient(90deg,#d63384,#9b59b6)",
        button_primary_background_fill_hover  = "linear-gradient(90deg,#e0559a,#a96fc4)",
        button_secondary_background_fill      = "#2e2838",
        button_secondary_background_fill_hover= "#3d3550",
    )

    with gr.Blocks(title="Face Transform Studio", theme=theme, css=css) as demo:

        # ---- Header ----
        gr.HTML("<h1 id='title' style='color:#e87aba'>✨ Face Transform Studio</h1>")
        gr.HTML("<p id='subtitle'>Real-time AI face transformation for live streaming</p>")

        # ---- Hidden state for selected source path ----
        selected_source = gr.State(value=sources[0] if sources else "")

        with gr.Row(equal_height=False):

            # ──────────────── LEFT: source face picker ────────────────
            with gr.Column(scale=1, min_width=200):
                gr.HTML("<div class='section-label'>Choose Your Look</div>")

                if source_thumbnails:
                    face_gallery = gr.Gallery(
                        value      = [img for img, _ in source_thumbnails],
                        label      = "Source faces",
                        show_label = False,
                        columns    = 2,
                        height     = 320,
                        object_fit = "cover",
                        allow_preview = False,
                    )

                    def _on_face_select(evt: gr.SelectData) -> str:
                        idx = evt.index
                        return sources[idx] if idx < len(sources) else (sources[0] if sources else "")

                    face_gallery.select(_on_face_select, outputs=selected_source)
                else:
                    gr.Markdown(
                        "_No source faces found._  \nAdd `.png` / `.jpg` images to the "
                        "`source_faces/` folder and restart.",
                        elem_id="no-faces"
                    )

                gr.HTML("<div style='margin-top:16px'></div>")

                # ---- Camera index ----
                gr.HTML("<div class='section-label'>Camera</div>")
                camera_input = gr.Number(
                    value   = 0,
                    label   = "Camera index",
                    minimum = 0,
                    maximum = 9,
                    step    = 1,
                    info    = "Usually 0 for built-in, 1+ for external",
                )

            # ──────────────── CENTER: live preview ─────────────────────
            with gr.Column(scale=3, min_width=400):

                status_box = gr.Textbox(
                    value       = "Ready — press Start to begin",
                    show_label  = False,
                    interactive = False,
                    elem_id     = "status",
                )

                preview_img = gr.Image(
                    value       = _placeholder_frame(),
                    label       = "Live Preview",
                    show_label  = False,
                    elem_id     = "preview",
                    height      = 480,
                    interactive = False,
                )

                with gr.Row():
                    start_btn      = gr.Button("▶  Start",      variant="primary",   size="lg")
                    stop_btn       = gr.Button("⏹  Stop",       variant="secondary", size="lg")
                    screenshot_btn = gr.Button("📸  Screenshot", variant="secondary", size="lg")

                screenshot_out = gr.Textbox(
                    value       = "",
                    show_label  = False,
                    interactive = False,
                    visible     = True,
                    max_lines   = 1,
                )

            # ──────────────── RIGHT: controls ──────────────────────────
            with gr.Column(scale=1, min_width=220):

                gr.HTML("<div class='section-label'>Beauty Controls</div>")

                beauty_toggle = gr.Checkbox(
                    label = "Beauty Filters",
                    value = True,
                    info  = "Toggle all beauty enhancements on/off",
                )

                skin_slider = gr.Slider(
                    minimum = 0.0, maximum = 1.0, value = 0.6, step = 0.05,
                    label   = "Skin Smoothing",
                    info    = "Soften skin texture",
                )

                eye_slider = gr.Slider(
                    minimum = 0.0, maximum = 1.0, value = 0.3, step = 0.05,
                    label   = "Eye Brightness",
                    info    = "Brighten and enhance eyes",
                )

                gr.HTML("<div style='margin-top:16px'></div>")
                gr.HTML("<div class='section-label'>Quality</div>")

                enhance_toggle = gr.Checkbox(
                    label = "AI Enhancement (CodeFormer)",
                    value = False,
                    info  = "Higher quality — slightly slower",
                )

                gr.HTML("<div style='margin-top:16px'></div>")
                gr.HTML("<div class='section-label'>Output</div>")

                obs_toggle = gr.Checkbox(
                    label = "Stream to OBS Virtual Camera",
                    value = False,
                    info  = "Send output to OBS for broadcasting",
                )

                gr.HTML("<div style='margin-top:24px'></div>")
                gr.HTML("<div class='section-label'>Keyboard Shortcuts</div>")
                gr.Markdown(
                    "In the preview window (not browser):  \n"
                    "**Q** quit &nbsp;·&nbsp; **S** screenshot  \n"
                    "**B** beauty on/off &nbsp;·&nbsp; **D** debug"
                )

        # ---- Wire events ----

        start_btn.click(
            fn      = _start,
            inputs  = [selected_source, camera_input,
                       obs_toggle, enhance_toggle,
                       skin_slider, eye_slider],
            outputs = [status_box, preview_img],
        )

        stop_btn.click(
            fn      = _stop,
            outputs = [status_box, preview_img],
        )

        screenshot_btn.click(
            fn      = _screenshot,
            outputs = [screenshot_out],
        )

        skin_slider.release(_update_skin,   inputs=[skin_slider])
        eye_slider.release (_update_eye,    inputs=[eye_slider])
        beauty_toggle.change(_toggle_beauty, inputs=[beauty_toggle])

        # ---- Live polling: refresh frame + status every ~100 ms ----
        demo.load(
            fn      = _get_frame,
            outputs = preview_img,
            every   = 0.1,
        )
        demo.load(
            fn      = _status_line,
            outputs = status_box,
            every   = 1.0,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        print("[GUI] Gradio not installed. Run:  pip install gradio>=4.0")
        sys.exit(1)

    print("=" * 56)
    print("  ✨ Face Transform Studio")
    print("=" * 56)
    print("  Opening browser at  http://localhost:7860")
    print("  Press Ctrl+C to quit.")
    print("=" * 56)

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        inbrowser   = True,
        share       = False,
        show_error  = True,
    )


if __name__ == "__main__":
    main()
