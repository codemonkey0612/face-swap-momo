"""Interactive beauty filter calibration.

Opens a live webcam preview with trackbars for all beauty parameters.
Saves final values to config/profiles/beauty_calibrated.yaml.

Usage:
    python scripts/calibrate_beauty.py [--device 0]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.beauty.skin_smoothing  import SkinSmoother
from src.beauty.eye_enhancement import EyeEnhancer
from src.beauty.color_correction import ColorCorrector


def run_calibration(device: int = 0) -> None:
    from insightface.app import FaceAnalysis
    from src.occlusion.face_parsing import FaceParser

    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    parser = FaceParser({"model_path": "models/face_parser.onnx"})
    smoother = SkinSmoother()
    enhancer = EyeEnhancer()
    corrector = ColorCorrector()

    win = "Beauty Calibration (q=quit, s=save)"
    cv2.namedWindow(win)
    cv2.createTrackbar("Smoothing %",   win, 60,  100, lambda v: None)
    cv2.createTrackbar("Eye enhance %", win, 30,  100, lambda v: None)
    cv2.createTrackbar("Sharpness %",   win, 20,  100, lambda v: None)
    cv2.createTrackbar("Contrast %",    win, 20,  100, lambda v: None)
    cv2.createTrackbar("Show split",    win, 1,   1,   lambda v: None)

    print("Controls:")
    print("  Trackbars: adjust each filter in real time")
    print("  s = save current settings to config/profiles/beauty_calibrated.yaml")
    print("  q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        smoothing  = cv2.getTrackbarPos("Smoothing %",   win) / 100.0
        eye_str    = cv2.getTrackbarPos("Eye enhance %", win) / 100.0
        sharpness  = cv2.getTrackbarPos("Sharpness %",   win) / 100.0
        contrast   = cv2.getTrackbarPos("Contrast %",    win) / 100.0
        show_split = cv2.getTrackbarPos("Show split",    win)

        result = frame.copy()
        faces = app.get(frame)
        if faces:
            f = faces[0]
            bbox = f.bbox
            lm   = f.kps
            face_mask = parser.get_face_region_mask(frame, bbox)
            result = smoother.smooth(result, face_mask, smoothing)
            result = enhancer.enhance(result, lm, eye_str)
            result = corrector.enhance_contrast(result, face_mask, contrast)
            # Sharpening
            if sharpness > 0:
                blur = cv2.GaussianBlur(result, (0, 0), 1.5)
                result = cv2.addWeighted(result, 1.0 + sharpness * 0.5, blur, -sharpness * 0.5, 0)

        if show_split:
            vis = np.hstack([frame, result])
        else:
            vis = result

        # Overlay current values
        overlay_txt = (
            f"Smooth={smoothing:.2f} Eye={eye_str:.2f} "
            f"Sharp={sharpness:.2f} Contrast={contrast:.2f}"
        )
        cv2.putText(vis, overlay_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            _save_profile(smoothing, eye_str, sharpness, contrast)
            print("[Saved] config/profiles/beauty_calibrated.yaml")

    cap.release()
    cv2.destroyAllWindows()


def _save_profile(smoothing, eye, sharpness, contrast):
    profile = {
        "beauty": {
            "enabled": True,
            "skin_smoothing_strength": round(float(smoothing), 2),
            "eye_brighten": round(float(eye), 2),
            "sharpness": round(float(sharpness), 2),
            "color_correction": True,
        }
    }
    os.makedirs("config/profiles", exist_ok=True)
    with open("config/profiles/beauty_calibrated.yaml", "w") as f:
        yaml.dump(profile, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive beauty filter calibration")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()
    run_calibration(args.device)
