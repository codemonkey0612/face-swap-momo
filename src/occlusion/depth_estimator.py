"""Layer 5: Monocular depth estimation for z-order occlusion reasoning.

If something is CLOSER to the camera than the face, it must be in front.
MiDaS v3.1 Small (DPT_SwinV2_T_256) or Depth Anything V2 (vits) provide
relative depth maps at ~20-30ms on an RTX 5080.

Relative depth from MiDaS: higher value = closer to camera (inverse depth).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


class DepthEstimator:
    """MiDaS monocular depth — Layer 5 of the occlusion system.

    Args:
        config: dict with optional keys:
            enabled (bool)
            model (str): 'midas_small' | 'depth_anything_v2_small'
            margin (float): fraction of depth range to use as threshold (0.05-0.15)
    """

    _INPUT_SIZE = 256  # for midas_small / DPT_SwinV2_T_256

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._model_name = cfg.get("model", "midas_small")
        self._margin = float(cfg.get("margin", 0.10))
        self._model = None
        self._transform = None
        self._device = None
        self._prev_depth: Optional[np.ndarray] = None

        if self._enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._model_name == "midas_small":
                self._model = torch.hub.load(
                    "intel-isl/MiDaS", "DPT_SwinV2_T_256",
                    pretrained=True, trust_repo=True,
                )
                transforms = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True,
                )
                self._transform = transforms.dpt_transform
            else:
                # Fallback to lightweight MiDaS
                self._model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small",
                    pretrained=True, trust_repo=True,
                )
                transforms = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True,
                )
                self._transform = transforms.small_transform

            self._model.to(self._device).eval()
            print(f"[DepthEstimator] Loaded {self._model_name} on {self._device}")
        except Exception as e:
            print(f"[DepthEstimator] Load failed: {e} — layer disabled")
            self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Return a float32 depth map (H, W) in [0, 1].

        0 = closest to camera, 1 = farthest (after inversion from MiDaS output).

        Args:
            frame: Full BGR frame.

        Returns:
            float32 depth map at original frame resolution.
        """
        if not self._enabled or self._model is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            inp = self._transform(rgb).to(self._device)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            depth = pred.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[DepthEstimator] Inference error: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)

        # MiDaS outputs INVERSE depth (larger = closer).
        # Invert so that 0 = closest, 1 = farthest.
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = 1.0 - (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        # Bilateral filter to reduce edge noise
        depth_u8 = (depth * 255).astype(np.uint8)
        depth_smooth = cv2.bilateralFilter(depth_u8, 9, 75, 75)
        return depth_smooth.astype(np.float32) / 255.0

    def get_foreground_mask(
        self,
        frame: np.ndarray,
        face_bbox: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return float32 mask (H, W) of pixels closer than the face plane.

        Args:
            frame:     Full BGR frame.
            face_bbox: [x1, y1, x2, y2].
            depth_map: Pre-computed depth map (avoids re-computing if you
                       already called estimate_depth this frame).

        Returns:
            float32 mask (H, W): 1.0 = something in front of the face.
        """
        H, W = frame.shape[:2]
        if not self._enabled:
            return np.zeros((H, W), dtype=np.float32)

        if depth_map is None:
            depth_map = self.estimate_depth(frame)

        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((H, W), dtype=np.float32)

        face_depth_roi = depth_map[y1:y2, x1:x2]
        face_depth_median = float(np.median(face_depth_roi))

        # Pixels significantly closer (lower value) than the face plane
        threshold = face_depth_median - self._margin
        closer_mask = (depth_map < threshold).astype(np.float32)

        # Restrict to face bbox region
        result = np.zeros((H, W), dtype=np.float32)
        result[y1:y2, x1:x2] = closer_mask[y1:y2, x1:x2]
        return result

    def smooth_depth(self, depth_map: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Temporal EMA smoothing to reduce flickering in the depth map."""
        if self._prev_depth is None or self._prev_depth.shape != depth_map.shape:
            self._prev_depth = depth_map.copy()
            return depth_map
        smoothed = alpha * depth_map + (1.0 - alpha) * self._prev_depth
        self._prev_depth = smoothed.copy()
        return smoothed

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from insightface.app import FaceAnalysis

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    estimator = DepthEstimator({"model": "midas_small", "margin": 0.1})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        depth = estimator.estimate_depth(frame)
        depth_vis = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        vis = frame.copy()
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            fg_mask = estimator.get_foreground_mask(frame, bbox, depth_map=depth)
            red = np.zeros_like(frame)
            red[:, :, 2] = (fg_mask * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 0.6, red, 0.4, 0)

        combined = np.hstack([vis, depth_vis])
        cv2.imshow("DepthEstimator Layer 5 | original+fg_mask | depth", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
