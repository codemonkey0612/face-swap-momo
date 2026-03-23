"""Layer 2: Hand detection via MediaPipe Hands.

Hands are the most common occluder during streaming (touching face,
resting chin, holding objects near face). MediaPipe detects hands
geometrically (21 landmarks → convex hull), not by skin colour, so
it correctly flags hands even when skin tone matches the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HandRegion:
    """A detected hand with its landmarks and bounding box."""
    landmarks: np.ndarray   # (21, 2) pixel coords
    bbox: np.ndarray        # [x1, y1, x2, y2]
    handedness: str         # "Left" or "Right"
    confidence: float


class HandDetector:
    """MediaPipe Hands wrapper — Layer 2 of the occlusion system.

    Args:
        config: dict with optional keys:
            enabled (bool)
            min_confidence (float): MediaPipe detection confidence (0-1)
            min_tracking_confidence (float)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._min_det = cfg.get("min_confidence", 0.5)
        self._min_track = cfg.get("min_tracking_confidence", 0.5)
        self._hands = None
        self._fallback_mp = None  # selfie segmentation backup

        if self._enabled:
            self._load_mediapipe()

    def _load_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
            )
            print("[HandDetector] MediaPipe Hands loaded")
        except Exception as e:
            print(f"[HandDetector] MediaPipe not available: {e} — layer disabled")
            self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_hands(self, frame: np.ndarray) -> list[HandRegion]:
        """Detect all hands in the frame.

        Args:
            frame: Full BGR frame.

        Returns:
            List of HandRegion objects (may be empty).
        """
        if not self._enabled or self._hands is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        H, W = frame.shape[:2]
        regions: list[HandRegion] = []
        handedness_list = results.multi_handedness or []

        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            pts = np.array(
                [[lm.x * W, lm.y * H] for lm in hand_lm.landmark],
                dtype=np.float32,
            )
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)

            hand_label = "Unknown"
            conf = 0.0
            if i < len(handedness_list):
                cl = handedness_list[i].classification[0]
                hand_label = cl.label
                conf = cl.score

            regions.append(HandRegion(
                landmarks=pts,
                bbox=np.array([x1, y1, x2, y2]),
                handedness=hand_label,
                confidence=conf,
            ))

        return regions

    def get_hand_mask(
        self,
        frame: np.ndarray,
        face_bbox: np.ndarray,
        face_parse_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return float32 mask (H, W) where 1.0 = hand overlapping face area.

        Args:
            frame:           Full BGR frame.
            face_bbox:       [x1, y1, x2, y2] of the face.
            face_parse_mask: Optional BiSeNet skin mask — used to filter
                             detections where the hand hull is mostly face skin
                             (MediaPipe occasionally misidentifies the face).

        Returns:
            float32 mask in [0, 1]. 1.0 = hand pixel over face.
        """
        H, W = frame.shape[:2]
        hand_mask = np.zeros((H, W), dtype=np.float32)

        hands = self.detect_hands(frame)
        if not hands:
            return self.get_hand_mask_fallback(frame, face_bbox)

        # Expanded face region (20% padding)
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        pw = int((x2 - x1) * 0.2)
        ph = int((y2 - y1) * 0.2)
        fx1, fy1 = max(0, x1 - pw), max(0, y1 - ph)
        fx2, fy2 = min(W, x2 + pw), min(H, y2 + ph)

        face_region = np.zeros((H, W), dtype=np.uint8)
        face_region[fy1:fy2, fx1:fx2] = 255

        for hand in hands:
            # Convex hull of the 21 hand landmarks
            pts_int = hand.landmarks.astype(np.int32)
            hull = cv2.convexHull(pts_int)
            hand_hull = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(hand_hull, hull, 255)

            # Optional: filter out if hand hull is mostly face-parsing skin
            if face_parse_mask is not None:
                hull_area = hand_hull.sum() / 255.0
                if hull_area > 0:
                    overlap = (hand_hull.astype(np.float32) / 255.0 * face_parse_mask).sum()
                    if overlap / hull_area > 0.80:
                        continue  # likely face detected as hand

            # Intersect with face region
            overlap_mask = cv2.bitwise_and(hand_hull, face_region)
            hand_mask = np.maximum(hand_mask, overlap_mask.astype(np.float32) / 255.0)

        return hand_mask

    def get_hand_mask_fallback(
        self, frame: np.ndarray, face_bbox: np.ndarray
    ) -> np.ndarray:
        """Skin-colour heuristic fallback when MediaPipe misses hands.

        Detects isolated skin-coloured blobs within the face bbox that
        are NOT contiguous with the main face skin — likely fingers/palms.

        Returns float32 mask (H, W).
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((H, W), dtype=np.float32)

        roi = frame[y1:y2, x1:x2]
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        # YCrCb skin range
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([235, 173, 127], dtype=np.uint8)
        skin = cv2.inRange(ycrcb, lower, upper)

        # Keep only the largest blob (the face) and flag the rest
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
        result_roi = np.zeros_like(skin)
        if n > 2:  # background + face + possible hand blobs
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = np.argmax(areas) + 1  # skip background
            for lbl in range(1, n):
                if lbl != largest:
                    result_roi[labels == lbl] = 255

        mask = np.zeros((H, W), dtype=np.float32)
        mask[y1:y2, x1:x2] = result_roi.astype(np.float32) / 255.0
        return mask

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

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
    detector = HandDetector({"min_confidence": 0.5})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = frame.copy()
        faces = app.get(frame)
        if faces:
            bbox = faces[0].bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue = face bbox
            hand_mask = detector.get_hand_mask(frame, bbox)
            # Green = hand mask
            green = np.zeros_like(frame)
            green[:, :, 1] = (hand_mask * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 0.7, green, 0.3, 0)
            # Red = overlap with face bbox
            overlap_mask = hand_mask.copy()
            overlap_mask[:y1, :] = 0; overlap_mask[y2:, :] = 0
            overlap_mask[:, :x1] = 0; overlap_mask[:, x2:] = 0
            red = np.zeros_like(frame)
            red[:, :, 2] = (overlap_mask * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 1.0, red, 0.5, 0)
        cv2.imshow("HandDetector Layer 2 (b=face, g=hand, r=overlap)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
