# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time face swap pipeline using MediaPipe FaceMesh and OpenCV. Swaps faces in a webcam feed while preserving the original mouth region. Supports two modes: swapping two detected webcam faces, or overlaying a source image face onto a webcam face. Target deployment: Mac M4 via OBS virtual camera.

## Commands

```bash
pip install -r requirements.txt

# Two-face webcam swap (default)
python main.py

# Swap a source image face onto your webcam face
python main.py --source path/to/face.jpg

# Fast mode (alpha blending instead of seamlessClone)
python main.py --no-seamless

# Multiband Laplacian blending (good quality, faster than seamless)
python main.py --blend multiband

# Debug overlay (landmarks + triangles), toggle with 'd' key at runtime
python main.py --debug
```

## Architecture

**Pipeline flow:** `main.py` webcam loop → `landmarks.py` extracts 468 MediaPipe landmarks → `face_swap.py` performs Delaunay triangulation + affine warping + mouth-excluded blending → display.

- **landmarks.py** — MediaPipe FaceMesh wrapper. Defines face oval, outer lip, and inner lip index constants. `extract_landmarks()` returns list of `(468, 2)` int32 arrays. `get_lip_polygon()` returns an expanded lip contour used to carve the mouth hole in the blend mask.
- **face_swap.py** — Core algorithm. `compute_triangles()` does Delaunay triangulation on landmark points, returns index triples. `swap_face()` orchestrates the 6-step pipeline: triangulation → triangle warping → face mask → mouth carve-out → seamless clone (or alpha blend fallback). `swap_faces_bidirectional()` handles the two-face swap case.
- **utils.py** — `FPSCounter` (rolling average), `FreezeFrameManager` (holds last good frame when detection drops), `draw_debug_overlay`.
- **main.py** — Entry point with argparse. Webcam capture loop, mode selection (source image vs two-face), freeze-frame fallback, debug toggle.

## Key Design Decisions

- Mouth exclusion works by filling the lip polygon with 0 on the blend mask before `cv2.seamlessClone`. The lip polygon is scaled 1.12x outward from its centroid to prevent edge artifacts.
- `cv2.seamlessClone` is the quality path but costs ~15-30ms/frame. `--no-seamless` falls back to alpha blending for higher FPS. `--blend multiband` uses Laplacian pyramid blending (good quality, moderate speed).
- Face mask is eroded inward from the convex hull to avoid edge artifacts at forehead/jawline. Distance-transform feathering (adaptive to face size) produces smooth edges.
- Color correction uses Reinhard LAB transfer. Post-swap sharpening is applied inside the face region.
- Landmark smoother is velocity-adaptive: heavy smoothing when still (reduces jitter), light smoothing during fast movement (reduces lag).
- Triangulation uses all 468 MediaPipe landmarks (not just face oval) for dense coverage. Points outside the frame bounds are skipped.
- `FreezeFrameManager` returns the last swapped frame for up to 90 frames of missed detection, preventing flicker during brief tracking loss.

