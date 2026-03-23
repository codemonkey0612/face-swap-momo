# Face-Swap Streaming Pipeline — Project Guide for Claude Code

## Overview

A real-time face-swapping pipeline optimized for **maximum video quality** on an RTX 5080 PC. The system captures webcam input, replaces the performer's face with a pre-generated AI face, applies beauty filters, handles occlusion, and outputs to a virtual camera for OBS.

---

## Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Language | Python 3.11+ | Best ML/CV ecosystem |
| Face Detection | RetinaFace (insightface) | High accuracy landmarks |
| Face Swap | InSwapper (insightface) | Best open-source quality |
| Occlusion Seg | BiSeNet / face-parsing | Real-time capable, accurate |
| Beauty Filters | OpenCV + custom shaders | Full control, GPU-accelerated |
| Temporal Smooth | Custom EMA blending | Reduces flicker |
| Virtual Camera | pyvirtualcam (OBS Virtual Cam) | OBS-compatible output on Windows |
| GPU Acceleration | ONNX Runtime (CUDA/TensorRT) | Maximizes RTX 5080 utilization |
| UI | Gradio or PyQt6 | Quick config interface |

---

## Project Structure

```
faceswap-stream/
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.bat                       # One-click environment setup (Windows)
│
├── config/
│   ├── default.yaml                # All tunable parameters
│   └── profiles/
│       ├── max_quality.yaml        # Quality-first preset
│       └── balanced.yaml           # Quality-latency balance
│
├── models/                         # Downloaded model weights (gitignored)
│   ├── .gitkeep
│   └── download_models.py          # Script to fetch all required models
│
├── source_faces/                   # Pre-generated AI face images
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   │
│   ├── pipeline.py                 # Main pipeline orchestrator
│   │
│   ├── capture/
│   │   ├── __init__.py
│   │   └── webcam.py               # Webcam capture with threading
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── face_detector.py        # RetinaFace wrapper
│   │
│   ├── swap/
│   │   ├── __init__.py
│   │   ├── face_swapper.py         # InSwapper wrapper
│   │   └── identity_manager.py     # Source face embedding cache
│   │
│   ├── occlusion/
│   │   ├── __init__.py
│   │   ├── face_parsing.py         # BiSeNet: identifies face regions
│   │   ├── hand_detector.py        # MediaPipe Hands: detects hands over face
│   │   ├── body_segmenter.py       # Selfie/body segmentation for arms, objects
│   │   ├── error_detector.py       # HEAR-Net style: finds swap anomalies
│   │   ├── depth_estimator.py      # Monocular depth: z-order reasoning
│   │   ├── occlusion_fuser.py      # Combines all signals into final mask
│   │   └── mask_refiner.py         # Mask smoothing & edge blending
│   │
│   ├── beauty/
│   │   ├── __init__.py
│   │   ├── skin_smoothing.py       # Bilateral + guided filter
│   │   ├── color_correction.py     # Skin tone matching & grading
│   │   ├── eye_enhancement.py      # Eye brightening & sharpening
│   │   └── filter_chain.py         # Composable beauty pipeline
│   │
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── stabilizer.py           # Landmark temporal smoothing
│   │   └── frame_blender.py        # EMA frame blending
│   │
│   ├── compositing/
│   │   ├── __init__.py
│   │   ├── blender.py              # Poisson / alpha blending
│   │   └── color_transfer.py       # Source-to-target color match
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── virtual_camera.py       # pyvirtualcam output
│   │   └── preview_window.py       # Local preview with stats
│   │
│   └── utils/
│       ├── __init__.py
│       ├── gpu_utils.py            # CUDA device management
│       ├── metrics.py              # FPS / latency tracking
│       └── image_utils.py          # Common image operations
│
├── tests/
│   ├── test_detection.py
│   ├── test_swap.py
│   ├── test_occlusion.py
│   ├── test_beauty.py
│   ├── test_pipeline_integration.py
│   └── fixtures/
│       └── sample_frames/          # Test images
│
├── scripts/
│   ├── benchmark.py                # Full pipeline benchmark
│   ├── calibrate_beauty.py         # Interactive beauty tuning
│   └── generate_source_face.py     # Helper to prep AI face
│
└── app.py                          # Entry point
```

---

## Claude Code Prompts — Step-by-Step

> **How to use:** Run each prompt sequentially in Claude Code.
> Wait for each step to complete and verify before proceeding.
> Adjust paths if your working directory differs.

---

### PHASE 0: Project Scaffolding

#### Prompt 0.1 — Initialize project

```
Create a new Python project called "faceswap-stream" with the following:

1. Create the full directory structure as specified:
   - src/ with subpackages: capture, detection, swap, occlusion, beauty, temporal, compositing, output, utils
   - config/, models/, source_faces/, tests/, scripts/
   - All __init__.py files

2. Create pyproject.toml with:
   - Python >=3.11
   - Project name: faceswap-stream
   - Entry point: app:main

3. Create requirements.txt with these pinned dependencies:
   - insightface>=0.7.3
   - onnxruntime-gpu>=1.17.0
   - opencv-python>=4.9.0
   - opencv-contrib-python>=4.9.0
   - numpy>=1.26.0
   - pyvirtualcam>=0.10.0
   - PyYAML>=6.0
   - Pillow>=10.0
   - scipy>=1.12.0
   - gradio>=4.0.0
   - mediapipe>=0.10.9
   - torch>=2.2.0 (with CUDA — install via pytorch.org instructions)
   - torchvision>=0.17.0
   - timm>=0.9.12 (for MiDaS/Depth Anything)

4. Create setup.bat (Windows batch script) that:
   - Creates a conda environment named faceswap with Python 3.11
   - Installs requirements via pip
   - Installs Visual C++ Build Tools reminder (needed for some packages)
   - Checks that CUDA toolkit is installed and prints version
   - Checks that OBS is installed (needed for virtual camera)
   - Creates models/ directory
   - Prints success message with next steps

5. Create config/default.yaml with all pipeline parameters:
   - capture: device_id, width (1920), height (1080), fps (30)
   - detection: model_name (buffalo_l), det_size (640), confidence_threshold (0.5)
   - swap: model_path, source_face_path
   - occlusion:
       face_parsing: enabled (true), model_path
       hand_detection: enabled (true), min_confidence (0.5)
       body_segmentation: enabled (true)
       error_detection: enabled (true), sensitivity (0.5)
       depth_estimation: enabled (true), model (midas_small), margin (0.1)
       fusion:
         hand_weight (0.9), depth_weight (0.7), error_weight (0.6),
         body_weight (0.5), parse_weight (0.3),
         voting_threshold (2), temporal_alpha (0.8)
       mask_refinement:
         feather_radius (15), morph_close_kernel (5),
         morph_open_kernel (3), adaptive_feather (true)
   - beauty: enabled (true), skin_smoothing_strength (0.6), eye_brighten (0.3), color_correction (true), sharpness (0.2)
   - temporal: landmark_smoothing (0.7), frame_blend_alpha (0.85)
   - compositing: blend_mode (poisson), feather_radius (11), color_transfer (true)
   - output: virtual_cam_backend (obs), virtual_cam_device (OBS Virtual Camera), preview (true), target_fps (25)

6. Create config/profiles/max_quality.yaml that overrides:
   - detection det_size: 960
   - occlusion: all 5 layers enabled, depth model: depth_anything_v2_small
   - occlusion fusion temporal_alpha: 0.85
   - occlusion mask_refinement feather_radius: 21, adaptive_feather: true
   - temporal frame_blend_alpha: 0.9
   - compositing feather_radius: 15
   - output target_fps: 15

Do NOT write any implementation logic yet — only scaffolding, configs, and empty files with docstrings.
```

#### Prompt 0.2 — Model downloader

```
Create models/download_models.py that:

1. Downloads the insightface buffalo_l model pack (for RetinaFace detection + ArcFace recognition)
   using insightface.app.FaceAnalysis which auto-downloads to ~/.insightface/
   - On Windows this resolves to %USERPROFILE%\.insightface\

2. Downloads the inswapper_128.onnx model from the standard insightface model zoo
   - Save to models/inswapper_128.onnx
   - Verify file hash after download

3. Downloads the BiSeNet face-parsing model (79999_iter.pth from the
   zllrunning/face-parsing.PyTorch repo) for face region parsing
   - Save to models/face_parsing.pth

4. Downloads MiDaS v3.1 small (DPT_SwinV2_T_256) for depth estimation
   - Use torch.hub.load('intel-isl/MiDaS', 'DPT_SwinV2_T_256')
   - This auto-downloads to torch hub cache
   - Alternatively download ONNX version for faster inference

5. Verify MediaPipe is installed (hand detection + selfie segmentation
   models download automatically on first use, but verify import works)

6. Add a verify_models() function that checks all files exist and prints status

7. Add __main__ block so it can be run as: python models/download_models.py

Include proper error handling, progress bars (tqdm), and retry logic.
Print a clear summary at the end showing which models are ready.
```

---

### PHASE 1: Capture & Detection

#### Prompt 1.1 — Webcam capture

```
Implement src/capture/webcam.py:

Create a WebcamCapture class that:

1. Opens a webcam via cv2.VideoCapture with configurable:
   - device_id, width, height, fps
   - On Windows, use cv2.CAP_DSHOW (DirectShow) backend for best
     compatibility: cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
   - CAP_PROP_FOURCC set to MJPG for maximum quality
   - CAP_PROP_BUFFERSIZE set to 1 (always get latest frame)
   - CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT to requested resolution

2. Runs capture in a separate daemon thread to prevent frame drops

3. Provides get_frame() -> tuple[np.ndarray, float]:
   - Returns (BGR frame, timestamp)
   - Thread-safe with threading.Lock
   - Returns the most recent frame (not queued — drop old frames)

4. Provides properties: fps, resolution, is_open

5. Implements context manager (__enter__, __exit__) for clean resource management

6. Includes warmup: discards first 10 frames to let auto-exposure settle

Write a quick self-test in __main__ that opens the webcam, captures 100 frames,
prints actual FPS, and displays a preview window.
```

#### Prompt 1.2 — Face detection

```
Implement src/detection/face_detector.py:

Create a FaceDetector class wrapping insightface.app.FaceAnalysis:

1. __init__(self, config: dict):
   - Initialize FaceAnalysis with model_pack_name from config
   - Prepare with ctx_id=0 (GPU), det_size from config
   - Store confidence_threshold

2. detect(self, frame: np.ndarray) -> list[DetectedFace]:
   - Run analysis, filter by confidence threshold
   - Return list of DetectedFace dataclasses containing:
     - bbox: np.ndarray (x1, y1, x2, y2)
     - landmarks: np.ndarray (5x2 - eyes, nose, mouth corners)
     - embedding: np.ndarray (512-d ArcFace vector)
     - confidence: float
     - age: int
     - gender: str

3. detect_primary(self, frame) -> Optional[DetectedFace]:
   - Returns the single largest face (by bbox area)
   - This is the performer's face to swap

4. Create the DetectedFace dataclass in the same file

5. Add distance-based face tracking: if a face was detected in the previous
   frame, prefer the detection closest to the previous position (prevents
   jumping to a different face if multiple appear)

Include a __main__ self-test that opens webcam, detects faces, draws bounding
boxes and landmarks, and shows FPS overlay.
```

---

### PHASE 2: Face Swapping Core

#### Prompt 2.1 — Identity manager

```
Implement src/swap/identity_manager.py:

Create an IdentityManager class that:

1. Loads a source face image (the AI-generated face) from a file path

2. Uses FaceDetector to extract the face embedding and aligned face from
   the source image

3. Caches the embedding (np.ndarray, 512-d) for fast repeated use

4. Provides methods:
   - load_source(path: str) -> bool
   - get_embedding() -> np.ndarray
   - get_source_face() -> insightface Face object
   - similarity(detected_face: DetectedFace) -> float
     (cosine similarity between source and detected)

5. Supports loading multiple source faces and switching between them
   via set_active(index: int)

6. Validates that the source image contains exactly one clear face,
   raises descriptive errors otherwise

Add __main__ test that loads a source face and prints embedding stats.
```

#### Prompt 2.2 — Face swapper

```
Implement src/swap/face_swapper.py:

Create a FaceSwapper class that:

1. Loads the inswapper_128.onnx model via insightface.model_zoo.get_model()
   - Configure ONNX Runtime session with CUDAExecutionProvider
   - Set graph_optimization_level to ORT_ENABLE_ALL
   - Enable TensorRT if available (try/except fallback to CUDA)

2. swap(self, frame: np.ndarray, detected_face: DetectedFace,
        source_face: Face) -> np.ndarray:
   - Performs the face swap using the InSwapper model
   - Returns the full frame with the face replaced
   - The model handles alignment internally

3. swap_with_raw_output(self, frame, detected_face, source_face)
        -> tuple[np.ndarray, np.ndarray]:
   - Returns (swapped_frame, swapped_face_crop)
   - The crop is useful for downstream beauty filter application

4. Add performance logging: track inference time per frame

5. Handle edge cases:
   - Face too small (< 48px) -> skip swap, return original
   - Face at frame edge (partially cut off) -> pad, swap, crop back
   - No face detected -> return original frame unchanged

Include __main__ test: load source face, open webcam, swap in real-time,
display side-by-side original vs swapped.
```

---

### PHASE 3: Occlusion Handling (Multi-Layer System)

> **Why this is hard:** BiSeNet face parsing only labels face PARTS (skin, nose,
> eyes, lips). It has NO class for "hand", "cup", "microphone", or "phone".
> When your hand covers your mouth, BiSeNet may label those hand pixels as
> "skin" — they ARE skin, just not face skin. So the swap bleeds through.
>
> The solution is to combine MULTIPLE independent detection methods. Each one
> catches different types of occlusion. Their combined output creates a robust
> mask that correctly identifies "this pixel is NOT the face to swap."

#### Prompt 3.1 — Face parsing (Layer 1: What IS the face)

```
Implement src/occlusion/face_parsing.py:

Create a FaceParser class using BiSeNet for face parsing:

1. Load the BiSeNet face-parsing model (models/face_parsing.pth)
   - Use the architecture from zllrunning/face-parsing.PyTorch
   - You will need to include or reference the BiSeNet model definition
   - Load onto GPU with torch

2. parse(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
   - Crop the face region with 40% padding around bbox
   - Resize to 512x512 for the model
   - Run inference -> 19-class segmentation map
   - Classes: 0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye,
     5=r_eye, 6=eye_glasses, 7=l_ear, 8=r_ear, 9=earring, 10=nose,
     11=mouth_inside, 12=u_lip, 13=l_lip, 14=neck, 15=necklace,
     16=cloth, 17=hair, 18=hat
   - Return the segmentation map resized back to FULL FRAME coordinates

3. get_face_region_mask(self, frame, bbox) -> np.ndarray:
   - From segmentation, create a binary mask where INNER face classes = 1:
     skin(1), l_brow(2), r_brow(3), l_eye(4), r_eye(5), nose(10),
     mouth_inside(11), u_lip(12), l_lip(13)
   - This tells us WHERE the face SHOULD be
   - Return as float32 [0, 1] at full frame resolution

4. IMPORTANT: This mask alone is NOT sufficient for occlusion handling.
   It tells us what face parts are VISIBLE, but it does NOT tell us
   if something is IN FRONT OF the face. A hand over your mouth
   will simply make those pixels show as "not face" — which is correct
   but incomplete. We need the other layers to know WHY they're not face.

Include visualization in __main__: show webcam with color-coded
segmentation overlay (each class a different color).
```

#### Prompt 3.2 — Hand detection (Layer 2: Catch the #1 occluder)

```
Implement src/occlusion/hand_detector.py:

Hands are BY FAR the most common thing that covers a face during a stream
(touching face, resting chin, holding items near face, waving, etc.).
We need explicit hand detection.

Create a HandDetector class using MediaPipe Hands:

1. __init__(self, config: dict):
   - Initialize mediapipe.solutions.hands.Hands with:
     - static_image_mode=False (video mode for temporal consistency)
     - max_num_hands=2
     - min_detection_confidence=0.5
     - min_tracking_confidence=0.5
   - Also initialize mediapipe.solutions.selfie_segmentation as backup

2. detect_hands(self, frame: np.ndarray) -> list[HandRegion]:
   - Run MediaPipe hand detection
   - For each detected hand, return a HandRegion dataclass containing:
     - landmarks: 21 hand keypoints (wrist, fingers, etc.)
     - bbox: bounding box around the hand
     - handedness: left or right

3. get_hand_mask(self, frame: np.ndarray, face_bbox: np.ndarray) -> np.ndarray:
   THIS IS THE CRITICAL METHOD:

   a. Detect all hands in the frame
   b. For each detected hand:
      - Create a filled convex hull polygon from the 21 hand landmarks
      - This gives a tight mask of the hand shape
   c. Combine all hand masks into one binary mask
   d. INTERSECT with the face region (expanded bbox with 20% padding):
      - We only care about hand pixels that overlap the face area
      - Hands elsewhere in the frame are irrelevant
   e. Return float32 mask at full frame resolution where:
      - 1.0 = hand pixel overlapping face area (DO NOT SWAP here)
      - 0.0 = not a hand / hand not over face

4. get_hand_mask_fallback(self, frame, face_bbox) -> np.ndarray:
   When MediaPipe fails to detect hands (partial occlusion, unusual angles):
   - Use skin color detection in YCrCb space
   - Find skin-colored regions WITHIN the face bbox that are NOT
     contiguous with the main face skin blob
   - These isolated skin patches are likely fingers/palms
   - This is less accurate but catches cases MediaPipe misses

5. Handle edge case: MediaPipe sometimes detects the face itself as a
   "hand" when fingers are near the face. Filter out hand detections
   where >80% of the hand hull overlaps with the face parsing skin region.

Add mediapipe to requirements.txt.

Include __main__ test: open webcam, show hand mask overlay in green,
face bbox in blue, overlap region in red.
```

#### Prompt 3.3 — Body and object segmentation (Layer 3: Arms, items, etc.)

```
Implement src/occlusion/body_segmenter.py:

Beyond hands, arms, shoulders, clothing, and held objects (cups, phones,
microphones) can also occlude the face. We need a general "what is in
front of the face that isn't the face" detector.

Create a BodySegmenter class:

1. __init__(self, config: dict):
   Option A (recommended — faster):
   - Use MediaPipe Selfie Segmentation (landscape model)
   - This gives a person vs background mask
   - Combined with face parsing, we can isolate "person but not face"
     = body parts that might occlude

   Option B (more accurate, slower):
   - Use a general-purpose segmentation model like SAM2
     (Segment Anything Model 2) or YOLO-Seg
   - Can identify specific object classes (hand, arm, cup, phone, etc.)
   - Heavier but catches held objects that aren't body parts

   Implement Option A as default with Option B as configurable upgrade.

2. get_body_over_face_mask(self, frame, face_bbox, face_parse_mask) -> np.ndarray:

   Strategy using MediaPipe Selfie Segmentation:
   a. Get person_mask from selfie segmentation (full body)
   b. Get face_mask from face parsing (just the face)
   c. Compute: body_not_face = person_mask AND (NOT face_mask)
   d. Compute: body_over_face = body_not_face AND face_bbox_region
   e. This catches arms, shoulders, hands (redundant with Layer 2
      but good for reinforcement), clothing, etc. that overlap
      the face bounding box

   The key insight: within the face bounding box, any pixel that is
   "person" but NOT "face" is almost certainly something in FRONT of
   the face (arm, hand, shoulder, collar, etc.)

3. get_object_mask(self, frame, face_bbox) -> np.ndarray:
   For non-body objects (microphone, cup, phone, straw):
   - Use edge detection (Canny) within the face bbox
   - Find strong edges that break the smooth face surface
   - Cluster edge-bounded regions
   - Regions with texture/color very different from surrounding
     skin are likely objects
   - This is a heuristic fallback, not a primary detector

Include __main__: show body-not-face regions highlighted in yellow overlay.
```

#### Prompt 3.4 — Error-based anomaly detection (Layer 4: The FaceShifter approach)

```
Implement src/occlusion/error_detector.py:

This implements the core idea from FaceShifter's HEAR-Net: instead of
trying to detect WHAT is occluding, detect WHERE the swap went wrong.

The insight: when the swap model tries to generate a face where there's
actually a hand, the output looks WRONG. By comparing the swapped result
against what we'd expect, we can find anomaly regions.

Create an ErrorDetector class:

1. __init__(self):
   - Stores reference statistics for the source face (color histogram,
     texture patterns, skin color range in LAB space)
   - These are computed once when the source face is loaded

2. compute_error_map(self, original_frame, swapped_frame,
                     face_bbox) -> np.ndarray:

   Within the face bounding box, compute a per-pixel anomaly score:

   a. STRUCTURAL ERROR: Compute the absolute difference between
      the original and swapped frames in LAB space.
      Where the face was successfully swapped, the L channel changes
      significantly (new identity). Where an occluder exists, the
      swap model produces artifacts — high-frequency noise,
      color bleeding, or texture discontinuities.

   b. COLOR ANOMALY: The swapped face should have consistent skin tone.
      Compute the skin color distribution of the clearly-swapped
      region (center of face). Pixels within the face bbox that
      deviate significantly from this skin color are anomalous.
      A hand may have different skin tone, veins, hair, rings, etc.

   c. TEXTURE ANOMALY: Compute local texture energy (Laplacian variance)
      in small patches (8x8) across the face region. The swap model
      produces smooth, consistent texture on face areas. Occluded
      areas show texture discontinuities — sudden edges, different
      skin texture, object surfaces.

   d. EDGE DISCONTINUITY: Run Sobel edge detection on the swapped
      face. Strong edges that DON'T correspond to expected face
      features (eyes, nose, mouth, jaw) suggest occlusion boundaries.

   e. Combine all scores: anomaly_map = weighted sum of (a, b, c, d)
      Normalize to [0, 1] where 1 = high anomaly = likely occluded

3. threshold_anomaly(self, anomaly_map, sensitivity=0.5) -> np.ndarray:
   - Apply adaptive thresholding (Otsu's method on the anomaly map)
   - OR use a fixed threshold scaled by sensitivity parameter
   - Return binary mask: 1 = anomalous region (don't swap here)

4. IMPORTANT: This layer works AFTER the swap has been applied.
   The pipeline becomes:
     swap face -> compute error map -> use error map to mask out
     bad regions -> re-composite with original pixels in those regions

   This is a REFINEMENT step, not a pre-filter.

Include __main__: show side-by-side of swapped face with anomaly map
heatmap overlay. Red = high anomaly.
```

#### Prompt 3.5 — Monocular depth estimation (Layer 5: Z-ordering)

```
Implement src/occlusion/depth_estimator.py:

Depth estimation provides the most physically-grounded occlusion signal:
if something is CLOSER to the camera than the face, it must be in front.

Create a DepthEstimator class:

1. __init__(self, config: dict):
   - Load MiDaS v3.1 small model (DPT_SwinV2_T_256) via torch.hub
     or download ONNX version for faster inference
   - MiDaS small runs at ~30ms on RTX 5080 — acceptable for our use case
   - Alternative: Depth Anything V2 (vits model) which is even faster

2. estimate_depth(self, frame: np.ndarray) -> np.ndarray:
   - Resize frame to model input size (256x256 for small)
   - Run inference -> relative depth map
   - Resize back to frame dimensions
   - Normalize to [0, 1] range (0 = closest, 1 = farthest)
   - Return float32 depth map

3. get_foreground_mask(self, frame, face_bbox) -> np.ndarray:
   THE KEY METHOD:

   a. Compute depth map for the full frame
   b. Extract depth values within the face bbox
   c. Compute the MEDIAN depth of face pixels (this is the "face plane")
   d. Find all pixels within the face bbox that are CLOSER (lower depth)
      than the face plane by a threshold margin
   e. These closer pixels are IN FRONT of the face = occluders

   depth_face = median(depth[face_bbox])
   occluder_mask = (depth[face_bbox] < depth_face - margin)

   The margin (configurable, ~0.05-0.15 of depth range) prevents
   noise from the face surface itself being flagged.

4. Handle limitations:
   - MiDaS gives RELATIVE depth, not absolute — this is fine for our
     purpose (we only need "closer than face")
   - Depth is noisy at edges — apply bilateral filtering to depth map
   - Hair at the sides of the face may appear closer — combine with
     face parsing to exclude hair regions from analysis

Add torch and timm to requirements.txt (for MiDaS/Depth Anything).

Include __main__: show webcam with depth map visualization and
foreground mask highlighted.
```

#### Prompt 3.6 — Occlusion fusion (Combining all layers)

```
Implement src/occlusion/occlusion_fuser.py:

This is the BRAIN of the occlusion system. It takes the output from
all 5 layers and produces a single, high-quality occlusion mask.

Create an OcclusionFuser class:

1. __init__(self, config: dict):
   - Store weights for each layer (configurable):
     - face_parsing_weight: 0.3
     - hand_detection_weight: 0.9  (most reliable for common case)
     - body_segmentation_weight: 0.5
     - error_detection_weight: 0.6
     - depth_estimation_weight: 0.7
   - Store which layers are enabled (all by default, but user can
     disable slower ones for better FPS)

2. fuse(self, face_parse_mask, hand_mask, body_mask,
        error_mask, depth_mask) -> np.ndarray:

   Weighted fusion strategy:

   a. Start with the face region from face_parsing as the BASE.
      This defines the maximum area where swapping CAN happen.

   b. SUBTRACT occlusion evidence from each layer:
      - Any pixel flagged by hand_detection is STRONGLY removed
        (hands are the most reliable signal)
      - Pixels flagged by depth_estimation are removed (physically
        grounded — something is closer than the face)
      - Pixels flagged by body_segmentation are moderately removed
      - Pixels flagged by error_detection refine the edges

   c. Voting system: a pixel is marked as OCCLUDED if:
      - hand_mask says yes (instant override — high confidence), OR
      - 2 or more other layers agree it's occluded, OR
      - depth says yes AND one other layer agrees

   d. The output is a SWAP MASK in [0, 1]:
      - 1.0 = safe to show swapped face
      - 0.0 = show original frame pixels (something is in front)
      - Intermediate values at boundaries for smooth blending

3. fuse_fast(self, face_parse_mask, hand_mask, body_mask) -> np.ndarray:
   - Lightweight version using only Layers 1-3 (no error or depth)
   - For when FPS matters more than perfection
   - Logic: face_parse_mask AND NOT (hand_mask OR body_mask)

4. Handle temporal consistency:
   - The fused mask should not flicker frame-to-frame
   - Apply EMA smoothing: fused = 0.8 * current + 0.2 * previous
   - BUT: when a new occlusion APPEARS (hand moves in front of face),
     react quickly (alpha=0.95) to avoid ghost trail
   - When occlusion DISAPPEARS (hand moves away), fade out smoothly
     (alpha=0.7) so the face doesn't pop in harshly
   - Detect these transitions by comparing mask area change between frames

5. Debug output: generate a color-coded visualization showing which
   layer is responsible for each masked pixel:
   - Green = hand detector
   - Blue = depth
   - Yellow = body segmentation
   - Orange = error detection
   - White = face parsing boundary

Include __main__: full demo with all layers running, show the debug
color-coded overlay alongside the final fused mask.
```

#### Prompt 3.7 — Mask refinement (Post-processing)

```
Implement src/occlusion/mask_refiner.py:

Create a MaskRefiner class that post-processes the fused occlusion mask
to produce clean, artifact-free blending boundaries:

1. refine(self, mask: np.ndarray, config: dict) -> np.ndarray:
   Apply these steps in sequence:

   a. Morphological closing (kernel=5) to fill small holes in the
      swap-safe region (prevents speckle artifacts on the face)
   b. Morphological opening (kernel=3) to remove small noise pixels
      at the boundary
   c. Connected component analysis: in the SWAP region (mask=1),
      keep only the largest connected component — this prevents
      small disconnected patches of swapped face appearing through
      gaps in fingers, etc.
   d. Feathered edges (see below)
   e. Normalize to [0, 1] float range

2. feather_edges(self, mask, radius) -> np.ndarray:
   - Compute distance transform from the mask boundary
   - Create smooth alpha falloff over the feather radius
   - This is CRITICAL for seamless blending — hard mask edges
     are the #1 visual giveaway of face swapping
   - Use a wider feather radius (15-21px) at occlusion boundaries
     specifically, narrower (7-11px) at the face/background boundary

3. adaptive_feathering(self, mask, frame) -> np.ndarray:
   - Analyze the frame's local contrast at mask boundaries
   - Where contrast is high (sharp edge between hand and face),
     use a NARROW feather — the edge is real and should stay sharp
   - Where contrast is low (face fading into background),
     use a WIDER feather — blend softly
   - This makes occlusion boundaries look natural rather than
     uniformly blurred

4. temporal_smooth_mask(self, current_mask, previous_mask, alpha=0.8):
   - EMA blend between consecutive masks to prevent mask flickering
   - The OcclusionFuser already does some temporal smoothing, but
     this is a final safety net after morphological operations

All operations work on float32 masks in [0, 1] range.
Include __main__ test showing raw fused mask -> refined mask comparison.
```

---

### PHASE 4: Beauty Filters

#### Prompt 4.1 — Skin smoothing

```
Implement src/beauty/skin_smoothing.py:

Create a SkinSmoother class:

1. smooth(self, frame, face_mask, strength=0.6) -> np.ndarray:

   Implement a high-quality multi-step skin smoothing algorithm:

   a. Convert to LAB color space (smooth luminance, preserve color)
   b. Apply bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
      on the L channel — preserves edges while smoothing skin
   c. Apply guided filter (opencv ximgproc.guidedFilter) with the
      original L channel as guide — this preserves texture better
      than bilateral alone
   d. Blend between original and smoothed using strength parameter
   e. Apply ONLY within the face_mask region (skip eyes, brows, lips
      to preserve sharp details there)

2. detect_skin_regions(self, frame, face_mask) -> np.ndarray:
   - Within the face mask, further isolate actual skin pixels
     using HSV color range thresholding
   - Exclude eyes, nostrils, lip interior
   - This prevents over-smoothing non-skin facial features

3. The strength parameter (0.0 - 1.0) should feel natural:
   - 0.3 = subtle, like good lighting
   - 0.6 = noticeable but natural, magazine-quality
   - 0.9 = heavy, like a strong phone beauty filter

Include __main__ with trackbar for real-time strength adjustment.
```

#### Prompt 4.2 — Eye enhancement & color correction

```
Implement src/beauty/eye_enhancement.py:

Create an EyeEnhancer class:

1. enhance(self, frame, landmarks, strength=0.3) -> np.ndarray:
   a. Extract eye regions using the 5-point landmarks
   b. Increase local contrast in iris area (CLAHE on L channel)
   c. Subtle brightening of sclera (whites of eyes)
   d. Light sharpening via unsharp mask on eye region only
   e. Blend with strength parameter

Then implement src/beauty/color_correction.py:

Create a ColorCorrector class:

1. match_skin_tone(self, swapped_frame, original_frame, face_mask) -> np.ndarray:
   - Calculate mean & std of skin pixels in LAB space for both frames
   - Apply Reinhard color transfer to match the swapped face's skin
     to the original frame's lighting conditions
   - This prevents the swapped face from looking pasted-on

2. auto_white_balance(self, frame, face_mask) -> np.ndarray:
   - Estimate illuminant from face skin pixels
   - Apply subtle correction to neutralize color cast

3. enhance_contrast(self, frame, face_mask, strength=0.2) -> np.ndarray:
   - Apply adaptive CLAHE to the face region only
   - Very subtle — just enough to add depth

Include __main__ with before/after comparison.
```

#### Prompt 4.3 — Beauty filter chain

```
Implement src/beauty/filter_chain.py:

Create a BeautyFilterChain class that:

1. __init__(self, config: dict):
   - Instantiate all beauty sub-modules based on config
   - Each filter can be individually enabled/disabled
   - Order matters: color_correction -> skin_smoothing -> eye_enhancement -> sharpening

2. process(self, frame, face_mask, landmarks) -> np.ndarray:
   - Run enabled filters in sequence
   - Each filter receives the output of the previous one
   - Track per-filter timing for profiling

3. update_config(self, config: dict):
   - Hot-reload filter parameters without restarting
   - Useful for the GUI to adjust in real-time

4. Add a subtle sharpening final pass:
   - Unsharp mask (amount=0.3, radius=1.5) on the face region
   - Restores detail lost by smoothing, gives a crisp look

Make sure all filters operate on the FACE REGION ONLY using the mask,
leaving the body and background completely untouched.
```

---

### PHASE 5: Temporal Stabilization

#### Prompt 5.1 — Temporal smoothing

```
Implement src/temporal/stabilizer.py:

Create a LandmarkStabilizer class:

1. Uses exponential moving average (EMA) to smooth landmark positions
   across frames, preventing jitter

2. stabilize(self, landmarks: np.ndarray) -> np.ndarray:
   - If first frame, store and return as-is
   - Otherwise: smoothed = alpha * current + (1 - alpha) * previous
   - alpha is configurable (0.7 = moderate smoothing)
   - Apply independently to each of the 5 landmark points

3. stabilize_bbox(self, bbox: np.ndarray) -> np.ndarray:
   - Same EMA approach for bounding box corners
   - Prevents the swap region from jumping frame-to-frame

4. Add adaptive alpha: when landmark movement exceeds a threshold
   (e.g., fast head turn), temporarily increase alpha toward 1.0
   to allow fast tracking without lag

Then implement src/temporal/frame_blender.py:

Create a FrameBlender class:

1. blend(self, current_frame, face_mask) -> np.ndarray:
   - Maintain a buffer of the previous blended face region
   - EMA blend: output = alpha * current + (1 - alpha) * previous
   - Apply ONLY to the face region (using mask), leave background sharp
   - alpha=0.85 gives good flicker reduction without ghosting

2. reset(self):
   - Clear buffer (call when source face changes or on scene cut)

3. detect_scene_cut(self, current, previous, threshold=30) -> bool:
   - If mean absolute difference between frames exceeds threshold,
     it's likely a scene cut — reset the blend buffer

This is critical for stream quality — without temporal smoothing,
the swapped face will visibly flicker frame-to-frame.
```

---

### PHASE 6: Compositing & Output

#### Prompt 6.1 — Final compositing

```
Implement src/compositing/blender.py:

Create a FrameCompositor class:

1. composite(self, original_frame, swapped_frame, final_mask,
             config) -> np.ndarray:

   This is the critical final assembly step. The final_mask comes from
   the OcclusionFuser + MaskRefiner and already encodes ALL occlusion
   information from all 5 layers.

   final_mask values:
     1.0 = show swapped face (this pixel IS the face, nothing in front)
     0.0 = show original frame (this pixel is a hand, arm, object, or background)
     0.0-1.0 = feathered boundary (smooth blend)

   a. Compute: output = final_mask * swapped_frame + (1 - final_mask) * original_frame
      This single operation handles EVERYTHING:
      - Face pixels show the new identity
      - Hand/arm/object pixels show the real webcam feed
      - Boundaries blend smoothly
   b. Optionally apply Poisson blending (cv2.seamlessClone) WITHIN the
      swap region (where final_mask > 0.5) for color-seamless integration
   c. After Poisson blending, re-apply the mask to ensure occluders
      are still preserved (Poisson can bleed slightly)

2. simple_blend(self, original, swapped, mask) -> np.ndarray:
   - Fast fallback: mask * swapped + (1 - mask) * original
   - Used when Poisson blending is too slow

3. poisson_blend(self, original, swapped, mask) -> np.ndarray:
   - cv2.seamlessClone with NORMAL_CLONE
   - Automatically matches color/lighting at boundaries
   - Higher quality but slower (~15ms per frame)

Then implement src/compositing/color_transfer.py:

1. transfer(self, source_region, target_frame, mask) -> np.ndarray:
   - Reinhard color transfer in LAB space
   - Matches mean and std of color channels
   - Ensures the swapped face doesn't look color-shifted

Include __main__ showing composite with debug overlays
(mask boundaries, occlusion regions highlighted).
```

#### Prompt 6.2 — Virtual camera output

```
Implement src/output/virtual_camera.py:

Create a VirtualCameraOutput class:

1. __init__(self, config: dict):
   - Initialize pyvirtualcam.Camera with configured width, height, fps
   - On Windows, pyvirtualcam uses the OBS Virtual Camera backend automatically
   - IMPORTANT: OBS must NOT be running when the virtual camera starts,
     because OBS and pyvirtualcam cannot share the OBS virtual camera device
   - Alternative: use the 'unitycapture' backend if OBS conflicts arise
     (pip install unitycapture, then set backend='unitycapture')
   - Print the device name for OBS configuration

2. send(self, frame: np.ndarray):
   - Convert BGR (OpenCV) -> RGB (pyvirtualcam)
   - Send frame to virtual camera
   - Handle frame rate limiting (don't send faster than target_fps)

3. Context manager support

Then implement src/output/preview_window.py:

Create a PreviewWindow class:

1. show(self, frame, metrics: dict):
   - Display frame in an OpenCV window
   - Overlay: FPS, latency (ms), swap active (yes/no),
     detection confidence, GPU memory usage
   - Small picture-in-picture of the original unswapped face
     in the corner (for the performer's reference)

2. Handle keyboard shortcuts:
   - 'q' = quit
   - 'b' = toggle beauty filters
   - 'o' = toggle ALL occlusion handling
   - '1' = toggle Layer 1 (face parsing)
   - '2' = toggle Layer 2 (hand detection)
   - '3' = toggle Layer 3 (body segmentation)
   - '4' = toggle Layer 4 (error detection)
   - '5' = toggle Layer 5 (depth estimation)
   - 'd' = toggle debug overlay (show color-coded occlusion layers)
   - 's' = screenshot (save current frame)
   - '+'/'-' = adjust beauty strength

Windows virtual camera setup instructions in docstring:

  Option A — OBS Virtual Camera (recommended):
    1. Install OBS Studio (https://obsproject.com)
    2. OBS includes a virtual camera since v26+
    3. IMPORTANT workflow: Start your faceswap-stream FIRST (it claims
       the OBS virtual camera device via pyvirtualcam), then open OBS
       and add "Video Capture Device" -> select "OBS Virtual Camera"
    4. If conflict occurs, use Option B instead

  Option B — Unity Capture (alternative, avoids OBS conflicts):
    1. Download UnityCapture from https://github.com/schellingb/UnityCapture
    2. Run Install.bat from the downloaded package
    3. Set virtual_cam_backend: unitycapture in config
    4. In OBS, add "Video Capture Device" -> select "Unity Video Capture"
```

---

### PHASE 7: Pipeline Orchestrator

#### Prompt 7.1 — Main pipeline

```
Implement src/pipeline.py:

Create a SwapPipeline class that orchestrates everything:

1. __init__(self, config_path: str):
   - Load YAML config (with profile override support)
   - Initialize ALL components:
     - WebcamCapture
     - FaceDetector
     - IdentityManager (load source face)
     - FaceSwapper
     - FaceParser (Layer 1)
     - HandDetector (Layer 2)
     - BodySegmenter (Layer 3)
     - ErrorDetector (Layer 4)
     - DepthEstimator (Layer 5)
     - OcclusionFuser (combines all layers)
     - MaskRefiner
     - BeautyFilterChain
     - LandmarkStabilizer
     - FrameBlender
     - FrameCompositor
     - VirtualCameraOutput
     - PreviewWindow
   - Print initialization summary with timing for each component
   - Print total VRAM usage after all models loaded

2. process_frame(self, frame: np.ndarray) -> np.ndarray:
   Single frame processing pipeline:

   a. Detect face -> get primary face
   b. If no face: return original frame (with brief grace period
      using last known position)
   c. Stabilize landmarks and bbox

   --- SWAP ---
   d. Swap face using InSwapper -> produces swapped_frame

   --- OCCLUSION (multi-layer) ---
   e. Layer 1: Face parsing -> face_region_mask (where the face IS)
   f. Layer 2: Hand detection -> hand_mask (hands over face)
   g. Layer 3: Body segmentation -> body_mask (arms, objects over face)
   h. Layer 4: Error detection -> compare original vs swapped -> error_mask
   i. Layer 5: Depth estimation -> depth_mask (things closer than face)
   j. FUSE all layers -> raw_occlusion_mask
   k. REFINE mask -> feathered, morphologically cleaned final_mask

   --- BEAUTY ---
   l. Apply beauty filters to swapped face region (using final_mask
      to know which pixels are actually face)

   --- COMPOSITE ---
   m. Temporal frame blending (face region only)
   n. Final compositing:
      - Where final_mask = 1.0: show swapped + beautified face
      - Where final_mask = 0.0: show ORIGINAL frame pixels
        (hand, arm, object preserved from real webcam feed)
      - Intermediate values: smooth blend at boundaries

   o. Return finished frame

3. run(self):
   Main loop:
   - Capture frame
   - Process frame (with full timing breakdown)
   - Send to virtual camera
   - Show preview
   - Handle keyboard input
   - Print periodic stats (every 5 seconds):
     avg FPS, per-stage latency breakdown, GPU memory,
     occlusion layers active, hand detections per second

4. Graceful shutdown: release all resources in correct order

5. Error resilience:
   - If any stage fails on a frame, fall back gracefully
   - Occlusion layer fallback chain:
     If depth estimation fails -> skip Layer 5, use Layers 1-4
     If error detection fails -> skip Layer 4, use remaining layers
     If hand detection fails -> use hand_mask_fallback (skin detection)
     If body segmentation fails -> skip Layer 3
     If face parsing fails -> use bbox-based rectangular mask only
   - Each layer is independent — one failing doesn't affect others
   - Never crash the stream — always output SOMETHING to virtual cam
   - Log layer failures at WARNING level for debugging

Include detailed logging at DEBUG level for troubleshooting.
```

---

### PHASE 8: Entry Point & GUI

#### Prompt 8.1 — Application entry point

```
Implement app.py:

1. Parse command-line arguments:
   - --config PATH (default: config/default.yaml)
   - --profile NAME (loads from config/profiles/)
   - --source PATH (source face image)
   - --device INT (webcam device id)
   - --preview / --no-preview
   - --gui (launch Gradio interface instead of CLI)

2. CLI mode (default):
   - Print banner with project name and version
   - Load config and apply profile overrides
   - Initialize pipeline
   - Print keyboard shortcuts
   - Run pipeline loop
   - Catch KeyboardInterrupt for clean shutdown

3. GUI mode (--gui):
   - Launch a Gradio interface with:
     - Source face upload/selection
     - Beauty filter sliders (smoothing, eye enhancement, sharpness)
     - Occlusion layer toggles (each of the 5 layers individually)
     - Occlusion debug view checkbox (show which layer is doing what)
     - Live preview panel (updating every ~500ms for the GUI)
     - Start/Stop stream button
     - FPS and per-stage latency display
     - Config save/load

4. Add a health check on startup:
   - Verify all models exist (run download if not)
   - Test GPU availability
   - Test webcam access
   - Test virtual camera device
   - Print clear actionable errors for any failures
```

---

### PHASE 9: Testing & Benchmarking

#### Prompt 9.1 — Tests

```
Create the test suite:

1. tests/test_detection.py:
   - Test face detection on sample images (include 3 test fixtures)
   - Test no-face case returns empty
   - Test multi-face returns sorted by size
   - Test confidence filtering

2. tests/test_swap.py:
   - Test swap produces same-size output
   - Test swap with small face is skipped
   - Test swap preserves non-face regions exactly

3. tests/test_occlusion.py:
   - Test face parsing produces valid 19-class map
   - Test face_region_mask is binary and covers expected area
   - Test hand detector returns empty list when no hands visible
   - Test hand detector mask overlaps face bbox correctly
   - Test hand_mask_fallback finds skin blobs within face region
   - Test body segmenter body_over_face excludes face pixels
   - Test error detector anomaly_map is in [0, 1] range
   - Test depth estimator returns valid depth map shape
   - Test depth foreground_mask flags closer pixels
   - Test occlusion fuser voting logic:
     - hand_mask alone is enough to mark occluded
     - 2+ layers agreeing marks occluded
     - single weak signal alone is NOT enough
   - Test fuser temporal smoothing reduces frame-to-frame variance
   - Test mask refinement preserves mask shape
   - Test feathering produces smooth edges (no hard 0->1 transitions)
   - Test full occlusion pipeline: synthetic hand-over-face image
     correctly produces mask that preserves hand region

4. tests/test_beauty.py:
   - Test skin smoothing reduces high-frequency noise
   - Test eye enhancement only modifies eye region
   - Test color correction preserves overall luminance
   - Test filter chain processes all enabled filters

5. tests/test_pipeline_integration.py:
   - Test full pipeline on a static image
   - Test pipeline handles missing face gracefully
   - Test pipeline handles occluded face
   - Test output dimensions match input

Use pytest. Create minimal test fixtures in tests/fixtures/sample_frames/.
Mock the webcam and virtual camera for CI-friendly tests.
```

#### Prompt 9.2 — Benchmark script

```
Create scripts/benchmark.py:

1. Run the full pipeline on 200 frames from webcam (or a test video)

2. Measure and report per-stage latency:
   - Face detection: X ms (avg / p95 / max)
   - Face swap: X ms
   - Occlusion Layer 1 (face parsing): X ms
   - Occlusion Layer 2 (hand detection): X ms
   - Occlusion Layer 3 (body segmentation): X ms
   - Occlusion Layer 4 (error detection): X ms
   - Occlusion Layer 5 (depth estimation): X ms
   - Occlusion fusion + refinement: X ms
   - Occlusion TOTAL: X ms
   - Beauty filters: X ms
   - Temporal blending: X ms
   - Compositing: X ms
   - Total pipeline: X ms
   - Effective FPS: X

3. Report GPU metrics:
   - Peak VRAM usage
   - Average GPU utilization
   - TensorRT vs CUDA comparison if both available

4. Generate a performance report as a markdown table

5. Suggest optimizations based on bottlenecks:
   - If detection is slow -> suggest smaller det_size
   - If swap is slow -> suggest TensorRT conversion
   - If depth estimation is slow -> suggest switching to Depth Anything V2 small
   - If total occlusion is slow -> suggest disabling Layer 4 (error) or
     Layer 5 (depth) first, as Layers 1-3 cover most cases
   - If compositing is slow -> suggest simple blend over Poisson
   - Print estimated FPS with each layer combination disabled

6. Save report to benchmark_results.md
```

---

## Post-Development Checklist

After all phases are complete, run through these with Claude Code:

```
Review the entire faceswap-stream codebase and:

1. Check for any resource leaks (unclosed cameras, GPU memory)
2. Verify all error handling is consistent
3. Check that GPU tensors are properly freed after inference
4. Verify thread safety in the webcam capture module
5. Check that all config parameters are actually used
6. Look for any hardcoded values that should be configurable
7. Verify the shutdown sequence releases resources in correct order
8. Add type hints to any functions missing them
9. Verify all __main__ self-tests still work
10. Run the full test suite and fix any failures
```

---

## OBS Configuration Notes (Windows)

Include these in the README:

1. Install OBS Studio from https://obsproject.com (v26+ includes virtual camera)
2. Start faceswap-stream FIRST, then open OBS
3. In OBS, add a "Video Capture Device" source
4. Select "OBS Virtual Camera" (or "Unity Video Capture" if using Option B)
5. Set OBS canvas to match webcam resolution (1920x1080)
6. Set output encoder to NVENC H.264 (RTX 5080 hardware encoder — much
   better performance than x264 on Windows)
7. For streaming: CBR at 6000-8000 kbps for 1080p
8. The face swap pipeline outputs the final composited frame — no additional
   OBS filters should be needed on the source
9. If OBS shows a black screen, check that:
   - faceswap-stream started successfully before OBS
   - The correct virtual camera device is selected
   - No other application is claiming the virtual camera

---

## Windows Prerequisites

Before starting, ensure you have these installed:

1. **Python 3.11+** — Download from https://python.org (check "Add to PATH" during install)
2. **CUDA Toolkit 12.x** — Download from https://developer.nvidia.com/cuda-downloads
   (must match your RTX 5080 driver version)
3. **cuDNN** — Download from https://developer.nvidia.com/cudnn and copy DLLs to CUDA bin folder
4. **Visual Studio Build Tools** — Required for compiling some Python packages
   Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
   Select "Desktop development with C++" workload
5. **OBS Studio v26+** — https://obsproject.com (for virtual camera)
6. **Git** — https://git-scm.com (for cloning model repos)
7. **Conda (recommended)** — https://docs.conda.io/en/latest/miniconda.html

### Common Windows Pitfalls

- **onnxruntime-gpu** requires matching CUDA version. If import fails, check:
  `nvidia-smi` for driver CUDA version, then install matching onnxruntime
- **insightface** may fail to build on Windows — install Visual C++ Build Tools first
- **pyvirtualcam** on Windows only works with OBS Virtual Camera or UnityCapture backend
- **OpenCV** `cv2.imshow()` windows may freeze if you don't call `cv2.waitKey()` in the loop
- **Antivirus** may flag ONNX model files — add the models/ folder to exclusions
- **Windows Defender Real-time Protection** can slow model loading significantly —
  consider adding the project folder to exclusions during development

---

## Important Notes

- **First run** will be slow as ONNX Runtime optimizes for your GPU
- **TensorRT** conversion happens automatically on first inference
  and is cached — subsequent starts will be much faster
- Place your **AI-generated source face** in `source_faces/` — use a
  front-facing, well-lit, neutral expression photo for best results
- If latency exceeds 200ms, switch config profile to `balanced.yaml`
- Monitor GPU temperature — sustained face swap can push the RTX 5080