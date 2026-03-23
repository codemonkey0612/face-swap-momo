"""Offline source face variant generator.

Generates a full set of lighting and pose variants from a SINGLE base AI
face image.  Run this ONCE before streaming.

Requirements (separate from main pipeline — heavy deps):
    pip install diffusers>=0.25.0 transformers>=4.36.0 accelerate>=0.25.0

IP-Adapter weights:
    Download from: https://huggingface.co/h94/IP-Adapter
    Place in: models/ip_adapter/

Usage:
    python scripts/generate_source_faces.py \\
        --base source_faces/base.png \\
        --output_dir source_faces/ \\
        --num_variants 1

The script also works WITHOUT Stable Diffusion (--sd_mode none):
    python scripts/generate_source_faces.py \\
        --base source_faces/base.png \\
        --output_dir source_faces/ \\
        --sd_mode none

In --sd_mode none the script just:
  1. Copies your base image as front_neutral.png
  2. Creates identity.json with detected properties
  3. Prints instructions for manual variant creation

This lets users create variants in ComfyUI / Midjourney and just drop
PNG files into source_faces/ — the runtime selector auto-detects their
properties from the images themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Variant specification table
# ---------------------------------------------------------------------------

LIGHTING_VARIANTS = [
    {
        "name":        "front_warm",
        "prompt_desc": "warm golden hour lighting, orange-yellow ambient, skin warmth",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 3500,
        "direction":   "front",
        "strength":    0.35,
    },
    {
        "name":        "front_cool",
        "prompt_desc": "cool blue fluorescent office lighting, slight blue tint",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 6800,
        "direction":   "front",
        "strength":    0.35,
    },
    {
        "name":        "front_bright",
        "prompt_desc": "bright key light from front, well lit, studio lighting",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 5500,
        "direction":   "front",
        "strength":    0.30,
    },
    {
        "name":        "front_dim",
        "prompt_desc": "dim ambient lighting, low light environment, soft shadows",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 4000,
        "direction":   "front",
        "strength":    0.40,
    },
    {
        "name":        "front_side_lit_left",
        "prompt_desc": "strong side lighting from left, dramatic shadows on right side",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 5500,
        "direction":   "left",
        "strength":    0.40,
    },
    {
        "name":        "front_side_lit_right",
        "prompt_desc": "strong side lighting from right, dramatic shadows on left side",
        "pose_yaw":    0.0,
        "pose_pitch":  0.0,
        "temperature": 5500,
        "direction":   "right",
        "strength":    0.40,
    },
]

POSE_VARIANTS = [
    {"name": "left_15",  "pose_yaw": -15.0, "pose_pitch": 0.0,  "temperature": 5500, "direction": "front"},
    {"name": "left_30",  "pose_yaw": -30.0, "pose_pitch": 0.0,  "temperature": 5500, "direction": "front"},
    {"name": "right_15", "pose_yaw":  15.0, "pose_pitch": 0.0,  "temperature": 5500, "direction": "front"},
    {"name": "right_30", "pose_yaw":  30.0, "pose_pitch": 0.0,  "temperature": 5500, "direction": "front"},
    {"name": "up_15",    "pose_yaw":   0.0, "pose_pitch": -15.0, "temperature": 5500, "direction": "front"},
    {"name": "down_15",  "pose_yaw":   0.0, "pose_pitch":  15.0, "temperature": 5500, "direction": "front"},
]

COMBO_VARIANTS = [
    {"name": "left_warm",  "pose_yaw": -15.0, "pose_pitch": 0.0, "temperature": 3500, "direction": "front",
     "prompt_desc": "warm golden hour lighting, slight left head turn"},
    {"name": "right_warm", "pose_yaw":  15.0, "pose_pitch": 0.0, "temperature": 3500, "direction": "front",
     "prompt_desc": "warm golden hour lighting, slight right head turn"},
]

DIRECTION_VEC = {
    "front": [0.0,  0.0],
    "left":  [-0.6, 0.0],
    "right": [ 0.6, 0.0],
    "top":   [0.0, -0.4],
    "bottom":[0.0,  0.4],
}


# ---------------------------------------------------------------------------
# ArcFace embedding (using insightface)
# ---------------------------------------------------------------------------

def _load_swapper():
    """Load a minimal insightface analyser for embedding extraction."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neural_swap import NeuralFaceSwapper
    model = "models/ghost_3_256.onnx"
    if not os.path.exists(model):
        # Try fallback
        for candidate in ["models/inswapper_128.onnx", "models/ghost_2_256.onnx"]:
            if os.path.exists(candidate):
                model = candidate
                break
    return NeuralFaceSwapper(model_path=model, occlusion=False)


def _extract_embedding(swapper, img: np.ndarray) -> Optional[np.ndarray]:
    faces = swapper.detect(img)
    if not faces:
        return None
    return faces[0].normed_embedding.astype(np.float32).copy()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


# ---------------------------------------------------------------------------
# Stable Diffusion + IP-Adapter generation
# ---------------------------------------------------------------------------

def _build_pipeline(ip_adapter_dir: str, sd_model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Load the SD + IP-Adapter pipeline (runs on GPU if available)."""
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from ip_adapter import IPAdapter
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install: pip install diffusers transformers accelerate ip_adapter")
        return None, None

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[Generate] Loading SD pipeline on {device} ...")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=__import__("torch").float16 if device == "cuda" else __import__("torch").float32,
    ).to(device)

    ip_ckpt = os.path.join(ip_adapter_dir, "ip-adapter_sd15.bin")
    if not os.path.exists(ip_ckpt):
        print(f"[WARN] IP-Adapter checkpoint not found at {ip_ckpt}")
        print("       Falling back to plain img2img (identity preservation reduced)")
        return pipe, None

    image_encoder_path = os.path.join(ip_adapter_dir, "models/image_encoder")
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    print("[Generate] IP-Adapter loaded")
    return pipe, ip_model


def _generate_variant(
    pipe,
    ip_model,
    base_pil,
    prompt_desc: str,
    strength: float,
    seed: int,
    num_inference_steps: int = 25,
) -> Optional[object]:
    """Generate one variant image. Returns PIL Image or None."""
    try:
        import torch
        from PIL import Image

        neg_prompt = "deformed, ugly, low quality, blurry, different person, wrong face"
        prompt = (
            f"portrait photo of a person, {prompt_desc}, "
            "high quality, photorealistic, natural skin, same person, "
            "detailed face, sharp focus"
        )
        generator = torch.Generator().manual_seed(seed)

        if ip_model is not None:
            result = ip_model.generate(
                pil_image=base_pil,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                seed=seed,
                prompt=prompt,
                negative_prompt=neg_prompt,
                scale=0.7,
                strength=strength,
                image=base_pil,
            )
            return result[0] if result else None
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=base_pil,
                strength=strength,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            return result.images[0] if result.images else None

    except Exception as e:
        print(f"[Generate] Generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate source face variants for multi-reference selector"
    )
    parser.add_argument("--base",         required=True, help="Base AI face image")
    parser.add_argument("--output_dir",   default="source_faces/", help="Output directory")
    parser.add_argument("--num_variants", type=int, default=1,
                        help="Variants to generate per spec (1=one attempt, 3=best of 3)")
    parser.add_argument("--sd_model",     default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace SD model ID")
    parser.add_argument("--ip_adapter_dir", default="models/ip_adapter/",
                        help="Directory with IP-Adapter weights")
    parser.add_argument("--sd_mode",      default="auto",
                        choices=["auto", "sd", "none"],
                        help="'auto'=try SD, 'sd'=require SD, 'none'=skip SD (manual mode)")
    parser.add_argument("--sim_threshold", type=float, default=0.6,
                        help="Min ArcFace cosine similarity to accept variant")
    parser.add_argument("--steps",        type=int, default=25,
                        help="SD inference steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load base image ---
    base_img = cv2.imread(args.base)
    if base_img is None:
        print(f"[ERROR] Cannot read: {args.base}")
        sys.exit(1)

    print(f"[Generate] Base image: {args.base} ({base_img.shape[1]}×{base_img.shape[0]})")

    # --- Extract base embedding ---
    swapper = _load_swapper()
    base_emb = _extract_embedding(swapper, base_img)
    if base_emb is None:
        print("[ERROR] No face detected in base image")
        sys.exit(1)
    print(f"[Generate] Base identity extracted (embedding norm={np.linalg.norm(base_emb):.3f})")

    # --- Copy base as front_neutral ---
    neutral_path = os.path.join(args.output_dir, "front_neutral.png")
    cv2.imwrite(neutral_path, base_img)
    print(f"[Generate] Saved: front_neutral.png")

    # --- Decide SD mode ---
    use_sd = False
    pipe = ip_model = None

    if args.sd_mode != "none":
        try:
            from PIL import Image  # noqa: F401
            from diffusers import StableDiffusionImg2ImgPipeline  # noqa: F401
            if args.sd_mode == "auto":
                pipe, ip_model = _build_pipeline(args.ip_adapter_dir, args.sd_model)
                use_sd = pipe is not None
            else:
                pipe, ip_model = _build_pipeline(args.ip_adapter_dir, args.sd_model)
                if pipe is None:
                    print("[ERROR] Could not load SD pipeline in --sd_mode sd"); sys.exit(1)
                use_sd = True
        except ImportError:
            if args.sd_mode == "sd":
                print("[ERROR] diffusers not installed — install requirements-generate.txt")
                sys.exit(1)
            print("[WARN] diffusers not installed — switching to manual mode")

    if not use_sd:
        print("\n[Generate] Running in MANUAL MODE (no Stable Diffusion).")
        print("  Only front_neutral.png has been created.")
        print("  Create additional variants manually using ComfyUI, Midjourney,")
        print("  or AUTOMATIC1111 — same identity, different lighting/poses.")
        print("  Drop PNG files into:", args.output_dir)
        print("  Then re-run this script with --sd_mode none to build identity.json.\n")

    # --- Build variant list ---
    all_specs = (
        [{"name": "front_neutral", "pose_yaw": 0.0, "pose_pitch": 0.0,
          "temperature": 5500, "direction": "front", "strength": 0.0}]
        + LIGHTING_VARIANTS
        + POSE_VARIANTS
        + COMBO_VARIANTS
    )

    # --- Generate / register variants ---
    accepted_variants = []

    for spec in all_specs:
        name  = spec["name"]
        out_p = os.path.join(args.output_dir, f"{name}.png")

        # Already exists? Verify and register.
        if os.path.exists(out_p):
            img = cv2.imread(out_p)
            if img is None:
                continue
            emb = _extract_embedding(swapper, img)
            if emb is None:
                print(f"  [skip] {name}.png — no face detected")
                continue
            sim = _cosine_sim(base_emb, emb)
            if sim < args.sim_threshold:
                print(f"  [skip] {name}.png — sim={sim:.2f} < {args.sim_threshold}")
                continue
            accepted_variants.append({
                "filename":            f"{name}.png",
                "pose_yaw":           spec["pose_yaw"],
                "pose_pitch":         spec["pose_pitch"],
                "lighting_temperature": spec["temperature"],
                "lighting_direction":  DIRECTION_VEC.get(spec.get("direction", "front"), [0, 0]),
                "lighting_intensity":  0.7,
                "identity_similarity": round(float(sim), 4),
            })
            print(f"  [ok]   {name}.png  sim={sim:.3f}")
            continue

        # Generate via SD
        if use_sd and spec.get("strength", 0) > 0:
            from PIL import Image
            base_pil = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
            prompt_desc = spec.get("prompt_desc",
                                   f"portrait photo, {spec.get('direction','front')} lighting")

            generated = None
            for attempt in range(args.num_variants):
                seed = int(time.time()) + attempt * 1000
                pil = _generate_variant(pipe, ip_model, base_pil,
                                        prompt_desc, spec.get("strength", 0.35),
                                        seed, args.steps)
                if pil is None:
                    continue
                gen_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                emb = _extract_embedding(swapper, gen_bgr)
                if emb is None:
                    print(f"  [retry] {name} attempt {attempt+1}: no face")
                    continue
                sim = _cosine_sim(base_emb, emb)
                print(f"  [gen]  {name} attempt {attempt+1}: sim={sim:.3f}")
                if sim >= args.sim_threshold:
                    generated = gen_bgr
                    accepted_variants.append({
                        "filename":            f"{name}.png",
                        "pose_yaw":           spec["pose_yaw"],
                        "pose_pitch":         spec["pose_pitch"],
                        "lighting_temperature": spec["temperature"],
                        "lighting_direction":  DIRECTION_VEC.get(spec.get("direction", "front"), [0, 0]),
                        "lighting_intensity":  0.7,
                        "identity_similarity": round(float(sim), 4),
                    })
                    cv2.imwrite(out_p, generated)
                    print(f"  [ok]   {name}.png saved")
                    break
            if generated is None:
                print(f"  [fail] {name} — no passing variant after {args.num_variants} attempts")
        else:
            if spec["name"] != "front_neutral":
                print(f"  [skip] {name} — SD disabled, create manually")

    # --- Scan for extra image files not in predefined specs ---
    known_filenames = {f"{s['name']}.png" for s in all_specs}
    extra_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    extra_files = sorted([
        f for f in os.listdir(args.output_dir)
        if f.lower().endswith(extra_extensions)
        and f not in known_filenames
        and f != "front_neutral.png"
    ])

    if extra_files:
        print(f"\n[Generate] Found {len(extra_files)} extra image(s) — auto-detecting properties...")

    for fname in extra_files:
        fpath = os.path.join(args.output_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [skip] {fname} — cannot read image")
            continue

        emb = _extract_embedding(swapper, img)
        if emb is None:
            print(f"  [skip] {fname} — no face detected")
            continue

        sim = _cosine_sim(base_emb, emb)
        if sim < args.sim_threshold:
            print(f"  [skip] {fname} — sim={sim:.2f} < {args.sim_threshold}")
            continue

        # Auto-detect pose from landmarks
        faces = swapper.detect(img)
        face = faces[0]
        pose_yaw, pose_pitch = 0.0, 0.0
        if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5:
            kps = face.kps
            left_eye, right_eye, nose = kps[0], kps[1], kps[2]
            left_mouth, right_mouth = kps[3], kps[4]
            # Yaw
            eye_center = (left_eye + right_eye) / 2
            eye_dist = np.linalg.norm(right_eye - left_eye)
            yaw_ratio = (nose[0] - eye_center[0]) / (eye_dist + 1e-6)
            pose_yaw = float(np.degrees(np.arcsin(np.clip(yaw_ratio * 2, -1, 1))))
            # Pitch
            mouth_center = (left_mouth + right_mouth) / 2
            face_vertical = mouth_center[1] - eye_center[1]
            eye_nose_vertical = nose[1] - eye_center[1]
            pitch_ratio = eye_nose_vertical / (face_vertical + 1e-6)
            pose_pitch = float((pitch_ratio - 0.35) * 90)

        # Auto-detect lighting from face region
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        pad_w = int((x2 - x1) * 0.3)
        pad_h = int((y2 - y1) * 0.3)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(img.shape[1], x2 + pad_w)
        cy2 = min(img.shape[0], y2 + pad_h)
        face_crop = img[cy1:cy2, cx1:cx2]

        # Color temperature from LAB B-channel
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        mean_b = float(np.mean(lab[:, :, 2]))
        temperature = 3000 + (mean_b / 255) * 7000

        # Lighting direction from brightness asymmetry
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h_fc, w_fc = gray.shape[:2]
        left_mean = float(np.mean(gray[:, :w_fc // 2]))
        right_mean = float(np.mean(gray[:, w_fc // 2:]))
        dir_x = (right_mean - left_mean) / (right_mean + left_mean + 1e-6)
        top_mean = float(np.mean(gray[:h_fc // 2, :]))
        bot_mean = float(np.mean(gray[h_fc // 2:, :]))
        dir_y = (bot_mean - top_mean) / (bot_mean + top_mean + 1e-6)

        mean_luminance = float(np.mean(gray) / 255.0)

        accepted_variants.append({
            "filename":             fname,
            "pose_yaw":            round(pose_yaw, 1),
            "pose_pitch":          round(pose_pitch, 1),
            "lighting_temperature": round(temperature, 0),
            "lighting_direction":  [round(dir_x, 3), round(dir_y, 3)],
            "lighting_intensity":  round(mean_luminance, 3),
            "identity_similarity": round(float(sim), 4),
            "auto_detected":       True,
        })
        print(f"  [ok]   {fname}  sim={sim:.3f}  yaw={pose_yaw:.1f}°  pitch={pose_pitch:.1f}°  temp={temperature:.0f}K")

    # --- Write identity.json ---
    identity_data = {
        "identity_embedding": base_emb.tolist(),
        "variants": accepted_variants,
    }
    json_path = os.path.join(args.output_dir, "identity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(identity_data, f, indent=2)
    print(f"\n[Generate] identity.json written — {len(accepted_variants)} variants")

    # --- Summary ---
    print("\n=== Generation Summary ===")
    print(f"  Accepted variants : {len(accepted_variants)}")
    print(f"  Output directory  : {os.path.abspath(args.output_dir)}")
    print(f"  identity.json     : {json_path}")
    print("\nNext steps:")
    print("  1. Review the generated images in", args.output_dir)
    print("  2. Delete any that look wrong (the selector will ignore missing files)")
    print("  3. Set source_face_path: source_faces/ in config/default.yaml")
    print("  4. Start streaming — the selector auto-detects variant properties")


if __name__ == "__main__":
    main()
