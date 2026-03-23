# Eye Region Transition Fix — Claude Code Prompt

## The Problem

The eye area is where viewers look first and where face swap artifacts
are most visible. Specific issues:

1. InSwapper works at 128x128 — iris detail, eyelashes, and light
   reflections (catchlights) are lost, making eyes look flat/dead
2. The orbital region (eye socket, under-eye, brow ridge) has different
   skin thickness, color, and shadow patterns between source and performer,
   creating a visible "panda mask" around the eyes
3. Sclera (eye white) color mismatch — source has clear whites, performer
   may have slightly yellow/red whites, or vice versa
4. The brow ridge shadow from the source face doesn't match the performer's
   actual lighting, creating a floating dark band above the eyes
5. The transition from swapped eye texture to surrounding original skin
   shows a sharp texture/color boundary around the orbital bone

## The Solution: Three-Part Eye Pipeline

Part 1: Face restoration with CodeFormer (reconstructs eye detail lost at 128x128)
Part 2: Region-specific color matching (eliminates panda mask effect)
Part 3: Gradient eye-orbit blending (smooth transition between eye zone and face)

---

## Claude Code Prompt

```
Implement src/enhancement/eye_region_fixer.py:

This module fixes the eye region transition after face swapping.
It runs on the ALIGNED 128x128 (or upscaled 256/512) crop AFTER
InSwapper and shape correction, but BEFORE XSeg masking and
final compositing.

Create an EyeRegionFixer class:

========================================================
PART 1: FACE RESTORATION (CodeFormer)
========================================================

The biggest single improvement for eye quality. CodeFormer was
specifically trained to reconstruct realistic facial details —
it excels at eyes, producing sharp irises, realistic eyelashes,
and natural catchlights that InSwapper's 128x128 output lacks.

1. __init__(self, config: dict):
   - Load CodeFormer model:
     Option A (recommended for speed): ONNX version
       Download codeformer.onnx from:
       https://github.com/facefusion/facefusion-assets/releases
       Load with ONNX Runtime + CUDAExecutionProvider
     
     Option B (more flexible): PyTorch version
       git clone https://github.com/sczhou/CodeFormer
       Load CodeFormer with torch, weights: codeformer-v0.1.0.pth
   
   - Store config parameters:
     - restore_strength: float (0.0-1.0, default 0.6)
       CodeFormer's "fidelity" parameter. Higher = more restoration
       (sharper eyes, cleaner skin) but risks changing identity.
       Lower = preserves identity better but less enhancement.
       0.5-0.7 is the sweet spot for face swap use.
     - apply_to: str ('full_face' or 'eyes_only', default 'eyes_only')
     - upscale_before_restore: bool (default True)
       Upscale 128→512 before CodeFormer for best eye detail
   
   - Load BiSeNet face parser (reuse from existing pipeline)
     for eye region extraction
   
   - Initialize GFPGAN as fallback if CodeFormer fails
     (GFPGAN v1.4 is lighter and faster, slightly less quality)

2. restore_face(self, crop: np.ndarray, fidelity: float) -> np.ndarray:
   
   Run CodeFormer on the face crop:
   
   a. If upscale_before_restore:
      # Upscale 128→512 with LANCZOS (fast, preserves what's there)
      crop_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
   else:
      crop_512 = cv2.resize(crop, (512, 512))
   
   b. Preprocess for CodeFormer:
      # Normalize to [-1, 1]
      input_tensor = (crop_512.astype(np.float32) / 255.0)
      input_tensor = (input_tensor - 0.5) / 0.5
      input_tensor = np.transpose(input_tensor, (2, 0, 1))  # CHW
      input_tensor = np.expand_dims(input_tensor, 0)  # NCHW
   
   c. Run inference:
      # For ONNX:
      output = session.run(None, {
          'input': input_tensor,
          'fidelity': np.array([fidelity], dtype=np.float32)
      })[0]
      
      # For PyTorch:
      # output = codeformer_net(input_tensor, w=fidelity)[0]
   
   d. Postprocess:
      restored = output.squeeze()
      restored = np.transpose(restored, (1, 2, 0))  # HWC
      restored = (restored * 0.5 + 0.5) * 255.0
      restored = np.clip(restored, 0, 255).astype(np.uint8)
   
   e. Resize back to original crop size if needed:
      restored = cv2.resize(restored, (crop.shape[1], crop.shape[0]))
   
   Return restored

3. extract_eye_regions(self, crop: np.ndarray) -> dict:
   
   Use BiSeNet face parsing to get precise eye region masks:
   
   a. Run BiSeNet on the crop → 19-class segmentation
   
   b. Extract specific regions:
      left_eye_mask = (seg == 4).astype(np.float32)   # class 4
      right_eye_mask = (seg == 5).astype(np.float32)  # class 5
      left_brow_mask = (seg == 2).astype(np.float32)  # class 2
      right_brow_mask = (seg == 3).astype(np.float32) # class 3
   
   c. Create the "orbital zone" — the region AROUND the eyes that
      includes the eye socket, under-eye, brow ridge, and temples:
      
      # Combine eye + brow masks
      eye_brow = np.maximum(
          np.maximum(left_eye_mask, right_eye_mask),
          np.maximum(left_brow_mask, right_brow_mask)
      )
      
      # Dilate to create the orbital zone (eye socket area)
      orbital_zone = cv2.dilate(
          eye_brow.astype(np.uint8),
          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
      ).astype(np.float32)
      
      # Smooth the mask edges
      orbital_zone = cv2.GaussianBlur(orbital_zone, (15, 15), 5)
      orbital_zone = np.clip(orbital_zone, 0, 1)
   
   Return {
       'left_eye': left_eye_mask,
       'right_eye': right_eye_mask,
       'left_brow': left_brow_mask,
       'right_brow': right_brow_mask,
       'orbital_zone': orbital_zone,   # the full eye area with soft edges
       'eye_brow_tight': eye_brow      # just the eyes + brows, no padding
   }

4. apply_restoration_to_eyes_only(
       self,
       original_crop: np.ndarray,    # swapped crop (unrestored)
       restored_crop: np.ndarray,    # CodeFormer output
       eye_regions: dict
   ) -> np.ndarray:
   
   Apply CodeFormer restoration ONLY to the eye region, leaving
   the rest of the face untouched. This preserves the swap's
   identity in the cheeks/jaw/forehead while getting sharp eyes:
   
   a. Use the orbital_zone mask as a blend weight:
      mask = eye_regions['orbital_zone'][:, :, None]
      
      result = mask * restored_crop.astype(np.float64) + \
               (1 - mask) * original_crop.astype(np.float64)
   
   Return result.astype(np.uint8)

========================================================
PART 2: REGION-SPECIFIC COLOR MATCHING (Anti-Panda-Mask)
========================================================

5. match_orbital_colors(
       self,
       swapped_crop: np.ndarray,     # the swapped face
       original_crop: np.ndarray,    # performer's original aligned face
       eye_regions: dict
   ) -> np.ndarray:
   
   The "panda mask" happens because the source face's orbital
   pigmentation differs from the performer's. The source might have
   darker under-eyes while the performer has lighter skin there,
   creating a visible dark ring around the swapped eyes.
   
   a. Convert both to LAB:
      swap_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
      orig_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
   
   b. Define the "surrounding skin zone":
      # Skin around the orbital area but NOT the eyes themselves
      skin_mask = eye_regions['orbital_zone'].copy()
      tight_eye = eye_regions['eye_brow_tight']
      # Erode the tight eye mask slightly so we get nearby skin
      skin_mask = skin_mask * (1 - tight_eye)
      # This gives us the under-eye, temple, and brow ridge skin
   
   c. Sample under-eye / orbital skin colors from BOTH faces:
      swap_orbital_pixels = swap_lab[skin_mask > 0.3]
      orig_orbital_pixels = orig_lab[skin_mask > 0.3]
      
      if len(swap_orbital_pixels) < 30 or len(orig_orbital_pixels) < 30:
          return swapped_crop  # not enough pixels
      
      swap_mean = np.mean(swap_orbital_pixels, axis=0)
      swap_std = np.std(swap_orbital_pixels, axis=0) + 1e-6
      orig_mean = np.mean(orig_orbital_pixels, axis=0)
      orig_std = np.std(orig_orbital_pixels, axis=0) + 1e-6
   
   d. Apply color transfer ONLY in the orbital zone:
      # Reinhard transfer in the orbital zone
      for ch in range(3):
          channel = swap_lab[:, :, ch]
          normalized = (channel - swap_mean[ch]) / swap_std[ch]
          transferred = normalized * orig_std[ch] + orig_mean[ch]
          
          # Apply only in orbital zone with a soft mask
          weight = skin_mask * 0.7  # 70% correction strength
          channel_new = (1 - weight) * channel + weight * transferred
          swap_lab[:, :, ch] = channel_new
   
   e. Convert back:
      result = cv2.cvtColor(
          np.clip(swap_lab, 0, 255).astype(np.uint8),
          cv2.COLOR_LAB2BGR
      )
   
   Return result

========================================================
PART 3: GRADIENT EYE-ORBIT BLENDING
========================================================

6. blend_orbital_transition(
       self,
       swapped_crop: np.ndarray,     # color-matched swapped face
       original_crop: np.ndarray,    # performer's original face
       eye_regions: dict
   ) -> np.ndarray:
   
   Create a smooth gradient transition in the orbital zone.
   The inner eye area shows 100% swapped content, the outer
   orbital rim blends toward the original, eliminating the
   hard boundary around the eye socket.
   
   a. Create a graduated orbital mask:
      # Start with tight eye+brow mask
      inner = eye_regions['eye_brow_tight'].copy()
      
      # Create multiple dilated rings
      ring1 = cv2.dilate(inner.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
      ring2 = cv2.dilate(inner.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
      ring3 = cv2.dilate(inner.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(np.float32)
      
      # Build gradient: 1.0 at center → 0.0 at outer ring
      gradient = inner * 1.0
      gradient = np.maximum(gradient, (ring1 - inner) * 0.75)
      gradient = np.maximum(gradient, (ring2 - ring1) * 0.45)
      gradient = np.maximum(gradient, (ring3 - ring2) * 0.15)
      
      # Smooth the gradient
      gradient = cv2.GaussianBlur(gradient, (11, 11), 4)
      gradient = np.clip(gradient, 0, 1)
   
   b. Apply gradient blending:
      # This blends the ORBITAL TRANSITION zone only
      # The rest of the face mask handles the overall face boundary
      
      g = gradient[:, :, None]
      # In the gradient zone, bias toward swapped content at center
      # and toward original content at the edges
      result = g * swapped_crop.astype(np.float64) + \
               (1 - g) * original_crop.astype(np.float64)
   
   Return result.astype(np.uint8)

7. fix_sclera_color(
       self,
       swapped_crop: np.ndarray,
       original_crop: np.ndarray,
       eye_regions: dict
   ) -> np.ndarray:
   
   Match the eye white (sclera) color between swapped and original:
   
   a. Create a sclera mask:
      # Within the eye region, sclera is the BRIGHTEST area
      # that is NOT the iris (which is darker)
      for eye_key in ['left_eye', 'right_eye']:
          eye_mask = eye_regions[eye_key]
          if eye_mask.sum() < 10:
              continue
          
          # Convert eye region to grayscale
          gray = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
          # In the eye region, bright pixels = sclera
          eye_pixels = gray * eye_mask
          threshold = np.percentile(eye_pixels[eye_mask > 0.5], 65)
          sclera_mask = ((gray > threshold) * eye_mask).astype(np.float32)
          sclera_mask = cv2.GaussianBlur(sclera_mask, (5, 5), 1.5)
   
   b. Sample sclera color from original performer's eyes:
      orig_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
      swap_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
      
      orig_sclera = orig_lab[sclera_mask > 0.3]
      swap_sclera = swap_lab[sclera_mask > 0.3]
      
      if len(orig_sclera) > 10 and len(swap_sclera) > 10:
          # Subtle correction: shift sclera color toward performer's
          sclera_diff = np.mean(orig_sclera, axis=0) - np.mean(swap_sclera, axis=0)
          # Apply 50% of the difference (subtle, avoid over-correction)
          for ch in range(3):
              correction = sclera_diff[ch] * 0.5 * sclera_mask
              swap_lab[:, :, ch] += correction
          
          result = cv2.cvtColor(
              np.clip(swap_lab, 0, 255).astype(np.uint8),
              cv2.COLOR_LAB2BGR
          )
          return result
   
   Return swapped_crop  # no correction needed or possible

========================================================
MAIN METHOD
========================================================

8. fix(
       self,
       swapped_crop: np.ndarray,     # 128x128 from InSwapper + shape adapter
       original_crop: np.ndarray,    # 128x128 aligned original (performer)
   ) -> np.ndarray:
   
   THE MAIN METHOD — orchestrates all three parts:
   
   a. Extract eye regions (BiSeNet parsing):
      eye_regions = self.extract_eye_regions(swapped_crop)
   
   b. Part 1: Face restoration (eye detail reconstruction)
      if self.apply_to == 'eyes_only':
          restored = self.restore_face(swapped_crop, self.restore_strength)
          result = self.apply_restoration_to_eyes_only(
              swapped_crop, restored, eye_regions
          )
      elif self.apply_to == 'full_face':
          result = self.restore_face(swapped_crop, self.restore_strength)
      else:
          result = swapped_crop.copy()
   
   c. Part 2: Orbital color matching (anti-panda-mask)
      result = self.match_orbital_colors(result, original_crop, eye_regions)
   
   d. Part 3: Sclera color fix
      result = self.fix_sclera_color(result, original_crop, eye_regions)
   
   e. Part 4: Orbital transition blending
      result = self.blend_orbital_transition(result, original_crop, eye_regions)
   
   Return result

---

INTEGRATION INTO THE PIPELINE:

Update src/swap/face_swapper.py swap() method.
The eye fixer runs AFTER shape correction and BEFORE XSeg masking:

  # Step B: Generate swapped face
  swapped_crop = self.model.get(frame, face, source, paste_back=False)
  
  # Step B2: Shape correction
  swapped_crop = self.shape_adapter.adapt(aimg, swapped_crop)
  
  # Step B3: EYE REGION FIX (new!)
  swapped_crop = self.eye_fixer.fix(swapped_crop, aimg)
  
  # Step C: Occlusion mask (XSeg) — unchanged
  occlusion_mask = self.face_occluder.create_occlusion_mask(aimg)
  
  # ... rest of pipeline unchanged

IMPORTANT: The eye fixer works on the 128x128 aligned crop.
CodeFormer internally upscales to 512 for processing, then
downscales back. This means the eye detail is reconstructed at
512x512 resolution even though the final crop is 128x128 — the
detail survives downscaling because it was generated with proper
high-frequency content that LANCZOS preserves.

---

MODEL DOWNLOADS:

Add to models/download_models.py:

7. Download CodeFormer model:
   - codeformer_v0.1.0.pth (PyTorch version, ~376MB)
     From: https://github.com/sczhou/CodeFormer/releases
     Save to: models/codeformer_v0.1.0.pth
   
   - OR codeformer.onnx (ONNX version for faster inference)
     From: https://github.com/facefusion/facefusion-assets/releases
     Save to: models/codeformer.onnx

8. Download GFPGAN v1.4 as fallback:
   - GFPGANv1.4.pth (~332MB)
     From: https://github.com/TencentARC/GFPGAN/releases
     Save to: models/GFPGANv1.4.pth

---

CONFIG UPDATES:

Update config/default.yaml:

eye_fix:
  enabled: true
  restore_model: codeformer       # 'codeformer' or 'gfpgan'
  restore_strength: 0.6           # 0.0-1.0, CodeFormer fidelity
  apply_restoration_to: eyes_only # 'eyes_only' or 'full_face'
  upscale_before_restore: true
  orbital_color_match: true
  orbital_color_strength: 0.7
  sclera_correction: true
  orbital_blend: true

Update config/profiles/max_quality.yaml:
  eye_fix:
    restore_strength: 0.7
    apply_restoration_to: full_face  # restore entire face for max quality

Update config/profiles/balanced.yaml:
  eye_fix:
    restore_strength: 0.5
    apply_restoration_to: eyes_only
    upscale_before_restore: false    # skip upscale for speed

Update Gradio GUI:
  - Eye fix on/off toggle
  - Restore model dropdown (CodeFormer / GFPGAN)
  - Restore strength slider (0.0 to 1.0)
  - Apply to dropdown (eyes only / full face)
  - Orbital color match toggle + strength slider
  - Before/after comparison in preview

Update keyboard shortcuts:
  - 'e' = toggle eye fix on/off
  - 'r' = cycle restore model (CodeFormer → GFPGAN → off)

---

TESTING:

Include __main__ self-test that:
1. Opens webcam, runs full swap pipeline with eye fix
2. Shows 4-panel comparison:
   - Top-left: raw InSwapper output (no eye fix)
   - Top-right: with CodeFormer restoration only
   - Bottom-left: with orbital color matching only
   - Bottom-right: full pipeline (all parts)
3. Zoom window showing 4x magnified eye region for detail comparison
4. Print per-part timing:
   - CodeFormer: expected ~20-30ms at 512x512
   - Orbital color match: expected ~2ms
   - Sclera fix: expected ~1ms
   - Orbital blend: expected ~2ms
5. Trackbar for restore_strength

Create tests/test_eye_region.py:

1. test_codeformer_increases_eye_detail():
   - Compute Laplacian variance (sharpness) of eye region before/after
   - Assert: restored eye sharpness > 1.5x original
   
2. test_orbital_color_match_reduces_panda():
   - Create synthetic swap with 15-unit LAB difference in orbital zone
   - Run match_orbital_colors()
   - Assert: difference reduced to < 5 units

3. test_sclera_correction():
   - Create swap with yellowish sclera, original with white sclera
   - Run fix_sclera_color()
   - Assert: sclera color shifted toward white

4. test_identity_preserved_after_restoration():
   - Extract ArcFace embedding before and after eye fix
   - Cosine similarity must be > 0.85 (same person)

5. test_eyes_only_mode_doesnt_touch_cheeks():
   - Run with apply_to='eyes_only'
   - Assert: pixel values outside orbital zone unchanged

6. test_full_face_mode():
   - Run with apply_to='full_face'
   - Assert: entire face is enhanced (sharpness increased everywhere)

7. test_fallback_to_gfpgan():
   - Simulate CodeFormer failure (bad input)
   - Assert: GFPGAN produces valid output

8. test_performance():
   - Full eye fix pipeline on 128x128 crop
   - Assert: < 40ms total (CodeFormer ~25ms + rest ~10ms)
```