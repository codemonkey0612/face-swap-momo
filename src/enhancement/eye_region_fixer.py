"""Eye region transition fixer — three-part pipeline for natural eye quality.

Fixes the most visible artifacts in face swapping:
  1. Lost eye detail (iris, eyelashes, catchlights) from low-res swap
  2. "Panda mask" — orbital skin tone mismatch between source & performer
  3. Hard texture boundary around the eye socket / orbital bone

Runs on the ALIGNED crop AFTER shape correction, BEFORE XSeg masking and
final compositing.  Works in the same 256×256 (or whatever swap size)
coordinate system as the swap model.

Three-part pipeline:
  Part 1: Face restoration (CodeFormer ONNX → PyTorch → GFPGAN fallback)
  Part 2: Region-specific LAB color matching (anti-panda-mask)
  Part 3: Gradient eye-orbit blending + sclera color correction
"""

import os
from typing import Dict, Optional

import cv2
import numpy as np


class EyeRegionFixer:
    """Fixes eye region transitions after face swapping."""

    def __init__(self, config: dict, face_parser=None):
        """
        Args:
            config: dict with keys:
                restore_strength (float): CodeFormer fidelity 0.0-1.0 (default 0.6)
                apply_to (str): 'eyes_only' or 'full_face' (default 'eyes_only')
                upscale_before_restore (bool): upscale to 512 before restore (default True)
                orbital_color_match (bool): enable orbital color matching (default True)
                orbital_color_strength (float): 0.0-1.0 (default 0.7)
                sclera_correction (bool): enable sclera color fix (default True)
                orbital_blend (bool): enable gradient orbital blending (default True)
            face_parser: Existing FaceParser instance (reuses BiSeNet model).
                         If None, eye region extraction uses a simpler fallback.
        """
        self._restore_strength = config.get("restore_strength", 0.6)
        self._apply_to = config.get("apply_to", "eyes_only")
        self._upscale = config.get("upscale_before_restore", True)
        self._orbital_color = config.get("orbital_color_match", True)
        self._orbital_strength = config.get("orbital_color_strength", 0.7)
        self._sclera_fix = config.get("sclera_correction", True)
        self._orbital_blend = config.get("orbital_blend", True)
        self._parser = face_parser  # FaceParser from neural_swap.py

        # Restoration model (try ONNX CodeFormer → PyTorch CodeFormer → GFPGAN)
        self._restorer = None
        self._restorer_type = None
        self._init_restorer(config.get("restore_model", "codeformer"))

    # ------------------------------------------------------------------
    # Restoration model init
    # ------------------------------------------------------------------

    def _init_restorer(self, model_pref: str) -> None:
        """Load face restoration model with fallback chain."""
        if model_pref == "codeformer":
            if self._try_codeformer_onnx():
                return
            if self._try_codeformer_torch():
                return
            # Fall through to GFPGAN
        if self._try_gfpgan():
            return
        print("[EyeRegionFixer] No restoration model available — "
              "eye detail reconstruction disabled")

    def _try_codeformer_onnx(self) -> bool:
        path = "models/codeformer.onnx"
        if not os.path.exists(path):
            return False
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(path, providers=providers)
            inp_names = [i.name for i in sess.get_inputs()]
            self._cf_onnx_sess = sess
            self._cf_onnx_inputs = inp_names
            self._cf_onnx_output = sess.get_outputs()[0].name
            self._restorer = "codeformer_onnx"
            self._restorer_type = "codeformer_onnx"
            print(f"[EyeRegionFixer] CodeFormer ONNX loaded (fidelity={self._restore_strength})")
            return True
        except Exception as e:
            print(f"[EyeRegionFixer] CodeFormer ONNX failed: {e}")
            return False

    def _try_codeformer_torch(self) -> bool:
        path = "models/codeformer.pth"
        if not os.path.exists(path):
            return False
        try:
            import torch
            CF = None
            for mod in ("codeformer.basicsr.archs.codeformer_arch",
                        "basicsr.archs.codeformer_arch"):
                try:
                    m = __import__(mod, fromlist=["CodeFormer"])
                    CF = getattr(m, "CodeFormer")
                    break
                except (ImportError, AttributeError):
                    continue
            if CF is None:
                return False
            net = CF(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=["32", "64", "128", "256"])
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(path, map_location=dev, weights_only=False)
            net.load_state_dict(state.get("params_ema", state))
            net.eval().to(dev)
            self._cf_torch_net = net
            self._cf_torch_dev = dev
            self._restorer = "codeformer_torch"
            self._restorer_type = "codeformer_torch"
            print(f"[EyeRegionFixer] CodeFormer PyTorch on {dev} (fidelity={self._restore_strength})")
            return True
        except Exception as e:
            print(f"[EyeRegionFixer] CodeFormer PyTorch failed: {e}")
            return False

    def _try_gfpgan(self) -> bool:
        path = "models/GFPGANv1.4.pth"
        if not os.path.exists(path):
            return False
        try:
            import torch
            from gfpgan import GFPGANer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._gfpgan = GFPGANer(
                model_path=path, upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None,
                device=torch.device(dev),
            )
            self._restorer = "gfpgan"
            self._restorer_type = "gfpgan"
            print(f"[EyeRegionFixer] GFPGAN v1.4 on {dev}")
            return True
        except Exception as e:
            print(f"[EyeRegionFixer] GFPGAN failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Part 1: Face restoration
    # ------------------------------------------------------------------

    def restore_face(self, crop: np.ndarray, fidelity: float) -> Optional[np.ndarray]:
        """Run face restoration on the aligned crop.

        Returns restored crop at same size as input, or None on failure.
        """
        if self._restorer is None:
            return None

        h, w = crop.shape[:2]

        # Upscale to 512 for better detail reconstruction
        if self._upscale:
            work = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        else:
            work = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)

        restored = None
        if self._restorer_type == "codeformer_onnx":
            restored = self._cf_onnx_infer(work, fidelity)
        elif self._restorer_type == "codeformer_torch":
            restored = self._cf_torch_infer(work, fidelity)
        elif self._restorer_type == "gfpgan":
            restored = self._gfpgan_infer(work)

        if restored is None:
            return None

        # Resize back to original crop size
        if restored.shape[0] != h or restored.shape[1] != w:
            restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return restored

    def _cf_onnx_infer(self, crop_512: np.ndarray, fidelity: float) -> Optional[np.ndarray]:
        try:
            # BGR → RGB, normalize to [-1, 1], NCHW
            rgb = crop_512[:, :, ::-1].astype(np.float32) / 255.0
            t = (rgb - 0.5) / 0.5
            t = np.ascontiguousarray(t.transpose(2, 0, 1)[np.newaxis])

            feed = {self._cf_onnx_inputs[0]: t}
            # CodeFormer ONNX has a scalar 'weight' input (fidelity parameter)
            if len(self._cf_onnx_inputs) > 1:
                feed[self._cf_onnx_inputs[1]] = np.array(fidelity, dtype=np.float64)

            out = self._cf_onnx_sess.run([self._cf_onnx_output], feed)[0]
            out = out.squeeze()
            if out.ndim == 3 and out.shape[0] == 3:
                out = out.transpose(1, 2, 0)  # CHW → HWC
            out = (out * 0.5 + 0.5) * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8)
            # RGB → BGR
            return out[:, :, ::-1].copy()
        except Exception:
            return None

    def _cf_torch_infer(self, crop_512: np.ndarray, fidelity: float) -> Optional[np.ndarray]:
        try:
            import torch
            rgb = crop_512[:, :, ::-1].copy().astype(np.float32) / 255.0
            t = (rgb - 0.5) / 0.5
            t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).to(self._cf_torch_dev)
            with torch.no_grad():
                o = self._cf_torch_net(t, w=fidelity, adain=True)[0]
            o = o.squeeze(0).cpu().clamp(-1, 1).numpy()
            o = ((o + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
            return o[:, :, ::-1].copy()
        except Exception:
            return None

    def _gfpgan_infer(self, crop_512: np.ndarray) -> Optional[np.ndarray]:
        try:
            _, _, enhanced = self._gfpgan.enhance(
                crop_512, has_aligned=True, only_center_face=True, paste_back=False
            )
            if enhanced is not None and enhanced.ndim == 3:
                return enhanced
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Eye region extraction (BiSeNet or fallback)
    # ------------------------------------------------------------------

    def extract_eye_regions(self, crop: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract eye-related region masks from the aligned crop.

        Uses BiSeNet face parser if available, otherwise falls back to
        a simpler landmark-free heuristic.

        Returns dict with keys:
            left_eye, right_eye, left_brow, right_brow,
            orbital_zone (dilated soft mask), eye_brow_tight
        """
        h, w = crop.shape[:2]

        if self._parser is not None:
            return self._extract_via_bisenet(crop, h, w)
        return self._extract_fallback(crop, h, w)

    def _extract_via_bisenet(self, crop: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
        """Use BiSeNet class probabilities for precise eye regions."""
        probs = self._parser._infer(crop)  # (C, H, W) softmax

        # Resize each class probability to crop dimensions
        def _cls(c):
            p = probs[c]
            if p.shape != (h, w):
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            return np.clip(p, 0, 1).astype(np.float32)

        left_eye = _cls(4)
        right_eye = _cls(5)
        left_brow = _cls(2)
        right_brow = _cls(3)

        # Tight eye+brow mask
        eye_brow = np.maximum(
            np.maximum(left_eye, right_eye),
            np.maximum(left_brow, right_brow),
        )

        # Dilated orbital zone with soft edges
        eb_u8 = (eye_brow > 0.3).astype(np.uint8)
        # Scale kernel size relative to crop size
        k_size = max(15, int(h * 0.1)) | 1
        orbital = cv2.dilate(
            eb_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)),
        ).astype(np.float32)
        blur_k = max(11, int(h * 0.06)) | 1
        orbital = cv2.GaussianBlur(orbital, (blur_k, blur_k), blur_k / 3.0)
        orbital = np.clip(orbital, 0, 1)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "orbital_zone": orbital,
            "eye_brow_tight": eye_brow,
        }

    def _extract_fallback(self, crop: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
        """Heuristic fallback when BiSeNet is not available.

        Assumes standard aligned face geometry: eyes are roughly in the
        upper-middle region of the 256×256 crop.
        """
        # Approximate eye regions for a standard aligned face crop
        # Eyes are roughly at y=35-45%, x=25-45% (left) and x=55-75% (right)
        le = np.zeros((h, w), dtype=np.float32)
        re = np.zeros((h, w), dtype=np.float32)
        lb = np.zeros((h, w), dtype=np.float32)
        rb = np.zeros((h, w), dtype=np.float32)

        ey_top, ey_bot = int(h * 0.35), int(h * 0.48)
        le_l, le_r = int(w * 0.22), int(w * 0.45)
        re_l, re_r = int(w * 0.55), int(w * 0.78)
        brow_top, brow_bot = int(h * 0.28), int(h * 0.37)

        le[ey_top:ey_bot, le_l:le_r] = 1.0
        re[ey_top:ey_bot, re_l:re_r] = 1.0
        lb[brow_top:brow_bot, le_l:le_r] = 1.0
        rb[brow_top:brow_bot, re_l:re_r] = 1.0

        # Smooth
        k = max(7, int(h * 0.04)) | 1
        for m in (le, re, lb, rb):
            m[:] = cv2.GaussianBlur(m, (k, k), k / 3.0)

        eye_brow = np.maximum(np.maximum(le, re), np.maximum(lb, rb))
        eb_u8 = (eye_brow > 0.3).astype(np.uint8)
        k_size = max(15, int(h * 0.1)) | 1
        orbital = cv2.dilate(
            eb_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)),
        ).astype(np.float32)
        blur_k = max(11, int(h * 0.06)) | 1
        orbital = cv2.GaussianBlur(orbital, (blur_k, blur_k), blur_k / 3.0)
        orbital = np.clip(orbital, 0, 1)

        return {
            "left_eye": le,
            "right_eye": re,
            "left_brow": lb,
            "right_brow": rb,
            "orbital_zone": orbital,
            "eye_brow_tight": eye_brow,
        }

    # ------------------------------------------------------------------
    # Part 1 helper: apply restoration to eyes only
    # ------------------------------------------------------------------

    def apply_restoration_to_eyes_only(
        self,
        original_crop: np.ndarray,
        restored_crop: np.ndarray,
        eye_regions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Blend CodeFormer output into only the orbital zone."""
        mask = eye_regions["orbital_zone"][:, :, np.newaxis]
        result = (
            mask * restored_crop.astype(np.float64)
            + (1.0 - mask) * original_crop.astype(np.float64)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Part 2: Orbital color matching (anti-panda-mask)
    # ------------------------------------------------------------------

    def match_orbital_colors(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
        eye_regions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Reinhard LAB color transfer in the orbital skin zone only."""
        if not self._orbital_color:
            return swapped_crop

        # Orbital skin = orbital_zone minus tight eye+brow area
        skin_mask = eye_regions["orbital_zone"].copy()
        tight = eye_regions["eye_brow_tight"]
        skin_mask = skin_mask * (1.0 - np.clip(tight, 0, 1))

        swap_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
        orig_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float64)

        threshold = 0.3
        swap_pixels = swap_lab[skin_mask > threshold]
        orig_pixels = orig_lab[skin_mask > threshold]

        if len(swap_pixels) < 30 or len(orig_pixels) < 30:
            return swapped_crop

        swap_mean = np.mean(swap_pixels, axis=0)
        swap_std = np.std(swap_pixels, axis=0) + 1e-6
        orig_mean = np.mean(orig_pixels, axis=0)
        orig_std = np.std(orig_pixels, axis=0) + 1e-6

        strength = self._orbital_strength
        result_lab = swap_lab.copy()
        for ch in range(3):
            channel = result_lab[:, :, ch]
            normalized = (channel - swap_mean[ch]) / swap_std[ch]
            transferred = normalized * orig_std[ch] + orig_mean[ch]
            weight = skin_mask * strength
            result_lab[:, :, ch] = (1.0 - weight) * channel + weight * transferred

        result = cv2.cvtColor(
            np.clip(result_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR,
        )
        return result

    # ------------------------------------------------------------------
    # Part 3a: Sclera (eye white) color correction
    # ------------------------------------------------------------------

    def fix_sclera_color(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
        eye_regions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Subtle sclera color shift toward performer's eye whites."""
        if not self._sclera_fix:
            return swapped_crop

        swap_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
        orig_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB).astype(np.float64)
        gray = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        combined_sclera = np.zeros_like(gray)

        for eye_key in ("left_eye", "right_eye"):
            eye_mask = eye_regions[eye_key]
            if eye_mask.sum() < 10:
                continue

            eye_pixels = gray * eye_mask
            valid = eye_mask > 0.5
            if valid.sum() < 10:
                continue

            # Sclera = bright pixels within eye region (above 65th percentile)
            threshold = np.percentile(eye_pixels[valid], 65)
            sclera = ((gray > threshold) * eye_mask).astype(np.float32)
            k = max(3, int(eye_mask.shape[0] * 0.02)) | 1
            sclera = cv2.GaussianBlur(sclera, (k, k), k / 3.0)
            combined_sclera = np.maximum(combined_sclera, sclera)

        if combined_sclera.sum() < 5:
            return swapped_crop

        orig_sclera_px = orig_lab[combined_sclera > 0.3]
        swap_sclera_px = swap_lab[combined_sclera > 0.3]

        if len(orig_sclera_px) < 10 or len(swap_sclera_px) < 10:
            return swapped_crop

        sclera_diff = np.mean(orig_sclera_px, axis=0) - np.mean(swap_sclera_px, axis=0)

        result_lab = swap_lab.copy()
        for ch in range(3):
            correction = sclera_diff[ch] * 0.5 * combined_sclera
            result_lab[:, :, ch] += correction

        return cv2.cvtColor(
            np.clip(result_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR,
        )

    # ------------------------------------------------------------------
    # Part 3b: Gradient eye-orbit blending
    # ------------------------------------------------------------------

    def blend_orbital_transition(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
        eye_regions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Gradient blend: inner eye area = 100% swapped, outer rim → original."""
        if not self._orbital_blend:
            return swapped_crop

        inner = eye_regions["eye_brow_tight"].copy()
        inner_u8 = (inner > 0.3).astype(np.uint8)
        h = inner.shape[0]

        # Create concentric dilated rings with decreasing blend strength
        def _dilate(src, px):
            k = max(3, px) | 1
            return cv2.dilate(
                src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            ).astype(np.float32)

        r1 = _dilate(inner_u8, max(5, int(h * 0.035)))
        r2 = _dilate(inner_u8, max(9, int(h * 0.065)))
        r3 = _dilate(inner_u8, max(13, int(h * 0.10)))

        inner_f = inner.copy()
        gradient = inner_f * 1.0
        gradient = np.maximum(gradient, (r1 - inner_f) * 0.75)
        gradient = np.maximum(gradient, (r2 - r1) * 0.45)
        gradient = np.maximum(gradient, (r3 - r2) * 0.15)

        blur_k = max(7, int(h * 0.04)) | 1
        gradient = cv2.GaussianBlur(gradient, (blur_k, blur_k), blur_k / 3.0)
        gradient = np.clip(gradient, 0, 1)

        g = gradient[:, :, np.newaxis]
        result = (
            g * swapped_crop.astype(np.float64)
            + (1.0 - g) * original_crop.astype(np.float64)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fix(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
    ) -> np.ndarray:
        """Run the full three-part eye fix pipeline.

        Args:
            swapped_crop: Aligned face crop after swap + shape correction.
            original_crop: Aligned original (performer) face crop.

        Returns:
            Fixed crop, same size and dtype as swapped_crop.
        """
        # Extract eye regions
        eye_regions = self.extract_eye_regions(swapped_crop)

        # Part 1: Face restoration (CodeFormer / GFPGAN)
        result = swapped_crop
        if self._restorer is not None:
            restored = self.restore_face(swapped_crop, self._restore_strength)
            if restored is not None:
                if self._apply_to == "eyes_only":
                    result = self.apply_restoration_to_eyes_only(
                        swapped_crop, restored, eye_regions
                    )
                else:  # full_face
                    result = restored

        # Part 2: Orbital color matching (anti-panda-mask)
        result = self.match_orbital_colors(result, original_crop, eye_regions)

        # Part 3a: Sclera color fix
        result = self.fix_sclera_color(result, original_crop, eye_regions)

        # Part 3b: Orbital transition blending
        result = self.blend_orbital_transition(result, original_crop, eye_regions)

        return result


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    # Quick smoke test with a static image
    cfg = {
        "restore_strength": 0.6,
        "apply_to": "eyes_only",
        "upscale_before_restore": True,
        "orbital_color_match": True,
        "orbital_color_strength": 0.7,
        "sclera_correction": True,
        "orbital_blend": True,
        "restore_model": "codeformer",
    }

    fixer = EyeRegionFixer(cfg)

    # Try webcam demo
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No webcam — testing with synthetic image")
        img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        out = fixer.fix(img, img)
        dt = (time.perf_counter() - t0) * 1000
        print(f"fix() on 256×256: {dt:.1f} ms")
        print(f"Input shape: {img.shape}  Output shape: {out.shape}")
        sys.exit(0)

    print("Webcam opened — press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use center crop as fake aligned face
        h, w = frame.shape[:2]
        sz = min(h, w)
        y0, x0 = (h - sz) // 2, (w - sz) // 2
        crop = cv2.resize(frame[y0:y0+sz, x0:x0+sz], (256, 256))

        t0 = time.perf_counter()
        fixed = fixer.fix(crop, crop)
        dt = (time.perf_counter() - t0) * 1000

        # Side-by-side
        vis = np.hstack([crop, fixed])
        cv2.putText(vis, f"{dt:.1f}ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("EyeRegionFixer: original | fixed", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
