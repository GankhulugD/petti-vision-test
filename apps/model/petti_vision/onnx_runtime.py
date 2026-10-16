"""Torch-free inference: DINOv2 and SigLIP2 vision towers as (INT8) ONNX models.

Produced by training/export_onnx.py into artifacts/onnx/:
  dinov2_core.onnx   pixels (B,3,H,W) + position table (1,1+N,384) → hidden (B,1+N,384)
                     (any H, W that are multiples of 14: one file for re-ID at 224
                     and for the segmenter at 448)
  siglip_vision.onnx pixels (B,3,224,224) → image embedding (B,768)
  meta.npz           DINOv2 base position embedding, SigLIP logit scale/bias,
                     cached text embeddings (text → vector) for runtime prompts

The position embedding is interpolated here in numpy with the same bicubic
kernel PyTorch uses (a = -0.75, align_corners=False), so the ONNX graph has no
data-dependent shapes.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .backbones import l2n

log = logging.getLogger("petti.onnx")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# PyTorch-compatible bicubic interpolation (separable, a = -0.75)
# --------------------------------------------------------------------------- #
def _cubic_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    def k(x):
        x = np.abs(x)
        return np.where(x <= 1, (a + 2) * x**3 - (a + 3) * x**2 + 1, np.where(x < 2, a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a, 0.0))

    return np.stack([k(t + 1), k(t), k(1 - t), k(2 - t)], axis=-1)


def _resize_axis(x: np.ndarray, out: int, axis: int, scale: float | None = None) -> np.ndarray:
    n = x.shape[axis]
    s = scale if scale is not None else n / out
    src = (np.arange(out) + 0.5) * s - 0.5
    i0 = np.floor(src).astype(int)
    w = _cubic_weights(src - i0)  # (out, 4)
    idx = np.clip(i0[:, None] + np.arange(-1, 3)[None, :], 0, n - 1)  # (out, 4)
    xm = np.moveaxis(x, axis, -1)
    y = (xm[..., idx] * w).sum(-1)
    return np.moveaxis(y, -1, axis)


def bicubic(x: np.ndarray, out_h: int, out_w: int, scale_h: float | None = None, scale_w: float | None = None) -> np.ndarray:
    """x (..., H, W) → (..., out_h, out_w), torch.nn.functional.interpolate(mode='bicubic', align_corners=False)."""
    return _resize_axis(_resize_axis(x, out_h, x.ndim - 2, scale_h), out_w, x.ndim - 1, scale_w)


def interpolate_pos(base: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """base (1, 1+S*S, C) DINOv2 position embedding → (1, 1+gh*gw, C), as HF Dinov2 does."""
    cls, patch = base[:, :1], base[:, 1:]
    s = int(round(patch.shape[1] ** 0.5))
    if gh == s and gw == s:
        return base
    grid = patch.reshape(1, s, s, -1).transpose(0, 3, 1, 2).astype(np.float64)  # (1,C,s,s)
    out = bicubic(grid, gh, gw).astype(np.float32)
    return np.concatenate([cls, out.transpose(0, 2, 3, 1).reshape(1, gh * gw, -1)], axis=1)


# --------------------------------------------------------------------------- #
# preprocessing identical to the HF processors used at training time
# --------------------------------------------------------------------------- #
def dino_preprocess(im: Image.Image, short: int = 256, crop: int = 224) -> np.ndarray:
    """DINOv2 processor: shortest edge → 256 (bicubic), centre crop 224, ImageNet norm."""
    w, h = im.size  # same rounding as transformers' get_resize_output_image_size
    if w <= h:
        nw, nh = short, int(short * h / w)
    else:
        nw, nh = int(short * w / h), short
    r = im.convert("RGB").resize((nw, nh), Image.BICUBIC)
    l, t = (nw - crop) // 2, (nh - crop) // 2
    arr = np.asarray(r.crop((l, t, l + crop, t + crop)), dtype=np.float32) / 255.0
    return ((arr - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def siglip_preprocess(im: Image.Image, size: int = 224) -> np.ndarray:
    """SigLIP processor: resize to 224×224 (bilinear), scale to [-1, 1]."""
    arr = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((arr - 0.5) / 0.5).transpose(2, 0, 1)


class OnnxEncoders:
    """Drop-in replacement for HFEncoders without torch/transformers."""

    torch = None
    device = "cpu"

    def __init__(self, cfg, onnx_dir: Path | None = None):
        import onnxruntime as ort

        self.cfg = cfg
        d = onnx_dir or (cfg.artifacts_dir / "onnx")
        t0 = time.perf_counter()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.dino = ort.InferenceSession(str(d / "dinov2_core.onnx"), so, providers=prov)
        self.sig = ort.InferenceSession(str(d / "siglip_vision.onnx"), so, providers=prov)
        meta = np.load(d / "meta.npz", allow_pickle=False)
        self.pos_base = meta["dino_pos"].astype(np.float32)
        self.logit_scale = float(meta["logit_scale"])
        self.logit_bias = float(meta["logit_bias"])
        self._texts = {str(t): e for t, e in zip(meta["texts"], meta["text_emb"])}
        self.reid_proj = None
        if cfg.reid_proj_path.exists():
            from .heads import load_reid_projection

            self.reid_proj = load_reid_projection(cfg.reid_proj_path)
        log.info("ONNX encoders loaded in %.1fs from %s", time.perf_counter() - t0, d)

    @lru_cache(maxsize=64)
    def _pos(self, gh: int, gw: int) -> np.ndarray:
        return interpolate_pos(self.pos_base, gh, gw)

    def dino_hidden(self, pixel: np.ndarray) -> np.ndarray:
        """pixel (B,3,H,W) normalised → last hidden state (B, 1+N, 384)."""
        gh, gw = pixel.shape[2] // 14, pixel.shape[3] // 14
        return self.dino.run(None, {"pixel": pixel.astype(np.float32), "pos": self._pos(gh, gw)})[0]

    def encode_images(self, crops: list[Image.Image], tta: bool = True, masks: list | None = None) -> dict[str, np.ndarray]:
        imgs = list(crops)
        if tta:
            imgs += [im.transpose(Image.FLIP_LEFT_RIGHT) for im in crops]
        n = len(crops)
        hs = self.dino_hidden(np.stack([dino_preprocess(im) for im in imgs]))
        cls = hs[:, 0]
        pt = hs[:n, 1:]
        ptn = pt / np.linalg.norm(pt, axis=-1, keepdims=True).clip(1e-6)
        attn = np.einsum("bnc,bc->bn", ptn, l2n(cls[:n]))
        reid = cls if self.cfg.reid_pool == "cls" else np.concatenate([cls, hs[:, 1:].mean(1)], axis=-1)
        vlm = self.sig.run(None, {"pixel": np.stack([siglip_preprocess(im) for im in imgs])})[0]
        if tta:
            reid = reid[:n] + reid[n:]
            vlm = l2n(vlm[:n]) + l2n(vlm[n:])
        reid = l2n(reid)
        if self.reid_proj is not None:
            reid = l2n(self.reid_proj(reid))
        return {"reid": reid, "vlm": l2n(vlm), "attn": attn, "attn_frac": 224 / 256}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t.lower() not in self._texts]
        if missing:
            raise KeyError(f"text embeddings not exported for {len(missing)} prompts, e.g. {missing[:2]} — re-run training/export_onnx.py")
        return l2n(np.stack([self._texts[t.lower()] for t in texts]))
