"""Pretrained building blocks: animal detector + two image encoders.

    image ──► Detector (YOLO, ~5-15ms) ──► animal crop
                                     ├──► ReID encoder  (DINOv2)  → identity vector
                                     └──► VLM encoder   (SigLIP2) → attribute vector
                                                                  (+ text tower for zero-shot)

All encoders return L2-normalised float32 numpy arrays so the rest of the
system (heads, index, matcher) is framework-agnostic.

`Fake*` classes give deterministic numpy-only stand-ins so that tests and UI
development run without torch or model downloads (PETTI_FAKE_MODELS=1).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .config import Config, resolve_device
from .taxonomy import COCO_TO_SPECIES

log = logging.getLogger("petti.backbones")

try:  # iPhone HEIC/HEIF photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1,y1,x2,y2 in pixels
    species: str
    confidence: float
    polygon: list[tuple[float, float]] | None = None  # segmentation outline (image coords)
    mask_small: np.ndarray | None = None  # patch-grid foreground, stretched onto the image

    def mask(self, size: tuple[int, int]) -> np.ndarray | None:
        """Boolean animal mask for an image of `size` (w,h)."""
        if self.mask_small is not None and self.mask_small.any():
            im = Image.fromarray((self.mask_small.astype(np.uint8) * 255)).resize(size, Image.NEAREST)
            return np.asarray(im) > 127
        if not self.polygon or len(self.polygon) < 3:
            return None
        from PIL import ImageDraw

        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).polygon(self.polygon, fill=255)
        return np.asarray(m) > 0

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_image(data: bytes | str | Image.Image, max_side: int = 1280) -> Image.Image:
    """Open, fix EXIF rotation (phone photos!), convert RGB, cap resolution."""
    import io

    if isinstance(data, Image.Image):
        img = data
    elif isinstance(data, (bytes, bytearray)):
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(data)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    return img


def crop_box(img: Image.Image, det: Detection | None, pad: float = 0.08) -> tuple[int, int, int, int]:
    """Padded detection box clipped to the image (the whole image without a detection)."""
    if det is None:
        return 0, 0, img.width, img.height
    x1, y1, x2, y2 = det.box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)),
            int(min(img.width, x2 + pad * w)), int(min(img.height, y2 + pad * h)))


def crop_detection(img: Image.Image, det: Detection | None, pad: float = 0.08) -> Image.Image:
    if det is None:
        return img
    return img.crop(crop_box(img, det, pad))


# =========================================================================== #
# Detector
# =========================================================================== #
class YoloDetector:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = None
        self.name = "yolo"
        # segmentation models give an animal mask → colours measured on the
        # animal only (not gravel/grass/walls). Fall back gracefully.
        for weights in dict.fromkeys([cfg.detector_weights, "yolo26n-seg.pt", "yolo11n-seg.pt", "yolo11n.pt"]):
            try:
                self.model = YOLO(weights)
                self.name = weights
                log.info("detector: %s", weights)
                break
            except Exception as exc:
                log.warning("could not load %s (%s)", weights, exc)
        if self.model is None:
            raise RuntimeError("no YOLO detector could be loaded")
        self.device = resolve_device(cfg.device)
        self.classes = sorted(COCO_TO_SPECIES)

    def detect(self, images: list[Image.Image]) -> list[list[Detection]]:
        kw = {"half": True} if (self.cfg.half and self.device == "cuda") else {}
        results = self.model.predict(
            source=images,
            conf=self.cfg.detector_conf,
            imgsz=self.cfg.detector_imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
            **kw,
        )
        out: list[list[Detection]] = []
        for r in results:
            dets = []
            if r.boxes is not None:
                polys = r.masks.xy if getattr(r, "masks", None) is not None else None
                for i, b in enumerate(r.boxes):
                    cls = int(b.cls.item())
                    poly = None
                    if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                        poly = [(float(x), float(y)) for x, y in polys[i]]
                    dets.append(
                        Detection(
                            box=tuple(float(v) for v in b.xyxy[0].tolist()),
                            species=COCO_TO_SPECIES.get(cls, "other"),
                            confidence=float(b.conf.item()),
                            polygon=poly,
                        )
                    )
            dets.sort(key=lambda d: d.confidence * (d.area() ** 0.5), reverse=True)
            out.append(dets)
        return out


# =========================================================================== #
# Encoders (HuggingFace transformers)
# =========================================================================== #
def _load(cls, name, dtype):
    try:  # transformers >= 4.56 uses `dtype`, older versions `torch_dtype`
        return cls.from_pretrained(name, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype)


def _features(out):
    """get_*_features returns a Tensor (transformers 4.x) or a ModelOutput (5.x)."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[1] if len(out) > 1 else out[0]
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, name, None)
        if v is not None:
            return v
    raise TypeError(f"unexpected feature output {type(out)}")


class HFEncoders:
    """DINOv2 (identity) + SigLIP2 (attributes, zero-shot) in one object."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        self.torch = torch
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        # fp16 only on CUDA: on Apple MPS fp16 transformer inference can silently
        # degrade embeddings (→ near-random zero-shot results). fp32 is safe there.
        self.dtype = torch.float16 if (cfg.half and self.device == "cuda") else torch.float32

        t0 = time.perf_counter()
        self.reid_proc = AutoImageProcessor.from_pretrained(cfg.reid_model)
        self.reid = _load(AutoModel, cfg.reid_model, self.dtype).to(self.device).eval()
        self.vlm_proc = AutoProcessor.from_pretrained(cfg.vlm_model)
        self.vlm = _load(AutoModel, cfg.vlm_model, self.dtype).to(self.device).eval()
        log.info("encoders loaded in %.1fs on %s/%s", time.perf_counter() - t0, self.device, self.dtype)

        # SigLIP calibration constants (sigmoid(logit_scale * cos + bias))
        self.logit_scale = float(self.vlm.logit_scale.exp().item()) if hasattr(self.vlm, "logit_scale") else 100.0
        self.logit_bias = float(self.vlm.logit_bias.item()) if hasattr(self.vlm, "logit_bias") else 0.0

        self.reid_proj = None  # optional learned projection (training/train_reid.py)
        if cfg.reid_proj_path.exists():
            from .heads import load_reid_projection

            self.reid_proj = load_reid_projection(cfg.reid_proj_path)
            log.info("loaded learned re-ID projection %s", cfg.reid_proj_path)

    def _to(self, batch):
        return {k: (v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)) for k, v in batch.items()}

    def _patch_weights(self, masks: list[np.ndarray | None], grid: int):
        """Square crop masks → (B, grid*grid) weights on DINOv2's patch grid,
        replicating the processor's resize + centre-crop geometry."""
        size = getattr(self.reid_proc, "size", {}) or {}
        crop = getattr(self.reid_proc, "crop_size", {}) or {}
        S = size.get("shortest_edge") or size.get("height") or 224
        C = crop.get("height", S) if getattr(self.reid_proc, "do_center_crop", False) else S
        off = (S - C) // 2
        out = []
        for m in masks:
            if m is None or not m.any():
                out.append(None)
                continue
            mi = Image.fromarray(m.astype(np.uint8) * 255).resize((S, S), Image.BILINEAR).crop((off, off, off + C, off + C))
            w = np.asarray(mi.resize((grid, grid), Image.BOX), dtype=np.float32).ravel() / 255.0
            out.append(w if w.sum() >= 3 else None)  # too few animal patches → plain mean
        return out

    def encode_images(self, crops: list[Image.Image], tta: bool = True, masks: list | None = None) -> dict[str, np.ndarray]:
        """masks: optional per-crop bool arrays (same size as crop). With
        PETTI_REID_POOL=masked the identity descriptor averages only patches on
        the animal, so the background (grass, sofa, street) cannot make two
        different dogs look alike or the same dog look different."""
        torch = self.torch
        imgs = list(crops)
        mks = list(masks) if masks is not None else [None] * len(crops)
        if tta:
            imgs += [ImageOps.mirror(c) for c in crops]
            mks += [None if m is None else m[:, ::-1] for m in mks]
        n = len(crops)
        with torch.inference_mode():
            r_in = self._to(self.reid_proc(images=imgs, return_tensors="pt"))
            r_out = self.reid(**r_in)
            hs = r_out.last_hidden_state
            cls = hs[:, 0]
            n_reg0 = int(getattr(self.reid.config, "num_register_tokens", 0) or 0)
            pt = torch.nn.functional.normalize(hs[:n, 1 + n_reg0 :].float(), dim=-1)
            # "where the identity model looks": similarity of every patch to the CLS token
            attn = (pt @ torch.nn.functional.normalize(cls[:n].float(), dim=-1).unsqueeze(-1)).squeeze(-1).cpu().numpy()
            n_reg = int(getattr(self.reid.config, "num_register_tokens", 0) or 0)
            patches = hs[:, 1 + n_reg :]
            patch = patches.mean(1)
            if self.cfg.reid_pool in ("masked", "masked_only", "masked_norm"):
                grid = int(round(patches.shape[1] ** 0.5))
                ws = self._patch_weights(mks, grid) if grid * grid == patches.shape[1] else [None] * len(mks)
                rows = []
                for i, w in enumerate(ws):
                    if w is None:
                        rows.append(patch[i])
                    else:
                        wt = torch.tensor(w, device=patches.device, dtype=patches.dtype)
                        rows.append((patches[i] * wt[:, None]).sum(0) / wt.sum())
                patch = torch.stack(rows)
            # CLS token + (animal-)mean patch token = robust global descriptor
            if self.cfg.reid_pool == "cls":
                reid = cls.float().cpu().numpy()
            elif self.cfg.reid_pool == "masked_only":
                reid = patch.float().cpu().numpy()
            else:
                reid = torch.cat([torch.nn.functional.normalize(cls, dim=-1), torch.nn.functional.normalize(patch, dim=-1)], dim=-1).float().cpu().numpy() if self.cfg.reid_pool == "masked_norm" else torch.cat([cls, patch], dim=-1).float().cpu().numpy()

            v_in = self._to(self.vlm_proc(images=imgs, return_tensors="pt"))
            vlm = _features(self.vlm.get_image_features(**v_in)).float().cpu().numpy()

        if tta:
            reid = reid[:n] + reid[n:]
            vlm = l2n(vlm[:n]) + l2n(vlm[n:])
        reid = l2n(reid)
        if self.reid_proj is not None:
            reid = l2n(self.reid_proj(reid))
        size = getattr(self.reid_proc, "size", {}) or {}
        crop = getattr(self.reid_proc, "crop_size", {}) or {}
        S = size.get("shortest_edge") or size.get("height") or 224
        C = crop.get("height", S) if getattr(self.reid_proc, "do_center_crop", False) else S
        return {"reid": reid, "vlm": l2n(vlm), "attn": attn, "attn_frac": C / S}

    def dino_hidden(self, pixel: np.ndarray) -> np.ndarray:
        """pixel (B,3,H,W) normalised → DINOv2 last hidden state (B, 1+N, C), registers dropped."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(pixel, dtype=np.float32)).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            hs = self.reid(pixel_values=x).last_hidden_state
        n_reg = int(getattr(self.reid.config, "num_register_tokens", 0) or 0)
        if n_reg:
            hs = torch.cat([hs[:, :1], hs[:, 1 + n_reg :]], 1)
        return hs.float().cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 256):
                t_in = self.vlm_proc(
                    text=[t.lower() for t in texts[i : i + 256]],
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                t_in = {k: v.to(self.device) for k, v in t_in.items()}
                feats = _features(self.vlm.get_text_features(**t_in))
                out.append(feats.float().cpu().numpy())
        return l2n(np.concatenate(out))


# =========================================================================== #
# Fakes (numpy only)
# =========================================================================== #
def _seeded(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class FakeDetector:
    def detect(self, images):
        return [[Detection((0, 0, im.width, im.height), "dog", 0.9)] for im in images]


class FakeEncoders:
    """Embeds the 16x16 thumbnail through fixed random projections, so similar
    images → similar vectors (good enough to test the matcher end-to-end)."""

    logit_scale = 10.0
    logit_bias = -1.0

    def __init__(self, reid_dim: int = 128, vlm_dim: int = 64):
        rng = np.random.default_rng(0)
        self.Pr = rng.standard_normal((16 * 16 * 3, reid_dim)).astype(np.float32)
        self.Pv = rng.standard_normal((16 * 16 * 3, vlm_dim)).astype(np.float32)
        self.vlm_dim = vlm_dim

    def _px(self, crops):
        return np.stack([np.asarray(c.resize((16, 16)), dtype=np.float32).ravel() / 255.0 - 0.5 for c in crops])

    def encode_images(self, crops, tta: bool = True, masks=None):
        x = self._px(crops)
        attn = np.abs(np.stack([np.asarray(c.convert("L").resize((16, 16)), dtype=np.float32).ravel() / 255.0 for c in crops]))
        return {"reid": l2n(x @ self.Pr), "vlm": l2n(x @ self.Pv), "attn": attn, "attn_frac": 1.0}

    def encode_texts(self, texts):
        return l2n(np.stack([_seeded(t, self.vlm_dim) for t in texts]))


def build(cfg: Config):
    if cfg.fake_models:
        return FakeDetector(), FakeEncoders()
    backend = cfg.detector_backend
    if cfg.runtime == "onnx":
        from .onnx_runtime import OnnxEncoders
        from .segmenter import DinoSegmenterV2

        enc = OnnxEncoders(cfg)
        if backend not in ("auto", "dino2"):
            log.warning("runtime=onnx supports only the dino2 segmenter; ignoring detector_backend=%s", backend)
        return DinoSegmenterV2(cfg, enc), enc
    if backend == "auto":
        # v2 (aspect-preserving, trained on cluttered COCO scenes) beats both v1 and YOLO
        # on COCO val2017: recall 0.95 vs 0.81/0.87, small-animal recall 0.83 vs 0.37/0.68
        backend = "dino2" if cfg.seg_head2_path.exists() else ("dino" if cfg.seg_head_path.exists() else "yolo")
    if backend == "yolo":
        log.warning("YOLO detector is AGPL-3.0 and dev-only (PETTI_DETECTOR_BACKEND=yolo)")
        return YoloDetector(cfg), HFEncoders(cfg)
    enc = HFEncoders(cfg)
    if backend == "dino2":
        from .segmenter import DinoSegmenterV2

        return DinoSegmenterV2(cfg, enc), enc
    from .segmenter import DinoSegmenter

    return DinoSegmenter(cfg, enc), enc
