"""Runtime configuration (env-overridable)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # apps/model


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- models ----------------------------------------------------------- #
    # Detector: Ultralytics YOLO (COCO has dog/cat/bird/horse). yolo26n is
    # NMS-free and the fastest; fall back to yolo11n.pt on older ultralytics.
    detector_weights: str = _env("PETTI_DETECTOR", "yolo26n-seg.pt")
    detector_conf: float = float(_env("PETTI_DETECTOR_CONF", "0.25"))
    detector_imgsz: int = int(_env("PETTI_DETECTOR_IMGSZ", "640"))

    # Identity / re-ID embedding (instance-level similarity). DINOv2 features
    # are the strongest open features for fine-grained instance retrieval.
    # Options: facebook/dinov2-small (fast) | facebook/dinov2-base (accurate)
    #          facebook/dinov3-vits16-pretrain-lvd1689m (gated on HF)
    reid_model: str = _env("PETTI_REID_MODEL", "facebook/dinov2-small")
    # identity descriptor pooling (leave-one-out R@1 on 168 owner photos of 46
    # dogs + 104 look-alikes, see training/evaluate.py --loo):
    #   "cls" 0.893 · "masked" (CLS ‖ animal-patch mean) 0.875 · "mean" 0.869
    # ⚠ changing it changes the embedding space → run training/reindex.py
    reid_pool: str = _env("PETTI_REID_POOL", "cls")

    # Vision-language model for zero-shot attributes (breed/colour/age/marks)
    # Options: google/siglip2-base-patch16-224 (default) |
    #          google/siglip2-so400m-patch14-384 (best, slower) |
    #          google/siglip-base-patch16-224
    vlm_model: str = _env("PETTI_VLM_MODEL", "google/siglip2-base-patch16-224")

    device: str = _env("PETTI_DEVICE", "auto")  # auto|cpu|cuda|mps
    half: bool = _env("PETTI_HALF", "1") == "1"  # fp16 on cuda only (mps runs fp32)

    # --- paths ------------------------------------------------------------ #
    data_dir: Path = Path(_env("PETTI_DATA_DIR", str(ROOT / "data")))
    artifacts_dir: Path = Path(_env("PETTI_ARTIFACTS_DIR", str(ROOT / "artifacts")))

    # --- matching --------------------------------------------------------- #
    # Weights of the final match score. Re-tune with training/evaluate.py.
    w_reid: float = float(_env("PETTI_W_REID", "0.50"))
    w_vlm: float = float(_env("PETTI_W_VLM", "0.15"))
    w_attr: float = float(_env("PETTI_W_ATTR", "0.35"))
    search_top_k: int = int(_env("PETTI_TOP_K", "20"))
    min_photos: int = int(_env("PETTI_MIN_PHOTOS", "5"))

    # test-time augmentation (horizontal flip) for embeddings: +~1-2% recall
    tta: bool = _env("PETTI_TTA", "1") == "1"

    # use fake numpy backbones (unit tests / UI dev without downloading weights)
    fake_models: bool = _env("PETTI_FAKE_MODELS", "0") == "1"

    extra: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "petti.db"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def heads_path(self) -> Path:
        return self.artifacts_dir / "heads.npz"

    @property
    def reid_proj_path(self) -> Path:
        return self.artifacts_dir / "reid_projection.npz"

    @property
    def text_cache_path(self) -> Path:
        safe = self.vlm_model.replace("/", "__")
        return self.artifacts_dir / f"text_embeddings_{safe}.npz"


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
