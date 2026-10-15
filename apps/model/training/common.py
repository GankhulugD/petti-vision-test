"""Shared training utilities: manifest I/O and feature extraction cache.

Manifest = CSV, one row per image (missing labels are just empty cells):

    path,species,breed,size,age,colors,pattern,coat,marks,pet_id,source
    data/custom/img1.jpg,dog,siberian_husky,medium,adult,black|white,bicolor,medium,blue_eyes,pet_42,owner

* colors / marks are `|`-separated multi-labels; marks="none" means "verified: no marks"
* pet_id groups photos of the same individual (needed for re-ID + leak-free splits)
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from petti_vision import taxonomy as T  # noqa: E402

FIELDS = ["path", "species", "breed", "size", "age", "colors", "pattern", "coat", "marks", "pet_id", "source"]
SINGLE = {
    "species": [s.key for s in T.SPECIES],
    "breed": [b.key for b in T.BREEDS],
    "size": [s.key for s in T.SIZES],
    "age": [a.key for a in T.AGES],
    "pattern": [p.key for p in T.PATTERNS],
    "coat": [c.key for c in T.COAT_LENGTH],
}
MULTI = {"color": [c.key for c in T.COLORS], "marks": [m.key for m in T.MARKS]}


def read_manifest(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                img = Path(r["path"])
                if not img.is_absolute():
                    img = (p.parent / img) if (p.parent / img).exists() else (ROOT / img)
                r["path"] = str(img)
                rows.append({k: (r.get(k) or "").strip() for k in FIELDS} | {"path": str(img)})
    return rows


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def split_of(row: dict, val_frac: float = 0.15) -> str:
    """Deterministic split by identity (or by file) → no leakage between splits."""
    key = row.get("pet_id") or row["path"]
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def encode_targets(rows: list[dict]) -> dict[str, np.ndarray]:
    """Single-label → int (-1 = unknown). Multi-label → float matrix (nan = unknown)."""
    out: dict[str, np.ndarray] = {}
    for task, keys in SINGLE.items():
        idx = {k: i for i, k in enumerate(keys)}
        out[task] = np.array([idx.get(r[task], -1) for r in rows], dtype=np.int64)
    col = {"color": "colors", "marks": "marks"}
    for task, keys in MULTI.items():
        idx = {k: i for i, k in enumerate(keys)}
        m = np.full((len(rows), len(keys)), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            v = r[col[task]]
            if not v:
                continue
            m[i] = 0.0
            if v != "none":
                for k in v.split("|"):
                    if k in idx:
                        m[i, idx[k]] = 1.0
        out[task] = m
    return out


def extract_features(rows: list[dict], cache: Path, batch: int = 32) -> dict[str, np.ndarray]:
    """Detect → crop → [SigLIP, DINOv2] features + zero-shot probabilities.

    Cached by manifest content hash, so re-running training is instant.
    """
    sig = hashlib.md5("\n".join(r["path"] for r in rows).encode()).hexdigest()[:10]
    if cache.exists():
        d = dict(np.load(cache, allow_pickle=False))
        if str(d.get("sig")) == sig:
            print(f"[features] cache hit {cache} ({len(rows)} rows)")
            return d

    from PIL import Image
    from tqdm import tqdm

    from petti_vision import Config, PettiVision
    from petti_vision.backbones import crop_detection, load_image
    from petti_vision.pipeline import _square

    cfg = Config()
    v = PettiVision(cfg)
    v.heads = None  # pure backbone + zero-shot features
    vlm, reid, ok = [], [], []
    zs = {t: [] for t in ("species", "breed", "size", "age", "pattern", "coat", "color")}
    for i in tqdm(range(0, len(rows), batch), desc="features"):
        chunk = rows[i : i + batch]
        imgs = []
        for r in chunk:
            try:
                imgs.append(load_image(r["path"], max_side=800))
                ok.append(True)
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
                ok.append(False)
        dets = v.detector.detect(imgs)
        crops = [_square(crop_detection(im, d[0] if d else None)) for im, d in zip(imgs, dets)]
        e = v.enc.encode_images(crops, tta=False)
        vlm.append(e["vlm"])
        reid.append(e["reid"])
        x = e["vlm"]
        zs["species"].append(v.zs.probs("species|any", x))
        # breed zero-shot on full breed list: dog block + cat block weighted by species prob
        sp = zs["species"][-1]
        pd, pc = v.zs.probs("breed|dog", x), v.zs.probs("breed|cat", x)
        zs["breed"].append(np.concatenate([pd * sp[:, :1], pc * sp[:, 1:2]], axis=1))
        cond = ["dog" if s.argmax() == 0 else "cat" if s.argmax() == 1 else "other" for s in sp]
        for t in ("size", "age", "pattern", "coat", "color"):
            zs[t].append(np.stack([v.zs.probs(f"{t}|{c}", x[j : j + 1])[0] for j, c in enumerate(cond)]))
    out = {
        "sig": np.array(sig),
        "vlm": np.concatenate(vlm).astype(np.float32),
        "reid": np.concatenate(reid).astype(np.float32),
        "ok": np.array(ok),
        **{f"zs_{t}": np.concatenate(z).astype(np.float32) for t, z in zs.items()},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
