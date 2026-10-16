#!/usr/bin/env python3
"""Compare the DINOv2 segmenter with YOLO on the Oxford test split.

    python3 training/eval_seg.py --backend yolo --limit 200
    python3 training/eval_seg.py --backend dino

Prints detection recall, recall on small animals (trimap foreground < 8% of
the image), mask IoU, no-animal false-positive rate, and latency. The Oxford
test list is not used for training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from common import ROOT
from PIL import Image

OX = ROOT / "data" / "public" / "oxford"
EVAL = ROOT / "data" / "eval"


def _names(limit: int) -> list[str]:
    names = (OX / "annotations" / "test.txt").read_text().split()[0::4]
    return names[:limit] if limit else names


def _gt(name: str) -> np.ndarray | None:
    p = OX / "annotations" / "trimaps" / f"{name}.png"
    img = OX / "images" / f"{name}.jpg"
    if not p.exists() or not img.exists():
        return None
    return np.asarray(Image.open(p).convert("L"))


def _iou(pred: np.ndarray, tri: np.ndarray) -> float:
    valid = tri != 3
    gt = (tri == 1) & valid
    pr = pred & valid
    inter = np.logical_and(pr, gt).sum()
    union = np.logical_or(pr, gt).sum()
    return float(inter) / float(union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("dino", "dino2", "yolo"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", choices=("oxford", "coco"), default="oxford",
                    help="coco = COCO val2017 dog/cat photos with real masks (cluttered scenes, eval only)")
    args = ap.parse_args()
    if args.set == "coco":
        return coco_eval(args)

    import os

    os.environ["PETTI_DETECTOR_BACKEND"] = args.backend
    from petti_vision import Config, PettiVision

    # common.py imports Config before this env var exists, so pass it explicitly.
    vision = PettiVision(Config(detector_backend=args.backend))
    det = vision.detector
    rows = []
    for name in _names(args.limit):
        tri = _gt(name)
        if tri is None:
            continue
        img = Image.open(OX / "images" / f"{name}.jpg").convert("RGB")
        t0 = time.perf_counter()
        found = det.detect([img])[0]
        ms = (time.perf_counter() - t0) * 1000
        pred = found[0].mask(img.size) if found else None
        if pred is None:
            pred = np.zeros(tri.shape[:2], dtype=bool)
        if pred.shape != tri.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize((tri.shape[1], tri.shape[0]), Image.NEAREST)) > 127
        fg = float((tri == 1).mean())
        rows.append({"hit": bool(found), "iou": _iou(pred, tri), "small": fg < 0.08, "ms": ms})
    neg_dir = EVAL / "negatives_photo"
    drawings = list((EVAL / "negatives").glob("*.jpg"))
    fps = []
    for p in list(neg_dir.glob("*.jpg")) + drawings:
        im = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        found = det.detect([im])[0]
        fps.append({"file": p.name, "fp": bool(found), "ms": (time.perf_counter() - t0) * 1000, "photo": p.parent.name == "negatives_photo"})

    def rate(xs, key):
        if not xs:
            return None
        return round(sum(1 for r in xs if r[key]) / len(xs), 3)

    small = [r for r in rows if r["small"]]
    ms = sorted(r["ms"] for r in rows)
    report = {
        "backend": args.backend,
        "n": len(rows),
        "recall": rate(rows, "hit"),
        "recall_small": rate(small, "hit"),
        "small_n": len(small),
        "mask_iou": round(float(np.mean([r["iou"] for r in rows])), 3) if rows else None,
        "latency_p50_ms": round(ms[len(ms) // 2], 1) if ms else None,
        "negative_fpr": rate(fps, "fp"),
        "negative_n": len(fps),
        "photo_fpr": rate([r for r in fps if r["photo"]], "fp"),
        "photo_n": sum(1 for r in fps if r["photo"]),
    }
    out = ROOT / "artifacts" / "reports" / f"seg_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def coco_eval(args):
    """Union-of-animals mask IoU, recall and box IoU on COCO val2017 dog/cat images."""
    from petti_vision import Config, PettiVision

    coco = ROOT / "data" / "public" / "coco"
    meta = json.loads((coco / "coco_pets.json").read_text())["val"]
    if args.limit:
        meta = meta[: args.limit]
    det = PettiVision(Config(detector_backend=args.backend)).detector
    rows = []
    for r in meta:
        img = Image.open(coco / "images" / r["file"]).convert("RGB")
        gt = np.asarray(Image.open(coco / "masks" / (Path(r["file"]).stem + ".png")))
        t0 = time.perf_counter()
        found = det.detect([img])[0]
        ms = (time.perf_counter() - t0) * 1000
        pred = np.zeros(gt.shape, bool)
        for d in found:
            m = d.mask(img.size)
            if m is None:  # box-only detector
                x1, y1, x2, y2 = (int(v) for v in d.box)
                m = np.zeros(gt.shape, bool)
                m[y1:y2, x1:x2] = True
            pred |= m
        valid = gt != 255
        g = (gt == 1) & valid
        inter, union = (pred & g & valid).sum(), ((pred | g) & valid).sum()
        frac = float(g.mean())
        rows.append({"hit": bool(found) and inter > 0, "iou": float(inter / union) if union else 0.0, "small": frac < 0.05, "ms": ms, "n_pred": len(found)})
    small = [r for r in rows if r["small"]]
    ms = sorted(r["ms"] for r in rows)
    rep = {
        "set": "coco_val2017", "backend": args.backend, "n": len(rows),
        "recall": round(float(np.mean([r["hit"] for r in rows])), 3),
        "mask_iou": round(float(np.mean([r["iou"] for r in rows])), 3),
        "iou>=0.5": round(float(np.mean([r["iou"] >= 0.5 for r in rows])), 3),
        "small_n": len(small),
        "small_recall": round(float(np.mean([r["hit"] for r in small])), 3) if small else None,
        "small_iou": round(float(np.mean([r["iou"] for r in small])), 3) if small else None,
        "latency_p50_ms": round(ms[len(ms) // 2], 1),
    }
    out = ROOT / "artifacts" / "reports" / f"seg_coco_{args.backend}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
