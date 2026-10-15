#!/usr/bin/env python3
"""Train the tiny DINOv2 foreground head. Backbone stays frozen.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 training/train_seg.py

Oxford-IIIT Pet trainval trimaps (CC BY-SA 4.0) plus photos in
data/public/scenes/train (no-animal). The official Oxford test list is never
loaded. Checkpoints each epoch to artifacts/seg_ckpt.pt and can be resumed.
The exported file is artifacts/seg_head.npz (numpy, no torch at inference
of the head itself).
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
from common import ROOT
from PIL import Image

OX = ROOT / "data" / "public" / "oxford"
ART = Path(__import__("os").environ.get("PETTI_ARTIFACTS_DIR", ROOT / "artifacts"))
NEG = ROOT / "data" / "public" / "scenes" / "train"


def _split_names() -> tuple[list[str], list[str]]:
    trainval = (OX / "annotations" / "trainval.txt").read_text().split()
    names = trainval[0::4]  # "Abyssinian_1 1 1 1"
    test = set((OX / "annotations" / "test.txt").read_text().split()[0::4])
    names = [n for n in names if n not in test]
    rng_names, val = [], []
    for n in names:
        h = int(hashlib.md5(n.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < 0.1 else rng_names).append(n)
    return rng_names, val


def _load_mask(name: str, size: int = 16) -> np.ndarray | None:
    """32×32 float target. nan = trimap border (ignored). None if files missing."""
    img_p = OX / "images" / f"{name}.jpg"
    tri_p = OX / "annotations" / "trimaps" / f"{name}.png"
    if not img_p.exists() or not tri_p.exists():
        return None
    tri = Image.open(tri_p).convert("L").resize((size, size), Image.NEAREST)
    a = np.asarray(tri, dtype=np.int16)
    # trimap: 1 pet, 2 background, 3 border
    target = np.full(a.shape, np.nan, dtype=np.float32)
    target[a == 1] = 1
    target[a == 2] = 0
    return target


def _blank(size: int = 16) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def _image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from petti_vision.config import resolve_device
    from petti_vision.segmenter import GRID, MEAN, SIDE, STD

    global _load_mask, _blank
    _load_mask.__defaults__ = (GRID,)
    _blank.__defaults__ = (GRID,)

    device = resolve_device("auto")
    if device == "mps":
        torch.set_default_dtype(torch.float32)

    train_names, val_names = _split_names()
    if args.limit:
        train_names = train_names[: args.limit]
        val_names = val_names[: max(1, args.limit // 5)]
    neg_paths = [p for p in NEG.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}] if NEG.exists() else []
    print(f"train {len(train_names)}  val {len(val_names)}  negatives {len(neg_paths)}  device {device}", flush=True)

    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(384, 64, 1)
            self.dec = nn.Conv2d(64, 1, 3, padding=1)

        def forward(self, tokens):
            b = tokens.shape[0]
            x = tokens.transpose(1, 2).reshape(b, 384, GRID, GRID)
            return self.dec(F.relu(self.proj(x)))

    head = Head().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    ckpt_path = ART / "seg_ckpt.pt"
    start = 1
    best = 1e9
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head"])
        opt.load_state_dict(ckpt["opt"])
        start = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best", best))
        print(f"resumed at epoch {start}", flush=True)

    def tokens_of(images: list[Image.Image]) -> torch.Tensor:
        batch = []
        for im in images:
            arr = np.asarray(im.resize((SIDE, SIDE), Image.BICUBIC), dtype=np.float32) / 255.0
            batch.append((arr - MEAN) / STD)
        pixel = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel)
            n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
            return out.last_hidden_state[:, 1 + n_reg :].float().contiguous()

    def run_epoch(names: list[str], train: bool) -> float:
        order = list(names)
        random.shuffle(order)
        # oversample negatives so a car is not a rare event
        extras = []
        if train and neg_paths:
            extras = [random.choice(neg_paths) for _ in range(max(1, len(order) // 4))]
        items: list[tuple[str, Path | None]] = [(n, None) for n in order] + [("neg", p) for p in extras]
        if train:
            random.shuffle(items)
        total = 0.0
        seen = 0
        head.train(train)
        for i in range(0, len(items), args.batch):
            if train and i and i % (args.batch * 40) == 0:
                print(f"  {i}/{len(items)}", flush=True)
            chunk = items[i : i + args.batch]
            images, targets = [], []
            for name, neg in chunk:
                if neg is not None:
                    im = _image(neg)
                    tgt = _blank()
                else:
                    tgt = _load_mask(name)
                    if tgt is None:
                        continue
                    im = _image(OX / "images" / f"{name}.jpg")
                flip = train and random.random() < 0.5
                if flip:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    tgt = np.fliplr(tgt).copy()
                images.append(im)
                targets.append(tgt)
            if not images:
                continue
            tok = tokens_of(images)
            y = torch.from_numpy(np.stack(targets)).to(device)
            logits = head(tok)[:, 0]
            weight = torch.ones_like(y)
            weight[torch.isnan(y)] = 0
            weight[y == 1] = 2
            y = torch.nan_to_num(y, nan=0.0)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weight).sum() / weight.sum().clamp(min=1)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += float(loss) * len(images)
            seen += len(images)
        return total / max(seen, 1)

    ART.mkdir(parents=True, exist_ok=True)
    stale = 0
    best_state = None
    for epoch in range(start, args.epochs + 1):
        tr = run_epoch(train_names, True)
        va = run_epoch(val_names, False)
        print(f"epoch {epoch}  train {tr:.4f}  val {va:.4f}", flush=True)
        if va < best - 1e-4:
            best = va
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        torch.save({"epoch": epoch, "head": head.state_dict(), "opt": opt.state_dict(), "best": best, "best_state": best_state}, ckpt_path)
        if stale >= 3:
            print("early stop", flush=True)
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    _export(head)
    print("wrote", ART / "seg_head.npz", flush=True)


def _export(head) -> None:
    head.eval()
    sd = head.state_dict()
    # proj is 1×1: (64, 384, 1, 1) → (64, 384). dec is 3×3: (1, 64, 3, 3)
    np.savez(
        ART / "seg_head.npz",
        w1=sd["proj.weight"].detach().cpu().numpy()[:, :, 0, 0].astype(np.float32),
        b1=sd["proj.bias"].detach().cpu().numpy().astype(np.float32),
        w2=sd["dec.weight"].detach().cpu().numpy().astype(np.float32),
        b2=sd["dec.bias"].detach().cpu().numpy().astype(np.float32),
        threshold=np.float32(0.5),
    )


if __name__ == "__main__":
    main()
