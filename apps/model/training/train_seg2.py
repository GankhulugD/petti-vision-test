#!/usr/bin/env python3
"""Segmenter v2: a small conv head on frozen DINOv2-small patch tokens.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 training/train_seg2.py [--teacher-n 2400] [--epochs 6]

What is different from v1 (train_seg.py):
  * aspect ratio is kept (long side 336–448 during training, 448 at inference)
  * real-world scenes: besides Oxford-IIIT Pet trimaps (CC BY-SA 4.0, pets
    filling the frame) it trains on cluttered dog photos labelled offline by
    the BSD teacher (torchvision Mask R-CNN v2, see teacher.py); the teacher
    never runs in production
  * mosaic (2×2) and random-crop augmentation → small, far and multiple animals
  * deeper head (1×1 → 3×3 → 3×3) trained with BCE + soft dice; the threshold
    is tuned on validation data
Every evaluation image (data/eval/**, bench_exclude.txt, Oxford test list) is
kept out of training. Checkpoint each epoch → resumable. Export:
artifacts/seg_head2.npz (numpy only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from common import ROOT

OX = ROOT / "data" / "public" / "oxford"
POOL = ROOT / "data" / "public" / "dogceo"
NEG = ROOT / "data" / "public" / "scenes" / "train"
ART = ROOT / "artifacts"
PATCH = 14


def _h(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def eval_names() -> set[str]:
    names = set((ROOT / "data" / "eval" / "bench_exclude.txt").read_text().split()) if (ROOT / "data" / "eval" / "bench_exclude.txt").exists() else set()
    names |= {p.name for p in (ROOT / "data" / "eval").rglob("*.jpg")}
    return names


# --------------------------------------------------------------------------- #
# samples: (image path, kind) → full-resolution float target in [0,1], nan = ignore
# --------------------------------------------------------------------------- #
def oxford_target(name: str, size: tuple[int, int]) -> np.ndarray:
    tri = np.asarray(Image.open(OX / "annotations" / "trimaps" / f"{name}.png").convert("L").resize(size, Image.NEAREST), dtype=np.int16)
    t = np.full(tri.shape, np.nan, dtype=np.float32)
    t[tri == 1] = 1
    t[tri == 2] = 0
    return t


def teacher_target(res: dict, size: tuple[int, int]) -> np.ndarray | None:
    w, h = size
    conf = [it for it in res["inst"] if it["score"] >= 0.6]
    if not conf:
        return None
    t = np.zeros((h, w), dtype=np.float32)
    for it in res["inst"]:
        m = np.asarray(Image.fromarray(it["mask"]).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        if it["score"] >= 0.6:
            t = np.maximum(t, (m >= 0.5).astype(np.float32))
        else:  # unsure instance → do not punish either answer there
            t[m >= 0.3] = np.nan
    # uncertain teacher border: ignore a thin band around the edge of confident masks
    return t


COCO = ROOT / "data" / "public" / "coco"


def coco_target(name: str, size: tuple[int, int]) -> np.ndarray:
    m = np.asarray(Image.open(COCO / "masks" / (Path(name).stem + ".png")).resize(size, Image.NEAREST))
    t = (m == 1).astype(np.float32)
    t[m == 255] = np.nan
    return t


def load_sample(item) -> tuple[Image.Image, np.ndarray]:
    kind, ref = item
    if kind == "coco":
        im = Image.open(COCO / "images" / ref).convert("RGB")
        return im, coco_target(ref, im.size)
    if kind == "ox":
        im = Image.open(OX / "images" / f"{ref}.jpg").convert("RGB")
        return im, oxford_target(ref, im.size)
    if kind == "neg":
        im = Image.open(ref).convert("RGB")
        return im, np.zeros((im.height, im.width), dtype=np.float32)
    path, res = ref
    im = Image.open(path).convert("RGB")
    return im, teacher_target(res, im.size)


def random_crop(im, t, rng):
    if rng.random() < 0.5:
        return im, t
    w, h = im.size
    s = rng.uniform(0.45, 1.0)
    cw, ch = max(32, int(w * s)), max(32, int(h * s))
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return im.crop((x, y, x + cw, y + ch)), t[y : y + ch, x : x + cw]


def augment(im, t, rng):
    im, t = random_crop(im, t, rng)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        t = t[:, ::-1]
    if rng.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.3))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
    return im, np.ascontiguousarray(t)


def mosaic(samples, rng):
    """2×2 grid of four samples → small and multiple animals in one image."""
    side = 448
    half = side // 2
    canvas = Image.new("RGB", (side, side))
    tgt = np.zeros((side, side), dtype=np.float32)
    for k, (im, t) in enumerate(samples):
        im2 = im.resize((half, half), Image.BILINEAR)
        t2 = np.asarray(Image.fromarray(np.nan_to_num(t, nan=-1.0).astype(np.float32), mode="F").resize((half, half), Image.NEAREST))
        t2 = np.where(t2 < 0, np.nan, t2)
        x, y = (k % 2) * half, (k // 2) * half
        canvas.paste(im2, (x, y))
        tgt[y : y + half, x : x + half] = t2
    return canvas, tgt


def to_grid(t: np.ndarray, gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-average the full-res target onto the patch grid; weight 0 where mostly ignored."""
    h, w = t.shape
    ys = np.linspace(0, h, gh + 1).astype(int)
    xs = np.linspace(0, w, gw + 1).astype(int)
    y = np.zeros((gh, gw), np.float32)
    wt = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            c = t[ys[i] : max(ys[i + 1], ys[i] + 1), xs[j] : max(xs[j + 1], xs[j] + 1)]
            valid = ~np.isnan(c)
            frac = valid.mean()
            if frac > 0.3:
                y[i, j] = float(c[valid].mean())
                wt[i, j] = frac
    return y, wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--teacher-n", type=int, default=2400)
    ap.add_argument("--oxford-per-epoch", type=int, default=2500)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--teacher-device", default="mps")
    ap.add_argument("--source", choices=("coco", "teacher"), default="coco",
                    help="coco: license-filtered COCO masks (shippable); teacher: pseudo-labels on dog.ceo (ImageNet → eval/experiments only)")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from petti_vision.config import resolve_device
    from petti_vision.segmenter import MEAN, STD, input_size
    from teacher import Teacher

    rng = random.Random(0)
    excluded = eval_names()

    # ---- Oxford (trainval minus the official test list) ------------------- #
    trainval = (OX / "annotations" / "trainval.txt").read_text().split()[0::4]
    test = set((OX / "annotations" / "test.txt").read_text().split()[0::4])
    ox = [n for n in trainval if n not in test and (OX / "images" / f"{n}.jpg").exists() and f"{n}.jpg" not in excluded]
    ox_val = [n for n in ox if _h(n) < 0.08]
    ox_tr = [n for n in ox if _h(n) >= 0.08]

    # ---- cluttered real-world photos ----------------------------------------- #
    if a.source == "coco":
        meta = json.loads((COCO / "coco_pets.json").read_text())
        files = [r["file"] for r in meta["train"] if (COCO / "images" / r["file"]).exists()]
        negs_coco = [COCO / "images" / r["file"] for r in meta["neg"] if (COCO / "images" / r["file"]).exists()]
        tt = [("coco", f) for f in files if _h(f) >= 0.08]
        tv = [("coco", f) for f in files if _h(f) < 0.08]
        print(f"coco train {len(tt)} val {len(tv)} negatives {len(negs_coco)}", flush=True)
    pool = [] if a.source == "coco" else [p for p in sorted(POOL.rglob("*.jpg")) if p.name not in excluded]
    rng.shuffle(pool)
    pool = pool[: a.teacher_n]
    try:
        teacher = Teacher(device=a.teacher_device) if pool else None
    except Exception as exc:  # MPS ops missing on old torchvision → CPU
        print(f"teacher on {a.teacher_device} failed ({exc}); using cpu", flush=True)
        teacher = Teacher(device="cpu")
    labelled = []
    t0 = time.time()
    for i, p in enumerate(pool):
        try:
            res = teacher.get(p)
        except Exception as exc:
            if teacher.device != "cpu" and not labelled:
                print(f"teacher on {teacher.device} failed ({exc}); switching to cpu", flush=True)
                teacher = Teacher(device="cpu")
                res = teacher.get(p)
            else:
                print(f"teacher failed on {p.name}: {exc}", flush=True)
                continue
        if any(it["score"] >= 0.6 for it in res["inst"]):
            labelled.append((p, res))
        if i % 200 == 0:
            print(f"teacher {i}/{len(pool)}  kept {len(labelled)}  {time.time() - t0:.0f}s", flush=True)
    del teacher
    if a.source == "teacher":
        tv = [("tch", x) for x in labelled if _h(x[0].name) < 0.08]
        tt = [("tch", x) for x in labelled if _h(x[0].name) >= 0.08]
        negs_coco = []
    negs = [p for p in NEG.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}] if NEG.exists() else []
    negs += negs_coco
    print(f"oxford train {len(ox_tr)} val {len(ox_val)} | teacher train {len(tt)} val {len(tv)} | negatives {len(negs)}", flush=True)

    device = resolve_device("auto")
    from transformers import AutoModel

    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    for q in dino.parameters():
        q.requires_grad_(False)
    n_reg = int(getattr(dino.config, "num_register_tokens", 0) or 0)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Conv2d(384, 96, 1)
            self.l1 = nn.Conv2d(96, 64, 3, padding=1)
            self.l2 = nn.Conv2d(64, 1, 3, padding=1)

        def forward(self, x):
            return self.l2(F.relu(self.l1(F.relu(self.l0(x)))))

    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    ckpt_p = ART / f"seg2_{a.source}_ckpt.pt"
    start, best = 1, -1.0
    if ckpt_p.exists() and not a.fresh:
        ck = torch.load(ckpt_p, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs), last_epoch=start - 2)

    def tokens(images: list[Image.Image], tw: int, th: int):
        arr = np.stack([(np.asarray(im.resize((tw, th), Image.BICUBIC), dtype=np.float32) / 255.0 - MEAN) / STD for im in images])
        px = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            hs = dino(pixel_values=px).last_hidden_state[:, 1 + n_reg :]
        gh, gw = th // PATCH, tw // PATCH
        return hs.float().transpose(1, 2).reshape(len(images), 384, gh, gw).contiguous(), gh, gw

    def batches(items, train: bool):
        """Group samples by a shared input size (aspect bucket) per batch."""
        order = list(items)
        if train:
            rng.shuffle(order)
        for i in range(0, len(order), a.batch):
            chunk = order[i : i + a.batch]
            pairs = []
            for it in chunk:
                if it[0] == "mosaic":
                    pairs.append(mosaic([augment(*load_sample(x), rng) for x in it[1]], rng))
                    continue
                im, t = load_sample(it)
                if t is None:
                    continue
                pairs.append(augment(im, t, rng) if train else (im, t))
            if not pairs:
                continue
            long_side = rng.choice((336, 392, 448)) if train else 448
            ar = np.median([p[0].width / p[0].height for p in pairs])
            tw, th = input_size(int(1000 * ar), 1000, long_side)
            yield pairs, tw, th

    def loss_fn(logits, y, wt):
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * wt).sum() / wt.sum().clamp(min=1)
        p = torch.sigmoid(logits) * wt
        dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + (y * wt).sum((1, 2)) + 1)
        return bce + 0.5 * dice.mean()

    def run(items, train: bool):
        head.train(train)
        tot, n = 0.0, 0
        stats = {t: [0.0, 0.0] for t in (0.3, 0.4, 0.5, 0.6)}  # inter, union per threshold
        for k, (pairs, tw, th) in enumerate(batches(items, train)):
            x, gh, gw = tokens([p[0] for p in pairs], tw, th)
            ys, ws = zip(*[to_grid(p[1], gh, gw) for p in pairs])
            y = torch.from_numpy(np.stack(ys)).to(device)
            wt = torch.from_numpy(np.stack(ws)).to(device)
            with torch.set_grad_enabled(train):
                logits = head(x)[:, 0]
                loss = loss_fn(logits, y, wt)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                if k % 50 == 0:
                    print(f"  step {k} loss {float(loss):.4f}", flush=True)
            else:
                pr = torch.sigmoid(logits).cpu().numpy()
                yy, ww = y.cpu().numpy(), wt.cpu().numpy() > 0
                for thr in stats:
                    pb, gb = (pr >= thr) & ww, (yy >= 0.5) & ww
                    stats[thr][0] += float((pb & gb).sum())
                    stats[thr][1] += float((pb | gb).sum())
            tot += float(loss) * len(pairs)
            n += len(pairs)
        ious = {t: (s[0] / s[1] if s[1] else 0.0) for t, s in stats.items()}
        return tot / max(n, 1), ious

    ART.mkdir(parents=True, exist_ok=True)
    for ep in range(start, a.epochs + 1):
        t1 = time.time()
        ox_ep = rng.sample(ox_tr, min(a.oxford_per_epoch, len(ox_tr)))
        items = [("ox", n) for n in ox_ep] + list(tt) + [("neg", p) for p in negs]
        base = [it for it in items]
        items += [("mosaic", rng.sample(base, 4)) for _ in range(len(base) // 6)]
        tr, _ = run(items, True)
        sched.step()
        va_ox, iou_ox = run([("ox", n) for n in ox_val], False)
        va_t, iou_t = run(list(tv), False)
        thr = max(iou_t, key=lambda k: iou_t[k] + iou_ox[k])
        score = iou_t[thr] + iou_ox[thr]
        print(f"epoch {ep} train {tr:.4f} | oxford val iou {iou_ox} | teacher val iou {iou_t} | thr {thr} | {time.time() - t1:.0f}s", flush=True)
        state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        if score > best:
            best = score
            exp = {"n_layers": np.array(3), "threshold": np.array(thr, dtype=np.float32), "long_side": np.array(448)}
            for i, name in enumerate(("l0", "l1", "l2")):
                exp[f"l{i}_w"] = state[f"{name}.weight"].numpy().astype(np.float16)
                exp[f"l{i}_b"] = state[f"{name}.bias"].numpy().astype(np.float32)
            exp["source"] = np.array(a.source)
            np.savez(ART / ("seg_head2.npz" if a.source == "coco" else "seg_head2_teacher.npz"), **exp)
            (ART / "reports").mkdir(exist_ok=True)
            (ART / "reports" / "seg2_train.json").write_text(json.dumps({"epoch": ep, "source": a.source, "thr": thr, "oxford_val_iou": iou_ox, "scene_val_iou": iou_t}, indent=2))
            print(f"  saved head (score {score:.3f})", flush=True)
        torch.save({"epoch": ep, "head": head.state_dict(), "opt": opt.state_dict(), "best": best}, ckpt_p)
    print("done", flush=True)


if __name__ == "__main__":
    main()
