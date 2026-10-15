#!/usr/bin/env python3
"""Train our attribute heads on frozen [SigLIP2 ‖ DINOv2] features.

    python3 training/train_heads.py                       # all manifests in data/manifests
    python3 training/train_heads.py --resume              # continue from artifacts/heads.npz
    python3 training/train_heads.py --manifest data/manifests/registry.csv --epochs 40

Because backbones are frozen and features are cached, a full run over ~30k
images takes ~1-2 min on an M-series Mac (MPS) or Colab GPU after the one-off
feature extraction.

For each task we also learn how much to trust the head vs. zero-shot
(`alpha`, chosen on the validation split) and a temperature for calibrated
confidences. Outputs: artifacts/heads.npz, heads.json, heads_report.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import MULTI, ROOT, SINGLE, encode_targets, extract_features, pick_device, read_manifest, split_of

ART = Path(__import__("os").environ.get("PETTI_ARTIFACTS_DIR", ROOT / "artifacts"))
MIN_LABELS = 30


class Head(nn.Module):
    def __init__(self, d_in: int, n_out: int, hidden: int = 512, p: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def geo_fuse(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    lp = (1 - alpha) * np.log(np.clip(a, 1e-9, None)) + alpha * np.log(np.clip(b, 1e-9, None))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def train_task(task, X, y, tr, va, multi, zs, args, device, init=None):
    d_in, n_out = X.shape[1], (y.shape[1] if multi else len(SINGLE[task]))
    if multi:
        lab = ~np.isnan(y).any(1)
    else:
        lab = y >= 0
    tr_i, va_i = np.where(tr & lab)[0], np.where(va & lab)[0]
    if len(tr_i) < MIN_LABELS or (not multi and len(np.unique(y[tr_i])) < 2):
        print(f"  [{task}] skipped ({len(tr_i)} labelled train samples)")
        return None

    mu = X[tr_i].mean(0)
    sd = X[tr_i].std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    head = Head(d_in, n_out).to(device)
    if init is not None:
        head.load_state_dict(init)
        print(f"  [{task}] resumed from previous weights")

    if multi:
        Y = torch.tensor(np.nan_to_num(y), dtype=torch.float32, device=device)
        pos = np.nanmean(y[tr_i], 0)
        pos_weight = torch.tensor(np.clip((1 - pos) / np.clip(pos, 1e-3, None), 1, 20), dtype=torch.float32, device=device)
    else:
        Y = torch.tensor(y, dtype=torch.long, device=device)
        counts = np.bincount(y[tr_i], minlength=n_out).astype(np.float32)
        w = 1.0 / np.sqrt(np.clip(counts, 1, None))
        w = w / w[counts > 0].mean()
        # classes without training data get weight 0: with label smoothing PyTorch
        # weights the smoothing term per class too, and an unseen class with a
        # large weight pulled every prediction towards it (species head → 23%)
        w[counts == 0] = 0.0
        cls_w = torch.tensor(w, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * ((len(tr_i) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps + 1, pct_start=0.1)

    def evaluate():
        head.eval()
        with torch.no_grad():
            logits = head(Xt[va_i]).float().cpu().numpy()
        head.train()
        if multi:
            pred = logits > 0
            return float((pred == y[va_i].astype(bool)).mean()), logits
        return float((logits.argmax(1) == y[va_i]).mean()), logits

    best, best_state, best_logits = -1.0, None, None
    for ep in range(args.epochs):
        perm = np.random.permutation(tr_i)
        for i in range(0, len(perm), args.batch):
            idx = torch.tensor(perm[i : i + args.batch], device=device)
            x = Xt[idx]
            x = x + args.noise * torch.randn_like(x)  # feature-space augmentation
            out = head(x)
            if multi:
                loss = F.binary_cross_entropy_with_logits(out, Y[idx], pos_weight=pos_weight)
            else:
                loss = F.cross_entropy(out, Y[idx], weight=cls_w, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if len(va_i) and (ep % 5 == 4 or ep == args.epochs - 1):
            acc, logits = evaluate()
            if acc > best:
                best, best_logits = acc, logits
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is None:  # no val split → keep last
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        best, best_logits = float("nan"), None

    # ---- calibration: temperature + fusion weight with zero-shot ---------- #
    labels_all = SINGLE.get(task) or MULTI[task]
    info = {"labels": labels_all, "multi_label": multi, "val_acc": round(best, 4),
            # classes with training data; unseen ones keep their zero-shot score at inference
            "seen": labels_all if multi else [labels_all[i] for i in np.unique(y[tr_i]).tolist()],
            "n_train": int(len(tr_i)), "n_val": int(len(va_i)), "alpha": 0.5, "temperature": 1.0}
    if best_logits is not None and not multi:
        yv = y[va_i]
        # T ≥ 1: never sharpen. Public datasets are cleaner than phone photos, so an
        # in-domain-optimal T<1 makes the head over-confident on real uploads.
        temps = np.linspace(1.0, 3.0, 21)
        nll = []
        for t in temps:
            z = best_logits / t
            z -= z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            nll.append(-lp[np.arange(len(yv)), yv].mean())
        T_ = float(temps[int(np.argmin(nll))])
        z = best_logits / T_
        z -= z.max(1, keepdims=True)
        hp = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        info["temperature"] = round(T_, 3)
        zk = f"zs_{task}"
        if zk in zs:
            zp = zs[zk][va_i]
            zp = zp / np.clip(zp.sum(1, keepdims=True), 1e-9, None)
            accs = [(geo_fuse(zp, hp, a).argmax(1) == yv).mean() for a in np.linspace(0, 1, 11)]
            # cap at 0.5: the val split is in-domain; on out-of-domain phone photos
            # zero-shot keeps an equal vote (validated on the live benchmark)
            info["alpha"] = min(0.5, round(float(np.linspace(0, 1, 11)[int(np.argmax(accs))]), 2))
            info["zs_val_acc"] = round(float(accs[0]), 4)
            info["fused_val_acc"] = round(float(max(accs)), 4)
    elif multi:
        info["alpha"] = 0.6
    # nn.Sequential linear layers are net.0 and net.3 → numpy MLP layers 0 and 1
    layer = {"0": 0, "3": 1}
    remap = {f"{task}.{layer[k.split('.')[1]]}.{k.split('.')[2]}": v.numpy() for k, v in best_state.items()}
    remap[f"{task}.mu"] = mu.astype(np.float32)
    remap[f"{task}.sd"] = sd.astype(np.float32)
    return remap, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--feature", choices=["vlm", "concat"], default="vlm")
    ap.add_argument("--tasks", default=",".join([*SINGLE, *MULTI]))
    args = ap.parse_args()

    manifests = args.manifest or sorted((ROOT / "data" / "manifests").glob("*.csv"))
    rows = read_manifest(manifests)
    print(f"{len(rows)} rows from {[m.name for m in manifests]}")
    feats = extract_features(rows, ART / "cache" / "features.npz")
    ok = feats["ok"]
    # "vlm" (default): SigLIP features only, so heads stay valid when the re-ID
    # descriptor changes; "concat" adds DINOv2 features (must match heads.features)
    X = feats["vlm"] if args.feature == "vlm" else np.concatenate([feats["vlm"], feats["reid"]], 1)
    Y = encode_targets(rows)
    split = np.array([split_of(r) for r in rows])
    tr, va = (split == "train") & ok, (split == "val") & ok
    device = pick_device()
    print(f"device={device} train={tr.sum()} val={va.sum()} dim={X.shape[1]}")

    prev_w, prev_meta = {}, {}
    if args.resume and (ART / "heads.npz").exists():
        prev_w = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(ART / "heads.npz").items()}
        prev_meta = json.loads((ART / "heads.json").read_text())["tasks"]

    weights, meta = {}, {}
    t0 = time.time()
    for task in args.tasks.split(","):
        multi = task in MULTI
        init = None
        if task in prev_meta and prev_meta[task]["labels"] == (SINGLE.get(task) or MULTI[task]) and f"{task}.0.weight" in prev_w:
            init = {"net.0.weight": torch.tensor(prev_w[f"{task}.0.weight"]), "net.0.bias": torch.tensor(prev_w[f"{task}.0.bias"]),
                    "net.3.weight": torch.tensor(prev_w[f"{task}.1.weight"]), "net.3.bias": torch.tensor(prev_w[f"{task}.1.bias"])}
        res = train_task(task, X, Y[task], tr, va, multi, feats, args, device, init)
        if res is None:
            if task in prev_meta:  # keep the old head rather than dropping it
                weights.update({k: v for k, v in prev_w.items() if k.startswith(task + ".")})
                meta[task] = prev_meta[task]
            continue
        w, info = res
        weights.update(w)
        meta[task] = info
        print(f"  [{task}] val_acc={info['val_acc']} zs={info.get('zs_val_acc')} fused={info.get('fused_val_acc')} "
              f"alpha={info['alpha']} T={info['temperature']} (n={info['n_train']})")

    ART.mkdir(parents=True, exist_ok=True)
    # fp16 halves the artifact (heads are tiny MLPs; no accuracy impact)
    np.savez_compressed(ART / "heads.npz", **{k: v.astype(np.float16) if v.ndim == 2 else v for k, v in weights.items()})
    (ART / "heads.json").write_text(json.dumps({"feature": args.feature, "tasks": meta, "trained_at": time.time()}, indent=2))
    lines = [f"{t}: {json.dumps(m)}" for t, m in meta.items()]
    (ART / "heads_report.txt").write_text("\n".join(lines))
    print(f"saved {ART/'heads.npz'} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
