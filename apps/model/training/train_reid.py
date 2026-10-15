#!/usr/bin/env python3
"""Metric-learning projection for pet re-identification ("is it the same animal?").

Every owner registration gives 5+ photos of ONE identity – exactly the
supervision re-ID needs. Export them with export_registry.py, optionally add
public identity datasets (e.g. PetFace, Cat Individual Images) in the
`<species>/<identity>/*.jpg` layout via prepare_data.py --folder, then:

    python3 training/train_reid.py --epochs 30

Trains a projection (DINOv2 768 → 256) with supervised-contrastive loss on
P×K batches, evaluates Recall@1/5 on held-out identities, and only saves
artifacts/reid_projection.npz if it beats the raw DINOv2 baseline.
⚠ After saving, run training/reindex.py so stored embeddings use the new space.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ROOT, extract_features, pick_device, read_manifest, split_of

ART = Path(__import__("os").environ.get("PETTI_ARTIFACTS_DIR", ROOT / "artifacts"))


class Proj(nn.Module):
    def __init__(self, d_in, d_out=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        self.skip = nn.Linear(d_in, d_out, bias=False)  # residual path keeps base geometry

    def forward(self, x):
        return F.normalize(self.net(x) + self.skip(x), dim=-1)


def supcon(z, labels, t=0.07):
    sim = z @ z.T / t
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def recall_at_k(emb: np.ndarray, ids: np.ndarray, ks=(1, 5)) -> dict:
    """Leave-one-out retrieval among val identities (query vs all others)."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -9)
    order = np.argsort(-sim, 1)
    hits = ids[order] == ids[:, None]
    valid = np.array([(ids == i).sum() > 1 for i in ids])
    return {f"R@{k}": round(float(hits[valid, :k].any(1).mean()), 4) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=32, help="identities per batch")
    ap.add_argument("--K", type=int, default=4, help="images per identity")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    manifests = args.manifest or sorted((ROOT / "data" / "manifests").glob("*.csv"))
    rows = [r for r in read_manifest(manifests) if r["pet_id"]]
    if not rows:
        raise SystemExit("No rows with pet_id. Export registry (export_registry.py) or add identity folders first.")
    feats = extract_features(rows, ART / "cache" / "features_reid.npz")
    X = feats["reid"]
    uniq = {p: i for i, p in enumerate(sorted({r["pet_id"] for r in rows}))}
    ids = np.array([uniq[r["pet_id"]] for r in rows])
    split = np.array([split_of(r, 0.2) for r in rows])
    tr, va = np.where(split == "train")[0], np.where(split == "val")[0]
    print(f"{len(uniq)} identities, train imgs={len(tr)} val imgs={len(va)}")

    base = recall_at_k(X[va], ids[va])
    print("baseline DINOv2:", base)

    by_id = defaultdict(list)
    for i in tr:
        by_id[ids[i]].append(i)
    good_ids = [k for k, v in by_id.items() if len(v) >= 2]
    device = pick_device()
    model = Proj(X.shape[1], args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    best, best_state = base["R@1"], None
    steps_per_epoch = max(1, len(good_ids) // args.P)
    for ep in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            chosen = np.random.choice(good_ids, size=min(args.P, len(good_ids)), replace=False)
            idx, lab = [], []
            for c in chosen:
                pick = np.random.choice(by_id[c], size=args.K, replace=len(by_id[c]) < args.K)
                idx += list(pick)
                lab += [c] * args.K
            x = Xt[idx] + 0.02 * torch.randn_like(Xt[idx])
            loss = supcon(model(x), torch.tensor(lab, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model(Xt[va]).cpu().numpy()
        r = recall_at_k(z, ids[va])
        print(f"epoch {ep+1}: loss={loss.item():.3f} {r}")
        if r["R@1"] > best:
            best, best_state = r["R@1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print("Projection did not beat the DINOv2 baseline → nothing saved (more identities needed).")
        return
    # fold residual into a 2-layer numpy MLP: out = W2·gelu(W1x+b1)+b2 + S·x
    # → export as separate 'proj' MLP + skip by concatenating into layer form
    w = {k: v.numpy() for k, v in best_state.items()}
    W1, b1, W2, b2, S = w["net.0.weight"], w["net.0.bias"], w["net.2.weight"], w["net.2.bias"], w["skip.weight"]
    np.savez(ART / "reid_projection.npz", **{"proj.0.weight": W1, "proj.0.bias": b1, "proj.1.weight": W2, "proj.1.bias": b2, "proj.skip": S})
    (ART / "reid_projection.json").write_text(json.dumps({"baseline": base, "best_R@1": best, "dim": args.dim}, indent=2))
    print(f"saved projection (R@1 {base['R@1']} → {best}). Now run: python3 training/reindex.py")


if __name__ == "__main__":
    main()
