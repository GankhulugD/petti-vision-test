#!/usr/bin/env python3
"""Evaluate the full lost&found matcher and fit its calibration.

Protocol (mirrors the real use-case): for every identity with ≥3 photos
(manifest rows sharing a pet_id), the LAST photo is the "found animal" query and
the remaining photos are the owner's registration. Every query is searched
against ALL registrations.

    python3 training/evaluate.py --manifest data/manifests/registry.csv --limit 300

Reports Recall@1/5/10 + MRR, fits the cosine→probability calibration
(logistic regression on positive/negative pairs) and grid-searches the score
weights, then writes artifacts/calibration.json (picked up by the server).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import ROOT, read_manifest
from tqdm import tqdm

from petti_vision import Config, PettiVision
from petti_vision.matcher import Matcher, _sig
from petti_vision.registry import PetRecord

ART = Path(__import__("os").environ.get("PETTI_ARTIFACTS_DIR", ROOT / "artifacts"))


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 3000, lr: float = 0.5) -> tuple[float, float]:
    """p = sigmoid((x - mu)/tau) fitted by gradient descent on log-loss."""
    m, s = float(x.mean()), float(x.std() + 1e-6)
    xs = (x - m) / s
    # balance classes (1 positive vs N-1 negatives per query)
    w = np.where(y > 0.5, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    a, b = 1.0, 0.0  # p = sigmoid(a xs + b)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * xs + b)))
        g = (p - y) * w
        a -= lr * float((g * xs).mean())
        b -= lr * float(g.mean())
    a = max(a, 1e-3)
    # floor tau: perfectly separable dev sets would otherwise give a step function
    return m - b * s / a, max(s / a, 0.02)  # mu, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--limit", type=int, default=300, help="max identities")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(list)
    for r in read_manifest(args.manifest):
        if r["pet_id"]:
            groups[r["pet_id"]].append(r)
    ids = [k for k, v in groups.items() if len(v) >= 3][: args.limit]
    print(f"{len(ids)} identities with ≥3 photos")

    cfg = Config()
    vision = PettiVision(cfg)
    gallery, queries = [], []
    for pid in tqdm(ids, desc="profiling"):
        rows = groups[pid]
        g = vision.analyze([r["path"] for r in rows[:-1]])
        q = vision.analyze([rows[-1]["path"]])
        gallery.append(PetRecord(pid, pid, "lost", g.species, g.data["category"], None, None, None, None, None, None, None,
                                 g.data, 0.0, [r["path"] for r in rows[:-1]], g.reid, g.vlm))
        queries.append((pid, q))

    matcher = Matcher(cfg)
    # collect raw components for every (query, candidate) pair
    comp = []  # (qi, ci, reid_cos, vlm_cos, attr, same)
    for qi, (pid, q) in enumerate(tqdm(queries, desc="matching")):
        for m in matcher.search(q, gallery, top_k=len(gallery)):
            comp.append((qi, ids.index(m.pet.id), m.details["reid_cos"], m.details["vlm_cos"], m.attr, m.pet.id == pid))
    if not comp:
        raise SystemExit("No candidate survived the species filter — check the detector / species predictions.")
    C = np.array(comp, dtype=np.float64)
    if not C[:, 5].any():
        raise SystemExit("True pets were never retrieved (species mismatch). Inspect profiles before calibrating.")
    same = C[:, 5].astype(bool)

    mu_r, tau_r = fit_logistic(C[:, 2], same.astype(float))
    mu_v, tau_v = fit_logistic(C[:, 3], same.astype(float))
    print(f"calibration reid: mu={mu_r:.3f} tau={tau_r:.3f} | vlm: mu={mu_v:.3f} tau={tau_v:.3f}")
    r_p = np.array([_sig((x - mu_r) / tau_r) for x in C[:, 2]])
    v_p = np.array([_sig((x - mu_v) / tau_v) for x in C[:, 3]])

    def metrics(score):
        ranks = []
        for qi in range(len(queries)):
            sel = C[:, 0] == qi
            s, y = score[sel], same[sel]
            if not y.any():
                ranks.append(10**6)  # species filter removed the true pet
                continue
            ranks.append(int((s > s[y].max()).sum()) + 1)
        ranks = np.array(ranks)
        return {"R@1": float((ranks <= 1).mean()), "R@5": float((ranks <= 5).mean()), "R@10": float((ranks <= 10).mean()),
                "MRR": float((1 / ranks).mean())}

    print("identity only  :", metrics(r_p))
    print("appearance only:", metrics(v_p))
    print("attributes only:", metrics(C[:, 4]))
    best, best_w = None, None
    for wr, wv in itertools.product(np.arange(0.2, 0.85, 0.05), np.arange(0.0, 0.4, 0.05)):
        wa = 1 - wr - wv
        if wa < 0:
            continue
        m = metrics(wr * r_p + wv * v_p + wa * C[:, 4])
        key = (m["R@1"], m["MRR"])
        if best is None or key > best[0]:
            best, best_w = (key, m), (round(wr, 2), round(wv, 2), round(wa, 2))
    print(f"best fused      : {best[1]} with w_reid={best_w[0]} w_vlm={best_w[1]} w_attr={best_w[2]}")

    if not args.no_write:
        ART.mkdir(parents=True, exist_ok=True)
        out = {"reid_mu": mu_r, "reid_tau": tau_r, "vlm_mu": mu_v, "vlm_tau": tau_v,
               "w_reid": best_w[0], "w_vlm": best_w[1], "w_attr": best_w[2], "metrics": best[1], "identities": len(ids)}
        (ART / "calibration.json").write_text(json.dumps(out, indent=2))
        print("wrote", ART / "calibration.json")


if __name__ == "__main__":
    main()
