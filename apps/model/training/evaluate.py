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
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import ROOT, read_manifest
from tqdm import tqdm

from petti_vision import Config, PettiVision
from petti_vision.matcher import Matcher
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


def loo_eval(vision, groups: dict, distractors: list):
    """Every photo of every identity is a query; gallery = all other photos
    (other identities + distractors + the identity's remaining photos).
    Pure identity-embedding metric with many more queries than the
    one-query-per-pet protocol, so small model changes become measurable."""
    paths, pid = [], []
    for k, rows in groups.items():
        for r in rows:
            paths.append(r["path"])
            pid.append(k)
    for i, r in enumerate(distractors):
        paths.append(r["path"])
        pid.append(f"_d{i}")
    E = []
    for p in tqdm(paths, desc="loo embed"):
        E.append(vision.analyze([p]).reid[0])
    E = np.stack(E)
    pid = np.array(pid)
    S = E @ E.T
    np.fill_diagonal(S, -9)
    ranks = []
    for q in range(len(paths)):
        if pid[q].startswith("_d"):
            continue
        same = pid == pid[q]
        same[q] = False
        if not same.any():
            continue
        # rank of the best same-identity photo among all other photos
        ranks.append(int((S[q] > S[q][same].max()).sum()) + 1)
    ranks = np.array(ranks)
    print(f"LOO identity ({len(ranks)} queries, {len(paths)} photos): R@1={np.mean(ranks <= 1):.3f} "
          f"R@5={np.mean(ranks <= 5):.3f} MRR={np.mean(1 / ranks):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--distractors", type=Path, action="append", default=[],
                    help="manifest of single photos of OTHER animals, added to the gallery only")
    ap.add_argument("--min-photos", type=int, default=2, help="photos per identity (1 query + ≥1 gallery)")
    ap.add_argument("--limit", type=int, default=300, help="max identities")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--loo", action="store_true",
                    help="only run image-level leave-one-out identity retrieval (every photo is a query once)")
    args = ap.parse_args()

    groups = defaultdict(list)
    for r in read_manifest(args.manifest):
        if r["pet_id"]:
            groups[r["pet_id"]].append(r)
    ids = [k for k, v in groups.items() if len(v) >= args.min_photos][: args.limit]
    print(f"{len(ids)} identities with ≥{args.min_photos} photos", flush=True)

    cfg = Config()
    vision = PettiVision(cfg)
    dis = read_manifest(args.distractors) if args.distractors else []
    if args.loo:  # quick model comparison: identity embeddings only, then stop
        loo_eval(vision, {k: groups[k] for k in ids}, dis)
        return
    gallery, queries = [], []

    def rec(pid, paths, prof):
        return PetRecord(pid, pid, "lost", prof.species, prof.data["category"], None, None, None, None, None, None, None,
                         prof.data, 0.0, list(paths), prof.reid, prof.vlm)

    for pid in tqdm(ids, desc="profiling"):
        rows = groups[pid]
        g = vision.analyze([r["path"] for r in rows[:-1]])
        q = vision.analyze([rows[-1]["path"]])
        gallery.append(rec(pid, [r["path"] for r in rows[:-1]], g))
        queries.append((pid, q))
    for i, r in enumerate(tqdm(dis, desc="distractors")):
        gallery.append(rec(f"distractor_{i}", [r["path"]], vision.analyze([r["path"]])))
    print(f"gallery: {len(gallery)} pets ({len(dis)} distractors)", flush=True)

    matcher = Matcher(cfg)
    comp = []  # (qi, reid_cos, vlm_cos, attr, production_score, same)
    for qi, (pid, q) in enumerate(tqdm(queries, desc="matching")):
        for m in matcher.search(q, gallery, top_k=len(gallery)):
            comp.append((qi, m.details["reid_cos"], m.details["vlm_cos"], m.attr, m.score, m.pet.id == pid))
    if not comp:
        raise SystemExit("No candidate survived the species filter — check the detector / species predictions.")
    C = np.array(comp, dtype=np.float64)
    same = C[:, 5].astype(bool)
    dump = ROOT / "data" / "eval" / "components.npy"  # (query, reid_cos, vlm_cos, attr, score, same) for offline analysis
    dump.parent.mkdir(parents=True, exist_ok=True)
    np.save(dump, C)
    if not same.any():
        raise SystemExit("True pets were never retrieved (species mismatch). Inspect profiles before calibrating.")

    def metrics(score, qsel=None):
        ranks = []
        for qi in range(len(queries)):
            if qsel is not None and qi not in qsel:
                continue
            sel = C[:, 0] == qi
            s, y = score[sel], same[sel]
            if not y.any():
                ranks.append(10**6)  # species filter removed the true pet
                continue
            ranks.append(int((s > s[y].max()).sum()) + 1)
        ranks = np.array(ranks)
        return {"R@1": round(float((ranks <= 1).mean()), 3), "R@5": round(float((ranks <= 5).mean()), 3),
                "R@10": round(float((ranks <= 10).mean()), 3), "MRR": round(float((1 / ranks).mean()), 3)}

    def fit(qsel):
        """identity calibration (μ, τ) + attribute weight β for logit fusion."""
        rows = np.isin(C[:, 0], list(qsel))
        mu_r, tau_r = fit_logistic(C[rows, 1], same[rows].astype(float))
        mu_v, tau_v = fit_logistic(C[rows, 2], same[rows].astype(float))
        best = None
        for beta in np.arange(0.0, 6.5, 0.5):
            m = metrics((C[:, 1] - mu_r) / tau_r + beta * (2 * C[:, 3] - 1), qsel)
            if best is None or (m["R@1"], m["MRR"]) > best[0]:
                best = ((m["R@1"], m["MRR"]), float(beta))
        return dict(reid_mu=mu_r, reid_tau=tau_r, vlm_mu=mu_v, vlm_tau=tau_v, attr_beta=best[1])

    def fused(p):
        return (C[:, 1] - p["reid_mu"]) / p["reid_tau"] + p["attr_beta"] * (2 * C[:, 3] - 1)

    print("production score :", metrics(C[:, 4]))
    print("identity (cos)   :", metrics(C[:, 1]))
    print("appearance (cos) :", metrics(C[:, 2]))
    print("attributes only  :", metrics(C[:, 3]))

    # 2-fold cross-validation of the fitted calibration (honest estimate)
    qs = list(range(len(queries)))
    A, B = set(qs[0::2]), set(qs[1::2])
    cv = [metrics(fused(fit(tr)), te) for tr, te in ((A, B), (B, A))]
    print("tuned (2-fold CV):", {k: round((cv[0][k] + cv[1][k]) / 2, 3) for k in cv[0]})

    p = fit(set(qs))
    z = fused(p)
    top_neg = [z[(C[:, 0] == qi) & ~same].max() for qi in qs if ((C[:, 0] == qi) & ~same).any()]
    p["display_offset"] = round(float(np.median(top_neg)), 2) if top_neg else 5.0
    print(f"fitted on all    : reid mu={p['reid_mu']:.3f} tau={p['reid_tau']:.3f}  attr_beta={p['attr_beta']}  "
          f"display_offset={p['display_offset']}  (vlm mu={p['vlm_mu']:.3f} tau={p['vlm_tau']:.3f})")

    if not args.no_write:
        ART.mkdir(parents=True, exist_ok=True)
        path = ART / "calibration.json"
        cal = json.loads(path.read_text()) if path.exists() else {}
        prod = metrics(C[:, 4])
        cal.update({k: p[k] for k in ("reid_mu", "reid_tau", "vlm_mu", "vlm_tau", "attr_beta", "display_offset")})
        cal["reid_eval"] = {"production_before": prod, "cv": cv, "identities": len(ids), "gallery": len(gallery)}
        path.write_text(json.dumps(cal, indent=2))
        print("wrote", path)


if __name__ == "__main__":
    main()
