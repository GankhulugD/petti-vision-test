#!/usr/bin/env python3
"""Breed accuracy on the frozen Oxford-IIIT Pet TEST split (12 cat + 25 dog breeds).

    python3 training/eval_breeds.py [--per-breed 40] [--tag oxford_test]

The heads never see this split (prepare_data.oxford() holds it out), so this is
the clean breed benchmark for cats and dogs. Reports top-1/top-3 per species,
ECE, accuracy per confidence level, how often the answer is "unsure", species
accuracy and the mark firing rates on these ordinary photos.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import ROOT
from prepare_data import OXFORD_MAP

OX = ROOT / "data" / "public" / "oxford"


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok, dtype=float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return round(float(e), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-breed", type=int, default=40)
    ap.add_argument("--tag", default="oxford_test")
    ap.add_argument("--heads", default="", help="evaluate artifacts/<this>.npz instead of the served heads")
    a = ap.parse_args()
    from petti_vision import Config, PettiVision

    rng = random.Random(0)
    by_breed = defaultdict(list)
    for n in (OX / "annotations" / "test.txt").read_text().split()[0::4]:
        by_breed[n.rsplit("_", 1)[0]].append(n)
    items = []
    for name, ns in sorted(by_breed.items()):
        key = OXFORD_MAP.get(name.lower())
        if not key:
            continue
        sp = "cat" if name[0].isupper() else "dog"
        for n in rng.sample(ns, min(a.per_breed, len(ns))):
            items.append((n, sp, key))
    print(f"{len(items)} images, {len({i[2] for i in items})} breeds", flush=True)

    v = PettiVision(Config())
    if a.heads:
        from petti_vision.heads import load_heads

        v.heads = load_heads(Path(v.cfg.artifacts_dir) / f"{a.heads}.npz")
        print(f"heads: {a.heads}", flush=True)
    # measure RAW probabilities (display calibration off) so fit_breed_calibration.py can
    # fit an absolute temperature; calibrated metrics are derived below
    T_cal, levels = v.breed_display_T, v.breed_levels
    v.breed_display_T = 1.0
    rows = []
    for k, (n, sp, key) in enumerate(items):
        d = v.analyze([(OX / "images" / f"{n}.jpg").read_bytes()]).to_json()
        b = d.get("breed") or {}
        top = [t["key"] for t in b.get("top", [])]
        rows.append({
            "name": n, "species": sp, "breed": key,
            "pred_species": d["species"]["key"],
            "top1": top[0] if top else None, "top3": top[:3],
            "conf": b["top"][0]["confidence"] if top else 0.0,
            "level": b.get("level"), "uncertain": bool(b.get("uncertain")), "mixed": bool(b.get("is_mixed")),
            "marks": [m["key"] for m in d.get("marks", [])],
            "probs": b.get("probs", {}),
        })
        if k % 200 == 0:
            print(f"{k}/{len(items)}", flush=True)

    out = {}
    for sp in ("dog", "cat"):
        rs = [r for r in rows if r["species"] == sp]
        if not rs:
            continue
        ok1 = [r["top1"] == r["breed"] for r in rs]
        lv = {}
        for level in ("high", "medium", "low"):
            sel = [o for o, r in zip(ok1, rs) if r["level"] == level]
            lv[level] = {"n": len(sel), "top1": round(float(np.mean(sel)), 3) if sel else None}
        cal_p1, cal_ok = [], []
        for r in rs:
            if r["probs"]:
                ks, ps = zip(*r["probs"].items())
                lg = np.log(np.clip(np.array(ps), 1e-9, 1)) / T_cal
                q = np.exp(lg - lg.max())
                q /= q.sum()
                cal_p1.append(float(q.max()))
                cal_ok.append(ks[int(q.argmax())] == r["breed"])
        cal_p1, cal_ok = np.array(cal_p1), np.array(cal_ok, dtype=float)
        hi_m = cal_p1 >= levels[0]
        out[sp] = {
            "ece_calibrated": ece(cal_p1, cal_ok) if len(cal_p1) else None,
            "high_calibrated": {"n": int(hi_m.sum()), "top1": round(float(cal_ok[hi_m].mean()), 3) if hi_m.any() else None},
            "display_T": T_cal,
            "n": len(rs),
            "species_acc": round(float(np.mean([r["pred_species"] == sp for r in rs])), 3),
            "top1": round(float(np.mean(ok1)), 3),
            "top3": round(float(np.mean([r["breed"] in r["top3"] for r in rs])), 3),
            "ece": ece([r["conf"] for r in rs], ok1),
            "by_level": lv,
            "uncertain_rate": round(float(np.mean([r["uncertain"] for r in rs])), 3),
        }
    marks = defaultdict(int)
    for r in rows:
        for m in r["marks"]:
            marks[m] += 1
    out["marks_fired_rate"] = {m: round(c / len(rows), 3) for m, c in sorted(marks.items(), key=lambda x: -x[1])}
    print(json.dumps(out, indent=1), flush=True)
    rep = ROOT / "artifacts" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{a.tag}.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
