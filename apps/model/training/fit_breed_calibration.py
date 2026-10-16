#!/usr/bin/env python3
"""Make breed confidences mean what they say.

    python3 training/fit_breed_calibration.py --report artifacts/reports/oxford_test.json

Uses the per-image breed probability vectors written by eval_breeds.py. Fits,
with 2-fold cross-fitting (so the numbers we report are on images the fit did
not see):
  * a display temperature T for the final breed distribution (min NLL)
  * level thresholds: "high" where ≥95% of answers are right, "medium" ≥70%,
    and the "unsure" cut below which top-1 is right <40% of the time
The final parameters (fit on all images) go into artifacts/calibration.json:
breed_display_temperature, breed_levels [high, medium], breed_unsure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT


def temp(P: np.ndarray, T: float) -> np.ndarray:
    lg = np.log(np.clip(P, 1e-9, 1)) / T
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def fit_T(P, y):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.25), np.log(3), 80)):
        nll = -np.log(np.clip(temp(P, T)[np.arange(len(y)), y], 1e-9, 1)).mean()
        best = min(best, (nll, float(T)))
    return best[1]


def thresholds(p1, ok):
    grid = np.linspace(0.2, 0.99, 80)
    hi = next((t for t in grid if (p1 >= t).sum() >= 20 and ok[p1 >= t].mean() >= 0.95), 0.9)
    mid = next((t for t in grid if t < hi and ((p1 >= t) & (p1 < hi)).sum() >= 20 and ok[(p1 >= t) & (p1 < hi)].mean() >= 0.7), min(0.5, hi))
    uns = max((t for t in grid if t < mid and (p1 < t).sum() >= 10 and ok[p1 < t].mean() < 0.4), default=0.2)
    return float(hi), float(mid), float(uns)


def ece(conf, ok, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def summarize(P, y, T, th):
    Q = temp(P, T)
    p1, pred = Q.max(1), Q.argmax(1)
    ok = (pred == y).astype(float)
    hi, mid, uns = th
    lv = {"high": p1 >= hi, "medium": (p1 >= mid) & (p1 < hi), "low": p1 < mid}
    return {
        "n": int(len(y)), "top1": round(float(ok.mean()), 3), "ece": round(ece(p1, ok), 4),
        "by_level": {k: {"n": int(m.sum()), "top1": round(float(ok[m].mean()), 3) if m.any() else None} for k, m in lv.items()},
        "unsure_rate": round(float((p1 < uns).mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=ROOT / "artifacts" / "reports" / "oxford_test.json")
    a = ap.parse_args()
    rows = [r for r in json.loads(a.report.read_text())["rows"] if r.get("probs") and r["breed"] in r["probs"]]
    keys = sorted({k for r in rows for k in r["probs"]})
    idx = {k: i for i, k in enumerate(keys)}
    P = np.zeros((len(rows), len(keys)))
    for i, r in enumerate(rows):
        for k, v in r["probs"].items():
            P[i, idx[k]] = v
    P /= P.sum(1, keepdims=True)
    y = np.array([idx[r["breed"]] for r in rows])
    fold = np.array([int(hashlib.md5(r["name"].encode()).hexdigest(), 16) % 2 for r in rows])

    before = summarize(P, y, 1.0, (0.6, 0.35, 0.3))
    cross = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        T = fit_T(P[tr], y[tr])
        Q = temp(P[tr], T)
        th = thresholds(Q.max(1), (Q.argmax(1) == y[tr]).astype(float))
        cross.append({"T": round(T, 3), "thresholds": [round(x, 3) for x in th], "heldout": summarize(P[te], y[te], T, th)})
    T = fit_T(P, y)
    Q = temp(P, T)
    th = thresholds(Q.max(1), (Q.argmax(1) == y).astype(float))
    out = {"before": before, "cross_fit": cross, "final": {"T": round(T, 3), "levels": [round(th[0], 3), round(th[1], 3)], "unsure": round(th[2], 3)}}
    print(json.dumps(out, indent=1), flush=True)
    cal_p = ROOT / "artifacts" / "calibration.json"
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else {}
    cal.update({"breed_display_temperature": out["final"]["T"], "breed_levels": out["final"]["levels"], "breed_unsure": out["final"]["unsure"]})
    cal_p.write_text(json.dumps(cal, indent=2))
    (ROOT / "artifacts" / "reports" / "breed_calibration.json").write_text(json.dumps(out, indent=2))
    print("wrote calibration.json", flush=True)


if __name__ == "__main__":
    main()
