#!/usr/bin/env python3
"""Calibrate distinctive-mark thresholds on ordinary pet photos.

    python3 training/calibrate_marks.py [--oxford 500]

Zero-shot mark scores saturate on some views (a dog seen from behind scores
"three legs: 0.94"). Ordinary photos almost never show a missing leg/eye, a
scar or a clipped ear, so the score distribution on them tells us how high a
threshold must be for a given false-alarm rate. Rare marks are capped at a
~0.5% firing rate, common ones at a plausible upper base rate. Writes
artifacts/marks_calibration.json, which the pipeline loads at start-up.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from common import ROOT

OX = ROOT / "data" / "public" / "oxford"
# highest share of ordinary photos on which a mark may plausibly be present
MAX_RATE = {
    "one_eye": 0.005, "three_legs": 0.005, "scar": 0.01, "heterochromia": 0.01, "ear_tip": 0.01,
    "clothes": 0.03, "eye_patch": 0.05, "blaze": 0.08, "blue_eyes": 0.08, "folded_ears": 0.08,
    "four_eyes": 0.12, "curled_tail": 0.2, "harness": 0.2, "face_mask": 0.3, "white_paws": 0.3,
    "white_chest": 0.35, "collar": 0.5, "floppy_ears": 0.65, "short_tail": 0.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxford", type=int, default=500)
    a = ap.parse_args()
    from petti_vision import Config, PettiVision
    from petti_vision import taxonomy as T

    rng = random.Random(0)
    test = (OX / "annotations" / "test.txt").read_text().split()[0::4]
    ox = [OX / "images" / f"{n}.jpg" for n in rng.sample(test, min(a.oxford, len(test)))]
    owner = sorted((ROOT / "data" / "eval" / "dogceo").rglob("*.jpg"))
    paths = [p for p in ox + owner if p.exists()]
    v = PettiVision(Config())
    v.mark_thresholds = {}  # measure raw scores
    probs: dict[str, list[float]] = {}
    for i, p in enumerate(paths):
        d = v.analyze([p.read_bytes()]).to_json()
        for k, s in d.get("mark_probs", {}).items():
            probs.setdefault(k, []).append(float(s))
        if i % 100 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    out = {}
    for m in T.MARKS:
        xs = np.array(probs.get(m.key, []))
        rate = MAX_RATE.get(m.key, 0.1)
        if rate <= 0 or len(xs) < 20:
            thr = max(m.threshold, 1.01)
        else:
            thr = float(np.quantile(xs, 1 - rate)) + 0.005
            thr = min(max(m.threshold, thr), 0.995)
        out[m.key] = {
            "threshold": round(thr, 4),
            "default": m.threshold,
            "n": int(len(xs)),
            "fired_before": round(float((xs >= m.threshold).mean()), 4) if len(xs) else None,
            "fired_after": round(float((xs >= thr).mean()), 4) if len(xs) else None,
            "q50": round(float(np.quantile(xs, 0.5)), 4) if len(xs) else None,
            "q99": round(float(np.quantile(xs, 0.99)), 4) if len(xs) else None,
        }
        print(m.key, out[m.key], flush=True)
    (ROOT / "artifacts" / "marks_calibration.json").write_text(json.dumps(out, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
