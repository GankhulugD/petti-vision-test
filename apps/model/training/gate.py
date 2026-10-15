#!/usr/bin/env python3
"""Fail a candidate eval report if it regresses past the frozen baseline.

    python3 training/gate.py artifacts/reports/v0.4.json
    python3 training/gate.py --candidate artifacts/reports/v0.5.json --baseline artifacts/reports/baseline.json

A metric that is null on either side is skipped (not yet measured). A metric
present on the baseline and missing on the candidate is a failure. Latency
fails when p50 or p95 grows by more than 15%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (dotted path, higher-is-better). Tolerance is absolute on a 0–1 score ("1 pt").
HIGHER = {
    "species.dog.accuracy": 0.01,
    "species.cat.accuracy": 0.01,
    "breed.dog.top1": 0.01,
    "breed.dog.top3": 0.01,
    "breed.cat.top1": 0.01,
    "breed.cat.top3": 0.01,
    "size.accuracy": 0.01,
    "size.within1": 0.01,
    "age.accuracy": 0.01,
    "age.within1": 0.01,
    "color.f1": 0.01,
    "marks.f1": 0.01,
    "reid.R@1": 0.01,
    "reid.R@5": 0.01,
    "reid.mAP": 0.01,
    "detection.recall": 0.01,
    "detection.mask_iou": 0.01,
    "confidence.high.accuracy": 0.01,
}
LOWER = {
    "species.ece": 0.01,
    "breed.ece": 0.01,
    "size.ece": 0.01,
    "age.ece": 0.01,
    "reid.false_match_rate": 0.01,
    "negative.false_positive_rate": 0.01,
}
LATENCY = ("latency.p50_ms", "latency.p95_ms")
LATENCY_FACTOR = 1.15


def get(doc: dict, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare(baseline: dict, candidate: dict, latency_factor: float = LATENCY_FACTOR) -> list[str]:
    failures: list[str] = []
    for path, tol in {**HIGHER, **LOWER}.items():
        b, c = get(baseline, path), get(candidate, path)
        if b is None:
            continue
        if c is None:
            failures.append(f"{path}: baseline {b} but candidate has no measurement")
            continue
        higher = path in HIGHER
        drop = (b - c) if higher else (c - b)
        if drop > tol + 1e-9:
            failures.append(f"{path}: {b} → {c} (allowed {tol})")
    for path in LATENCY:
        b, c = get(baseline, path), get(candidate, path)
        if b is None or c is None or b <= 0:
            continue
        if c > b * latency_factor + 1e-6:
            failures.append(f"{path}: {b} ms → {c} ms (>{int((latency_factor - 1) * 100)}%)")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", type=Path)
    ap.add_argument("--candidate", dest="candidate_opt", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    args = ap.parse_args()
    cand_path = args.candidate_opt or args.candidate
    if cand_path is None:
        raise SystemExit("pass a candidate report path")
    base_path = args.baseline or cand_path.parent / "baseline.json"
    baseline = json.loads(base_path.read_text())
    candidate = json.loads(cand_path.read_text())
    failures = compare(baseline, candidate)
    if failures:
        print(f"GATE FAIL ({len(failures)})", flush=True)
        for line in failures:
            print(" -", line, flush=True)
        return 1
    print(f"GATE PASS  {cand_path.name} vs {base_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
