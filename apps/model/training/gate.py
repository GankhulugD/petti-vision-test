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


def sample_n(doc: dict, path: str) -> int | None:
    """Look for an 'n' (or 'queries') next to the metric or one level up."""
    parts = path.split(".")
    for depth in (len(parts) - 1, len(parts) - 2):
        if depth < 1:
            continue
        node = get(doc, ".".join(parts[:depth]))
        if isinstance(node, dict):
            for key in ("n", "queries"):
                if isinstance(node.get(key), int) and node[key] > 0:
                    return node[key]
    return None


def compare(baseline: dict, candidate: dict, latency_factor: float = LATENCY_FACTOR) -> list[str]:
    """A drop fails when it exceeds the fixed tolerance AND two standard errors of
    the measurement: with 54 breed images one photo is already 1.9 points, so a
    fixed 1-point rule would flag pure noise."""
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
        n = sample_n(candidate, path) or sample_n(baseline, path)
        allowed = tol
        if n and 0.0 <= b <= 1.0:
            p = min(max(b, 0.02), 0.98)
            allowed = max(tol, 2 * (p * (1 - p) / n) ** 0.5)
        if drop > allowed + 1e-9:
            failures.append(f"{path}: {b} → {c} (allowed {allowed:.3f}{f', n={n}' if n else ''})")
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
    ap.add_argument("--waive", action="append", default=[], metavar="METRIC=REASON",
                    help="accept a known regression explicitly; the waiver is written into the candidate report")
    args = ap.parse_args()
    cand_path = args.candidate_opt or args.candidate
    if cand_path is None:
        raise SystemExit("pass a candidate report path")
    base_path = args.baseline or cand_path.parent / "baseline.json"
    baseline = json.loads(base_path.read_text())
    candidate = json.loads(cand_path.read_text())
    failures = compare(baseline, candidate)
    waivers = dict(w.split("=", 1) for w in args.waive if "=" in w)
    waived = [f for f in failures if f.split(":")[0] in waivers]
    failures = [f for f in failures if f.split(":")[0] not in waivers]
    if waived:
        candidate.setdefault("gate_waivers", {}).update({f.split(":")[0]: {"result": f, "reason": waivers[f.split(":")[0]]} for f in waived})
        cand_path.write_text(json.dumps(candidate, indent=2))
        for f in waived:
            print(" ~ waived:", f, "—", waivers[f.split(":")[0]], flush=True)
    if failures:
        print(f"GATE FAIL ({len(failures)})", flush=True)
        for line in failures:
            print(" -", line, flush=True)
        return 1
    print(f"GATE PASS  {cand_path.name} vs {base_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
