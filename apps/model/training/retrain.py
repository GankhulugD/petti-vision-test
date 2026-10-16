#!/usr/bin/env python3
"""The data flywheel in one command: new data → train → evaluate → gate → promote or roll back.

    python3 training/retrain.py               # monthly retrain
    python3 training/retrain.py --rollback    # restore the previous version

1. snapshot the current model files into artifacts/versions/<timestamp>/
2. collect new labelled data: registrations (export_registry.py) and consented
   user corrections (build_feedback_set.py)
3. retrain the attribute heads (auto_train.py --extra ...)
4. evaluate on the frozen Oxford test split (eval_breeds.py)
5. gate: dog/cat top-1 may not drop by more than 1 point, ECE may not rise by
   more than 0.02, compared with the last accepted report. Pass → keep the new
   files and record the report; fail → restore the snapshot automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

from common import ROOT

ART = ROOT / "artifacts"
VERSIONS = ART / "versions"
REPORTS = ART / "reports"
MODEL_FILES = ["heads.npz", "heads.json", "calibration.json", "marks_calibration.json", "seg_head2.npz"]
ACCEPTED = REPORTS / "oxford_test_accepted.json"


def run(script: str, *args: str):
    print(f"\n$ {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "training" / f"{script}.py"), *args], cwd=ROOT, check=True)


def snapshot() -> str:
    tag = time.strftime("%Y%m%d-%H%M%S")
    d = VERSIONS / tag
    d.mkdir(parents=True)
    for f in MODEL_FILES:
        if (ART / f).exists():
            shutil.copy2(ART / f, d / f)
    return tag


def restore(tag: str):
    d = VERSIONS / tag
    for f in MODEL_FILES:
        if (d / f).exists():
            shutil.copy2(d / f, ART / f)
    print(f"restored model files from versions/{tag}", flush=True)


def gate(new: dict, old: dict) -> list[str]:
    bad = []
    for sp in ("dog", "cat"):
        n, o = new.get(sp), old.get(sp)
        if not n or not o:
            continue
        if n["top1"] < o["top1"] - 0.01:
            bad.append(f"{sp} top-1 {o['top1']} → {n['top1']}")
        ne, oe = n.get("ece_calibrated", n["ece"]), o.get("ece_calibrated", o["ece"])
        if ne is not None and oe is not None and ne > oe + 0.02:
            bad.append(f"{sp} ECE {oe} → {ne}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--per-breed", default="40")
    a = ap.parse_args()
    if a.rollback:
        tags = sorted(p.name for p in VERSIONS.iterdir()) if VERSIONS.exists() else []
        if not tags:
            sys.exit("no saved versions")
        restore(tags[-1])
        return

    tag = snapshot()
    print(f"snapshot: versions/{tag}", flush=True)
    try:
        run("export_registry")
        run("build_feedback_set")
        extra = []
        for m in ("registry.csv", "feedback.csv"):
            p = ROOT / "data" / "manifests" / m
            if p.exists() and p.stat().st_size > 100:
                extra += ["--extra", str(p)]
        run("auto_train", "--datasets", "oxford", *extra)  # license-clean: no ImageNet/Stanford photos
        run("eval_breeds", "--per-breed", a.per_breed, "--tag", "oxford_test_candidate")
    except subprocess.CalledProcessError as exc:
        restore(tag)
        sys.exit(f"step failed ({exc}); previous model restored")

    new = json.loads((REPORTS / "oxford_test_candidate.json").read_text())["summary"]
    base = ACCEPTED if ACCEPTED.exists() else REPORTS / "oxford_test.json"  # first run: the v0.5 report
    if base.exists():
        problems = gate(new, json.loads(base.read_text())["summary"])
        if problems:
            restore(tag)
            print("GATE FAILED → rolled back:\n  " + "\n  ".join(problems), flush=True)
            sys.exit(1)
    # refit the display calibration for the new heads (raw probabilities in the report)
    run("fit_breed_calibration", "--report", str(REPORTS / "oxford_test_candidate.json"))
    shutil.copy2(REPORTS / "oxford_test_candidate.json", ACCEPTED)
    print("GATE PASSED → new model kept. POST /dev/reload (or restart) to serve it.", flush=True)


if __name__ == "__main__":
    main()
