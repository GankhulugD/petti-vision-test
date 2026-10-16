#!/usr/bin/env python3
"""Serve a tagged head variant: heads_<tag>.npz/json → heads.npz/json.

    python3 training/promote_heads.py --tag clean

The files being replaced are saved first under artifacts/versions/<timestamp>/
(restore with: python3 training/retrain.py --rollback).
"""

from __future__ import annotations

import argparse
import shutil

from retrain import ART, snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    src = [ART / f"heads_{a.tag}.npz", ART / f"heads_{a.tag}.json"]
    missing = [p.name for p in src if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}")
    tag = snapshot()
    for p in src:
        shutil.copy2(p, ART / p.name.replace(f"heads_{a.tag}", "heads"))
    print(f"previous heads saved in versions/{tag}; now serving heads_{a.tag}. POST /dev/reload to apply.", flush=True)


if __name__ == "__main__":
    main()
