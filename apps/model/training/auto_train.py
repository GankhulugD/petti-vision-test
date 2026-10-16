#!/usr/bin/env python3
"""One command: public datasets → features → trained heads → report.

    python3 training/auto_train.py                    # Oxford-IIIT Pet + Stanford Dogs
    python3 training/auto_train.py --max-per-class 80 # quicker run
    python3 training/auto_train.py --skip-download    # datasets already in data/public

What it does
  1. downloads Oxford-IIIT Pet (37 cat/dog breeds, ~800 MB) and Stanford Dogs
     (120 dog breeds, ~750 MB) once into data/public/
  2. builds a balanced manifest (≤ --max-per-class images per breed) →
     data/manifests/public_auto.csv
  3. extracts frozen-backbone features (cached in artifacts/cache/)
  4. trains the attribute heads (species, breed) + picks the
     head/zero-shot fusion weight on a held-out split
  5. prints and saves artifacts/auto_train_report.json

On an Apple-silicon Mac (MPS) a full run takes ~15-25 min, most of it step 3.
Restart the server (or POST /dev/reload) afterwards to load the new heads.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_data as P  # noqa: E402
from common import ROOT, write_manifest  # noqa: E402


def _progress_download(url: str, dest: Path):
    if dest.exists():
        print(f"[download] cached {dest.name}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = [0.0]

    def hook(blocks, bs, total):
        done = blocks * bs
        if time.time() - last[0] > 5:
            last[0] = time.time()
            pct = f"{100 * done / total:.0f}%" if total > 0 else f"{done / 1e6:.0f} MB"
            print(f"[download] {dest.name} {pct}", flush=True)

    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dest)


def build_rows(datasets: list[str], max_per_class: int, max_unmapped: int, seed: int = 0, exclude: set[str] | None = None) -> list[dict]:
    P._download = _progress_download  # nicer progress output
    rows: list[dict] = []
    if "oxford" in datasets:
        rows += P.oxford()
    if "stanford" in datasets:
        rows += P.stanford_dogs()
    if "dogceo" in datasets:
        rows += P.dogceo(max_per_breed=max_per_class, exclude=exclude)
    if exclude:  # keep benchmark images out of training (dog.ceo serves Stanford Dogs files)
        before = len(rows)
        rows = [r for r in rows if Path(r["path"]).name not in exclude]
        print(f"[data] excluded {before - len(rows)} benchmark images", flush=True)
    rng = random.Random(seed)
    rng.shuffle(rows)
    per = defaultdict(int)
    unmapped = defaultdict(int)
    out = []
    for r in rows:
        if r["breed"]:
            if per[r["breed"]] < max_per_class:
                per[r["breed"]] += 1
                out.append(r)
        elif unmapped[r["species"]] < max_unmapped:  # still useful for the species head
            unmapped[r["species"]] += 1
            out.append(r)
    print(f"[data] {len(out)} images, {len(per)} breeds, unmapped/species {dict(unmapped)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="oxford",
                    help="oxford (CC BY-SA, default: shippable); dogceo / stanford are ImageNet photos → non-commercial, experiments only")
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--max-unmapped", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--tag", default="", help="save as artifacts/heads_<tag>.npz/json and keep the served heads untouched")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="extra manifest CSV(s), e.g. data/manifests/feedback.csv (owner/user-confirmed labels)")
    ap.add_argument("--exclude", type=Path, default=ROOT / "data" / "eval" / "bench_exclude.txt",
                    help="file names (one per line) that must never be trained on")
    a = ap.parse_args()

    t0 = time.time()
    if a.skip_download:
        P._download = lambda url, dest: None
    exclude = set(a.exclude.read_text().split()) if a.exclude and a.exclude.exists() else None
    rows = build_rows(a.datasets.split(","), a.max_per_class, a.max_unmapped, exclude=exclude)
    if a.extra:
        from common import read_manifest

        extra = [r for r in read_manifest(a.extra) if Path(r["path"]).exists() and (r["species"] or r["breed"])]
        if exclude:
            extra = [r for r in extra if Path(r["path"]).name not in exclude]
        print(f"[data] +{len(extra)} rows from {', '.join(map(str, a.extra))}", flush=True)
        rows += extra
    manifest = ROOT / "data" / "manifests" / "public_auto.csv"
    write_manifest(manifest, rows)

    import shutil

    import train_heads

    art = Path(train_heads.ART)
    backup = {}
    if a.tag:
        for f in ("heads.npz", "heads.json"):
            if (art / f).exists():
                backup[f] = (art / f).read_bytes()
    sys.argv = ["train_heads.py", "--manifest", str(manifest), "--epochs", str(a.epochs), "--tasks", "species,breed"]
    train_heads.main()

    meta = json.loads((art / "heads.json").read_text())["tasks"]
    if a.tag:
        for f in ("heads.npz", "heads.json"):
            shutil.copy2(art / f, art / f.replace("heads", f"heads_{a.tag}"))
            if f in backup:
                (art / f).write_bytes(backup[f])
        print(f"[tag] saved heads_{a.tag}.npz/json; served heads restored", flush=True)
    report = {
        "images": len(rows),
        "minutes": round((time.time() - t0) / 60, 1),
        "tasks": {t: {k: m.get(k) for k in ("n_train", "n_val", "zs_val_acc", "val_acc", "fused_val_acc", "alpha", "temperature")}
                  for t, m in meta.items()},
    }
    (art / (f"auto_train_report_{a.tag}.json" if a.tag else "auto_train_report.json")).write_text(json.dumps(report, indent=2))
    print("\n=== REPORT ===", flush=True)
    for t, m in report["tasks"].items():
        print(f"{t:8s} zero-shot={m['zs_val_acc']}  head={m['val_acc']}  fused={m['fused_val_acc']}  (train {m['n_train']}, val {m['n_val']})", flush=True)
    print(f"done in {report['minutes']} min → {art / 'heads.npz'}", flush=True)


if __name__ == "__main__":
    main()
