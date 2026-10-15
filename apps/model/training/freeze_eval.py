#!/usr/bin/env python3
"""Hash the local eval images and write data/eval/manifest.json.

    python3 training/freeze_eval.py
    python3 training/freeze_eval.py --fetch-bench   # dog.ceo files named in bench_exclude.txt

Images stay on disk (data/ is large). The manifest is the freeze: sha256,
perceptual hash, source, license. Training (prepare_data.dogceo) drops any
photo whose perceptual hash is in this file.

A small synthetic negative set (drawings of a car, a person, furniture, a toy)
is written under data/eval/negatives/. It measures "a sketch is not a pet".
It is not a photograph of those things; that limitation is recorded on the set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, ahash, file_sha256, read_manifest, write_manifest
from PIL import Image, ImageDraw

EVAL = ROOT / "data" / "eval"

# Evaluation only. ImageNet / Stanford Dogs does not grant commercial training.
LICENSES = {
    "dogceo_id": {
        "source": "dog.ceo owner-named phone photos (same-name files in one breed folder)",
        "license": "uploaded by owners to dog.ceo; no commercial grant found",
        "allowed_use": "evaluation only",
    },
    "dogceo_distractor": {
        "source": "Stanford Dogs images served by dog.ceo (ImageNet synsets)",
        "license": "ImageNet: non-commercial research and education",
        "allowed_use": "evaluation only",
    },
    "bench_stanford": {
        "source": "Stanford Dogs filenames listed in bench_exclude.txt, via dog.ceo",
        "license": "ImageNet: non-commercial research and education",
        "allowed_use": "evaluation only",
    },
    "synthetic_negative": {
        "source": "generated in freeze_eval.py (line drawings, not photographs)",
        "license": "generated for this repo",
        "allowed_use": "evaluation only",
    },
}


def _rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _files(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        p = Path(r["path"])
        if not p.exists():
            continue
        out.append({
            "path": _rel(p),
            "sha256": file_sha256(p),
            "phash": ahash(p),
            "species": r.get("species") or "",
            "breed": r.get("breed") or "",
            "pet_id": r.get("pet_id") or "",
            "source": r.get("source") or "",
        })
    return out


def _draw(kind: str, dest: Path):
    im = Image.new("RGB", (320, 240), (245, 245, 240))
    d = ImageDraw.Draw(im)
    if kind == "car":
        d.rounded_rectangle((40, 110, 280, 180), radius=12, outline=(20, 20, 20), width=3)
        d.ellipse((70, 165, 110, 205), outline=(20, 20, 20), width=3)
        d.ellipse((210, 165, 250, 205), outline=(20, 20, 20), width=3)
    elif kind == "person":
        d.ellipse((140, 40, 180, 80), outline=(20, 20, 20), width=3)
        d.line((160, 80, 160, 160), fill=(20, 20, 20), width=3)
        d.line((160, 100, 110, 140), fill=(20, 20, 20), width=3)
        d.line((160, 100, 210, 140), fill=(20, 20, 20), width=3)
        d.line((160, 160, 120, 220), fill=(20, 20, 20), width=3)
        d.line((160, 160, 200, 220), fill=(20, 20, 20), width=3)
    elif kind == "furniture":
        d.rectangle((60, 80, 260, 150), outline=(20, 20, 20), width=3)
        d.line((70, 150, 70, 210), fill=(20, 20, 20), width=3)
        d.line((250, 150, 250, 210), fill=(20, 20, 20), width=3)
    elif kind == "toy":
        d.ellipse((110, 70, 210, 170), outline=(180, 40, 40), width=4)
        d.ellipse((140, 100, 155, 115), fill=(20, 20, 20))
        d.ellipse((165, 100, 180, 115), fill=(20, 20, 20))
    else:  # drawing of a house, still not an animal
        d.polygon([(160, 30), (40, 120), (280, 120)], outline=(20, 20, 20))
        d.rectangle((90, 120, 230, 210), outline=(20, 20, 20), width=3)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=90)


def write_negatives() -> list[dict]:
    rows = []
    for kind in ("car", "person", "furniture", "toy", "drawing"):
        for i in range(2):
            dest = EVAL / "negatives" / f"{kind}_{i}.jpg"
            _draw(kind, dest)
            rows.append({
                "path": str(dest),
                "species": "",
                "breed": "",
                "size": "",
                "age": "",
                "colors": "",
                "pattern": "",
                "coat": "",
                "marks": "",
                "pet_id": "",
                "source": f"synthetic_negative:{kind}",
            })
    write_manifest(EVAL / "negatives.csv", rows)
    return rows


def fetch_bench(names: set[str]) -> list[dict]:
    """Download Stanford Dogs files whose names are already on the exclude list."""
    import json
    import urllib.request

    from prepare_data import DOGCEO_MAP

    want = {n for n in names if n.startswith("n")}
    rows = []
    for slug, key in DOGCEO_MAP.items():
        if not want:
            break
        try:
            with urllib.request.urlopen(f"https://dog.ceo/api/breed/{slug}/images", timeout=30) as r:
                urls = json.loads(r.read())["message"]
        except Exception as exc:
            print(f"  skip {slug}: {exc}", flush=True)
            continue
        for url in urls:
            name = url.rsplit("/", 1)[-1]
            if name not in want:
                continue
            dest = EVAL / "bench" / key / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(url, timeout=30) as r:
                    dest.write_bytes(r.read())
            want.discard(name)
            rows.append({
                "path": str(dest), "species": "dog", "breed": key, "size": "", "age": "",
                "colors": "", "pattern": "", "coat": "", "marks": "", "pet_id": "",
                "source": "bench_stanford",
            })
    if want:
        print(f"bench files not on dog.ceo: {len(want)}", flush=True)
    write_manifest(EVAL / "attributes.csv", rows)
    print(f"bench dogs: {len(rows)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-bench", action="store_true")
    args = ap.parse_args()
    exclude_path = EVAL / "bench_exclude.txt"
    names = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    if args.fetch_bench:
        fetch_bench(names)
    neg_rows = write_negatives()
    sets = {}
    for key, path in (
        ("identity", EVAL / "dogceo_ids.csv"),
        ("distractors", EVAL / "dogceo_distractors.csv"),
        ("attributes", EVAL / "attributes.csv"),
        ("negatives", EVAL / "negatives.csv"),
    ):
        rows = read_manifest([path]) if path.exists() else []
        if key == "negatives":
            rows = neg_rows
        source = (rows[0].get("source") if rows else "") or key
        lic_key = source.split(":", 1)[0]
        sets[key] = {
            **LICENSES.get(lic_key, {"source": source, "license": "unspecified", "allowed_use": "evaluation only"}),
            "n": len(rows),
            "files": _files(rows),
        }
    doc = {
        "version": "v0.4",
        "role": "frozen evaluation — excluded from training by filename (bench_exclude.txt) and perceptual hash",
        "bench_exclude_names": sorted(names),
        "sets": sets,
    }
    dest = EVAL / "manifest.json"
    dest.write_text(json.dumps(doc, indent=2))
    n = sum(len(s["files"]) for s in sets.values())
    print(f"wrote {dest} ({n} files)", flush=True)


if __name__ == "__main__":
    main()
