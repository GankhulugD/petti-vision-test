#!/usr/bin/env python3
"""Build a small re-identification test set from dog.ceo user photos.

Besides Stanford-Dogs images (n0xxxxxxx_*.jpg), dog.ceo contains phone photos
uploaded by owners, named after the dog: cooper1.jpg, cooper2.jpg, pippin-3.jpg…
Photos sharing a name inside one breed folder are (almost always) the same
individual → a realistic "is it THIS dog?" benchmark. ImageNet photos of the
same breeds are added as look-alike distractors.

    python3 training/fetch_identity_set.py
    python3 training/evaluate.py --manifest data/eval/dogceo_ids.csv --distractors data/eval/dogceo_distractors.csv
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, write_manifest

OUT = ROOT / "data" / "eval"
IMG = OUT / "dogceo"
# generic words or breed names (e.g. "jack" in a Jack Russell folder) are not individuals
SKIP = re.compile(r"dog|img|image|photo|pic|puppy|pup|screen|whatsapp|dsc|pxl|unnamed|black|white|brown|daschund|^jack$", re.I)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def download(url: str, dest: Path) -> Path | None:
    if dest.exists():
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r:
            dest.write_bytes(r.read())
        return dest
    except Exception:
        return None


def main(max_per_id: int = 8, distractors_per_breed: int = 8, seed: int = 0):
    breeds = []
    for b, subs in get_json("https://dog.ceo/api/breeds/list/all")["message"].items():
        breeds += [f"{b}/{s}" for s in subs] if subs else [b]

    def listing(b):
        try:
            return b, get_json(f"https://dog.ceo/api/breed/{b}/images")["message"]
        except Exception:
            return b, []

    with ThreadPoolExecutor(16) as ex:
        lists = dict(ex.map(listing, breeds))

    groups: dict[str, list[str]] = {}
    imagenet: dict[str, list[str]] = {}
    for b, urls in lists.items():
        for u in urls:
            f = u.rsplit("/", 1)[-1]
            if re.match(r"^n\d{8}_", f):
                imagenet.setdefault(b, []).append(u)
                continue
            stem = re.sub(r"\.(jpe?g|png)$", "", f, flags=re.I)
            m = re.match(r"^([A-Za-z][A-Za-z_\-]*?)[_\- ]?\(?(\d{1,3})\)?$", stem)
            if not m:
                continue
            name = m.group(1).lower().strip("-_")
            breed_words = re.split(r"[/_\-]", b.lower())
            if len(name) < 3 or SKIP.search(name) or any(w and (w in name or name in w) for w in breed_words):
                continue
            groups.setdefault(f"{b.replace('/', '-')}__{name}", []).append(u)
    groups = {k: sorted(v)[:max_per_id] for k, v in groups.items() if len(v) >= 2}
    print(f"{len(groups)} identities, {sum(map(len, groups.values()))} photos", flush=True)

    rng = random.Random(seed)
    jobs, rows, drows = [], [], []
    for gid, urls in groups.items():
        for i, u in enumerate(urls):
            dest = IMG / gid / f"{i}{Path(u).suffix.lower()}"
            jobs.append((u, dest))
            rows.append({"path": str(dest), "species": "dog", "pet_id": gid, "source": "dogceo_id"})
    for b in {k.split("__")[0].replace("-", "/") for k in groups}:
        pool = imagenet.get(b, [])
        for u in rng.sample(pool, min(distractors_per_breed, len(pool))):
            dest = IMG / "_distractors" / u.rsplit("/", 1)[-1]
            jobs.append((u, dest))
            drows.append({"path": str(dest), "species": "dog", "source": "dogceo_distractor"})
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda j: download(*j), jobs))
    good = {str(p) for p in ok if p}
    rows = [r for r in rows if r["path"] in good]
    drows = [r for r in drows if r["path"] in good]
    write_manifest(OUT / "dogceo_ids.csv", rows)
    write_manifest(OUT / "dogceo_distractors.csv", drows)
    print(f"wrote {len(rows)} identity photos, {len(drows)} distractors → {OUT}", flush=True)


if __name__ == "__main__":
    main()
