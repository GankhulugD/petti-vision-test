#!/usr/bin/env python3
"""Download a small no-animal photo set from Wikimedia Commons.

Train images go to data/public/scenes/train. A held-out slice goes to
data/eval/negatives_photo and is listed in the frozen manifest. Licenses
that forbid commercial use (NC) are skipped. Nothing here is larger than
a few megabytes.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from common import ROOT, ahash, file_sha256

EVAL = ROOT / "data" / "eval"
TRAIN = ROOT / "data" / "public" / "scenes" / "train"
CATEGORIES = {
    "car": "Category:Automobiles",
    "person": "Category:People",
    "furniture": "Category:Furniture",
    "toy": "Category:Toys",
}
ALLOWED = ("cc0", "public domain", "cc by 2", "cc by 3", "cc by 4", "cc by-sa")


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "PettiVision/0.4 (local research; contact: local)"})
    return urllib.request.urlopen(req, timeout=40)


def _api(title: str) -> dict:
    q = urllib.parse.urlencode({
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": title,
        "gcmtype": "file",
        "gcmlimit": "30",
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "384",
        "format": "json",
    })
    with _open(f"https://commons.wikimedia.org/w/api.php?{q}") as r:
        return json.loads(r.read())


def _license(info: dict) -> str:
    meta = info.get("extmetadata") or {}
    return (meta.get("LicenseShortName") or {}).get("value") or ""


def _ok(lic: str) -> bool:
    s = lic.lower()
    if "nc" in s or "noncommercial" in s or "non-commercial" in s:
        return False
    return any(k in s for k in ALLOWED)


def main():
    TRAIN.mkdir(parents=True, exist_ok=True)
    photo_dir = EVAL / "negatives_photo"
    photo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, title in CATEGORIES.items():
        try:
            pages = _api(title).get("query", {}).get("pages", {})
        except Exception as exc:
            print(f"skip {kind}: {exc}", flush=True)
            continue
        kept = 0
        for page in pages.values():
            info = (page.get("imageinfo") or [None])[0]
            if not info or not str(info.get("mime", "")).startswith("image/"):
                continue
            lic = _license(info)
            if not _ok(lic):
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            split = "train" if kept < 8 else "eval"
            if kept >= 12:
                break
            dest_dir = TRAIN if split == "train" else photo_dir
            dest = dest_dir / f"{kind}_{kept}.jpg"
            if not dest.exists():
                try:
                    with _open(url) as r:
                        dest.write_bytes(r.read())
                except Exception as exc:
                    print(f"  fail {url}: {exc}", flush=True)
                    continue
            rec = {"file": dest.name, "kind": kind, "split": split, "license": lic, "url": info.get("descriptionurl") or url,
                   "sha256": file_sha256(dest), "phash": ahash(dest)}
            rows.append(rec)
            kept += 1
        print(f"{kind}: {kept}", flush=True)
    (EVAL / "negatives_photo_licenses.json").write_text(json.dumps(rows, indent=2))
    # extend the frozen manifest without dropping existing sets
    man_path = EVAL / "manifest.json"
    doc = json.loads(man_path.read_text()) if man_path.exists() else {"version": "v0.4", "sets": {}}
    doc["sets"]["negatives_photo"] = {
        "source": "Wikimedia Commons categories Automobiles, People, Furniture, Toys",
        "license": "per-file, see negatives_photo_licenses.json; NC skipped",
        "allowed_use": "evaluation only",
        "n": sum(1 for r in rows if r["split"] == "eval"),
        "files": [
            {"path": f"data/eval/negatives_photo/{r['file']}", "sha256": r["sha256"], "phash": r["phash"],
             "species": "", "breed": "", "pet_id": "", "source": f"commons:{r['kind']}"}
            for r in rows if r["split"] == "eval"
        ],
    }
    man_path.write_text(json.dumps(doc, indent=2))
    print(f"wrote {len(rows)} scene photos", flush=True)


if __name__ == "__main__":
    main()
