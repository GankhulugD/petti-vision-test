#!/usr/bin/env python3
"""License-clean photos of the dog breeds Oxford-IIIT Pet does not cover, from Wikimedia Commons.

    python3 training/fetch_commons_breeds.py [--per-breed 150] [--breeds bankhar,siberian_husky]

Why: the shipped heads are trained on Oxford only (25 dog breeds). The Bankhar,
husky, malamute, the big shepherds and mastiffs common in Mongolia were left to
zero-shot, and SigLIP barely knows "Bankhar" — so a fluffy black-and-tan husky
could come out as "Монгол банхар". These photos let the heads learn them.

Only CC0 / public domain / CC BY / CC BY-SA files are kept; title, author,
license and page URL go to data/commons/attribution.json (credit them when the
weights ship). Files are split train/test by photo series (digits and
"(cropped)" stripped from the title), so near-duplicates never straddle the
split:

    data/manifests/commons_breeds.csv   train rows  → auto_train.py --extra
    data/eval/commons_test.csv          test rows   (never trained on)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from common import ROOT, write_manifest

OUT = ROOT / "data" / "commons"
CATEGORIES = {
    "bankhar": ["Category:Mongolian Mastiff", "Category:Hotosho", "Category:Bankhar"],
    "siberian_husky": ["Category:Siberian Husky"],
    "alaskan_malamute": ["Category:Alaskan Malamute"],
    "samoyed": ["Category:Samoyed (dog)"],
    "german_shepherd": ["Category:German Shepherd Dog"],
    "tibetan_mastiff": ["Category:Tibetan Mastiff"],
    "central_asian_shepherd": ["Category:Central Asia Shepherd Dog", "Category:Alabai"],
    "caucasian_shepherd": ["Category:Caucasian Shepherd"],
    "east_european_shepherd": ["Category:East-European Shepherd"],
}
OK_LICENSE = re.compile(r"^(CC0|Public domain|PD\b|CC BY(-SA)? [0-9.]+)", re.I)
# not a photo of a living dog's body: drawings, stamps, statues, maps, tail close-ups
SKIP = re.compile(r"\.(png|svg|gif|tif|tiff|webm|ogv|pdf)$|stamp|марка|statue|sculpt|drawing|illustration|painting|logo|map|"
                  r"skull|skelet|хвост|tail|coat of arms|emblem|poster|book|cover|figurine", re.I)
SKIP_CAT = re.compile(r"in art|on stamps|heraldry|statue|sculpture|people with|events|shows", re.I)
UA = {"User-Agent": "PettiVision/0.5 (https://github.com/GankhulugD/petti-vision-test; dataset builder)"}  # Wikimedia UA policy: name + contact, or 429s
TEST_FRAC = 0.3
# only follow sub-categories that are about the same breed ("Sled dogs" would pull in every spitz)
KEYWORD = {
    "bankhar": r"bankhar|banhar|mongol|hotosho|buryat", "siberian_husky": r"husk", "alaskan_malamute": r"malamut",
    "samoyed": r"samoy", "german_shepherd": r"german shepherd|deutscher sch", "tibetan_mastiff": r"tibetan mastiff",
    "central_asian_shepherd": r"central asia|alabai", "caucasian_shepherd": r"caucasian|ovcharka", "east_european_shepherd": r"east.european",
}


def get(url: str, timeout=30, tries=4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or k == tries - 1:
                raise
            time.sleep(3 * 2 ** k)  # Wikimedia rate limit: back off
    raise RuntimeError("unreachable")


def commons(params: dict) -> dict:
    return json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"format": "json", **params})))


def list_files(roots: list[str], keyword: str, max_depth: int = 2, cap: int = 800) -> dict[str, dict]:
    seen, todo, files = set(), [(c, 0) for c in roots], {}
    while todo:
        cat, depth = todo.pop(0)
        if cat in seen or len(files) >= cap:
            continue
        seen.add(cat)
        cont = {}
        while True:
            r = commons({"action": "query", "generator": "categorymembers", "gcmtitle": cat, "gcmlimit": 50,
                         "gcmtype": "file|subcat", "prop": "imageinfo", "iiprop": "extmetadata|url|mime",
                         "iiurlwidth": 500, **cont})  # 500 px: a standard thumbnail size (w.wiki/GHai)
            for p in (r.get("query") or {}).get("pages", {}).values():
                t = p["title"]
                if t.startswith("Category:"):
                    if depth < max_depth and not SKIP_CAT.search(t) and re.search(keyword, t, re.I):
                        todo.append((t, depth + 1))
                    continue
                ii = (p.get("imageinfo") or [{}])[0]
                md = ii.get("extmetadata") or {}
                lic = (md.get("LicenseShortName") or {}).get("value", "")
                if ii.get("mime") != "image/jpeg" or not OK_LICENSE.match(lic) or SKIP.search(t) or "thumburl" not in ii:
                    continue
                files[t] = {"title": t, "license": lic, "page": ii.get("descriptionurl"), "thumb": ii["thumburl"],
                            "author": re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", "")).strip()[:200]}
            if "continue" not in r or len(files) >= cap:
                break
            cont = dict(r["continue"])  # pass the whole block back (incl. "continue": "||"), or the API repeats itself
            time.sleep(0.5)
    return files


def series(title: str) -> str:
    """Photo series key: 'Tibetanmastiff-sable3.jpg' and 'Bankhar (8367802157) (cropped).jpg' → stable group."""
    t = re.sub(r"\.[a-z]+$", "", title.lower().removeprefix("file:"))
    t = re.sub(r"\(cropped\)", " ", t)
    t = re.sub(r"[\d_\-\s.,()]+", " ", t)
    return " ".join(t.split()[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-breed", type=int, default=150)
    ap.add_argument("--breeds", default=",".join(CATEGORIES))
    a = ap.parse_args()

    attrib_p = OUT / "attribution.json"
    attrib = json.loads(attrib_p.read_text()) if attrib_p.exists() else {}
    train, test = [], []
    for key in a.breeds.split(","):
        files = list_files(CATEGORIES[key], KEYWORD[key])
        print(f"{key}: listed {len(files)} licensed photos", flush=True)
        # deterministic, spread over the category (not just the alphabetically first files)
        picked = sorted(files.values(), key=lambda f: hashlib.sha1(f["title"].encode()).hexdigest())[: a.per_breed]
        d = OUT / key
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in picked:
            name = hashlib.sha1(f["title"].encode()).hexdigest()[:16] + ".jpg"
            dest = d / name
            if not dest.exists():
                try:
                    time.sleep(1.0)
                    dest.write_bytes(get(f["thumb"]))
                except Exception as exc:
                    print(f"  skip {f['title']}: {exc}", flush=True)
                    continue
            attrib[f"{key}/{name}"] = f
            split = "test" if int(hashlib.sha1(series(f["title"]).encode()).hexdigest(), 16) % 100 < TEST_FRAC * 100 else "train"
            row = {"path": str(dest), "species": "dog", "breed": key, "source": "commons"}
            (test if split == "test" else train).append(row)
            n += 1
        print(f"{key}: {len(files)} licensed files listed, {n} kept", flush=True)
        attrib_p.write_text(json.dumps(attrib, ensure_ascii=False, indent=1))

    write_manifest(ROOT / "data" / "manifests" / "commons_breeds.csv", train)
    write_manifest(ROOT / "data" / "eval" / "commons_test.csv", test)
    print(f"train {len(train)}  test {len(test)}  (attribution: {attrib_p.relative_to(ROOT)})", flush=True)


if __name__ == "__main__":
    main()
