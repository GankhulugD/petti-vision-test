#!/usr/bin/env python3
"""Mongolian Bankhar check: is a real Bankhar recognised, and is a husky NOT called one?

    python3 training/bankhar_check.py --fetch        # download the sets once
    python3 training/bankhar_check.py --tag before   # evaluate the served code

Positives: Wikimedia Commons categories about the Bankhar (only CC0 / public
domain / CC BY / CC BY-SA files; author + license kept in manifest.json, so the
photos may also be used for training with attribution).
Look-alikes: dog.ceo photos of husky, malamute, samoyed, keeshond, German
shepherd, Tibetan mastiff, Leonberger, Newfoundland, chow chow. These are
ImageNet/Stanford files → evaluation only, never training.

Reports Bankhar recall (top-1 / top-3) and how often a look-alike is called
"Монгол банхар" (top-1), with raw and calibrated confidence.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import ROOT

D = ROOT / "data" / "eval" / "bankhar"
CATS = ["Category:Mongolian Mastiff", "Category:Hotosho", "Category:Bankhar"]
OK_LICENSE = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? [0-9.]+)", re.I)
SKIP_TITLE = re.compile(r"хвост|tail|transparent|\.png$|\.svg$|\.gif$|map|флаг|flag", re.I)
LOOKALIKES = {
    "husky": "siberian_husky", "malamute": "alaskan_malamute", "samoyed": "samoyed", "keeshond": "spitz",
    "german/shepherd": "german_shepherd", "mastiff/tibetan": "tibetan_mastiff", "leonberg": "leonberger",
    "newfoundland": "newfoundland", "chow": "chow_chow",
}
UA = {"User-Agent": "PettiVision/0.5 (https://github.com/GankhulugD/petti-vision-test; dataset builder)"}  # Wikimedia UA policy: name + contact, or 429s


def get(url: str, timeout=30, tries=4) -> bytes:
    import urllib.error

    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or k == tries - 1:
                raise
            time.sleep(3 * 2 ** k)  # Wikimedia rate limit: back off
    raise RuntimeError("unreachable")


def commons(params: dict) -> dict:
    return json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"format": "json", **params})))


def fetch():
    pos = D / "bankhar"
    pos.mkdir(parents=True, exist_ok=True)
    seen_cat, todo, files = set(), [(c, 0) for c in CATS], {}
    while todo:
        cat, depth = todo.pop()
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        r = commons({"action": "query", "generator": "categorymembers", "gcmtitle": cat, "gcmlimit": 500,
                     "gcmtype": "file|subcat", "prop": "imageinfo", "iiprop": "extmetadata|url", "iiurlwidth": 500})
        for p in (r.get("query") or {}).get("pages", {}).values():
            t = p["title"]
            if t.startswith("Category:"):
                if depth < 2:
                    todo.append((t, depth + 1))
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata") or {}
            lic = (md.get("LicenseShortName") or {}).get("value", "")
            if not OK_LICENSE.match(lic) or SKIP_TITLE.search(t) or "thumburl" not in ii:
                continue
            files[t] = {"title": t, "license": lic, "url": ii.get("descriptionurl"), "thumb": ii["thumburl"],
                        "author": re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", ""))[:200], "category": cat}
    man = []
    for k, (t, f) in enumerate(sorted(files.items())):
        dest = pos / f"commons_{k:03d}.jpg"
        if not dest.exists():
            try:
                time.sleep(1.0)
                dest.write_bytes(get(f["thumb"]))
            except Exception as exc:
                print(f"  skip {t}: {exc}", flush=True)
                continue
        man.append({**f, "file": dest.name})
    (D / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"bankhar positives: {len(man)} files from {len(seen_cat)} categories", flush=True)

    for b, key in LOOKALIKES.items():
        out = D / "lookalike" / key
        out.mkdir(parents=True, exist_ok=True)
        try:
            urls = sorted(json.loads(get(f"https://dog.ceo/api/breed/{b}/images"))["message"])
        except Exception as exc:
            print(f"  skip {b}: {exc}", flush=True)
            continue
        for u in urls[:: max(1, len(urls) // 25)][:25]:
            dest = out / u.rsplit("/", 1)[-1]
            if not dest.exists():
                try:
                    dest.write_bytes(get(u))
                except Exception:
                    pass
        print(f"  {key}: {len(list(out.glob('*.jpg')))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--tag", default="bankhar_check")
    ap.add_argument("--heads", default="", help="evaluate artifacts/<this>.npz instead of the served heads")
    ap.add_argument("--all-positives", action="store_true", help="use every Commons Bankhar photo (before any training on them)")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    from petti_vision import Config, PettiVision

    v = PettiVision(Config())
    if a.heads:
        from petti_vision.heads import load_heads

        v.heads = load_heads(Path(v.cfg.artifacts_dir) / f"{a.heads}.npz")
    T, v.breed_display_T = v.breed_display_T, 1.0  # raw probabilities; calibrated derived below
    ct = ROOT / "data" / "eval" / "commons_test.csv"
    if ct.exists():
        # after fetch_commons_breeds.py: held-out Commons photos of Bankhar and its look-alikes.
        # --all-positives: every Commons Bankhar photo (fair only while no heads were trained on them)
        from common import read_manifest

        items = [(Path(r["path"]), r["breed"]) for r in read_manifest([ct]) if Path(r["path"]).exists()
                 and not (a.all_positives and r["breed"] == "bankhar")]
        if a.all_positives:
            items += [(p, "bankhar") for p in sorted((ROOT / "data" / "commons" / "bankhar").glob("*.jpg"))]
    else:
        items = [(p, "bankhar") for p in sorted((D / "bankhar").glob("*.jpg"))]
    items += [(p, p.parent.name) for p in sorted((D / "lookalike").glob("*/*.jpg"))]
    rows = []
    for p, truth in items:
        try:
            d = v.analyze([p.read_bytes()]).to_json()
        except Exception as exc:
            print(f"  fail {p.name}: {exc}", flush=True)
            continue
        b = d.get("breed") or {}
        probs = b.get("probs", {})
        top = [t["key"] for t in b.get("top", [])]
        pb = probs.get("bankhar", 0.0)
        if probs:
            ks, ps = zip(*probs.items())
            lg = np.log(np.clip(np.array(ps), 1e-9, 1)) / T
            q = np.exp(lg - lg.max())
            q /= q.sum()
            pb_cal = float(q[list(ks).index("bankhar")]) if "bankhar" in ks else 0.0
        else:
            pb_cal = 0.0
        rows.append({"file": str(p.relative_to(ROOT)), "truth": truth, "species": d["species"]["key"], "top1": top[0] if top else None,
                     "top3": top[:3], "p_bankhar": round(pb, 4), "p_bankhar_cal": round(pb_cal, 4)})

    by = defaultdict(list)
    for r in rows:
        by[r["truth"]].append(r)
    pos = by.pop("bankhar", [])
    summ = {}
    if pos:
        summ["bankhar"] = {"n": len(pos), "top1": round(np.mean([r["top1"] == "bankhar" for r in pos]), 3),
                           "top3": round(np.mean(["bankhar" in r["top3"] for r in pos]), 3),
                           "mean_p_cal": round(np.mean([r["p_bankhar_cal"] for r in pos]), 3),
                           "wrong_top1": dict(sorted(defaultdict(int, {k: sum(r["top1"] == k for r in pos) for k in {r["top1"] for r in pos if r["top1"] != "bankhar"}}).items(), key=lambda x: -x[1]))}
    neg_all = [r for rs in by.values() for r in rs]
    summ["lookalikes"] = {"n": len(neg_all), "called_bankhar_top1": round(np.mean([r["top1"] == "bankhar" for r in neg_all]), 3) if neg_all else None,
                          "mean_p_cal": round(np.mean([r["p_bankhar_cal"] for r in neg_all]), 3) if neg_all else None,
                          "own_breed_top1": round(np.mean([r["top1"] == r["truth"] for r in neg_all]), 3) if neg_all else None}
    summ["per_lookalike"] = {k: {"n": len(rs), "called_bankhar": round(np.mean([r["top1"] == "bankhar" for r in rs]), 3),
                                 "correct": round(np.mean([r["top1"] == k for r in rs]), 3),
                                 "mean_p_bankhar_cal": round(np.mean([r["p_bankhar_cal"] for r in rs]), 3)} for k, rs in sorted(by.items())}
    print(json.dumps(summ, indent=1, ensure_ascii=False), flush=True)
    rep = ROOT / "artifacts" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{a.tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
