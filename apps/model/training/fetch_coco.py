#!/usr/bin/env python3
"""Real-world dog/cat masks from COCO 2017, license-filtered for training.

    python3 training/fetch_coco.py [--max-train 3000] [--neg 800]

COCO annotations are CC BY 4.0; every image carries its own Flickr license.
Training uses only images whose license allows commercial use:
  4 CC BY 2.0 · 5 CC BY-SA 2.0 · 7 No known copyright restrictions · 8 US Government work
(NC and ND licenses are skipped). COCO val2017 dog/cat images are kept as an
evaluation set regardless of license (evaluation only, never trained on).

Output under data/public/coco/:
  images/<file>.jpg                 downloaded photos
  masks/<file>.png                  0 background, 1 dog/cat, 255 ignore (crowd regions)
  coco_pets.json                    {"train": [...], "neg": [...], "val": [...]} with license per file
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT

OUT = ROOT / "data" / "public" / "coco"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CAT, DOG = 17, 18
ANIMALS = set(range(16, 26))  # bird … giraffe: never use these as "no animal" negatives
COMMERCIAL_OK = {4: "CC BY 2.0", 5: "CC BY-SA 2.0", 7: "No known copyright restrictions", 8: "United States Government Work"}


def download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    pos = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if pos else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + pos
        done, last = pos, 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if done - last > 20 << 20:
                last = done
                print(f"  {dest.name} {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    tmp.rename(dest)


def mask_of(img: dict, anns: list[dict]) -> np.ndarray:
    m = Image.new("L", (img["width"], img["height"]), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        if a["category_id"] not in (CAT, DOG):
            continue
        val = 255 if a.get("iscrowd") else 1
        seg = a["segmentation"]
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) >= 6:
                    d.polygon(list(zip(poly[0::2], poly[1::2])), fill=val)
        else:  # RLE (crowd): ignore its bounding box
            x, y, w, h = a["bbox"]
            d.rectangle([x, y, x + w, y + h], fill=255)
    return np.asarray(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--neg", type=int, default=800)
    ap.add_argument("--max-val", type=int, default=600)
    a = ap.parse_args()
    zp = OUT / "annotations_trainval2017.zip"
    print("downloading COCO annotations (241 MB, once)…", flush=True)
    download(ANN_URL, zp)
    sets = {}
    with zipfile.ZipFile(zp) as z:
        for split in ("train2017", "val2017"):
            sets[split] = json.loads(z.read(f"annotations/instances_{split}.json"))
    rng = random.Random(0)
    plan = {"train": [], "neg": [], "val": []}
    for split, data in sets.items():
        by_img: dict[int, list] = {}
        for ann in data["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)
        imgs = {im["id"]: im for im in data["images"]}
        pets = [i for i, anns in by_img.items() if any(x["category_id"] in (CAT, DOG) and not x.get("iscrowd") for x in anns)]
        if split == "train2017":
            ok = [i for i in pets if imgs[i]["license"] in COMMERCIAL_OK]
            rng.shuffle(ok)
            plan["train"] = [(imgs[i], by_img[i]) for i in ok[: a.max_train]]
            neg = [i for i, anns in by_img.items() if imgs[i]["license"] in COMMERCIAL_OK and not any(x["category_id"] in ANIMALS for x in anns)]
            rng.shuffle(neg)
            plan["neg"] = [(imgs[i], []) for i in neg[: a.neg]]
        else:
            rng.shuffle(pets)
            plan["val"] = [(imgs[i], by_img[i]) for i in pets[: a.max_val]]
        print(f"{split}: {len(pets)} dog/cat images, commercial-OK {sum(imgs[i]['license'] in COMMERCIAL_OK for i in pets)}", flush=True)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    def fetch(item):
        img, anns = item
        p = OUT / "images" / img["file_name"]
        try:
            download(img["coco_url"], p)
            Image.fromarray(mask_of(img, anns)).save(OUT / "masks" / (Path(img["file_name"]).stem + ".png"))
            return True
        except Exception as exc:
            print(f"  skip {img['file_name']}: {exc}", flush=True)
            return False

    out = {}
    for part, items in plan.items():
        with ThreadPoolExecutor(8) as ex:
            oks = list(ex.map(fetch, items))
        out[part] = [
            {"file": im["file_name"], "license": COMMERCIAL_OK.get(im["license"], f"coco license id {im['license']} (eval only)"), "flickr_url": im.get("flickr_url")}
            for (im, _), k in zip(items, oks) if k
        ]
        print(f"{part}: {len(out[part])} images ready", flush=True)
    (OUT / "coco_pets.json").write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
