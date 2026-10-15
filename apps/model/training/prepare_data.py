#!/usr/bin/env python3
"""Download public pet datasets and convert them into our manifest format.

    python3 training/prepare_data.py --oxford --stanford-dogs
    python3 training/prepare_data.py --folder data/my_photos   # <label>/<img>.jpg folders

Datasets:
  * Oxford-IIIT Pet  (37 cat & dog breeds, ~7.4k images)       CC BY-SA 4.0
  * Stanford Dogs    (120 dog breeds, ~20k images)              research use
  * your own folders (data/<anything>/<breed_key>/*.jpg)

Breeds that are not in our taxonomy keep the species label only (still useful
for the species head). Size labels are derived from the breed's typical size
(weak labels, marked source=*_weak).
"""

from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
from pathlib import Path

from common import ROOT, T, write_manifest

DATA = ROOT / "data" / "public"
OUT = ROOT / "data" / "manifests"

OXFORD_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
STANFORD_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

# dataset breed name (lower-case) → our key
OXFORD_MAP = {
    "abyssinian": "abyssinian", "bengal": "bengal", "birman": "birman", "bombay": "bombay",
    "british_shorthair": "british_shorthair", "egyptian_mau": "egyptian_mau", "maine_coon": "maine_coon",
    "persian": "persian", "ragdoll": "ragdoll", "russian_blue": "russian_blue", "siamese": "siamese",
    "sphynx": "sphynx", "american_pit_bull_terrier": "pit_bull", "staffordshire_bull_terrier": "pit_bull",
    "beagle": "beagle", "boxer": "boxer", "chihuahua": "chihuahua", "english_cocker_spaniel": "cocker_spaniel",
    "miniature_pinscher": "miniature_pinscher", "pomeranian": "pomeranian", "pug": "pug",
    "saint_bernard": "saint_bernard", "samoyed": "samoyed", "shiba_inu": "shiba_inu",
    "yorkshire_terrier": "yorkshire_terrier",
}
STANFORD_MAP = {
    "chihuahua": "chihuahua", "japanese_spaniel": None, "maltese_dog": "maltese", "pekinese": "pekingese",
    "shih-tzu": "shih_tzu", "blenheim_spaniel": "cavalier_king_charles", "papillon": "papillon",
    "toy_terrier": "miniature_pinscher", "beagle": "beagle", "yorkshire_terrier": "yorkshire_terrier",
    "american_staffordshire_terrier": "pit_bull", "staffordshire_bullterrier": "pit_bull",
    "miniature_schnauzer": "miniature_schnauzer", "golden_retriever": "golden_retriever",
    "labrador_retriever": "labrador_retriever", "cocker_spaniel": "cocker_spaniel",
    "english_springer": None, "shetland_sheepdog": "shetland_sheepdog", "border_collie": "border_collie",
    "german_shepherd": "german_shepherd", "malinois": "belgian_malinois", "doberman": "doberman",
    "rottweiler": "rottweiler", "bernese_mountain_dog": "bernese_mountain", "boxer": "boxer",
    "tibetan_mastiff": "tibetan_mastiff", "french_bulldog": "french_bulldog", "great_dane": "great_dane",
    "saint_bernard": "saint_bernard", "eskimo_dog": "siberian_husky", "malamute": "alaskan_malamute",
    "siberian_husky": "siberian_husky", "dalmatian": "dalmatian", "pug": "pug", "samoyed": "samoyed",
    "pomeranian": "pomeranian", "chow": "chow_chow", "pembroke": "welsh_corgi", "cardigan": "welsh_corgi",
    "toy_poodle": "toy_poodle", "miniature_poodle": "toy_poodle", "weimaraner": "weimaraner",
    "kelpie": None, "keeshond": "spitz",
}


def _download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")  # an interrupted download is never mistaken for a finished one
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _extract(tar: Path, into: Path):
    marker = into / ".extracted"
    if marker.exists():
        return
    print(f"extracting {tar.name}", flush=True)
    try:
        with tarfile.open(tar) as t:
            t.extractall(into, filter="data")
    except (tarfile.TarError, EOFError, OSError) as exc:
        tar.unlink(missing_ok=True)  # truncated download → fetch again next run
        raise SystemExit(f"{tar.name} is incomplete ({exc}); deleted, re-run to download again")
    marker.touch()


def _row(path: Path, species: str, breed: str | None, source: str, pet_id: str = "") -> dict:
    size = T.BREED_BY_KEY[breed].size if breed and breed in T.BREED_BY_KEY and breed not in ("mixed_dog",) else ""
    return {"path": str(path), "species": species, "breed": breed or "", "size": size, "pet_id": pet_id,
            "source": source + ("_weak" if size else "")}


def oxford() -> list[dict]:
    tar = DATA / "oxford_images.tar.gz"
    _download(OXFORD_URL, tar)
    _extract(tar, DATA / "oxford")
    rows = []
    for p in sorted((DATA / "oxford" / "images").glob("*.jpg")):
        name = p.stem.rsplit("_", 1)[0]
        species = "cat" if name[0].isupper() else "dog"
        rows.append(_row(p, species, OXFORD_MAP.get(name.lower()), "oxford"))
    return rows


def stanford_dogs() -> list[dict]:
    tar = DATA / "stanford_dogs.tar"
    _download(STANFORD_URL, tar)
    _extract(tar, DATA / "stanford")
    rows = []
    for d in sorted((DATA / "stanford" / "Images").iterdir()):
        name = d.name.split("-", 1)[1].lower()
        breed = STANFORD_MAP.get(name)
        for p in sorted(d.glob("*.jpg")):
            rows.append(_row(p, "dog", breed, "stanford"))
    return rows


def folder(root: Path) -> list[dict]:
    """<root>/<breed_key or species>/*.jpg ; <root>/<breed_key>/<pet_id>/*.jpg also works."""
    rows = []
    species_keys = {s.key for s in T.SPECIES}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        rel = p.relative_to(root).parts
        label = rel[0]
        pet_id = rel[1] if len(rel) > 2 else ""
        if label in T.BREED_BY_KEY:
            rows.append(_row(p, T.BREED_BY_KEY[label].species, label, "folder", pet_id))
        elif label in species_keys:
            rows.append(_row(p, label, None, "folder", pet_id))
    return rows


# dog.ceo breed path → our taxonomy key. dog.ceo serves the Stanford Dogs
# images one by one from a fast CDN (plus owner phone photos), which is far
# more reliable than the 750 MB Stanford tarball on slow connections.
DOGCEO_MAP = {
    "labrador": "labrador_retriever", "husky": "siberian_husky", "malamute": "alaskan_malamute", "samoyed": "samoyed",
    "retriever/golden": "golden_retriever", "rottweiler": "rottweiler", "doberman": "doberman", "pug": "pug",
    "chihuahua": "chihuahua", "beagle": "beagle", "poodle/toy": "toy_poodle", "poodle/miniature": "toy_poodle",
    "pomeranian": "pomeranian", "shihtzu": "shih_tzu", "corgi/cardigan": "welsh_corgi", "pembroke": "welsh_corgi",
    "dalmatian": "dalmatian", "chow": "chow_chow", "shiba": "shiba_inu", "bulldog/french": "french_bulldog",
    "bulldog/english": "bulldog", "boxer": "boxer", "dachshund": "dachshund", "collie/border": "border_collie",
    "maltese": "maltese", "mastiff/tibetan": "tibetan_mastiff", "stbernard": "saint_bernard",
    "terrier/yorkshire": "yorkshire_terrier", "akita": "akita", "papillon": "papillon", "german/shepherd": "german_shepherd",
    "spaniel/cocker": "cocker_spaniel", "malinois": "belgian_malinois", "pekinese": "pekingese",
    "schnauzer/miniature": "miniature_schnauzer", "weimaraner": "weimaraner", "sheepdog/shetland": "shetland_sheepdog",
    "australian/shepherd": "australian_shepherd", "spaniel/blenheim": "cavalier_king_charles",
    "pinscher/miniature": "miniature_pinscher", "frise/bichon": "bichon_frise", "terrier/russell": "jack_russell",
    "pitbull": "pit_bull", "terrier/staffordshire": "pit_bull", "terrier/american": "pit_bull",
    "spitz/japanese": "spitz", "keeshond": "spitz", "dane/great": "great_dane", "mountain/bernese": "bernese_mountain",
    "ovcharka/caucasian": "caucasian_shepherd", "sharpei": "shar_pei",
}


def dogceo(max_per_breed: int = 150, exclude: set[str] | None = None) -> list[dict]:
    import json
    from concurrent.futures import ThreadPoolExecutor

    root = DATA / "dogceo"

    def get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()

    jobs = []
    for b, key in DOGCEO_MAP.items():
        try:
            urls = json.loads(get(f"https://dog.ceo/api/breed/{b}/images"))["message"]
        except Exception as exc:
            print(f"  skip {b}: {exc}", flush=True)
            continue
        # Stanford (ImageNet) files only: owner photos (cooper1.jpg…) form the
        # re-ID test set (fetch_identity_set.py) and must stay unseen
        urls = [u for u in sorted(urls) if re.match(r"^n\d{8}_", u.rsplit("/", 1)[-1])
                and u.rsplit("/", 1)[-1] not in (exclude or set())]
        for u in urls[:: max(1, len(urls) // max_per_breed)][:max_per_breed]:
            jobs.append((u, root / b.replace("/", "-") / u.rsplit("/", 1)[-1], key))

    def fetch(job):
        u, dest, key = job
        if not dest.exists():
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(get(u))
            except Exception:
                return None
        return _row(dest, "dog", key, "dogceo")

    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(fetch, jobs) if r]
    print(f"dog.ceo: {len(rows)} images in {len(DOGCEO_MAP)} breed folders", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxford", action="store_true")
    ap.add_argument("--stanford-dogs", action="store_true")
    ap.add_argument("--dogceo", action="store_true", help="Stanford Dogs + owner photos via the dog.ceo CDN (fast)")
    ap.add_argument("--folder", type=Path, action="append", default=[])
    a = ap.parse_args()
    if a.oxford:
        r = oxford()
        write_manifest(OUT / "oxford.csv", r)
        print(f"oxford: {len(r)} images, {sum(bool(x['breed']) for x in r)} with mapped breed")
    if a.stanford_dogs:
        r = stanford_dogs()
        write_manifest(OUT / "stanford_dogs.csv", r)
        print(f"stanford dogs: {len(r)} images, {sum(bool(x['breed']) for x in r)} with mapped breed")
    if a.dogceo:
        ex = ROOT / "data" / "eval" / "bench_exclude.txt"
        r = dogceo(exclude=set(ex.read_text().split()) if ex.exists() else None)
        write_manifest(OUT / "dogceo.csv", r)
    for f in a.folder:
        r = folder(f)
        write_manifest(OUT / f"folder_{f.name}.csv", r)
        print(f"{f}: {len(r)} images")
    print("manifests:", *[p.name for p in OUT.glob("*.csv")])


if __name__ == "__main__":
    main()
