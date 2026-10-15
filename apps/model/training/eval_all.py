#!/usr/bin/env python3
"""One-shot eval of the frozen sets. Writes artifacts/reports/<version>.json and .md.

    python3 training/eval_all.py --version v0.4
    python3 training/eval_all.py --version v0.4 --limit 8          # smoke
    PETTI_FAKE_MODELS=1 python3 training/eval_all.py --version fake

Does not download data and does not write calibration.json (that stays in
evaluate.py). Images missing from a manifest are counted as skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import ROOT, read_manifest
from PIL import Image

from metrics import (
    ADULT_SIZE,
    AGE_ORDER,
    SIZE_ORDER,
    accuracy,
    expected_calibration_error,
    mean_f1,
    percentile,
    retrieval,
    round3,
    within_bins,
)

ART = Path(__import__("os").environ.get("PETTI_ARTIFACTS_DIR", ROOT / "artifacts"))
EVAL = ROOT / "data" / "eval"

# dog.ceo folder slug (hyphens) → taxonomy breed. Identity folders look like
# "spaniel-cocker__guusje"; mixes stay unlabeled.
BREED_FROM_SLUG = {
    "labrador": "labrador_retriever", "husky": "siberian_husky", "malamute": "alaskan_malamute",
    "samoyed": "samoyed", "retriever-golden": "golden_retriever", "rottweiler": "rottweiler",
    "doberman": "doberman", "pug": "pug", "chihuahua": "chihuahua", "beagle": "beagle",
    "poodle-toy": "toy_poodle", "pomeranian": "pomeranian", "shihtzu": "shih_tzu",
    "corgi-cardigan": "welsh_corgi", "dalmatian": "dalmatian", "chow": "chow_chow",
    "shiba": "shiba_inu", "bulldog-french": "french_bulldog", "bulldog-english": "bulldog",
    "boxer": "boxer", "dachshund": "dachshund", "collie-border": "border_collie",
    "maltese": "maltese", "mastiff-tibetan": "tibetan_mastiff", "stbernard": "saint_bernard",
    "terrier-yorkshire": "yorkshire_terrier", "akita": "akita", "papillon": "papillon",
    "german-shepherd": "german_shepherd", "spaniel-cocker": "cocker_spaniel",
    "malinois": "belgian_malinois", "pekinese": "pekingese",
    "schnauzer-miniature": "miniature_schnauzer", "weimaraner": "weimaraner",
}


def breed_of(row: dict) -> str:
    if row.get("breed"):
        return row["breed"]
    slug = (row.get("pet_id") or "").split("__", 1)[0]
    if not slug or slug == "mix":
        return ""
    return BREED_FROM_SLUG.get(slug, "")


def size_of(row: dict, breed: str) -> tuple[str, str]:
    if row.get("size"):
        return row["size"], "label"
    if breed in ADULT_SIZE:
        return ADULT_SIZE[breed], "adult_breed_prior"
    return "", ""


def _split_labels(cell: str) -> set[str]:
    return {p for p in (cell or "").split("|") if p and p != "none"}


def _luma(path: str) -> float:
    im = Image.open(path).convert("L")
    im.thumbnail((64, 64))
    return float(np.asarray(im, dtype=np.float32).mean())


def _blur(path: str) -> float:
    im = Image.open(path).convert("L")
    im.thumbnail((128, 128))
    a = np.asarray(im, dtype=np.float32)
    gy, gx = np.diff(a, axis=0), np.diff(a, axis=1)
    return float(gx.var() + gy.var())


def analyze_rows(vision, rows: list[dict], limit: int | None) -> list[dict]:
    out = []
    use = rows[:limit] if limit else rows
    for n, row in enumerate(use, start=1):
        if n == 1 or n % 25 == 0 or n == len(use):
            print(f"  analyze {n}/{len(use)}", flush=True)
        path = row["path"]
        if not Path(path).exists():
            out.append({"row": row, "missing": True})
            continue
        t0 = time.perf_counter()
        profile = vision.analyze([path], explain=True)
        wall = (time.perf_counter() - t0) * 1000
        data = profile.data
        explain = data.get("explain") or {}
        dets = explain.get("detections") or []
        chosen = data.get("detection") or {}
        box = chosen.get("box")
        wh = explain.get("image") or {}
        area_frac = None
        if box and wh.get("w") and wh.get("h"):
            area_frac = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (wh["w"] * wh["h"])
        out.append({
            "row": row,
            "missing": False,
            "species": (data.get("species") or {}).get("key"),
            "species_p": (data.get("species") or {}).get("confidence"),
            "breed_top": [t["key"] for t in ((data.get("breed") or {}).get("top") or [])],
            "breed_p": ((data.get("breed") or {}).get("top") or [{}])[0].get("confidence"),
            "breed_level": (data.get("breed") or {}).get("level"),
            "size": (data.get("size") or {}).get("key"),
            "size_p": (data.get("size") or {}).get("confidence"),
            "age": (data.get("age") or {}).get("key"),
            "age_p": (data.get("age") or {}).get("confidence"),
            "colors": {c["key"] for c in data.get("colors") or []},
            "marks": {m["key"] for m in data.get("marks") or []},
            "detected": bool(chosen.get("found")),
            "n_dets": len(dets),
            "area_frac": area_frac,
            "reid": np.asarray(profile.reid, dtype=np.float32).reshape(-1),
            "ms": (data.get("timing_ms") or {}).get("total", wall),
            "luma": _luma(path),
            "blur": _blur(path),
        })
    return out


def _task_block(items: list[dict], truth_fn, pred_fn, conf_fn, order: list[str] | None = None) -> dict:
    n = correct = within = 0
    confs, oks = [], []
    for it in items:
        truth = truth_fn(it)
        pred = pred_fn(it)
        if not truth or pred is None:
            continue
        n += 1
        hit = pred == truth
        correct += int(hit)
        conf = conf_fn(it)
        if conf is not None:
            confs.append(float(conf))
            oks.append(float(hit))
        if order is not None:
            w = within_bins(pred, truth, order, 1)
            if w is not None:
                within += int(w)
    block = {"n": n, "accuracy": accuracy(correct, n), "ece": expected_calibration_error(np.asarray(confs), np.asarray(oks)) if confs else None}
    if order is not None:
        block["within1"] = accuracy(within, n)
    return block


def _breed_block(items: list[dict], species: str) -> dict:
    n = top1 = top3 = 0
    confs, oks = [], []
    high_n = high_ok = 0
    for it in items:
        if it["row"].get("species") != species:
            continue
        truth = breed_of(it["row"])
        if not truth:
            continue
        n += 1
        ranked = it["breed_top"]
        hit1 = bool(ranked) and ranked[0] == truth
        top1 += int(hit1)
        top3 += int(truth in ranked[:3])
        if it["breed_p"] is not None:
            confs.append(float(it["breed_p"]))
            oks.append(float(hit1))
        if it["breed_level"] == "high":
            high_n += 1
            high_ok += int(hit1)
    return {
        "n": n,
        "top1": accuracy(top1, n),
        "top3": accuracy(top3, n),
        "ece": expected_calibration_error(np.asarray(confs), np.asarray(oks)) if confs else None,
        "high_n": high_n,
        "high_accuracy": accuracy(high_ok, high_n),
    }


def _slice_breed(items: list[dict], pred) -> dict | None:
    picked = [it for it in items if pred(it)]
    if not picked:
        return None
    dog = _breed_block(picked, "dog")
    return {"n": len(picked), "dog_top1": dog["top1"], "detected": round3(sum(it["detected"] for it in picked) / len(picked))}


def aggregate(scored: list[dict], reid_ids: set[str], negative: bool) -> dict:
    present = [s for s in scored if not s["missing"]]
    missing = sum(1 for s in scored if s["missing"])
    if negative:
        fired = sum(1 for s in present if s["detected"] or s["species"] in ("dog", "cat"))
        return {
            "n": len(present),
            "missing": missing,
            "false_positive_rate": accuracy(fired, len(present)),
            "by_source": _by_source_fpr(present),
        }
    species = {}
    for sp in ("dog", "cat"):
        rows = [s for s in present if s["row"].get("species") == sp]
        hit = sum(s["species"] == sp for s in rows)
        confs = [float(s["species_p"]) for s in rows if s["species_p"] is not None]
        oks = [float(s["species"] == sp) for s in rows if s["species_p"] is not None]
        species[sp] = {
            "n": len(rows),
            "accuracy": accuracy(hit, len(rows)),
            "ece": expected_calibration_error(np.asarray(confs), np.asarray(oks)) if confs else None,
        }
    breed = {"dog": _breed_block(present, "dog"), "cat": _breed_block(present, "cat")}
    n = correct = within = 0
    confs, oks, sources = [], [], set()
    for s in present:
        truth, src = size_of(s["row"], breed_of(s["row"]))
        if not truth or not s["size"]:
            continue
        sources.add(src)
        n += 1
        hit = s["size"] == truth
        correct += int(hit)
        w = within_bins(s["size"], truth, SIZE_ORDER, 1)
        if w is not None:
            within += int(w)
        if s["size_p"] is not None:
            confs.append(float(s["size_p"]))
            oks.append(float(hit))
    size = {
        "n": n,
        "accuracy": accuracy(correct, n),
        "within1": accuracy(within, n),
        "ece": expected_calibration_error(np.asarray(confs), np.asarray(oks)) if confs else None,
        "label_source": sorted(sources),
    }
    age = _task_block(present, lambda it: it["row"].get("age") or "", lambda it: it["age"], lambda it: it["age_p"], AGE_ORDER)
    color_pairs, mark_pairs = [], []
    for s in present:
        if s["row"].get("colors"):
            color_pairs.append((s["colors"], _split_labels(s["row"]["colors"])))
        if s["row"].get("marks"):
            mark_pairs.append((s["marks"], _split_labels(s["row"]["marks"])))
    detected = [s for s in present if s["row"].get("species") in ("dog", "cat")]
    small = [s for s in detected if s["area_frac"] is not None and s["area_frac"] < 0.08]
    return {
        "n": len(present),
        "missing": missing,
        "species": species,
        "breed": breed,
        "size": size,
        "age": age,
        "color": {"n": len(color_pairs), "f1": mean_f1(color_pairs)},
        "marks": {"n": len(mark_pairs), "f1": mean_f1(mark_pairs)},
        "detection": {
            "n": len(detected),
            "recall": accuracy(sum(s["detected"] for s in detected), len(detected)),
            "mask_iou": None,
            "small_box_n": len(small),
            "note": "mask_iou needs a frozen mask set (Oxford trimaps are not in data/eval yet)",
        },
        "slices": {
            "small_box": _slice_breed(present, lambda s: s["area_frac"] is not None and s["area_frac"] < 0.08),
            "dark": _slice_breed(present, lambda s: s["luma"] < 50),
            "blurry": _slice_breed(present, lambda s: s["blur"] < 80),
            "multi_animal": _slice_breed(present, lambda s: s["n_dets"] >= 2),
            "mixed_breed": _slice_breed(present, lambda s: (s["row"].get("pet_id") or "").startswith("mix") or breed_of(s["row"]) in ("", "mixed_dog")),
        },
        "latency_ms": [s["ms"] for s in present],
        "reid_rows": [s for s in present if (s["row"].get("pet_id") or "") in reid_ids or s["row"].get("source", "").endswith("distractor")],
    }


def _by_source_fpr(present: list[dict]) -> dict:
    groups = defaultdict(list)
    for s in present:
        groups[s["row"].get("source") or "unknown"].append(s)
    out = {}
    for src, rows in groups.items():
        fired = sum(1 for s in rows if s["detected"] or s["species"] in ("dog", "cat"))
        out[src] = {"n": len(rows), "false_positive_rate": accuracy(fired, len(rows))}
    return out


def build_report(version: str, animal_scored: list[dict], reid_scored: list[dict], neg_scored: list[dict], meta: dict, *, attribute_latency: bool = False) -> dict:
    reid_ids = {s["row"]["pet_id"] for s in reid_scored if s["row"].get("pet_id")}
    attr = aggregate(animal_scored, reid_ids, negative=False)
    ident = aggregate(reid_scored, reid_ids, negative=False)
    neg = aggregate(neg_scored, set(), negative=True)
    # re-ID uses the identity manifest only (gallery includes distractors)
    emb, ids, dist = [], [], []
    for s in reid_scored:
        if s["missing"]:
            continue
        emb.append(s["reid"])
        ids.append(s["row"].get("pet_id") or s["row"]["path"])
        dist.append(s["row"].get("source", "").endswith("distractor") or not s["row"].get("pet_id"))
    reid = retrieval(np.stack(emb), np.array(ids), np.array(dist)) if emb else retrieval(np.zeros((0, 4)), np.array([]), np.array([]))
    # Identity photos are inside reid_scored. Add the attribute pass only when
    # it is a different manifest, so latency is one measurement per photo.
    lat = [s["ms"] for s in reid_scored if not s["missing"]]
    lat += [s["ms"] for s in neg_scored if not s["missing"]]
    if attribute_latency:
        lat += attr["latency_ms"]
    high_n = attr["breed"]["dog"]["high_n"] + attr["breed"]["cat"]["high_n"]
    high_ok_dog = (attr["breed"]["dog"]["high_accuracy"] or 0) * attr["breed"]["dog"]["high_n"]
    high_ok_cat = (attr["breed"]["cat"]["high_accuracy"] or 0) * attr["breed"]["cat"]["high_n"]
    report = {
        "version": version,
        "meta": meta,
        "species": {"dog": attr["species"]["dog"], "cat": attr["species"]["cat"]},
        "breed": attr["breed"],
        "size": attr["size"],
        "age": attr["age"],
        "color": attr["color"],
        "marks": attr["marks"],
        "species_ece_note": "per-species ECE is under species.*.ece; breed ECE under breed.*.ece",
        "reid": reid,
        "detection": attr["detection"],
        "negative": {k: neg[k] for k in ("n", "missing", "false_positive_rate", "by_source")},
        "slices": attr["slices"],
        "latency": {"n": len(lat), "p50_ms": percentile(lat, 0.50), "p95_ms": percentile(lat, 0.95), "device": meta.get("device")},
        "confidence": {
            "high": {
                "n": high_n,
                "accuracy": round3((high_ok_dog + high_ok_cat) / high_n) if high_n else None,
            }
        },
        "coverage": {
            "attributes_n": attr["n"],
            "attributes_missing": attr["missing"],
            "identity_n": ident["n"],
            "identity_missing": ident["missing"],
            "negatives_n": neg["n"],
        },
    }
    # flat ECE keys the gate reads
    report["species"]["ece"] = _mean_present(attr["species"]["dog"]["ece"], attr["species"]["cat"]["ece"])
    report["breed"]["ece"] = _mean_present(attr["breed"]["dog"]["ece"], attr["breed"]["cat"]["ece"])
    return report


def _mean_present(*vals) -> float | None:
    xs = [v for v in vals if v is not None]
    return round3(sum(xs) / len(xs)) if xs else None


def render_md(report: dict) -> str:
    b = report["breed"]
    lines = [
        f"# Petti Vision eval {report['version']}",
        "",
        f"Device: {report['latency'].get('device')}. Latency p50 {report['latency']['p50_ms']} ms, p95 {report['latency']['p95_ms']} ms (n={report['latency']['n']}).",
        "",
        "## Scores",
        "",
        f"- Species dog {report['species']['dog']['accuracy']} (n={report['species']['dog']['n']}), cat {report['species']['cat']['accuracy']} (n={report['species']['cat']['n']}).",
        f"- Breed dog top-1 {b['dog']['top1']} / top-3 {b['dog']['top3']} (n={b['dog']['n']}); cat top-1 {b['cat']['top1']} / top-3 {b['cat']['top3']} (n={b['cat']['n']}).",
        f"- Size acc {report['size']['accuracy']} ±1 {report['size'].get('within1')} (source {report['size'].get('label_source')}).",
        f"- Age acc {report['age']['accuracy']} ±1 {report['age'].get('within1')}.",
        f"- Color F1 {report['color']['f1']} (n={report['color']['n']}), marks F1 {report['marks']['f1']} (n={report['marks']['n']}).",
        f"- ECE species {report['species'].get('ece')}, breed {report['breed'].get('ece')}, size {report['size'].get('ece')}, age {report['age'].get('ece')}.",
        f"- Re-ID R@1 {report['reid'].get('R@1')} R@5 {report['reid'].get('R@5')} mAP {report['reid'].get('mAP')} false-match {report['reid'].get('false_match_rate')} (queries {report['reid'].get('queries')}).",
        f"- Detection recall {report['detection']['recall']} (n={report['detection']['n']}). Mask IoU {report['detection']['mask_iou']}. {report['detection'].get('note', '')}",
        f"- No-animal false positive rate {report['negative']['false_positive_rate']} (n={report['negative']['n']}).",
        f"- High-confidence breed accuracy {report['confidence']['high']['accuracy']} (n={report['confidence']['high']['n']}).",
        "",
        "## Slices",
        "",
    ]
    for name, sl in (report.get("slices") or {}).items():
        lines.append(f"- {name}: {sl}")
    lines += ["", "## Coverage", "", json.dumps(report["coverage"], indent=2), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.4")
    ap.add_argument("--attributes", type=Path, default=EVAL / "attributes.csv")
    ap.add_argument("--identity", type=Path, default=EVAL / "dogceo_ids.csv")
    ap.add_argument("--distractors", type=Path, default=EVAL / "dogceo_distractors.csv")
    ap.add_argument("--negatives", type=Path, default=EVAL / "negatives.csv")
    ap.add_argument("--limit", type=int, default=0, help="max images per manifest (0 = all)")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    limit = args.limit or None

    from petti_vision import Config, PettiVision

    cfg = Config()
    vision = PettiVision(cfg)
    attr_rows = read_manifest([args.attributes]) if args.attributes.exists() else []
    id_rows = read_manifest([args.identity]) if args.identity.exists() else []
    dis_rows = read_manifest([args.distractors]) if args.distractors.exists() else []
    neg_rows = read_manifest([args.negatives]) if args.negatives.exists() else []
    # Attribute metrics on the identity photos as well when attributes.csv is absent:
    # breed is inferred from the pet_id slug, and those photos are not the Oxford train set.
    meta = {
        "device": cfg.device if hasattr(cfg, "device") else None,
        "detector": cfg.detector_weights,
        "reid_model": cfg.reid_model,
        "vlm_model": cfg.vlm_model,
        "fake": cfg.fake_models,
        "attributes_manifest": str(args.attributes) if attr_rows else "identity photos (breed from folder slug)",
        "notes": [
            "Re-ID is leave-one-out cosine on DINOv2, not the production matcher score.",
            "Size on identity photos uses the adult breed prior when the CSV has no size cell (same as bench.js).",
            "No-animal images are line drawings, so a 0% false-positive rate does not cover photos of cars or people.",
            "Mask IoU is unset until a trimap/COCO mask manifest is frozen.",
            "Color and marks stay unset until a labeled manifest exists.",
        ],
    }
    from petti_vision.config import resolve_device

    meta["device"] = "fake" if cfg.fake_models else resolve_device(cfg.device)
    id_scored = analyze_rows(vision, id_rows, limit)
    dis_scored = analyze_rows(vision, dis_rows, limit)
    animal = analyze_rows(vision, attr_rows, limit) if attr_rows else id_scored
    reid = id_scored + dis_scored
    neg = analyze_rows(vision, neg_rows, limit)
    report = build_report(args.version, animal, reid, neg, meta, attribute_latency=bool(attr_rows))
    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.version}.json"
    dest.write_text(json.dumps(report, indent=2))
    (out_dir / f"{args.version}.md").write_text(render_md(report))
    if args.write_baseline:
        (out_dir / "baseline.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {dest}", flush=True)
    print(render_md(report), flush=True)


if __name__ == "__main__":
    main()
