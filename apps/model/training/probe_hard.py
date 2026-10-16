#!/usr/bin/env python3
"""Real-world probe: cluttered owner photos, detector A/B against a teacher.

    python3 training/probe_hard.py --backends dino,yolo [--limit 400]

Images: data/eval/dogceo/** (owner phone photos + ImageNet look-alikes) and
apps/web/public/test/hard/*. Reference boxes come from the offline teacher
(torchvision Mask R-CNN v2). For every backend it records box IoU with the
teacher's main animal, the full-pipeline answer (species, breed, marks) and
latency, then writes artifacts/reports/probe_hard.json and a contact sheet of
the worst cases (green = teacher, red = first backend, blue = second).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT
from teacher import Teacher, box_iou, main_instance

REPORTS = ROOT / "artifacts" / "reports"


def images(limit: int) -> list[Path]:
    ps = sorted((ROOT / "data" / "eval" / "dogceo").rglob("*.jpg"))
    hard = sorted((ROOT.parent / "web" / "public" / "test" / "hard").glob("*.jpg"))
    ps = hard + ps
    return ps[:limit] if limit else ps


def sheet(rows, backends, out: Path, n: int = 24):
    worst = sorted(rows, key=lambda r: r["res"][backends[0]]["iou"] if r["ref"] else 2)[:n]
    tw = 256
    cols = 6
    canvas = Image.new("RGB", (tw * cols, tw * ((len(worst) + cols - 1) // cols)), "white")
    colors = ["#e11d48", "#2563eb"]
    for k, r in enumerate(worst):
        im = Image.open(r["path"]).convert("RGB")
        s = tw / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        d = ImageDraw.Draw(im)
        if r["ref"]:
            d.rectangle([v * s for v in r["ref"]], outline="#16a34a", width=3)
        for b, c in zip(backends, colors):
            if r["res"][b]["box"]:
                d.rectangle([v * s for v in r["res"][b]["box"]], outline=c, width=2)
        d.text((3, 3), f"{r['res'][backends[0]]['iou']:.2f}", fill="yellow")
        canvas.paste(im, ((k % cols) * tw, (k // cols) * tw))
    canvas.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="dino,yolo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="probe_hard")
    a = ap.parse_args()
    backends = a.backends.split(",")
    paths = images(a.limit)
    print(f"{len(paths)} images", flush=True)

    teacher = Teacher()
    refs = {}
    for i, p in enumerate(paths):
        res = teacher.get(p)
        m = main_instance(res)
        refs[p] = (m["box"] if m else None, m["species"] if m else None, len(res["inst"]))
        if i % 50 == 0:
            print(f"teacher {i}/{len(paths)}", flush=True)
    del teacher

    from petti_vision import Config, PettiVision

    rows = [{"path": str(p), "ref": refs[p][0], "ref_species": refs[p][1], "n_teacher": refs[p][2], "res": {}} for p in paths]
    summary = {}
    for b in backends:
        v = PettiVision(Config(detector_backend=b))
        lat = []
        for r in rows:
            img = Image.open(r["path"]).convert("RGB")
            t0 = time.perf_counter()
            dets = v.detector.detect([img])[0]
            lat.append((time.perf_counter() - t0) * 1000)
            box = list(dets[0].box) if dets else None
            prof = v.analyze([Path(r["path"]).read_bytes()]).to_json()
            r["res"][b] = {
                "box": box,
                "n": len(dets),
                "iou": box_iou(box, r["ref"]) if (box and r["ref"]) else 0.0,
                "species": prof["species"]["key"],
                "breed": [(t["key"], t["confidence"]) for t in (prof.get("breed") or {}).get("top", [])[:2]],
                "breed_level": (prof.get("breed") or {}).get("level"),
                "marks": [m["key"] for m in prof.get("marks", [])],
                "mark_probs": prof.get("mark_probs", {}),
                "no_animal": bool(prof["quality"].get("no_animal")),
            }
        with_ref = [r for r in rows if r["ref"]]
        ious = np.array([r["res"][b]["iou"] for r in with_ref])
        small = [r for r in with_ref if (r["ref"][2] - r["ref"][0]) * (r["ref"][3] - r["ref"][1]) < 0.1 * np.prod(Image.open(r["path"]).size)]
        mark_counts: dict[str, int] = {}
        for r in rows:
            for k in r["res"][b]["marks"]:
                mark_counts[k] = mark_counts.get(k, 0) + 1
        summary[b] = {
            "n_ref": len(with_ref),
            "hit@0.5": round(float((ious >= 0.5).mean()), 3),
            "hit@0.7": round(float((ious >= 0.7).mean()), 3),
            "mean_iou": round(float(ious.mean()), 3),
            "missed": int(sum(1 for r in with_ref if not r["res"][b]["box"])),
            "small_n": len(small),
            "small_hit@0.5": round(float(np.mean([r["res"][b]["iou"] >= 0.5 for r in small])), 3) if small else None,
            "species_dog_acc": round(float(np.mean([r["res"][b]["species"] == r["ref_species"] for r in with_ref])), 3),
            "no_animal_on_animals": int(sum(r["res"][b]["no_animal"] for r in with_ref)),
            "marks_fired_per_image": {k: round(c / len(rows), 3) for k, c in sorted(mark_counts.items(), key=lambda x: -x[1])},
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        }
        print(b, json.dumps(summary[b]), flush=True)
        del v

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    sheet(rows, backends, REPORTS / f"{a.tag}_worst.jpg")
    print("done", flush=True)


if __name__ == "__main__":
    main()
