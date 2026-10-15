#!/usr/bin/env python3
"""Explain what every stage of the pipeline sees for one photo.

    python3 diagnose.py path/to/dog.jpg

Prints versions/device, a text-tower sanity check, detector output, raw
zero-shot top-5 per attribute (before fusion), mark scores and pixel colours,
and writes artifacts/debug/<name>_crop.jpg (+ _mask.png) so you can see the crop.
Paste the output when reporting bad results.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

from petti_vision import Config, PettiVision
from petti_vision import taxonomy as T
from petti_vision.backbones import crop_detection, load_image
from petti_vision.color import coat_colors
from petti_vision.pipeline import _square


def top(keys, p, k=5):
    o = np.argsort(-p)[:k]
    return ", ".join(f"{keys[i]} {p[i]:.0%}" for i in o)


def main(paths: list[str]):
    import torch
    import transformers

    cfg = Config()
    v = PettiVision(cfg)
    enc = v.enc
    print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"device={getattr(enc, 'device', '?')} dtype={getattr(enc, 'dtype', '?')} detector={type(v.detector.model).__name__ if hasattr(v.detector, 'model') else '?'}")
    print(f"siglip logit_scale={enc.logit_scale:.2f} logit_bias={enc.logit_bias:.2f}  heads={'yes' if v.heads else 'no'}")

    # text tower sanity: related captions must be closer than unrelated ones
    t = enc.encode_texts(["a photo of a dog", "a photo of a puppy", "a photo of a cat", "a photo of a car"])
    print(f"text sanity: cos(dog,puppy)={t[0] @ t[1]:.3f}  cos(dog,cat)={t[0] @ t[2]:.3f}  cos(dog,car)={t[0] @ t[3]:.3f}"
          f"  -> {'OK' if t[0] @ t[1] > t[0] @ t[3] else 'BROKEN'}")

    out = Path(cfg.artifacts_dir) / "debug"
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        print("\n" + "=" * 80 + f"\n{path}")
        img = load_image(path)
        dets = v.detector.detect([img])[0]
        print(f"image {img.size}; detections: " + (", ".join(f"{d.species} {d.confidence:.2f} mask={'y' if d.polygon else 'n'}" for d in dets) or "NONE"))
        d = dets[0] if dets else None
        crop = crop_detection(img, d)
        crop.save(out / f"{Path(path).stem}_crop.jpg")
        mask = None
        if d is not None and d.polygon:
            from PIL import Image

            m = d.mask(img.size)
            mask = np.asarray(crop_detection(Image.fromarray(m), d))
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{Path(path).stem}_mask.png")
        e = enc.encode_images([_square(crop)], tta=False)
        x = e["vlm"]

        # image-text sanity: "dog" caption should beat others for a dog photo
        cs = v.zs.cosines("species|any", x)[0]
        print("species raw cos: " + ", ".join(f"{s.key}={c:.3f}" for s, c in zip(T.SPECIES, cs)))
        print("species zs     : " + top([s.key for s in T.SPECIES], v.zs.probs("species|any", x)[0]))
        sp = T.SPECIES[int(np.argmax(cs))].key if d is None else d.species
        cond = sp if sp in ("dog", "cat") else "other"
        if sp in ("dog", "cat"):
            print("breed zs       : " + top([b.key for b in T.breeds_for(sp)], v.zs.probs(f"breed|{sp}", x)[0]))
        for task, labels in (("size", T.SIZES), ("age", T.AGES), ("color", T.COLORS), ("pattern", T.PATTERNS), ("coat", T.COAT_LENGTH)):
            print(f"{task:7s} zs     : " + top([lab.key for lab in labels], v.zs.probs(f"{task}|{cond}", x)[0], 3))
        px = coat_colors(crop, mask=mask)
        print("pixel colours  : " + ", ".join(f"{k} {p:.0%}" for k, p in sorted(px.items(), key=lambda kv: -kv[1])[:4]) + f"  (mask={'y' if mask is not None else 'n'})")
        mk = v.zs.mark_probs(sp, x)
        print("marks          : " + ", ".join(f"{k} {p[0]:.2f}" for k, p in sorted(mk.items(), key=lambda kv: -kv[1][0])[:8]))
        prof = v.analyze([path])
        print("FINAL          : " + prof.data["description_en"])
        print("timing         :", prof.data["timing_ms"])
    print(f"\ncrops saved in {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 diagnose.py image.jpg [more.jpg ...]")
    main(sys.argv[1:])
