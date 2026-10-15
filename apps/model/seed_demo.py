#!/usr/bin/env python3
"""Fill the registry with demo pets from a folder (one sub-folder per animal).

    demo_pets/
      Bantan/   1.jpg 2.jpg 3.jpg 4.jpg 5.jpg
      Tsagaan/  ...
    python3 seed_demo.py demo_pets --status lost --location "СБД"

Then search with a held-out photo of one of them on /search.
"""

import argparse
from pathlib import Path

from petti_vision import Config, PettiVision
from petti_vision.registry import Registry

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--status", default="lost", choices=["home", "lost", "found"])
ap.add_argument("--location", default=None)
a = ap.parse_args()

cfg = Config()
vision = PettiVision(cfg)
reg = Registry(cfg.db_path, cfg.images_dir)
for d in sorted(p for p in a.folder.iterdir() if p.is_dir()):
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not imgs:
        continue
    raw = [p.read_bytes() for p in imgs]
    prof = vision.analyze(raw)
    rec = reg.add(profile=prof, images=raw, status=a.status, name=d.name, location=a.location)
    print(f"{d.name:20s} {len(raw)} photos  {rec.category:40s} {prof.data['timing_ms']['total']} ms")
print(f"registry: {len(reg.pets)} pets")
