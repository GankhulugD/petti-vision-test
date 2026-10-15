#!/usr/bin/env python3
"""Recompute every stored profile + embedding with the current models.

Run after changing backbones (PETTI_REID_MODEL / PETTI_VLM_MODEL), training a
new re-ID projection or new heads, so old and new registrations live in the
same embedding space.

    python3 training/reindex.py
"""

from __future__ import annotations

import json

import numpy as np
from common import ROOT  # noqa: F401  (sets sys.path)
from tqdm import tqdm

from petti_vision import Config, PettiVision
from petti_vision.pipeline import apply_owner_labels
from petti_vision.registry import Registry


def main():
    cfg = Config()
    vision = PettiVision(cfg)
    reg = Registry(cfg.db_path, cfg.images_dir)
    for pet in tqdm(list(reg.pets.values()), desc="reindex"):
        prof = vision.analyze([str(cfg.images_dir / p) for p in pet.photo_paths])
        data = prof.to_json()
        if pet.profile.get("owner_labels"):
            data = apply_owner_labels(data, pet.profile["owner_labels"])
        with reg.lock:
            reg.db.execute("UPDATE pets SET profile_json=?, category=?, species=? WHERE id=?",
                           (json.dumps(data, ensure_ascii=False), data["category"], data["species"]["key"], pet.id))
            rows = reg.db.execute("SELECT id FROM photos WHERE pet_id=? ORDER BY idx", (pet.id,)).fetchall()
            for (ph_id,), r, v in zip(rows, prof.reid, prof.vlm):
                reg.db.execute("UPDATE photos SET reid=?, vlm=? WHERE id=?",
                               (r.astype(np.float32).tobytes(), v.astype(np.float32).tobytes(), ph_id))
        reg.db.commit()
    print(f"reindexed {len(reg.pets)} pets — restart the server")


if __name__ == "__main__":
    main()
