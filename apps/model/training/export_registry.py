#!/usr/bin/env python3
"""Turn the live registry into training data (the data flywheel).

* every photo becomes a manifest row with pet_id = the pet → re-ID supervision
* owner-confirmed labels (`owner_labels` from the registration form) become
  ground-truth attribute labels; pets without them contribute identity only
* reunited lost/found pairs can be added by hand as the same pet_id

    python3 training/export_registry.py            # → data/manifests/registry.csv
"""

from __future__ import annotations

import json
import sqlite3

from common import ROOT, write_manifest

from petti_vision import Config


def main():
    cfg = Config()
    db = sqlite3.connect(str(cfg.db_path))
    rows = []
    pets = db.execute("SELECT id, profile_json FROM pets").fetchall()
    for pid, pj in pets:
        prof = json.loads(pj)
        own = prof.get("owner_labels") or {}
        photos = db.execute("SELECT path FROM photos WHERE pet_id=? ORDER BY idx", (pid,)).fetchall()
        for (path,) in photos:
            rows.append({
                "path": str(cfg.images_dir / path),
                "species": own.get("species") or (prof["species"]["key"] if own else ""),
                "breed": own.get("breed", ""),
                "size": own.get("size", ""),
                "age": own.get("age", ""),
                "colors": "|".join(own.get("colors") or []),
                "pattern": own.get("pattern", ""),
                "coat": own.get("coat", ""),
                "marks": "|".join(own.get("marks") or []) if "marks" in own else "",
                "pet_id": f"reg_{pid}",
                "source": "owner" if own else "registry",
            })
    out = ROOT / "data" / "manifests" / "registry.csv"
    write_manifest(out, rows)
    print(f"{len(pets)} pets → {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
