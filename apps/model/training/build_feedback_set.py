#!/usr/bin/env python3
"""Consented user corrections → training manifest (data/manifests/feedback.csv).

Only records with consent=true whose photo is still on disk are used (a deleted
photo — DELETE /feedback/<id> — drops out automatically). Photos are already
de-duplicated by content hash; the last correction for a photo wins.
"""

from __future__ import annotations

import json

from common import ROOT, write_manifest

FB = ROOT / "data" / "feedback"


def main():
    rows: dict[str, dict] = {}
    log = FB / "feedback.jsonl"
    if not log.exists():
        print("no feedback yet", flush=True)
        write_manifest(ROOT / "data" / "manifests" / "feedback.csv", [])
        return
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("consent"):
            continue
        c = rec.get("corrections") or {}
        for h in rec.get("images", []):
            p = FB / "images" / f"{h}.jpg"
            if not p.exists():
                continue
            rows[h] = {
                "path": str(p), "species": c.get("species", ""), "breed": c.get("breed", ""),
                "size": c.get("size", ""), "age": c.get("age", ""),
                "colors": "|".join(c.get("colors") or []), "pattern": c.get("pattern", ""),
                "coat": c.get("coat", ""), "marks": "|".join(c.get("marks") or []) if "marks" in c else "",
                "pet_id": "", "source": "feedback",
            }
    write_manifest(ROOT / "data" / "manifests" / "feedback.csv", list(rows.values()))
    print(f"feedback rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
