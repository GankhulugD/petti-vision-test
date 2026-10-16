#!/usr/bin/env python3
"""Analyse a few photos with the code on disk (fresh imports) and print the breed answer.

    python3 training/try_images.py path1.jpg [path2.jpg ...] [--heads heads_x]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT  # noqa: F401  (sets sys.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--heads", default="")
    ap.add_argument("--server", default="", help="ask a running server instead, e.g. http://localhost:8765")
    a = ap.parse_args()
    if a.server:
        import uuid
        import urllib.request

        for p in a.paths:
            bd = uuid.uuid4().hex
            body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"images\"; filename=\"x.jpg\"\r\n"
                    f"Content-Type: image/jpeg\r\n\r\n").encode() + Path(p).read_bytes() + f"\r\n--{bd}--\r\n".encode()
            req = urllib.request.Request(a.server.rstrip("/") + "/analyze", data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={bd}"})
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            b = d.get("breed") or {}
            print(json.dumps({"file": Path(p).name, "server": True, "level": b.get("level"),
                              "top": [(t["key"], round(t["confidence"], 3)) for t in b.get("top", [])[:4]]}, ensure_ascii=False), flush=True)
        return
    from petti_vision import Config, PettiVision

    v = PettiVision(Config())
    if a.heads:
        from petti_vision.heads import load_heads

        v.heads = load_heads(Path(v.cfg.artifacts_dir) / f"{a.heads}.npz")
    for p in a.paths:
        d = v.analyze([Path(p).read_bytes()]).to_json()
        b = d.get("breed") or {}
        print(json.dumps({"file": Path(p).name, "level": b.get("level"),
                          "top": [(t["key"], round(t["confidence"], 3)) for t in b.get("top", [])[:4]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
