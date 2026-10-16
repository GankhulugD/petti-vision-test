"""Offline teacher for dog/cat boxes and masks: torchvision Mask R-CNN v2.

BSD-3 code, COCO-trained weights. Used only to label/measure data during
training and evaluation — never loaded by the server.

Results are cached per image (sha1 of bytes) under data/teacher/ so every
script can reuse them for free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT

CACHE = ROOT / "data" / "teacher"
COCO_CAT, COCO_DOG = 17, 18
MASK_SIDE = 128  # cached mask resolution (long side)


def _key(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class Teacher:
    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = device
        self.model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device).eval()
        CACHE.mkdir(parents=True, exist_ok=True)

    def _run(self, img: Image.Image) -> dict:
        torch = self.torch
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        with torch.inference_mode():
            out = self.model([x])[0]
        keep = [(i, int(l)) for i, l in enumerate(out["labels"].tolist()) if int(l) in (COCO_CAT, COCO_DOG)]
        inst = []
        w, h = img.size
        s = MASK_SIDE / max(w, h)
        mw, mh = max(1, round(w * s)), max(1, round(h * s))
        for i, lab in keep:
            score = float(out["scores"][i])
            if score < 0.3:
                continue
            m = out["masks"][i, 0].float().cpu().numpy()
            small = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((mw, mh), Image.BILINEAR), dtype=np.uint8)
            inst.append({
                "species": "cat" if lab == COCO_CAT else "dog",
                "score": round(score, 4),
                "box": [round(float(v), 1) for v in out["boxes"][i].tolist()],
                "mask": small,
            })
        return {"size": [w, h], "inst": inst}

    def get(self, path: Path) -> dict:
        """{'size': [w,h], 'inst': [{'species','score','box','mask'(uint8 0..255, long side 128)}]}"""
        k = _key(path)
        meta_p, mask_p = CACHE / f"{k}.json", CACHE / f"{k}.npz"
        if meta_p.exists() and mask_p.exists():
            meta = json.loads(meta_p.read_text())
            z = np.load(mask_p)
            for j, it in enumerate(meta["inst"]):
                it["mask"] = z[f"m{j}"]
            return meta
        img = Image.open(path).convert("RGB")
        res = self._run(img)
        np.savez_compressed(mask_p, **{f"m{j}": it["mask"] for j, it in enumerate(res["inst"])})
        meta_p.write_text(json.dumps({"size": res["size"], "inst": [{k2: v for k2, v in it.items() if k2 != "mask"} for it in res["inst"]]}))
        return res


def main_instance(res: dict, min_score: float = 0.5) -> dict | None:
    """The animal a user means: largest confident dog/cat."""
    c = [it for it in res["inst"] if it["score"] >= min_score]
    if not c:
        return None
    return max(c, key=lambda it: (it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]) * it["score"])


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
