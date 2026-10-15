"""Lost & Found matcher: query profile → ranked candidate pets with reasons.

score = w_reid · identity(DINOv2 multi-view)        – "is it THIS animal?"
      + w_vlm  · appearance(SigLIP multi-view)       – "does it look alike?"
      + w_attr · attribute compatibility             – breed/colour/size/age/marks

* species is a hard filter (a cat is never suggested for a lost dog)
* multi-view: every query photo is compared with every registered photo
  (5+ angles at registration ⇒ robust to pose); best view per query photo,
  then averaged over query photos.
* Identity similarities are mapped to [0,1] with a logistic calibration whose
  parameters `training/evaluate.py` fits on your data (artifacts/calibration.json).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import taxonomy as T
from .config import Config
from .registry import PetRecord

DEFAULT_CAL = {
    # cosine → probability-of-same-animal; defaults for dinov2-small CLS‖mean (re-fit with evaluate.py)
    "reid_mu": 0.62, "reid_tau": 0.06,
    "vlm_mu": 0.80, "vlm_tau": 0.05,
}

ATTR_WEIGHTS = {"breed": 0.24, "color": 0.26, "pattern": 0.14, "size": 0.14, "coat": 0.12, "age": 0.10}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bc(p: dict, q: dict) -> float:
    """Bhattacharyya coefficient of two distributions given as {key: prob}."""
    keys = set(p) | set(q)
    s = sum(math.sqrt(max(p.get(k, 0.0), 0) * max(q.get(k, 0.0), 0)) for k in keys)
    return float(min(1.0, s))


def _ordinal_compat(p: dict, q: dict, order: dict, scale: float) -> float:
    """1 when expected ordinal distance is 0, decays with distance."""
    d = 0.0
    for a, pa in p.items():
        for b, qb in q.items():
            d += pa * qb * abs(order[a] - order[b])
    return float(math.exp(-d / scale))


@dataclass
class Match:
    pet: PetRecord
    score: float
    reid: float
    vlm: float
    attr: float
    details: dict
    reasons: list[str]
    mismatches: list[str]

    def to_json(self) -> dict:
        return {
            "pet": self.pet.public(),
            "score": round(self.score, 4),
            "percent": int(round(self.score * 100)),
            "components": {"identity": round(self.reid, 4), "appearance": round(self.vlm, 4), "attributes": round(self.attr, 4)},
            "details": {k: round(v, 3) for k, v in self.details.items()},
            "reasons": self.reasons,
            "mismatches": self.mismatches,
        }


class Matcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cal = dict(DEFAULT_CAL)
        self.w = {"reid": cfg.w_reid, "vlm": cfg.w_vlm, "attr": cfg.w_attr}
        cal_path = cfg.artifacts_dir / "calibration.json"
        if cal_path.exists():
            c = json.loads(cal_path.read_text())
            self.cal.update({k: v for k, v in c.items() if k in self.cal})
            for k in ("reid", "vlm", "attr"):
                if f"w_{k}" in c:
                    self.w[k] = float(c[f"w_{k}"])

    # ------------------------------------------------------------------ #
    def attribute_compat(self, q: dict, c: dict) -> tuple[float, dict, list[str], list[str]]:
        det: dict[str, float] = {}
        reasons, mism = [], []

        if q.get("breed") and c.get("breed"):
            det["breed"] = _bc(q["breed"]["probs"], c["breed"]["probs"])
        det["color"] = _bc(q["color_probs"], c["color_probs"])
        det["pattern"] = _bc(q["pattern"]["probs"], c["pattern"]["probs"])
        det["coat"] = _bc(q["coat"]["probs"], c["coat"]["probs"])
        det["size"] = _ordinal_compat(q["size"]["probs"], c["size"]["probs"], T.SIZE_INDEX, 0.9)
        # a lost puppy keeps growing, so age differences are tolerated more
        det["age"] = _ordinal_compat(q["age"]["probs"], c["age"]["probs"], T.AGE_INDEX, 1.3)

        ws = {k: v for k, v in ATTR_WEIGHTS.items() if k in det}
        base = sum(det[k] * w for k, w in ws.items()) / sum(ws.values())

        # distinctive marks: shared rare marks are strong evidence
        bonus = 0.0
        qm, cm = q.get("mark_probs", {}), c.get("mark_probs", {})
        for k in set(qm) & set(cm):
            m = T.MARK_BY_KEY.get(k)
            if not m:
                continue
            a, b = qm[k], cm[k]
            if a >= m.threshold and b >= m.threshold:
                bonus += 0.04 * m.weight
                reasons.append(f"Онцгой шинж таарсан: {m.mn}")
            elif m.weight >= 1.5 and ((a >= m.threshold + 0.1 and b < 0.2) or (b >= m.threshold + 0.1 and a < 0.2)):
                # one view might hide it (scar on the other side) → soft penalty only
                bonus -= 0.03 * m.weight
                mism.append(f"Онцгой шинж зөрж байна: {m.mn}")
        det["marks_bonus"] = bonus
        score = float(np.clip(base + bonus, 0.0, 1.0))

        if det.get("breed", 0) > 0.6 and c.get("breed"):
            reasons.append("Үүлдэр төстэй: " + ", ".join(x["mn"] for x in c["breed"]["looks_like"][:2]))
        if det["color"] > 0.75:
            reasons.append("Өнгө таарч байна: " + ", ".join(x["mn"] for x in c["colors"]))
        elif det["color"] < 0.4:
            mism.append("Өнгө зөрүүтэй")
        if det["size"] > 0.8:
            reasons.append(f"Хэмжээ ижил: {c['size']['mn']}")
        elif det["size"] < 0.4:
            mism.append("Хэмжээ зөрүүтэй")
        if det["pattern"] > 0.75:
            reasons.append(f"Зүс ижил: {c['pattern']['mn']}")
        return score, det, reasons, mism

    # ------------------------------------------------------------------ #
    def search(self, query, candidates: list[PetRecord], top_k: int | None = None) -> list[Match]:
        """query: AnimalProfile. candidates: pets to rank (e.g. status='lost')."""
        top_k = top_k or self.cfg.search_top_k
        qd = query.data
        q_species_p = qd["species_probs"]

        # hard species filter (keep if query gives the species ≥15%)
        cands = [c for c in candidates if q_species_p.get(c.species, 0) >= 0.15 and len(c.photo_paths) > 0]
        if not cands:
            return []

        # vectorised multi-view similarity
        R = np.concatenate([c.reid for c in cands])  # (sumP, D)
        V = np.concatenate([c.vlm for c in cands])
        offsets = np.cumsum([0] + [len(c.reid) for c in cands])[:-1]
        sim_r = query.reid @ R.T  # (Pq, sumP)
        sim_v = query.vlm @ V.T
        best_r = np.maximum.reduceat(sim_r, offsets, axis=1)  # (Pq, Npets)
        best_v = np.maximum.reduceat(sim_v, offsets, axis=1)
        # average over query views, but let the single best view count too
        reid_raw = 0.5 * best_r.mean(0) + 0.5 * best_r.max(0)
        vlm_raw = 0.5 * best_v.mean(0) + 0.5 * best_v.max(0)

        out: list[Match] = []
        for i, c in enumerate(cands):
            r = _sig((reid_raw[i] - self.cal["reid_mu"]) / self.cal["reid_tau"])
            v = _sig((vlm_raw[i] - self.cal["vlm_mu"]) / self.cal["vlm_tau"])
            a, det, reasons, mism = self.attribute_compat(qd, c.profile)
            score = self.w["reid"] * r + self.w["vlm"] * v + self.w["attr"] * a
            score *= 0.85 + 0.15 * q_species_p.get(c.species, 0)  # species uncertainty
            if r > 0.8:
                reasons.insert(0, "Зураг дээрх төрх маш төстэй (ижил амьтан байх магадлал өндөр)")
            elif r > 0.5:
                reasons.insert(0, "Зураг дээрх төрх төстэй")
            det.update({"reid_cos": float(reid_raw[i]), "vlm_cos": float(vlm_raw[i])})
            out.append(Match(c, float(score), r, v, a, det, reasons, mism))
        out.sort(key=lambda m: -m.score)
        return out[:top_k]


def load_calibration(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else dict(DEFAULT_CAL)
