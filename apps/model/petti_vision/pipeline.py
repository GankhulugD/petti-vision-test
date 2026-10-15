"""Image(s) → AnimalProfile (species, breed mix, size, age, colours, pattern,
coat, distinctive marks, text description, embeddings).

Signal fusion per attribute (log-linear / product-of-experts):

    species  = detector class prior × zero-shot × trained head
    breed    = zero-shot (species-conditioned) × trained head
    size     = zero-shot × breed→size prior × trained head
    age      = zero-shot × trained head
    colours  = pixel estimator (Lab) ⊕ zero-shot ⊕ head
    pattern / coat / marks = zero-shot × trained head

With no trained heads (day 0) the system works purely zero-shot; every time
you retrain heads on registered pets the fused accuracy goes up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from . import taxonomy as T
from .backbones import Detection, build, crop_detection, l2n, load_image
from .color import coat_colors
from .config import Config
from .heads import load_heads
from .zeroshot import ZeroShot

log = logging.getLogger("petti.pipeline")


# --------------------------------------------------------------------------- #
def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def fuse(dists: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted geometric mean of distributions (product of experts)."""
    logp = sum(w * np.log(np.clip(d, 1e-9, None)) for d, w in zip(dists, weights))
    logp = logp - logp.max()
    return _norm(np.exp(logp))


def _square(img: Image.Image, fill=(124, 116, 104)) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def _label_dict(lab: T.Label, p: float) -> dict:
    return {"key": lab.key, "en": lab.en, "mn": lab.mn, "confidence": round(float(p), 4)}


@dataclass
class PhotoResult:
    """Per-photo raw outputs (probabilities kept un-thresholded for aggregation)."""

    detection: Detection | None
    crop_ratio: float
    reid: np.ndarray
    vlm: np.ndarray
    probs: dict[str, np.ndarray]  # task -> distribution over that task's labels
    breed_keys: list[str]
    color_shares: np.ndarray
    marks: dict[str, float]
    weight: float = 1.0


@dataclass
class AnimalProfile:
    data: dict[str, Any]
    reid: np.ndarray  # (P, D) one per photo
    vlm: np.ndarray  # (P, D)
    photos: list[PhotoResult] = field(default_factory=list)

    @property
    def species(self) -> str:
        return self.data["species"]["key"]

    def to_json(self) -> dict:
        return self.data


# --------------------------------------------------------------------------- #
class PettiVision:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        t0 = time.perf_counter()
        self.detector, self.enc = build(self.cfg)
        self.zs = ZeroShot(self.enc, None if self.cfg.fake_models else self.cfg.text_cache_path)
        self.heads = None if self.cfg.fake_models else load_heads(self.cfg.heads_path)
        if self.heads:
            log.info("trained heads loaded: %s", list(self.heads.tasks))
        self.load_seconds = time.perf_counter() - t0
        self._species_keys = [s.key for s in T.SPECIES]
        self._breeds = {sp: T.breeds_for(sp) for sp in ("dog", "cat")}

    # ------------------------------------------------------------------ #
    def warmup(self):
        img = Image.new("RGB", (320, 320), (120, 100, 80))
        self.analyze([img])

    def _head(self, task: str, feats: np.ndarray, keys: list[str]) -> tuple[np.ndarray | None, float]:
        """Head probabilities re-indexed to `keys` (renormalised); alpha weight."""
        if not self.heads or not self.heads.has(task):
            return None, 0.0
        labels, p = self.heads.predict(task, feats[None])
        p = p[0]
        idx = {k: i for i, k in enumerate(labels)}
        if self.heads.tasks[task]["multi"]:
            return np.array([p[idx[k]] if k in idx else np.nan for k in keys]), self.heads.tasks[task]["alpha"]
        out = np.array([p[idx[k]] if k in idx else 1e-4 for k in keys])
        return _norm(out), self.heads.tasks[task]["alpha"]

    def _fuse_task(self, task: str, zs_p: np.ndarray, feats: np.ndarray, keys: list[str], extra=None, zs_w: float = 1.0) -> np.ndarray:
        dists, ws = [zs_p], [zs_w]
        hp, alpha = self._head(task, feats, keys)
        if hp is not None:
            dists, ws = [zs_p, hp], [zs_w * (1.0 - alpha), alpha]
        if extra is not None:
            dists.append(extra[0])
            ws.append(extra[1])
        return fuse(dists, ws)

    # ------------------------------------------------------------------ #
    def _photo(self, img: Image.Image, det: Detection | None, reid: np.ndarray, vlm: np.ndarray, crop: Image.Image,
               mask: np.ndarray | None = None) -> PhotoResult:
        feats = self.heads.features(vlm[None], reid[None])[0] if self.heads else None
        v = vlm[None]
        probs: dict[str, np.ndarray] = {}

        # ---- species -------------------------------------------------- #
        p_sp = self.zs.probs("species|any", v)[0]
        extra = None
        if det is not None:
            prior = np.full(len(self._species_keys), (1 - det.confidence) / len(self._species_keys))
            prior[self._species_keys.index(det.species)] += det.confidence
            # COCO-trained detector is very reliable for dog vs cat: trust it more when confident
            extra = (_norm(prior), 2.0 if det.confidence >= 0.5 else 1.0)
        probs["species"] = self._fuse_task("species", p_sp, feats, self._species_keys, extra)
        species = self._species_keys[int(np.argmax(probs["species"]))]
        sp_cond = species if species in ("dog", "cat") else "other"

        # ---- breed (dogs & cats only) ---------------------------------- #
        breed_keys: list[str] = []
        if species in ("dog", "cat"):
            breeds = self._breeds[species]
            breed_keys = [b.key for b in breeds]
            p_b = self.zs.probs(f"breed|{species}", v)[0]
            probs["breed"] = self._fuse_task("breed", p_b, feats, breed_keys)

        # ---- size (breed prior helps a lot: a husky is "medium") ------- #
        size_keys = [s.key for s in T.SIZES]
        p_size = self.zs.probs(f"size|{sp_cond}", v)[0]
        size_extra = None
        if "breed" in probs:
            prior = np.full(len(size_keys), 1e-3)
            for b, pb in zip(self._breeds[species], probs["breed"]):
                i = T.SIZE_INDEX[b.size]
                prior[i] += pb
                for j in (i - 1, i + 1):  # individuals vary ±1 class
                    if 0 <= j < len(size_keys):
                        prior[j] += 0.25 * pb
            size_extra = (_norm(prior), 1.5)
        # a single photo has no scale reference → zero-shot size is weak; the
        # breed prior carries most of the signal until a size head is trained
        probs["size"] = self._fuse_task("size", p_size, feats, size_keys, size_extra, zs_w=0.4)

        # ---- age / pattern / coat ------------------------------------- #
        for task, labels in (("age", T.AGES), ("pattern", T.PATTERNS), ("coat", T.COAT_LENGTH)):
            keys = [lab.key for lab in labels]
            probs[task] = self._fuse_task(task, self.zs.probs(f"{task}|{sp_cond}", v)[0], feats, keys)

        # ---- colours: pixels ⊕ zero-shot ⊕ head ---------------------- #
        color_keys = [c.key for c in T.COLORS]
        px = coat_colors(crop, mask=mask)
        shares = np.array([px[k] for k in color_keys])
        zs_c = self.zs.probs(f"color|{sp_cond}", v)[0]
        mix = 0.55 * shares + 0.45 * zs_c
        hc, alpha = self._head("color", feats, color_keys) if feats is not None else (None, 0)
        if hc is not None and not np.isnan(hc).any():
            mix = (1 - alpha) * mix + alpha * (hc / max(hc.sum(), 1e-6))
        shares = mix / mix.sum()

        # ---- marks (multi-label) ------------------------------------- #
        marks = {k: float(p[0]) for k, p in self.zs.mark_probs(species, v).items()}
        hm, alpha = self._head("marks", feats, list(marks)) if feats is not None else (None, 0)
        if hm is not None:
            for (k, zp), hp in zip(list(marks.items()), hm):
                if not np.isnan(hp):
                    marks[k] = (1 - alpha) * zp + alpha * float(hp)

        crop_ratio = 1.0 if det is None else det.area() / (img.width * img.height)
        weight = (det.confidence if det else 0.3) * (0.5 + min(crop_ratio, 0.5))
        return PhotoResult(det, crop_ratio, reid, vlm, probs, breed_keys, shares, marks, weight)

    # ------------------------------------------------------------------ #
    def analyze(self, images: list[bytes | Image.Image | str]) -> AnimalProfile:
        """Analyse 1..N photos of the SAME animal and return one profile."""
        t0 = time.perf_counter()
        imgs = [load_image(i) for i in images]
        dets = self.detector.detect(imgs)
        t_det = time.perf_counter()

        chosen = [d[0] if d else None for d in dets]
        crops = [crop_detection(im, d) for im, d in zip(imgs, chosen)]
        masks = []
        for im, d in zip(imgs, chosen):
            m = d.mask(im.size) if d is not None else None
            masks.append(None if m is None else np.asarray(crop_detection(Image.fromarray(m), d)))
        emb = self.enc.encode_images([_square(c) for c in crops], tta=self.cfg.tta)
        t_enc = time.perf_counter()

        photos = [self._photo(im, d, emb["reid"][i], emb["vlm"][i], crops[i], masks[i]) for i, (im, d) in enumerate(zip(imgs, chosen))]
        profile = self._aggregate(photos)
        t_end = time.perf_counter()
        profile.data["timing_ms"] = {
            "detect": round((t_det - t0) * 1000, 1),
            "embed": round((t_enc - t_det) * 1000, 1),
            "reason": round((t_end - t_enc) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
            "photos": len(imgs),
        }
        return profile

    # ------------------------------------------------------------------ #
    def _aggregate(self, photos: list[PhotoResult]) -> AnimalProfile:
        w = np.array([p.weight for p in photos], dtype=np.float64)
        w = w / w.sum()

        def avg(task):
            arr = [p.probs[task] for p in photos if task in p.probs]
            ww = [wi for p, wi in zip(photos, w) if task in p.probs]
            return _norm(np.average(np.stack(arr), axis=0, weights=ww)) if arr else None

        sp_p = avg("species")
        species_lab = T.SPECIES[int(np.argmax(sp_p))]
        species = species_lab.key
        warnings: list[str] = []

        # consistency: all photos should agree on species
        per_photo_sp = {T.SPECIES[int(np.argmax(p.probs["species"]))].key for p in photos}
        if len(per_photo_sp) > 1:
            warnings.append("Зургууд өөр өөр амьтан шиг харагдаж байна: " + ", ".join(sorted(per_photo_sp)))
        if any(p.detection is None for p in photos):
            warnings.append(f"{sum(p.detection is None for p in photos)} зурагт амьтан тод илрээгүй (бүтэн зургаар шинжлэв)")
        if any(p.detection is not None and p.crop_ratio < 0.03 for p in photos):
            warnings.append("Зарим зурагт амьтан хэт жижиг харагдаж байна — ойроос авна уу")

        # identity outlier check (photo of a different animal in the set)
        reid = np.stack([p.reid for p in photos])
        if len(photos) >= 3:
            sim = reid @ reid.T
            mean_sim = (sim.sum(1) - 1) / (len(photos) - 1)
            outliers = [i + 1 for i, s in enumerate(mean_sim) if s < mean_sim.mean() - 2.5 * mean_sim.std() - 0.05]
            if outliers:
                warnings.append(f"{outliers}-р зураг бусдаасаа их ялгаатай — өөр амьтан байж магадгүй")

        data: dict[str, Any] = {
            "species": _label_dict(species_lab, sp_p.max()),
            "species_probs": {s.key: round(float(p), 4) for s, p in zip(T.SPECIES, sp_p)},
        }

        # ---- breed ---------------------------------------------------- #
        breed_p = None
        if species in ("dog", "cat"):
            want = [b.key for b in T.breeds_for(species)]
            sel = [(p.probs["breed"], wi) for p, wi in zip(photos, w) if p.breed_keys == want]
            if sel:
                breed_p = _norm(np.average(np.stack([s[0] for s in sel]), axis=0, weights=[s[1] for s in sel]))
        if breed_p is not None:
            breeds = T.breeds_for(species)
            order = np.argsort(-breed_p)[:5]
            top = [_label_dict(breeds[i], breed_p[i]) for i in order]
            p1, p2 = breed_p[order[0]], breed_p[order[1]]
            first_is_mixed = breeds[order[0]].key in ("mixed_dog", "domestic_shorthair", "domestic_longhair")
            mixed = bool(first_is_mixed or p1 < 0.45 or (p1 - p2) < 0.15)
            looks_like = [t for t in top if t["key"] not in ("mixed_dog",)][:3 if mixed else 1]
            data["breed"] = {
                "top": top,
                "is_mixed": mixed,
                "looks_like": looks_like,
                "probs": {b.key: round(float(p), 4) for b, p in zip(breeds, breed_p)},
            }
        else:
            data["breed"] = None

        for task, labels in (("size", T.SIZES), ("age", T.AGES), ("pattern", T.PATTERNS), ("coat", T.COAT_LENGTH)):
            p = avg(task)
            i = int(np.argmax(p))
            data[task] = {**_label_dict(labels[i], p[i]), "probs": {l.key: round(float(v), 4) for l, v in zip(labels, p)}}

        shares = np.average(np.stack([p.color_shares for p in photos]), axis=0, weights=w)
        order = np.argsort(-shares)
        colors = [
            {"key": T.COLORS[i].key, "en": T.COLORS[i].en, "mn": T.COLORS[i].mn, "share": round(float(shares[i]), 3)}
            for i in order
            if shares[i] >= 0.12
        ][:3] or [{"key": T.COLORS[order[0]].key, "en": T.COLORS[order[0]].en, "mn": T.COLORS[order[0]].mn, "share": round(float(shares[order[0]]), 3)}]
        data["colors"] = colors
        data["color_probs"] = {c.key: round(float(s), 4) for c, s in zip(T.COLORS, shares)}

        mark_keys = set().union(*[p.marks for p in photos])
        mark_p = {}
        for k in mark_keys:
            vals = [(p.marks[k], wi) for p, wi in zip(photos, w) if k in p.marks]
            # max-pooling across views matters: a scar is visible from one side only
            mean = sum(v * wi for v, wi in vals) / sum(wi for _, wi in vals)
            mark_p[k] = 0.5 * mean + 0.5 * max(v for v, _ in vals)
        data["marks"] = sorted(
            [_label_dict(T.MARK_BY_KEY[k], v) for k, v in mark_p.items() if v >= T.MARK_BY_KEY[k].threshold],
            key=lambda d: -d["confidence"],
        )[:4]  # never flood the UI: the 4 most certain marks
        data["mark_probs"] = {k: round(float(v), 4) for k, v in mark_p.items()}

        data["category"] = self.category(data)
        data["description_mn"], data["description_en"] = describe(data)
        best = max(photos, key=lambda p: p.weight)
        data["detection"] = {
            "found": best.detection is not None,
            "box": list(best.detection.box) if best.detection else None,
            "confidence": round(best.detection.confidence, 3) if best.detection else 0.0,
        }
        data["quality"] = {"warnings": warnings, "photos": len(photos)}
        return AnimalProfile(data=data, reid=reid, vlm=np.stack([p.vlm for p in photos]), photos=photos)

    @staticmethod
    def category(d: dict) -> str:
        parts = [d["species"]["key"], d["size"]["key"], d["age"]["key"], "+".join(c["key"] for c in d["colors"][:2]), d["pattern"]["key"]]
        return "/".join(parts)


# --------------------------------------------------------------------------- #
def describe(d: dict) -> tuple[str, str]:
    mn, en = [], []
    mn.append(f"Төрөл: {d['species']['mn']} ({d['species']['confidence']:.0%})")
    en.append(f"Species: {d['species']['en']} ({d['species']['confidence']:.0%})")
    b = d.get("breed")
    if b:
        ll = ", ".join(f"{x['mn']} {x['confidence']:.0%}" for x in b["looks_like"])
        lle = ", ".join(f"{x['en']} {x['confidence']:.0%}" for x in b["looks_like"])
        if b["is_mixed"]:
            mn.append(f"Үүлдэр: эрлийз байж магадгүй — {ll}-тэй төстэй")
            en.append(f"Breed: likely mixed — resembles {lle}")
        else:
            mn.append(f"Үүлдэр: {ll}")
            en.append(f"Breed: {lle}")
    mn.append(f"Хэмжээ: {d['size']['mn']}")
    en.append(f"Size: {d['size']['en']}")
    mn.append(f"Нас: {d['age']['mn']}")
    en.append(f"Age: {d['age']['en']}")
    mn.append("Өнгө: " + ", ".join(c["mn"] for c in d["colors"]) + f" · Зүс: {d['pattern']['mn']} · Үс: {d['coat']['mn']}")
    en.append("Colour: " + ", ".join(c["en"] for c in d["colors"]) + f" · Pattern: {d['pattern']['en']} · Coat: {d['coat']['en']}")
    if d["marks"]:
        mn.append("Онцгой шинж: " + ", ".join(m["mn"] for m in d["marks"]))
        en.append("Distinctive: " + ", ".join(m["en"] for m in d["marks"]))
    return ". ".join(mn) + ".", ". ".join(en) + "."


__all__ = ["PettiVision", "AnimalProfile", "fuse", "describe", "l2n"]


# --------------------------------------------------------------------------- #
def apply_owner_labels(data: dict, labels: dict) -> dict:
    """Owner-confirmed labels (registration form) override/blend predictions.

    labels: {"breed": "siberian_husky", "size": "medium", "age": "adult",
             "colors": ["black", "white"], "pattern": "bicolor", "coat": "medium",
             "marks": ["blue_eyes"]}
    They also become ground truth for the next training round (export_registry.py).
    """
    data = dict(data)
    data["owner_labels"] = labels
    for task, table in (("size", T.SIZES), ("age", T.AGES), ("pattern", T.PATTERNS), ("coat", T.COAT_LENGTH)):
        key = labels.get(task)
        if key and data.get(task) and key in data[task]["probs"]:
            probs = {k: 0.2 * v + (0.8 if k == key else 0.0) for k, v in data[task]["probs"].items()}
            lab = next(l for l in table if l.key == key)
            data[task] = {**_label_dict(lab, probs[key]), "probs": probs, "owner": True}
    b = labels.get("breed")
    if b and data.get("breed") and b in data["breed"]["probs"]:
        probs = {k: 0.2 * v + (0.8 if k == b else 0.0) for k, v in data["breed"]["probs"].items()}
        lab = T.BREED_BY_KEY[b]
        top = sorted(probs, key=lambda k: -probs[k])[:5]
        tops = [_label_dict(T.BREED_BY_KEY[k], probs[k]) for k in top]
        data["breed"] = {"top": tops, "is_mixed": b in ("mixed_dog",), "looks_like": [_label_dict(lab, probs[b])], "probs": probs, "owner": True}
    cols = [c for c in labels.get("colors") or [] if c in data["color_probs"]]
    if cols:
        probs = {k: 0.2 * v + (0.8 / len(cols) if k in cols else 0.0) for k, v in data["color_probs"].items()}
        data["color_probs"] = probs
        data["colors"] = [{"key": k, "en": T.label_by_key("color", k).en, "mn": T.label_by_key("color", k).mn, "share": round(probs[k], 3)} for k in cols]
    for m in labels.get("marks") or []:
        if m in T.MARK_BY_KEY:
            data.setdefault("mark_probs", {})[m] = 0.95
    if labels.get("marks"):
        data["marks"] = [_label_dict(T.MARK_BY_KEY[k], v) for k, v in data["mark_probs"].items() if v >= T.MARK_BY_KEY[k].threshold]
    data["category"] = PettiVision.category(data)
    data["description_mn"], data["description_en"] = describe(data)
    return data
