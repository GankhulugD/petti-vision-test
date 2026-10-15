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
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from . import taxonomy as T
from .backbones import Detection, build, crop_box, crop_detection, l2n, load_image
from .color import PALETTE_LAB, coat_colors
from .config import Config
from .heads import load_heads
from .zeroshot import ZeroShot, softmax

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


def confidence_level(p: float, cuts: tuple[float, float]) -> str:
    """high / medium / low – what the UI should say ("итгэлтэй" / "магадгүй" / "тодорхойгүй")."""
    return "high" if p >= cuts[0] else "medium" if p >= cuts[1] else "low"


def _square_mask(m: np.ndarray) -> np.ndarray:
    """Pad a crop mask exactly like _square pads the crop."""
    h, w = m.shape
    sz = max(h, w)
    out = np.zeros((sz, sz), dtype=bool)
    y, x = (sz - h) // 2, (sz - w) // 2
    out[y : y + h, x : x + w] = m
    return out


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
    extras: dict = field(default_factory=dict)  # explainability (zero-shot vs head votes)


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


def _despeckle(a: np.ndarray) -> np.ndarray:
    """ViTs without registers have a few high-norm 'artifact' patches that light up
    in any CLS-similarity map; replace isolated spikes by their neighbourhood."""
    p = np.pad(a, 1, mode="edge")
    nb = np.stack([p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    med = np.median(nb, axis=0)
    spike = a > nb.max(0) + 0.5 * (a.std() + 1e-6)
    return np.where(spike, med, a)


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
        self._lock = threading.Lock()
        self._species_keys = [s.key for s in T.SPECIES]
        # zero-shot breed softmax is over-confident; T fitted on a labelled set
        # (training/calibrate.py writes artifacts/calibration.json)
        self.breed_temperature = 1.5
        self._gate_pos = None
        self._gate_neg = None
        cal = self.cfg.artifacts_dir / "calibration.json"
        if cal.exists():
            import json

            self.breed_temperature = float(json.loads(cal.read_text()).get("breed_temperature", 1.5))
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
        seen = self.heads.tasks[task].get("seen")  # classes that had training data
        if self.heads.tasks[task]["multi"]:
            return np.array([p[idx[k]] if k in idx else np.nan for k in keys]), self.heads.tasks[task]["alpha"]
        # classes the head never saw (e.g. Bankhar before we collect photos) are
        # NaN → they keep their zero-shot probability instead of being zeroed
        out = np.array([p[idx[k]] if (k in idx and (seen is None or k in seen)) else np.nan for k in keys])
        return out, self.heads.tasks[task]["alpha"]

    def _fuse_task(self, task: str, zs_p: np.ndarray, feats: np.ndarray, keys: list[str], extra=None, zs_w: float = 1.0) -> np.ndarray:
        dists, ws = [zs_p], [zs_w]
        if extra is not None:
            dists.append(extra[0])
            ws.append(extra[1])
        base = fuse(dists, ws)
        hp, alpha = self._head(task, feats, keys) if feats is not None else (None, 0.0)
        if hp is None:
            return base
        known = ~np.isnan(hp)
        if not known.any():
            return base
        out = base.copy()
        mass = base[known].sum()  # probability the answer is one of the head's classes
        h = _norm(hp[known])
        out[known] = mass * fuse([_norm(base[known]), h], [1.0 - alpha, alpha])
        return _norm(out)

    # ------------------------------------------------------------------ #
    def _photo(self, img: Image.Image, det: Detection | None, reid: np.ndarray, vlm: np.ndarray, crop: Image.Image,
               mask: np.ndarray | None = None) -> PhotoResult:
        feats = self.heads.features(vlm[None], reid[None])[0] if self.heads else None
        if feats is not None and self.heads.dim not in (None, feats.shape[-1]):
            # heads were trained on another backbone → ignore them instead of crashing
            log.warning("trained heads expect %s-d features, got %s-d: ignoring heads (retrain)", self.heads.dim, feats.shape[-1])
            self.heads, feats = None, None
        v = vlm[None]
        probs: dict[str, np.ndarray] = {}

        # ---- species -------------------------------------------------- #
        p_sp = self.zs.probs("species|any", v)[0]
        extra = None
        if det is not None and det.species in ("dog", "cat"):
            prior = np.full(len(self._species_keys), (1 - det.confidence) / len(self._species_keys))
            prior[self._species_keys.index(det.species)] += det.confidence
            # COCO-trained detector is very reliable for dog vs cat: trust it more when confident
            # trust a confident dog/cat detection more; COCO "horse"/"bird" boxes on
            # pets are a known YOLO confusion (a Doberman can come back as a horse)
            strong = det.confidence >= 0.5 and det.species in ("dog", "cat")
            extra = (_norm(prior), 2.0 if strong else 1.0)
        probs["species"] = self._fuse_task("species", p_sp, feats, self._species_keys, extra)
        # fine-grained species evidence: the best-fitting BREED caption competes
        # with the other species' captions on the same logit scale. "Sphynx
        # hairless cat" fits a sphynx far better than "a photo of a cat" (which
        # loses to "other animal"), and "Doberman Pinscher" beats "horse".
        ls = self.zs.logits("species|any", v)[0]
        fine = ls.copy()
        i_dog, i_cat = self._species_keys.index("dog"), self._species_keys.index("cat")
        fine[i_dog] = max(ls[i_dog], self.zs.logits("breed|dog", v)[0].max())
        fine[i_cat] = max(ls[i_cat], self.zs.logits("breed|cat", v)[0].max())
        probs["species"] = fuse([probs["species"], softmax(fine)], [1.0, 1.0])
        species = self._species_keys[int(np.argmax(probs["species"]))]
        sp_cond = species if species in ("dog", "cat") else "other"

        # ---- breed (dogs & cats only) ---------------------------------- #
        breed_keys: list[str] = []
        if species in ("dog", "cat"):
            breeds = self._breeds[species]
            breed_keys = [b.key for b in breeds]
            p_b = self.zs.probs(f"breed|{species}", v, temperature=self.breed_temperature)[0]
            probs["breed"] = self._fuse_task("breed", p_b, feats, breed_keys)
            hp, _ = self._head("breed", feats, breed_keys) if feats is not None else (None, 0.0)
            extras = {"breed_zs": p_b, "breed_head": hp}
        else:
            extras = {}

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

        # Breed size is the adult weight. A baby of a large breed is not 25-45kg.
        age_i = int(np.argmax(probs["age"]))
        shift = 2 if age_i == 0 else 1 if age_i == 1 else 0
        if shift:
            src = probs["size"]
            dst = np.zeros_like(src)
            for i, mass in enumerate(src):
                dst[max(0, i - shift)] += mass
            probs["size"] = dst / max(dst.sum(), 1e-6)

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

        # ---- pattern sanity: species-specific patterns + measured pixels ---- #
        pat_keys = [p.key for p in T.PATTERNS]
        prior = np.ones(len(pat_keys))
        for i, k in enumerate(pat_keys):
            if k in T.PATTERN_SPECIES and species in ("dog", "cat") and species not in T.PATTERN_SPECIES[k]:
                prior[i] = 1e-3  # e.g. "colour-point" on a dog
        sig = [k for k in color_keys if px[k] >= 0.2]
        lightness = {k: float(L) for k, L in zip(color_keys, PALETTE_LAB[:, 0])}
        spread = max(lightness[k] for k in sig) - min(lightness[k] for k in sig) if sig else 0.0
        if spread >= 35:  # black + white/tan → not a solid coat
            prior[pat_keys.index("solid")] *= 0.25
        elif max(px.values()) >= 0.45:  # one dominant shade (cream+golden lab) → solid
            prior[pat_keys.index("solid")] *= 5.0
        probs["pattern"] = fuse([probs["pattern"], _norm(prior)], [1.0, 1.0])

        # ---- marks (multi-label) ------------------------------------- #
        marks = {k: float(p[0]) for k, p in self.zs.mark_probs(species, v).items()}
        hm, alpha = self._head("marks", feats, list(marks)) if feats is not None else (None, 0)
        if hm is not None:
            for (k, zp), hp in zip(list(marks.items()), hm):
                if not np.isnan(hp):
                    marks[k] = (1 - alpha) * zp + alpha * float(hp)

        crop_ratio = 1.0 if det is None else det.area() / (img.width * img.height)
        weight = (det.confidence if det else 0.3) * (0.5 + min(crop_ratio, 0.5))
        return PhotoResult(det, crop_ratio, reid, vlm, probs, breed_keys, shares, marks, weight, extras)

    # ------------------------------------------------------------------ #
    def analyze(self, images: list[bytes | Image.Image | str], explain: bool = False) -> AnimalProfile:
        """Analyse 1..N photos of the SAME animal and return one profile.

        Thread-safe: FastAPI runs sync endpoints in a thread pool, and torch
        models on MPS/CUDA must not be driven by two threads at once (on Apple
        MPS concurrent calls can deadlock the server). Requests are serialised;
        each takes ~0.2 s so a queue is far cheaper than a hang.
        """
        with self._lock:
            return self._analyze(images, explain)

    def _analyze(self, images: list[bytes | Image.Image | str], explain: bool = False) -> AnimalProfile:
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
        sq_masks = [None if m is None else _square_mask(m) for m in masks]
        emb = self.enc.encode_images([_square(c) for c in crops], tta=self.cfg.tta, masks=sq_masks)
        t_enc = time.perf_counter()

        photos = [self._photo(im, d, emb["reid"][i], emb["vlm"][i], crops[i], masks[i]) for i, (im, d) in enumerate(zip(imgs, chosen))]
        profile = self._aggregate(photos)
        self._apply_gates(profile, imgs, dets, photos)
        t_end = time.perf_counter()
        profile.data["timing_ms"] = {
            "detect": round((t_det - t0) * 1000, 1),
            "embed": round((t_enc - t_det) * 1000, 1),
            "reason": round((t_end - t_enc) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
            "photos": len(imgs),
        }
        if explain:
            profile.data["explain"] = self._explain(imgs, dets, chosen, emb, photos, profile.data)
        return profile

    def _animal_scores(self, vlm: np.ndarray) -> tuple[float, float]:
        from .segmenter import ANIMAL_PROMPTS, NOT_ANIMAL_PROMPTS

        if self._gate_pos is None:
            self._gate_pos = self.enc.encode_texts(list(ANIMAL_PROMPTS))
            self._gate_neg = self.enc.encode_texts(list(NOT_ANIMAL_PROMPTS))
        scale = float(getattr(self.enc, "logit_scale", 1.0))
        bias = float(getattr(self.enc, "logit_bias", 0.0))

        def peak(text_emb: np.ndarray) -> float:
            logits = scale * (vlm @ text_emb.T) + bias
            return float(1 / (1 + np.exp(-logits.max())))

        return peak(self._gate_pos), peak(self._gate_neg)

    def _apply_gates(self, profile: AnimalProfile, imgs, dets, photos) -> None:
        from .segmenter import MULTI_MN, apply_no_animal, decide_no_animal

        data = profile.data
        if any(len(ds) >= 2 for ds in dets):
            warnings = data["quality"]["warnings"]
            if MULTI_MN not in warnings:
                warnings.insert(0, MULTI_MN)
            data["quality"]["multi_animal"] = True
        if self.cfg.fake_models:
            return
        i = max(range(len(photos)), key=lambda k: photos[k].weight)
        det = dets[i][0] if dets[i] else None
        if det is not None and det.mask_small is not None:
            fg = float(det.mask_small.mean())
        elif det is not None:
            fg = det.area() / max(imgs[i].width * imgs[i].height, 1)
        else:
            fg = 0.0
        animal, other = self._animal_scores(photos[i].vlm)
        if decide_no_animal(fg, animal, other, det is not None):
            apply_no_animal(data)

    # ------------------------------------------------------------------ #
    def _explain(self, imgs, dets, chosen, emb, photos, data) -> dict:
        """What the UI needs to show *how* the result was reached, for the most
        informative photo: detections, mask outline, where the identity model
        looked, zero-shot vs trained-head votes and the identity fingerprint."""
        i = max(range(len(photos)), key=lambda k: photos[k].weight)
        im, det = imgs[i], chosen[i]
        out: dict[str, Any] = {
            "photo_index": i,
            "image": {"w": im.width, "h": im.height},
            "detections": [{"box": [round(v, 1) for v in d.box], "species": d.species, "confidence": round(d.confidence, 3)} for d in dets[i][:6]],
            "chosen": next((j for j, d in enumerate(dets[i][:6]) if d is det), None),
            "polygon": None,
            "attention": None,
            "models": {"detector": getattr(self.detector, "name", self.cfg.detector_weights), "identity": self.cfg.reid_model, "vision_language": self.cfg.vlm_model,
                       "heads": sorted(self.heads.tasks) if self.heads else []},
        }
        if det is not None and det.polygon:
            step = max(1, len(det.polygon) // 90)
            out["polygon"] = [[round(x, 1), round(y, 1)] for x, y in det.polygon[::step]]
        attn = emb.get("attn")
        if attn is not None:
            a = np.asarray(attn[i], dtype=np.float32)
            g = int(round(len(a) ** 0.5))
            if g * g == len(a):
                a = _despeckle(a.reshape(g, g)).ravel()
                lo, hi = np.percentile(a, 5), np.percentile(a, 99)
                a = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)
                # DINOv2's processor keeps aspect ratio (short side → 256) and centre-crops
                # 224², i.e. a centred square of side short_side·224/256 of our crop
                x1, y1, x2, y2 = crop_box(im, det)
                cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, min(x2 - x1, y2 - y1) * float(emb.get("attn_frac", 1.0))
                out["attention"] = {"grid": g, "values": [round(float(v), 3) for v in a],
                                    "box": [round(cx - side / 2, 1), round(cy - side / 2, 1), round(cx + side / 2, 1), round(cy + side / 2, 1)]}
        # zero-shot vs trained head vs final, for the final top breeds
        b = data.get("breed")
        ex = photos[i].extras
        if b and "breed_zs" in ex and photos[i].breed_keys:
            keys = photos[i].breed_keys
            idx = {k: j for j, k in enumerate(keys)}
            hp = ex.get("breed_head")
            votes = []
            for t in b["top"][:5]:
                j = idx.get(t["key"])
                if j is None:
                    continue
                h = None if hp is None or np.isnan(hp[j]) else round(float(hp[j]), 4)
                votes.append({"key": t["key"], "mn": t["mn"], "zero_shot": round(float(ex["breed_zs"][j]), 4), "head": h, "final": t["confidence"]})
            out["breed_votes"] = votes
        # identity fingerprint: the re-ID embedding folded to 48 bins
        r = photos[i].reid
        bins = np.array_split(r, 48)
        f = np.array([float(c.mean()) for c in bins])
        out["fingerprint"] = [round(float(v), 3) for v in f / max(float(np.abs(f).max()), 1e-6)]
        return out

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
            "species": {**_label_dict(species_lab, sp_p.max()), "level": confidence_level(sp_p.max(), (0.9, 0.6))},
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
            first_is_mixed = breeds[order[0]].key in T.MIXED_KEYS
            # thresholds from the calibration set (T=1.5): top-1 ≥0.6 → ~93% correct,
            # 0.35-0.6 → top-1 71% / top-3 92% (show alternatives), <0.35 → unsure
            mixed = bool(first_is_mixed or p1 < 0.6 or (p1 - p2) < 0.15)
            looks_like = [t for t in top if t["key"] not in ("mixed_dog",) and t["confidence"] >= 0.05][:3 if mixed else 1] or top[:1]
            data["breed"] = {
                "top": top,
                "is_mixed": mixed,
                "level": confidence_level(p1, (0.6, 0.35)),
                "looks_like": looks_like,
                "probs": {b.key: round(float(p), 4) for b, p in zip(breeds, breed_p)},
            }
        else:
            data["breed"] = None

        for task, labels in (("size", T.SIZES), ("age", T.AGES), ("pattern", T.PATTERNS), ("coat", T.COAT_LENGTH)):
            p = avg(task)
            i = int(np.argmax(p))
            data[task] = {**_label_dict(labels[i], p[i]), "level": confidence_level(p[i], (0.6, 0.35)),
                          "probs": {l.key: round(float(v), 4) for l, v in zip(labels, p)}}

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
            data.setdefault("mark_probs", {})[m] = 0.99
    if labels.get("marks"):
        owner = set(labels.get("marks") or [])
        data["marks"] = [_label_dict(T.MARK_BY_KEY[k], v) for k, v in data["mark_probs"].items()
                         if k in owner or v >= T.MARK_BY_KEY[k].threshold]
    data["category"] = PettiVision.category(data)
    data["description_mn"], data["description_en"] = describe(data)
    return data
