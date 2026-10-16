"""Zero-shot attribute classifier on top of the SigLIP2 text tower.

* Prompt ensembling + species-conditioned prompts ("a photo of a black cat").
* Age uses the words people actually caption photos with (puppy / kitten /
  old dog with a grey muzzle) instead of abstract classes.
* Distinctive marks are scored against a NEUTRAL caption ("a photo of a dog"),
  never against a negated caption: vision-language models do not understand
  "without a scar", so pos-vs-negation always favours the positive caption.

Text embeddings are cached in artifacts/ (keyed by model + prompt version).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from .backbones import l2n
from .taxonomy import AGES, BREEDS, COAT_LENGTH, COLORS, MARKS, PATTERNS, SIZES, SPECIES, Label

log = logging.getLogger("petti.zeroshot")

PROMPT_VERSION = "v3"  # v3: Bankhar vs spitz prompts

SPECIES_WORD = {"dog": "dog", "cat": "cat", "bird": "bird", "rabbit": "rabbit", "rodent": "small pet", "horse": "horse", "other": "animal"}
CONDITIONED = ("dog", "cat", "other")  # prompt sets we pre-compute

TEMPLATES = {
    "species": ["a photo of a {l}.", "a photo of a {l} outdoors.", "a close-up photo of a {l}.", "a phone photo of a {l} on the street."],
    "breed": ["a photo of a {l}, a type of {s}.", "a photo of a {l}.", "a phone photo of a {l} outdoors."],
    "size": ["a photo of a {l} {s}."],
    "color": ["a photo of a {l} {s}.", "a {s} with {l} fur."],
    "pattern": ["a photo of a {s} with a {l}."],
    "coat": ["a photo of a {l} {s}.", "a close-up photo of a {l} {s}."],
    "age": ["a photo of a {l}.", "a phone photo of a {l} outdoors."],
    "marks": ["a photo of a {s} {l}.", "a close-up photo of a {s} {l}."],
    "neutral": ["a photo of a {s}.", "a close-up photo of a {s}."],
}

AGE_WORDS = {
    "dog": {"baby": ["tiny puppy", "young puppy"], "young": ["young dog", "adolescent dog"], "adult": ["adult dog", "grown dog"],
            "senior": ["old dog with a grey muzzle", "elderly senior dog"]},
    "cat": {"baby": ["tiny kitten", "young kitten"], "young": ["young cat", "adolescent cat"], "adult": ["adult cat", "grown cat"],
            "senior": ["old senior cat", "elderly cat"]},
    "other": {"baby": ["baby animal"], "young": ["young animal"], "adult": ["adult animal"], "senior": ["old animal"]},
}


def _phrases(lab: Label) -> list[str]:
    return [lab.en, *lab.alt]


def _fill(task: str, lab: Label, species: str) -> list[str]:
    s = SPECIES_WORD[species]
    return [t.format(l=p, s=s) for t in TEMPLATES[task] for p in _phrases(lab)]


class ZeroShot:
    """Holds one averaged text embedding per (task, species, label)."""

    def __init__(self, encoders, cache_path=None):
        self.enc = encoders
        self.scale = float(getattr(encoders, "logit_scale", 100.0))
        self.bias = float(getattr(encoders, "logit_bias", 0.0))
        self.bank: dict[str, np.ndarray] = {}
        sig = self._signature()
        if cache_path is not None and cache_path.exists():
            try:
                data = np.load(cache_path)
                if str(data["__sig__"]) == sig:
                    self.bank = {k: data[k] for k in data.files if k != "__sig__"}
                    log.info("loaded %d cached text embeddings", len(self.bank))
            except Exception as exc:  # corrupted / old cache
                log.warning("text cache unusable (%s), rebuilding", exc)
        if not self._complete():
            self._build()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, __sig__=np.array(sig), **self.bank)

    # ------------------------------------------------------------------ #
    def _jobs(self):
        yield "species|any", [(lab.key, [t.format(l=p) for t in TEMPLATES["species"] for p in _phrases(lab)]) for lab in SPECIES]
        for sp in ("dog", "cat"):
            yield f"breed|{sp}", [(b.key, _fill("breed", b, sp)) for b in BREEDS if b.species == sp]
        for sp in CONDITIONED:
            for task, labels in (("size", SIZES), ("color", COLORS), ("pattern", PATTERNS), ("coat", COAT_LENGTH)):
                yield f"{task}|{sp}", [(lab.key, _fill(task, lab, sp)) for lab in labels]
            yield f"age|{sp}", [(a.key, [t.format(l=w) for t in TEMPLATES["age"] for w in AGE_WORDS[sp][a.key]]) for a in AGES]
            s = SPECIES_WORD[sp]
            yield f"marks|{sp}", [(m.key, [t.format(l=m.en, s=s) for t in TEMPLATES["marks"]]) for m in MARKS]
            yield f"neutral|{sp}", [("neutral", [t.format(s=s) for t in TEMPLATES["neutral"]])]

    def _signature(self) -> str:
        txt = PROMPT_VERSION + repr([(n, items) for n, items in self._jobs()])
        return hashlib.md5(txt.encode()).hexdigest()[:12]

    def _complete(self) -> bool:
        return bool(self.bank) and all(k in self.bank for k, _ in self._jobs())

    def _build(self):
        jobs = list(self._jobs())
        flat: list[str] = []
        spans = []
        for name, items in jobs:
            for key, prompts in items:
                spans.append((name, key, len(flat), len(flat) + len(prompts)))
                flat.extend(prompts)
        log.info("encoding %d prompts for zero-shot bank", len(flat))
        emb = self.enc.encode_texts(flat)
        grouped: dict[str, list[np.ndarray]] = {}
        for name, _key, a, b in spans:
            grouped.setdefault(name, []).append(l2n(emb[a:b].mean(0)))
        self.bank = {k: np.stack(v) for k, v in grouped.items()}

    # ------------------------------------------------------------------ #
    def cosines(self, task_key: str, img: np.ndarray) -> np.ndarray:
        return img @ self.bank[task_key].T

    def logits(self, task_key: str, img: np.ndarray) -> np.ndarray:
        """img: (N,D) normalised → (N,C) logits."""
        return self.scale * self.cosines(task_key, img)

    def probs(self, task_key: str, img: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.logits(task_key, img) / temperature)

    def mark_probs(self, species: str, img: np.ndarray) -> dict[str, np.ndarray]:
        """P(mark) = sigmoid(scale·(cos(mark caption) − cos(neutral caption)) − margin).

        Positive only when the specific caption fits the image *better* than a
        plain "a photo of a dog" – robust and needs no negated prompts.
        """
        sp = species if species in ("dog", "cat") else "other"
        pos = self.cosines(f"marks|{sp}", img)  # (N, M)
        neu = self.cosines(f"neutral|{sp}", img)[:, :1]  # (N, 1)
        delta = self.scale * (pos - neu)
        p = 1.0 / (1.0 + np.exp(-(delta - MARK_MARGIN)))
        out = {}
        for j, m in enumerate(MARKS):
            if sp == "other" or species in m.species:
                out[m.key] = p[:, j]
        return out


# a mark caption must beat the neutral caption by this many logits to reach 50%
MARK_MARGIN = 1.0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
