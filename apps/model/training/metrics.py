"""Pure metric helpers for the frozen eval. No model imports."""

from __future__ import annotations

import numpy as np

# Adult size prior used by the browser bench (bench.js). A photo of a puppy
# of a "large" breed is still labeled large — same protocol as v0.4.
ADULT_SIZE = {
    "labrador_retriever": "large", "siberian_husky": "medium", "alaskan_malamute": "large",
    "samoyed": "medium", "golden_retriever": "large", "rottweiler": "large", "doberman": "large",
    "pug": "small", "chihuahua": "toy", "beagle": "small", "toy_poodle": "toy", "pomeranian": "toy",
    "shih_tzu": "toy", "welsh_corgi": "small", "dalmatian": "large", "chow_chow": "medium",
    "shiba_inu": "small", "french_bulldog": "small", "boxer": "large", "dachshund": "small",
    "border_collie": "medium", "maltese": "toy", "tibetan_mastiff": "giant", "saint_bernard": "giant",
    "yorkshire_terrier": "toy", "akita": "large", "papillon": "toy", "german_shepherd": "large",
    "cocker_spaniel": "small", "belgian_malinois": "large", "pekingese": "toy",
    "miniature_schnauzer": "small", "weimaraner": "large",
}
SIZE_ORDER = ["toy", "small", "medium", "large", "giant"]
AGE_ORDER = ["baby", "young", "adult", "senior"]


def safe_div(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return float(n) / float(d)


def round3(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


def accuracy(correct: int, n: int) -> float | None:
    return round3(safe_div(correct, n))


def topk_hit(ranked_keys: list[str], truth: str, k: int) -> bool:
    return truth in ranked_keys[:k]


def within_bins(pred: str, truth: str, order: list[str], tol: int) -> bool | None:
    if pred not in order or truth not in order:
        return None
    return abs(order.index(pred) - order.index(truth)) <= tol


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float | None:
    """ECE of a single-label head: confidence is P(predicted class), correct is 0/1."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confidence) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        hi = edges[i + 1]
        m = (confidence >= edges[i]) & (confidence < hi if i < bins - 1 else confidence <= hi)
        if not m.any():
            continue
        ece += float(m.mean()) * abs(float(correct[m].mean()) - float(confidence[m].mean()))
    return round3(ece)


def binary_f1(pred: set[str], truth: set[str]) -> float | None:
    if not truth and not pred:
        return None
    tp = len(pred & truth)
    prec = safe_div(tp, len(pred)) if pred else 0.0
    rec = safe_div(tp, len(truth)) if truth else 0.0
    if prec is None or rec is None or (prec + rec) == 0:
        return 0.0 if (pred or truth) else None
    return round3(2 * prec * rec / (prec + rec))


def mean_f1(pairs: list[tuple[set[str], set[str]]]) -> float | None:
    scores = [binary_f1(p, t) for p, t in pairs]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    return round3(sum(scores) / len(scores))


def retrieval(emb: np.ndarray, ids: np.ndarray, distractor: np.ndarray) -> dict:
    """Leave-one-out identity retrieval. Distractors are gallery-only."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be (n, d), got {emb.shape}")
    ids = np.asarray(ids)
    distractor = np.asarray(distractor, dtype=bool)
    if len(emb) < 2:
        return {"queries": 0, "R@1": None, "R@5": None, "mAP": None, "false_match_rate": None}
    sim = emb @ emb.T
    np.fill_diagonal(sim, -9.0)
    ranks, aps, false_top = [], [], []
    for q in range(len(emb)):
        if distractor[q]:
            continue
        positive = (ids == ids[q]) & ~distractor
        positive[q] = False
        if not positive.any():
            continue
        order = np.argsort(-sim[q])
        rel = positive[order]
        hit = int(np.argmax(rel)) + 1  # rank of the best positive (1-based)
        ranks.append(hit)
        # average precision over every other photo of this identity
        hits = 0
        sum_prec = 0.0
        npos = int(rel.sum())
        for i, y in enumerate(rel, start=1):
            if y:
                hits += 1
                sum_prec += hits / i
        aps.append(sum_prec / npos)
        top_is_distractor = bool(distractor[order[0]])
        false_top.append(top_is_distractor)
    ranks_a = np.asarray(ranks, dtype=np.float64)
    n = len(ranks)
    return {
        "queries": n,
        "gallery": int(len(emb)),
        "R@1": round3(float((ranks_a <= 1).mean())) if n else None,
        "R@5": round3(float((ranks_a <= 5).mean())) if n else None,
        "mAP": round3(float(np.mean(aps))) if aps else None,
        "false_match_rate": round3(float(np.mean(false_top))) if false_top else None,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = np.sort(np.asarray(values, dtype=np.float64))
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
    return round(float(xs[idx]), 1)
