"""Our own trainable models that sit on top of the frozen backbones.

* `AttributeHeads` – small MLPs (species / breed / size / age / colour /
  pattern / coat / marks) trained on cached [SigLIP ‖ DINOv2] features.
  Trained by `training/train_heads.py`, continuously re-trainable as owners
  register pets with verified labels.
* `ReidProjection` – metric-learning projection on DINOv2 features, trained by
  `training/train_reid.py` on pet identities (each registered pet = 5+ photos
  of one identity → perfect supervision for re-identification).

Weights are exported to .npz so inference needs only numpy (≈0.1 ms per head).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _MLP:
    def __init__(self, w: dict[str, np.ndarray], prefix: str):
        self.layers = []
        i = 0
        while f"{prefix}.{i}.weight" in w:
            self.layers.append((w[f"{prefix}.{i}.weight"], w[f"{prefix}.{i}.bias"]))
            i += 1
        # optional input normalisation (LayerNorm without affine is folded in training)
        self.mu = w.get(f"{prefix}.mu")
        self.sd = w.get(f"{prefix}.sd")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mu is not None:
            x = (x - self.mu) / self.sd
        for j, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if j < len(self.layers) - 1:
                x = gelu(x)
        return x


class AttributeHeads:
    """Loads artifacts/heads.npz + heads.json written by train_heads.py."""

    def __init__(self, path: Path):
        self.path = Path(path)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        w = dict(np.load(self.path))
        self.meta = meta
        self.feature = meta.get("feature", "concat")  # vlm | reid | concat
        self.tasks: dict[str, dict] = {}
        for task, info in meta["tasks"].items():
            self.tasks[task] = {
                "mlp": _MLP(w, task),
                "labels": info["labels"],
                "multi": info.get("multi_label", False),
                # fusion weight with zero-shot, chosen on the validation split
                "alpha": float(info.get("alpha", 0.5)),
                "temperature": float(info.get("temperature", 1.0)),
                "val_acc": info.get("val_acc"),
            }

    def features(self, vlm: np.ndarray, reid: np.ndarray) -> np.ndarray:
        if self.feature == "vlm":
            return vlm
        if self.feature == "reid":
            return reid
        return np.concatenate([vlm, reid], axis=-1)

    def has(self, task: str) -> bool:
        return task in self.tasks

    def predict(self, task: str, feats: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Returns (label_keys, probabilities (N,C))."""
        t = self.tasks[task]
        logits = t["mlp"](feats) / t["temperature"]
        if t["multi"]:
            return t["labels"], sigmoid(logits)
        logits = logits - logits.max(-1, keepdims=True)
        e = np.exp(logits)
        return t["labels"], e / e.sum(-1, keepdims=True)


class ReidProjection:
    def __init__(self, path: Path):
        w = dict(np.load(path))
        self.mlp = _MLP(w, "proj")
        self.skip = w.get("proj.skip")  # residual linear path (see train_reid.Proj)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.mlp(x)
        if self.skip is not None:
            out = out + x @ self.skip.T
        return out


def load_reid_projection(path: Path) -> ReidProjection:
    return ReidProjection(path)


def load_heads(path: Path) -> AttributeHeads | None:
    path = Path(path)
    if path.exists() and path.with_suffix(".json").exists():
        return AttributeHeads(path)
    return None
