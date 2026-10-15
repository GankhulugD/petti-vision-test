"""Deterministic coat-colour estimator (numpy only, ~1 ms).

Vision-language models know *names* of colours but are biased by lighting and
background. This estimator measures actual pixel colours inside the animal
crop (centre-weighted, background-suppressed), in CIE-Lab space, and returns
the share of each palette colour. The pipeline fuses both signals.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .taxonomy import COLOR_RGB, COLORS

_KEYS = [c.key for c in COLORS]


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


_PALETTE_LAB = _srgb_to_lab(np.array([COLOR_RGB[k] for k in _KEYS], dtype=np.float32))


def coat_colors(crop: Image.Image, size: int = 64, mask: np.ndarray | None = None) -> dict[str, float]:
    """mask: optional bool array (crop H×W) from the segmentation model."""
    img = np.asarray(crop.resize((size, size), Image.BILINEAR), dtype=np.float32)
    lab = _srgb_to_lab(img)

    weight = None
    if mask is not None:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        if m.mean() > 0.05:  # trust the segmentation
            weight = (m > 0.5).astype(np.float32)
            # ignore the darkest/brightest 3% (eyes, nose, specular highlights)
            L = lab[..., 0][weight > 0]
            lo, hi = np.percentile(L, [3, 97])
            weight *= (lab[..., 0] >= lo) & (lab[..., 0] <= hi)
    if weight is None:
        # centre-weighted elliptical mask (animal is usually centred in its box)
        yy, xx = np.mgrid[0:size, 0:size]
        r = ((xx - size / 2) / (size * 0.42)) ** 2 + ((yy - size / 2) / (size * 0.46)) ** 2
        weight = np.clip(1.2 - r, 0, 1)
        # suppress pixels that look like the border (= background)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.median(border, axis=0)
        dist_bg = np.linalg.norm(lab - bg, axis=-1)
        weight = weight * np.clip(dist_bg / 25.0, 0.15, 1.0)

    # assign each pixel to nearest palette colour (lightness weighted more)
    d = lab[..., None, :] - _PALETTE_LAB[None, None]
    d = d * np.array([1.4, 1.0, 1.0])
    idx = np.argmin(np.linalg.norm(d, axis=-1), axis=-1)

    hist = np.bincount(idx.ravel(), weights=weight.ravel(), minlength=len(_KEYS))
    hist = hist / max(hist.sum(), 1e-6)
    return {k: float(v) for k, v in zip(_KEYS, hist)}
