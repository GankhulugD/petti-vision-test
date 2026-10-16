"""Foreground segmenter on frozen DINOv2-small patch tokens.

The image is resized to 224×224 (patch size 14 → a 16×16 grid). 448×448 scores a
higher mask IoU but costs about 55 ms on an M2, which is slower than YOLO.
224×224 stays under YOLO's detect time. A tiny conv
head, trained in training/train_seg.py and stored as numpy in seg_head.npz,
predicts a pet mask. Connected components become instances. Species is left
to SigLIP: these boxes are labeled "pet", not dog or cat.

YOLO stays available only when PETTI_DETECTOR_BACKEND=yolo (AGPL, dev only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .backbones import Detection

log = logging.getLogger("petti.segmenter")

GRID = 16
SIDE = 224  # 16 * 14. 448 (32×32) was measured at ~55 ms vs YOLO ~25 ms.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NO_ANIMAL_MN = "Энэ зурагт нохой эсвэл муур харагдахгүй байна. Амьтныхаа зургийг дахин оруулна уу."
NO_ANIMAL_EN = "No dog or cat in this photo. Please upload a photo of the animal."
MULTI_MN = "Зурагт хэд хэдэн амьтан байна. Хамгийн томыг нь сонголоо."

# SigLIP prompts for the no-animal gate. Not part of the taxonomy.
ANIMAL_PROMPTS = (
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a puppy",
    "a photo of a kitten",
)
NOT_ANIMAL_PROMPTS = (
    "a photo of a car",
    "a photo of a person",
    "a photo of furniture",
    "a photo of a toy",
    "a drawing of an object",
)


def decide_no_animal(fg_frac: float, animal: float, other: float, found: bool) -> bool:
    """True when the photo should not be labeled as a pet.

    A real animal that the segmenter missed still passes if SigLIP prefers
    the animal prompts. A car passes the gate only when the non-animal
    prompts win and the foreground is small.
    """
    if found and fg_frac >= 0.04 and animal >= other:
        return False
    if other > animal + 0.05 and fg_frac < 0.08:
        return True
    if not found and other >= animal:
        return True
    return False


def apply_no_animal(data: dict) -> dict:
    """Keep the response shape, replace the answer with the Mongolian message."""
    data = data
    q = data.setdefault("quality", {"warnings": [], "photos": 1})
    q["no_animal"] = True
    warnings = [NO_ANIMAL_MN] + [w for w in q.get("warnings", []) if w != NO_ANIMAL_MN]
    q["warnings"] = warnings
    data["description_mn"] = NO_ANIMAL_MN
    data["description_en"] = NO_ANIMAL_EN
    data["breed"] = None
    return data


def label_components(mask: np.ndarray) -> np.ndarray:
    """4-connected labels on a small boolean grid. Returns int32, 0 = background."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        left = labels[y, x - 1] if x else 0
        up = labels[y - 1, x] if y else 0
        if left and up and left != up:
            a, b = find(left), find(up)
            parent[b] = a
            labels[y, x] = a
        elif left or up:
            labels[y, x] = left or up
        else:
            n += 1
            parent.append(n)
            labels[y, x] = n
    if n == 0:
        return labels
    for y, x in zip(ys.tolist(), xs.tolist()):
        labels[y, x] = find(int(labels[y, x]))
    # compact ids
    uniq = {lab: i + 1 for i, lab in enumerate(sorted(set(labels.ravel()) - {0}))}
    out = np.zeros_like(labels)
    for lab, i in uniq.items():
        out[labels == lab] = i
    return out


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (H, W, C), w (Cout, C, 3, 3), b (Cout,) → (H, W, Cout)."""
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(0, 1))
    return np.einsum("hwcij,ocij->hwo", windows, w, optimize=True) + b


def foreground_prob(tokens: np.ndarray, head: dict) -> np.ndarray:
    """tokens (GRID*GRID, 384) → (GRID, GRID) probability."""
    x = tokens.reshape(GRID, GRID, -1).astype(np.float32)
    h = np.maximum(x @ head["w1"].T + head["b1"], 0)  # 1×1, ReLU
    logit = _conv3x3(h, head["w2"], head["b2"])[..., 0]
    return 1 / (1 + np.exp(-logit))


def load_head(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fill one-cell gaps so a single animal is not split into body and head."""
    padded = np.pad(mask, radius)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def instances_from_prob(prob: np.ndarray, size: tuple[int, int], thr: float = 0.5, min_frac: float = 0.008) -> list[Detection]:
    """prob is GRID×GRID. size is (width, height) of the original image.

    Connectivity uses a 1-cell closing, so a thin gap does not become a second
    animal. A leftover piece smaller than about a third of the main animal is
    dropped instead of being described on its own.
    """
    binary = prob >= thr
    labels = label_components(_close(binary, 1))
    g = prob.shape[0]
    w, h = size
    dets = []
    for lab in range(1, int(labels.max()) + 1):
        region = labels == lab
        sel = region & binary
        if float(sel.mean()) < min_frac or not sel.any():
            continue
        # box the real foreground; the closed map only groups the pieces
        ys, xs = np.nonzero(sel)
        x1 = float(xs.min()) / g * w
        x2 = float(xs.max() + 1) / g * w
        y1 = float(ys.min()) / g * h
        y2 = float(ys.max() + 1) / g * h
        conf = float(prob[sel].mean()) if sel.any() else 0.0
        dets.append(Detection(
            box=(x1, y1, x2, y2),
            species="pet",
            confidence=conf,
            polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            mask_small=sel,
        ))
    if dets:
        biggest = max(d.area() for d in dets)
        dets = [d for d in dets if d.area() >= 0.35 * biggest]
    dets.sort(key=lambda d: d.confidence * (d.area() ** 0.5), reverse=True)
    return dets


class DinoSegmenter:
    name = "dinov2-small-seg"

    def __init__(self, cfg, encoders):
        self.cfg = cfg
        self.enc = encoders
        path = cfg.seg_head_path
        if not path.exists():
            raise FileNotFoundError(f"segmentation head not found: {path}")
        self.head = load_head(path)
        self.thr = float(self.head.get("threshold", np.array(cfg.detector_conf)))
        log.info("segmenter: %s threshold %.2f", path.name, self.thr)

    def _tokens(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.enc.torch
        batch = []
        for im in images:
            arr = np.asarray(im.resize((SIDE, SIDE), Image.BICUBIC), dtype=np.float32) / 255.0
            batch.append((arr - MEAN) / STD)
        pixel = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
        pixel = pixel.to(self.enc.device, dtype=self.enc.dtype)
        with torch.inference_mode():
            out = self.enc.reid(pixel_values=pixel)
            hs = out.last_hidden_state
            n_reg = int(getattr(self.enc.reid.config, "num_register_tokens", 0) or 0)
            patches = hs[:, 1 + n_reg :]
        tokens = patches.float().cpu().numpy()
        if tokens.shape[1] != GRID * GRID:
            raise RuntimeError(f"expected {GRID * GRID} patches at {SIDE}px, got {tokens.shape[1]}")
        return tokens

    def detect(self, images: list[Image.Image]) -> list[list[Detection]]:
        if not images:
            return []
        tokens = self._tokens(images)
        out = []
        for i, im in enumerate(images):
            prob = foreground_prob(tokens[i], self.head)
            out.append(instances_from_prob(prob, im.size, thr=self.thr))
        return out


# =========================================================================== #
# v2: aspect-preserving input, deeper head, sub-patch masks and real outlines
# =========================================================================== #
PATCH = 14
UPS = 4  # mask resolution = 4 × patch grid


def input_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Keep the aspect ratio; both sides a multiple of 14, at least 4 patches."""
    s = long_side / max(w, h)
    tw = max(4 * PATCH, int(round(w * s / PATCH)) * PATCH)
    th = max(4 * PATCH, int(round(h * s / PATCH)) * PATCH)
    return tw, th


def head_forward(tokens: np.ndarray, gh: int, gw: int, head: dict) -> np.ndarray:
    """Generic numpy conv stack stored as l{i}_w (Cout,Cin,k,k), l{i}_b. ReLU between layers."""
    x = tokens.reshape(gh, gw, -1).astype(np.float32)
    n = int(head["n_layers"])
    for i in range(n):
        w, b = head[f"l{i}_w"].astype(np.float32), head[f"l{i}_b"].astype(np.float32)
        if w.shape[2] == 1:
            x = x @ w[:, :, 0, 0].T + b
        else:
            x = _conv3x3(x, w, b)
        if i < n - 1:
            x = np.maximum(x, 0)
    return x[..., 0]


def _upsample(prob: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(prob.astype(np.float32), mode="F").resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def trace_outline(mask: np.ndarray, max_points: int = 90) -> list[tuple[int, int]]:
    """Outer boundary of the largest blob (Moore neighbour tracing), as (x, y) points."""
    h, w = mask.shape
    if not mask.any():
        return []
    m = np.pad(mask, 1)
    ys, xs = np.nonzero(m)
    start = (int(ys[0]), int(xs[0]))  # top-most, then left-most foreground pixel
    nbrs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pts = [start]
    cur, back = start, 6  # we "came from" the west
    for _ in range(4 * (h + w) * 8):
        found = False
        for k in range(8):
            d = (back + 1 + k) % 8
            ny, nx = cur[0] + nbrs[d][0], cur[1] + nbrs[d][1]
            if m[ny, nx]:
                back = (d + 4) % 8
                cur = (ny, nx)
                found = True
                break
        if not found or cur == start:
            break
        pts.append(cur)
    step = max(1, -(-len(pts) // max_points))
    return [(x - 1, y - 1) for y, x in pts[::step]]


def instances_v2(prob: np.ndarray, size: tuple[int, int], thr: float, min_frac: float = 0.004) -> list[Detection]:
    """prob: (gh, gw) grid probability. Components on the grid, masks/boxes/outlines at 4×."""
    gh, gw = prob.shape
    w, h = size
    binary = prob >= thr
    labels = label_components(_close(binary, 1))
    fine = _upsample(prob, (gh * UPS, gw * UPS))
    sx, sy = w / (gw * UPS), h / (gh * UPS)
    dets = []
    for lab in range(1, int(labels.max()) + 1):
        region = labels == lab
        if float((region & binary).mean()) < min_frac:
            continue
        # this component's area at fine resolution (dilate by one cell for the edges)
        grow = np.asarray(Image.fromarray((_close(region, 1) * 255).astype(np.uint8)).resize((gw * UPS, gh * UPS), Image.NEAREST)) > 0
        mk = grow & (fine >= thr)
        if not mk.any():
            continue
        ys, xs = np.nonzero(mk)
        box = (float(xs.min()) * sx, float(ys.min()) * sy, float(xs.max() + 1) * sx, float(ys.max() + 1) * sy)
        poly = [(x * sx, y * sy) for x, y in trace_outline(mk)]
        dets.append(Detection(box=box, species="pet", confidence=float(fine[mk].mean()), polygon=poly or None, mask_small=mk))
    if dets:
        biggest = max(d.area() for d in dets)
        dets = [d for d in dets if d.area() >= 0.15 * biggest]
    dets.sort(key=lambda d: d.confidence * (d.area() ** 0.5), reverse=True)
    return dets


class DinoSegmenterV2:
    """Frozen DINOv2-small patch tokens (aspect kept, long side 448) → small conv head."""

    name = "dinov2-small-seg2"

    def __init__(self, cfg, encoders, path: Path | None = None):
        self.cfg = cfg
        self.enc = encoders
        path = path or cfg.seg_head2_path
        if not path.exists():
            raise FileNotFoundError(f"segmentation head not found: {path}")
        self.head = load_head(path)
        self.thr = float(self.head.get("threshold", np.array(0.5)))
        self.long_side = int(self.head.get("long_side", np.array(448)))
        log.info("segmenter v2: %s threshold %.2f long side %d", path.name, self.thr, self.long_side)

    def _tokens(self, im: Image.Image) -> tuple[np.ndarray, int, int]:
        tw, th = input_size(im.width, im.height, self.long_side)
        arr = np.asarray(im.convert("RGB").resize((tw, th), Image.BICUBIC), dtype=np.float32) / 255.0
        pixel = ((arr - MEAN) / STD).transpose(2, 0, 1)[None]
        hs = self.enc.dino_hidden(pixel)  # torch or ONNX, same numbers
        return hs[0, 1:], th // PATCH, tw // PATCH

    def probs(self, im: Image.Image) -> np.ndarray:
        tok, gh, gw = self._tokens(im)
        return 1 / (1 + np.exp(-head_forward(tok, gh, gw, self.head)))

    def detect(self, images: list[Image.Image]) -> list[list[Detection]]:
        return [instances_v2(self.probs(im), im.size, self.thr) for im in images]
