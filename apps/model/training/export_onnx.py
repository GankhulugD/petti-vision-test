#!/usr/bin/env python3
"""Export DINOv2 + SigLIP2 vision towers to ONNX, quantize to INT8, verify.

    python3 training/export_onnx.py            # → artifacts/onnx/, report in artifacts/reports/onnx_export.json
    python3 training/export_onnx.py --no-int8  # keep fp32 only

After this, the server can run with PETTI_RUNTIME=onnx: no torch, no
transformers, ~4× smaller memory, CPU-only hosting. The script refuses to
finish if the ONNX outputs drift from PyTorch (cosine < 0.99 for INT8,
0.9999 for fp32) on real photos.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT

OUT = ROOT / "artifacts" / "onnx"


def ensure(pkgs: list[str]):
    import importlib.util

    missing = [p for p in pkgs if importlib.util.find_spec(p.split("==")[0].replace("-", "_")) is None]
    if missing:
        print(f"pip install {' '.join(missing)}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def sample_images(n: int) -> list[Path]:
    ps = sorted((ROOT / "data" / "eval" / "dogceo").rglob("*.jpg"))[:: max(1, 1)]
    ox = ROOT / "data" / "public" / "oxford"
    if (ox / "annotations" / "test.txt").exists():
        names = (ox / "annotations" / "test.txt").read_text().split()[0::4][::60]
        ps += [ox / "images" / f"{k}.jpg" for k in names]
    import random

    random.Random(0).shuffle(ps)
    return [p for p in ps if p.exists()][:n]


def cos(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--check", type=int, default=40)
    a = ap.parse_args()
    ensure(["onnx", "onnxruntime"])

    import torch
    import torch.nn as nn
    from petti_vision import Config
    from petti_vision.backbones import HFEncoders, _features
    from petti_vision.segmenter import ANIMAL_PROMPTS, NOT_ANIMAL_PROMPTS

    cfg = Config(device="cpu", half=False)
    enc = HFEncoders(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    class DinoCore(nn.Module):
        """patch-embed + CLS + (precomputed, interpolated) position table + encoder + final norm."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel, pos):
            e = self.m.embeddings
            x = e.patch_embeddings(pixel)
            x = torch.cat([e.cls_token.expand(x.shape[0], -1, -1), x], 1) + pos
            x = self.m.encoder(x)
            x = x.last_hidden_state if hasattr(x, "last_hidden_state") else x[0]
            return self.m.layernorm(x)

    class SigVision(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel):
            return _features(self.m.get_image_features(pixel_values=pixel))

    dino = DinoCore(enc.reid).eval()
    sig = SigVision(enc.vlm).eval()
    if int(getattr(enc.reid.config, "num_register_tokens", 0) or 0):
        sys.exit("DINOv2 with register tokens is not supported by the ONNX core yet")

    base_pos = enc.reid.embeddings.position_embeddings.detach().float().numpy()
    from petti_vision.onnx_runtime import interpolate_pos

    # 1) the numpy position interpolation must equal what this transformers version does
    for H, W in ((224, 224), (448, 336), (336, 448)):
        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            ref = enc.reid(pixel_values=x).last_hidden_state.numpy()
            mine = dino(x, torch.from_numpy(interpolate_pos(base_pos, H // 14, W // 14))).numpy()
        d = float(np.abs(ref - mine).max())
        print(f"pos-interp check {H}x{W}: max abs diff {d:.2e}", flush=True)
        if d > 1e-3:
            sys.exit("position interpolation differs from transformers — aborting")

    # 2) export fp32
    t0 = time.time()
    kw = {"opset_version": 17, "do_constant_folding": True}
    x = torch.randn(2, 3, 224, 224)
    pos = torch.from_numpy(interpolate_pos(base_pos, 16, 16))

    def export(model, args, path, names, dyn):
        try:
            torch.onnx.export(model, args, str(path), input_names=names[0], output_names=names[1], dynamic_axes=dyn, dynamo=False, **kw)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(model, args, str(path), input_names=names[0], output_names=names[1], dynamic_axes=dyn, **kw)

    export(dino, (x, pos), OUT / "dinov2_core_fp32.onnx", (["pixel", "pos"], ["hidden"]),
           {"pixel": {0: "b", 2: "h", 3: "w"}, "pos": {1: "n"}, "hidden": {0: "b", 1: "n"}})
    export(sig, (x,), OUT / "siglip_vision_fp32.onnx", (["pixel"], ["emb"]), {"pixel": {0: "b"}, "emb": {0: "b"}})
    print(f"exported fp32 in {time.time() - t0:.0f}s", flush=True)

    # 3) runtime text embeddings + constants
    texts = sorted({t.lower() for t in (*ANIMAL_PROMPTS, *NOT_ANIMAL_PROMPTS)})
    np.savez(OUT / "meta.npz", dino_pos=base_pos, logit_scale=np.float32(enc.logit_scale), logit_bias=np.float32(enc.logit_bias),
             texts=np.array(texts), text_emb=enc.encode_texts(texts).astype(np.float32))

    # 4) INT8 dynamic quantisation, chosen per model by measured drift.
    #    Naive all-op INT8 wrecks SigLIP (cos 0.60): its patch-embedding conv and
    #    attention-pooling head carry outliers. Quantising only the transformer
    #    MatMuls keeps most of the size/speed gain; fp32 is the safe fallback.
    import onnxruntime as ort

    probe = [Image.open(p).convert("RGB") for p in sample_images(12)]
    from petti_vision.onnx_runtime import dino_preprocess, siglip_preprocess

    px_sig = np.stack([siglip_preprocess(im) for im in probe])
    px_dino = np.stack([dino_preprocess(im) for im in probe])
    pos16 = interpolate_pos(base_pos, 16, 16)

    def run(path, name):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return sess.run(None, {"pixel": px_sig} if name == "siglip_vision" else {"pixel": px_dino, "pos": pos16})[0]

    def score(name, path):
        ref = run(OUT / f"{name}_fp32.onnx", name)
        got = run(path, name)
        if name == "dinov2_core":
            return float(min(cos(ref[:, 0], got[:, 0]).min(), cos(ref[:, 1:], got[:, 1:]).mean()))
        return float(cos(ref, got).min())

    chosen = {}
    for name in ("dinov2_core", "siglip_vision"):
        src = OUT / f"{name}_fp32.onnx"
        pick, best = "fp32", None
        if not a.no_int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            g = __import__("onnx").load(str(src), load_external_data=False)
            head_nodes = [n.name for n in g.graph.node if "head" in n.name.lower() or "pooler" in n.name.lower()]
            variants = {
                "int8_all": dict(),
                "int8_matmul": dict(op_types_to_quantize=["MatMul"], nodes_to_exclude=head_nodes),
            }
            for vname, kw in variants.items():
                dst = OUT / f"{name}_{vname}.onnx"
                quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True, **kw)
                c = score(name, dst)
                print(f"{name} {vname}: cosine vs fp32 {c:.5f}", flush=True)
                if c >= 0.995 and (best is None or c > best[1]):
                    best = (vname, c)
            if best:
                pick = best[0]
        final = OUT / f"{name}.onnx"
        final.write_bytes((OUT / (f"{name}_{pick}.onnx" if pick != "fp32" else f"{name}_fp32.onnx")).read_bytes())
        chosen[name] = pick
        print(f"{name}: serving {pick}", flush=True)

    # 5) verify on real photos: torch encoders vs ONNX encoders (exactly what the server will run)
    from petti_vision.onnx_runtime import OnnxEncoders
    from petti_vision.segmenter import MEAN, STD, input_size

    onx = OnnxEncoders(cfg, OUT)
    imgs = [Image.open(p).convert("RGB") for p in sample_images(a.check)]
    rep = {"n": len(imgs)}
    t1 = time.time()
    ref = enc.encode_images(imgs, tta=True)
    t_torch = (time.time() - t1) / max(1, len(imgs))
    t1 = time.time()
    got = onx.encode_images(imgs, tta=True)
    t_onnx = (time.time() - t1) / max(1, len(imgs))
    for k in ("reid", "vlm"):
        c = cos(ref[k], got[k])
        rep[f"{k}_cos_min"], rep[f"{k}_cos_mean"] = round(float(c.min()), 5), round(float(c.mean()), 5)
    segc = []
    for im in imgs[:12]:
        tw, th = input_size(im.width, im.height, 448)
        arr = np.asarray(im.resize((tw, th), Image.BICUBIC), dtype=np.float32) / 255.0
        px = ((arr - MEAN) / STD).transpose(2, 0, 1)[None]
        segc.append(float(cos(enc.dino_hidden(px)[0, 1:], onx.dino_hidden(px)[0, 1:]).mean()))
    rep["seg_tokens_cos_mean"] = round(float(np.mean(segc)), 5)

    # diagnostics: separate preprocessing error from export error from INT8 error
    import onnxruntime as ort

    from petti_vision.onnx_runtime import dino_preprocess, siglip_preprocess

    sub = imgs[:12]
    hf_sig = enc.vlm_proc(images=sub, return_tensors="pt")["pixel_values"].numpy()
    my_sig = np.stack([siglip_preprocess(im) for im in sub])
    hf_dino = enc.reid_proc(images=sub, return_tensors="pt")["pixel_values"].numpy()
    my_dino = np.stack([dino_preprocess(im) for im in sub])
    rep["prep_maxdiff_siglip"] = round(float(np.abs(hf_sig - my_sig).max()), 4) if hf_sig.shape == my_sig.shape else f"shape {hf_sig.shape} vs {my_sig.shape}"
    rep["prep_maxdiff_dino"] = round(float(np.abs(hf_dino - my_dino).max()), 4) if hf_dino.shape == my_dino.shape else f"shape {hf_dino.shape} vs {my_dino.shape}"
    with torch.no_grad():
        t_sig = sig(torch.from_numpy(hf_sig)).numpy()
    for tag, path in (("fp32", "siglip_vision_fp32.onnx"), ("int8", "siglip_vision.onnx")):
        sess = ort.InferenceSession(str(OUT / path), providers=["CPUExecutionProvider"])
        rep[f"siglip_{tag}_cos_vs_torch_same_input"] = round(float(cos(t_sig, sess.run(None, {"pixel": hf_sig})[0]).min()), 5)
    rep["ms_per_image_torch_cpu"] = round(t_torch * 1000, 1)
    rep["ms_per_image_onnx"] = round(t_onnx * 1000, 1)
    rep["sizes_mb"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(OUT.glob("*.onnx"))}
    rep["chosen"] = chosen
    print(json.dumps(rep, indent=1), flush=True)
    (ROOT / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "reports" / "onnx_export.json").write_text(json.dumps(rep, indent=2))
    need = 0.99
    if rep["reid_cos_min"] < need or rep["vlm_cos_min"] < need or rep["seg_tokens_cos_mean"] < need:
        sys.exit(f"ONNX outputs drift from torch (need cosine ≥ {need}) — do not deploy with PETTI_RUNTIME=onnx")
    print("ONNX export verified ✓", flush=True)


if __name__ == "__main__":
    main()
