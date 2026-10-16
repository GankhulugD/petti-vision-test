# Petti Vision model card (v0.5)

## Intended use

Match a photo of a lost dog or cat to registered pets, and describe species, breed (or the breeds a mix resembles), size, age, colors and distinctive marks. UI copy is Mongolian. Assistive lookup only — not proof of ownership.

## What runs

| Stage | Model | License | Notes |
|---|---|---|---|
| Find the animal | DINOv2-small patch tokens (frozen) + our conv head `seg_head2.npz` | Apache-2.0 + ours | Aspect ratio kept, long side 448 px. Real outlines (polygons), sub-patch masks. Trained on Oxford-IIIT Pet trimaps + license-filtered COCO dog/cat masks. |
| Identity | DINOv2-small CLS on the crop | Apache-2.0 | Used by the matcher. Learned projection (`train_reid.py`) is gated and activates only when it beats the raw embedding. |
| Attributes | SigLIP2-base zero-shot + our heads `heads.npz` | Apache-2.0 + ours | Heads trained on Oxford trainval only (no ImageNet/Stanford photos). Breed confidences calibrated (`calibration.json`). |
| Runtime | PyTorch (dev) or ONNX Runtime (`PETTI_RUNTIME=onnx`, torch-free) | MIT (ORT) | ONNX export verified identical to PyTorch (cosine 1.000). |

YOLO (AGPL-3.0) is no longer in production requirements; it remains available only through `requirements-dev.txt` for A/B comparisons.

## Metrics (2026-10-16, Apple M2 / MPS unless noted)

### Finding the animal

| Set | Metric | YOLO26n-seg | Segmenter v1 (224², Oxford only) | **Segmenter v2 (shipped)** |
|---|---|---|---|---|
| COCO val2017 dog/cat (349 cluttered photos, real masks) | recall | 0.868 | 0.805 | **0.946** |
| | mask IoU | 0.742 | 0.563 | **0.762** |
| | IoU ≥ 0.5 | 0.837 | 0.719 | **0.905** |
| | small animals (<5% of image, n=103): recall / IoU | 0.680 / 0.551 | 0.369 / 0.194 | **0.825 / 0.557** |
| Owner phone photos (371, reference = BSD Mask R-CNN teacher) | box hit@0.5 / hit@0.7 | 0.938 / 0.898 | 0.968 / 0.868 | **0.981 / 0.906** |
| | missed animals | 17 | 0 | **0** |
| Oxford-IIIT Pet test (800) | recall / mask IoU | — | 1.0 / 0.938* | **1.0 / 0.987** |
| No-animal photos + drawings (22) | false positives | 0 | 0 | **0** |
| Detect time p50 | | 32 ms | 29 ms | 45–60 ms |

\* v1's Oxford IoU was measured on the 16×16 grid.

### Breeds — frozen Oxford-IIIT Pet **test** split (never trained on), 40 photos per breed

| | n | top-1 | top-3 | ECE raw → calibrated | "High confidence" answers correct |
|---|---|---|---|---|---|
| Dogs (25 breeds) | 520 | **94.8%** | 99.8% | 0.140 → **0.016** | 97.5% (484) |
| Cats (12 breeds) | 480 | **85.6%** | 94.2% | 0.189 → **0.047** | 93.2% (411) |

Calibration (display temperature 0.53, levels high ≥ 0.67, medium ≥ 0.50, "unsure" < 0.49) was cross-fitted: on held-out halves ECE was 0.022 and 0.037 and "high" was 94–95% correct. Dog breeds outside Oxford's 25 rely on zero-shot; on 54 owner photos of such breeds top-1 is 85.2% (the ImageNet-trained variant: 87.0%).

### Whole system — frozen owner-photo set (`eval_all.py`, gate vs v0.4 baseline)

| Metric | v0.4 (YOLO) | **v0.5** |
|---|---|---|
| Species, dogs (168) | 1.000 | 0.994 |
| Breed dog top-1 / top-3 (54) | 0.815 / 1.0 | 0.852 / 1.0 |
| Size exact / ±1 | 0.712 / 0.885 | 0.750 / 0.923 |
| Breed ECE | 0.222 | 0.092 |
| Re-ID R@1 / R@5 / mAP (168 queries) | 0.893 / 0.976 / 0.829 | 0.881 / 0.988 / 0.821 |
| Look-alike false match | 3.0% | 3.0% |
| Pipeline latency p50 / p95 (MPS) | 120 / 331 ms | 141 / 183 ms |

`gate.py` passes with one written waiver: p50 latency +21 ms for the new segmenter.

### Marks

Calibrated on 709 ordinary photos (`marks_calibration.json`): every mark fires on ≤4.4% of ordinary photos; rare marks (one eye, three legs, scar, heterochromia, clipped ear) on ≤0.3%. "Three legs" and "one eye" are only asserted when at least two photos agree — a hidden leg or eye (back view, lying down) no longer becomes a distinctive mark.

### ONNX runtime

fp32 ONNX matches PyTorch exactly (cosine 1.000 for identity, SigLIP and segmenter tokens). Dynamic INT8 was rejected by the export gate: SigLIP drifts to cosine 0.59–0.69 and DINOv2 to 0.990. On the M2 the ONNX CPU path is slower than PyTorch+Accelerate (324 vs 142 ms/image with test-time augmentation). Its value is a torch-free, smaller production image (`requirements-runtime.txt`). Measure on the target x86 host before switching.

## Limitations

- Touching animals can merge into one region; the description then covers both (a multi-animal warning is shown).
- Dogs in clothes or blankets: the mask follows the visible animal, so the crop may exclude covered body parts.
- Cats are weaker than dogs (85.6% top-1); breeds outside Oxford's 37 rely on zero-shot.
- Re-ID has only 46 test identities. Fine-tuning is wired (`train_reid.py`, gated) and needs registrations with consent.
- Calibration and level thresholds were fitted on the Oxford test photos (cross-fitted numbers above are the honest ones). Refit after every head retrain (`retrain.py` does this).

## Operating it

```bash
cd apps/model
python3 training/eval_seg.py --backend dino2 --set coco      # detector on cluttered scenes
python3 training/eval_breeds.py                              # breeds, frozen Oxford test
python3 training/eval_all.py --version v0.5 && python3 training/gate.py artifacts/reports/v0.5.json
python3 training/retrain.py                                  # data flywheel: feedback → train → eval → gate → keep or roll back
python3 training/retrain.py --rollback
python3 training/export_onnx.py && PETTI_RUNTIME=onnx python3 server.py
```
