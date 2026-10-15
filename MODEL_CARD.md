# Petti Vision model card

## Intended use

Match a photo of a lost dog or cat to pets an owner has registered, and describe species, breed mix, size, age, color, and marks. The UI copy is Mongolian. This is an assistive lookup, not a determination of ownership.

## What runs today

Detector: a small conv head on frozen DINOv2-small patch tokens at 224 px (`artifacts/seg_head.npz`, Apache-2.0 backbone). It finds a pet region and does not decide dog versus cat. Species comes from SigLIP. YOLO26n-seg (AGPL-3.0) loads only when `PETTI_DETECTOR_BACKEND=yolo`. Identity: frozen DINOv2-small CLS on the crop. Attributes: SigLIP2-base zero-shot fused with `artifacts/heads.npz`. If the foreground is small and SigLIP prefers car/person/furniture/toy prompts, the answer is the Mongolian sentence "Энэ зурагт нохой эсвэл муур харагдахгүй байна" and `quality.no_animal` is set. Two or more instances add `quality.multi_animal` and a warning.

## Metrics

The regression gate reads `apps/model/artifacts/reports/baseline.json`, produced by:

```bash
cd apps/model && python3 training/eval_all.py --version v0.4 --write-baseline
```

Measured on the frozen identity set (168 owner photos, 46 dogs, 104 ImageNet distractors) on Apple MPS, 2026-10-15. This is not the browser bench (that image set is not on disk, only its filenames are).

| Metric | Measured | Notes |
|---|---|---|
| Species, dogs | 100% (n=168) | No cat photos in this set |
| Breed top-1 / top-3, dogs | 81.5% / 100% (n=54) | Non-mix folder slugs only. Browser bench 92.9% is a different set |
| Breed, cats | not measured | No labeled cat eval images frozen |
| Size exact / ±1 | 71.2% / 88.5% | Adult breed prior, not a measured weight |
| Age, color, marks | not measured | No labels on the frozen photos |
| ECE species / breed / size | 0.001 / 0.222 / 0.105 | Breed confidence is over-sure |
| Re-ID R@1 / R@5 / mAP | 0.893 / 0.976 / 0.829 | Leave-one-out cosine, 168 queries |
| Look-alike false match | 3.0% | Top hit is a distractor |
| Detection recall | 96.4% | Mask IoU has no ground truth here |
| No-animal false positive | 0% (n=10) | Line drawings only, not photos of cars or people |
| High-confidence breed | 93.9% (n=33) | Below the 95% bar |
| Server latency p50 / p95 | 120 ms / 331 ms | MPS, not the 2 vCPU host |

The DINOv2 segmenter (`artifacts/reports/v0.5-seg.json`, same identity set, MPS) compared with that baseline and with YOLO on the official Oxford-IIIT Pet test split (3669 images, not used to train the head):

| Metric | YOLO / v0.4 | DINOv2 segmenter |
|---|---|---|
| Oxford test mask IoU | 0.945 (200 images) | 0.938 (full test) |
| Oxford detection recall, all / small | 0.985 / 1.0 (200 images, 8 small) | 1.0 / 1.0 (3669 images, 233 small) |
| Detect time p50 | 25 ms | 17 ms |
| Identity-set species, dogs | 1.000 | 0.994 |
| Breed top-1, dogs | 0.815 | 0.907 |
| Size exact | 0.712 | 0.808 |
| Re-ID R@1 / false match | 0.893 / 0.030 | 0.887 / 0.048 |
| Size ECE | 0.105 | 0.121 |
| Pipeline p50 | 120 ms | 115 ms |
| No-animal false positive | 0 on 10 drawings | 0 on those drawings plus 12 Wikimedia photos |

`training/gate.py` still compares against `baseline.json` (the YOLO run). It fails this candidate on size ECE (0.105 → 0.121) and look-alike false-match rate (0.030 → 0.048). The baseline file was not replaced.

## Limitations

- Oxford "small" pets (foreground under 8% of the image, 233 test photos) were all detected. The identity-set small slice is still only 12 photos.
- Photographic no-animal coverage is 12 Wikimedia images (people, furniture, toys) plus 10 drawings. There is no held-out photo of a car; one CC0 car was used in training.
- Several animals in one photo produce a warning. The description is still the largest animal.
- Look-alike false matches rose from 3.0% to 4.8% because the crop changed. Size ECE rose from 0.105 to 0.121 even though size accuracy rose.
- Size labels on the identity set are the adult breed prior, so a puppy of a large breed is scored against the adult bin.
- High-confidence breed predictions are not yet shown to be ≥95% correct.

## Versions

`/health` reports the running models. Changing the identity embedding requires `Registry.reembed_stale` (Phase 2). This card does not change API shapes.
