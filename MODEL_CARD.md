# Petti Vision model card

## Intended use

Match a photo of a lost dog or cat to pets an owner has registered, and describe species, breed mix, size, age, color, and marks. The UI copy is Mongolian. This is an assistive lookup, not a determination of ownership.

## What runs today (v0.4)

Detector: Ultralytics YOLO26n-seg (AGPL-3.0, not acceptable in the production image). Identity: frozen DINOv2-small CLS (Apache-2.0). Attributes: SigLIP2-base zero-shot (Apache-2.0) fused with small MLP heads in `artifacts/heads.npz`. Matcher: cosine identity plus attribute agreement, calibrated in `artifacts/calibration.json`.

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

`training/gate.py` compares a later report to `artifacts/reports/baseline.json` (this run). A metric with no number is not gated.

## Limitations

- Small or distant animals are often missed.
- There is no photograph-level "no animal" gate. A synthetic drawing set measures false positives; photos of cars and people are not in the frozen set yet.
- The detector can label a husky as a cat. Species is still taken from that detector.
- Multiple animals in one photo are not separated for the user.
- Size labels on the identity set are the adult breed prior, so a puppy of a large breed is scored against the adult bin.
- High-confidence breed predictions are not yet shown to be ≥95% correct.

## Versions

`/health` reports the running models. Changing the identity embedding requires `Registry.reembed_stale` (Phase 2). This card does not change API shapes.
