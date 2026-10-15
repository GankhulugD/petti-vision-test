# Data and model licenses

Weights that ship must be Apache-2.0, MIT, or BSD. A dataset without a commercial grant is evaluation only and must not be used to train those weights.

| Asset | License | Allowed use in this repo |
|---|---|---|
| Ultralytics YOLO26n-seg (`yolo26n-seg.pt`) | AGPL-3.0 | Dev / current demo runtime only. Must not be in the shipped production image (Phase 1 replaces it). |
| DINOv2-small (`facebook/dinov2-small`) | Apache-2.0 | Runtime and training. |
| SigLIP2-base (`google/siglip2-base-patch16-224`) | Apache-2.0 | Runtime and training. |
| Oxford-IIIT Pet images | CC BY-SA 4.0 | Already used to train `artifacts/heads.npz`. Share-alike may attach to a model trained on this set. Do not train new shipped weights on it until that is resolved. Evaluation is fine. |
| Stanford Dogs / ImageNet (dog.ceo `n########_*.jpg`, distractors, browser bench) | ImageNet: non-commercial research and education | Evaluation only. `prepare_data.py` still downloads Stanford images for the current heads; new production weights must not train on them. |
| dog.ceo owner-named phone photos (`data/eval/dogceo/`, identity set) | Uploaded by owners; no commercial grant found | Evaluation only. |
| Synthetic negatives (`data/eval/negatives/`) | Generated in `training/freeze_eval.py` | Evaluation only. Line drawings, not photographs. |
| Owner registrations in `data/` (live registry) | User photos | Not in git. Training use requires explicit consent (Phase 2). Deletion on request (Phase 5). |

Frozen eval contract: `apps/model/data/eval/manifest.json` (sha256 + perceptual hash). Filename exclude list: `apps/model/data/eval/bench_exclude.txt`.
