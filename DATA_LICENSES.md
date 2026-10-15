# Data and model licenses

Weights that ship must be Apache-2.0, MIT, or BSD. A dataset without a commercial grant is evaluation only and must not be used to train those weights.

| Asset | License | Allowed use in this repo |
|---|---|---|
| Ultralytics YOLO26n-seg (`yolo26n-seg.pt`) | AGPL-3.0 | Dev only, `PETTI_DETECTOR_BACKEND=yolo`. The default server does not import it. |
| DINOv2-small (`facebook/dinov2-small`) | Apache-2.0 | Runtime and training. |
| SigLIP2-base (`google/siglip2-base-patch16-224`) | Apache-2.0 | Runtime and training. |
| Oxford-IIIT Pet images and trimaps | CC BY-SA 4.0 | Trainval trimaps trained `artifacts/seg_head.npz`. The official test list was not used for that head. Share-alike may attach to a model trained on this set. The same images were already used for `artifacts/heads.npz`. |
| Stanford Dogs / ImageNet (dog.ceo `n########_*.jpg`, distractors, browser bench) | ImageNet: non-commercial research and education | Evaluation only. `prepare_data.py` still downloads Stanford images for the current heads; new production weights must not train on them. |
| dog.ceo owner-named phone photos (`data/eval/dogceo/`, identity set) | Uploaded by owners; no commercial grant found | Evaluation only. |
| Synthetic negatives (`data/eval/negatives/`) | Generated in `training/freeze_eval.py` | Evaluation only. Line drawings, not photographs. |
| Wikimedia scene photos (`data/public/scenes/train`, `data/eval/negatives_photo`) | Per file in `negatives_photo_licenses.json`: CC0, public domain, CC BY, CC BY-SA. NC licenses were skipped | Train split may influence `seg_head.npz`. Eval split is evaluation only. |
| Owner registrations in `data/` (live registry) | User photos | Not in git. Training use requires explicit consent (Phase 2). Deletion on request (Phase 5). |

Frozen eval contract: `apps/model/data/eval/manifest.json` (sha256 + perceptual hash). Filename exclude list: `apps/model/data/eval/bench_exclude.txt`.
