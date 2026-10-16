# Data and model licenses

Weights that ship must be Apache-2.0, MIT, or BSD. A dataset without a commercial grant is evaluation only and must not be used to train those weights.

| Asset | License | Allowed use in this repo |
|---|---|---|
| Ultralytics YOLO26n-seg (`yolo26n-seg.pt`) | AGPL-3.0 | Dev-only A/B comparisons (`requirements-dev.txt`, `PETTI_DETECTOR_BACKEND=yolo`). Not in production requirements or images. |
| Segmenter v2 head (`artifacts/seg_head2.npz`) | Ours | Trained on Oxford-IIIT Pet trainval trimaps + COCO train2017 dog/cat masks with commercial-OK licenses only (see below) + COCO/Wikimedia no-animal photos. Default detector. |
| torchvision Mask R-CNN v2 (COCO weights) | BSD-3 | Offline teacher for *evaluation* (`training/teacher.py`, `probe_hard.py`). Never loaded by the server; its pseudo-labels are not used for shipped weights. |
| COCO 2017 annotations | CC BY 4.0 | Masks for training (train2017, filtered) and evaluation (val2017). |
| COCO 2017 images, train split | Per-image Flickr license | **Training only with license ids 4 (CC BY 2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions), 8 (US Gov)** — filtered in `training/fetch_coco.py`, recorded per file in `data/public/coco/coco_pets.json`. NC/ND images are never downloaded for training. |
| COCO 2017 images, val split | Per-image Flickr license (incl. NC) | Evaluation only (`eval_seg.py --set coco`). |
| Attribute heads (`artifacts/heads.npz`) | Ours | v0.5: Oxford-IIIT Pet **trainval only** (CC BY-SA 4.0); the official Oxford test split is held out for evaluation. A variant trained additionally on Stanford/ImageNet photos (`heads_full`, +1.4 pt dog top-1) is kept local and git-ignored. |
| User feedback photos (`data/feedback/`) | User photos | Stored only when the user ticks the consent box; deletable via `DELETE /feedback/<sha1>`; used by `retrain.py` through `build_feedback_set.py`. |
| DINOv2-small (`facebook/dinov2-small`) | Apache-2.0 | Runtime and training. |
| SigLIP2-base (`google/siglip2-base-patch16-224`) | Apache-2.0 | Runtime and training. |
| Oxford-IIIT Pet images and trimaps | CC BY-SA 4.0 | Trainval trimaps trained `artifacts/seg_head.npz`. The official test list was not used for that head. Share-alike may attach to a model trained on this set. The same images were already used for `artifacts/heads.npz`. |
| Stanford Dogs / ImageNet (dog.ceo `n########_*.jpg`, distractors, browser bench) | ImageNet: non-commercial research and education | Evaluation only. v0.5 production weights (heads, segmenter) are not trained on them; `auto_train.py` defaults to `--datasets oxford`. |
| dog.ceo owner-named phone photos (`data/eval/dogceo/`, identity set) | Uploaded by owners; no commercial grant found | Evaluation only. |
| Synthetic negatives (`data/eval/negatives/`) | Generated in `training/freeze_eval.py` | Evaluation only. Line drawings, not photographs. |
| Wikimedia scene photos (`data/public/scenes/train`, `data/eval/negatives_photo`) | Per file in `negatives_photo_licenses.json`: CC0, public domain, CC BY, CC BY-SA. NC licenses were skipped | Train split may influence `seg_head.npz`. Eval split is evaluation only. |
| Owner registrations in `data/` (live registry) | User photos | Not in git. Training use requires explicit consent (Phase 2). Deletion on request (Phase 5). |

Frozen eval contract: `apps/model/data/eval/manifest.json` (sha256 + perceptual hash). Filename exclude list: `apps/model/data/eval/bench_exclude.txt`.
