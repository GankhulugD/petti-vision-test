You are a senior ML engineer taking the Petti Vision pet-recognition system (this repo) from a working demo to a production-grade, license-clean, free-to-run model. Work in the existing code; do not rewrite from scratch.

## Context (current state)
- Python FastAPI model server in `apps/model` (`petti_vision/`: backbones, pipeline, zeroshot, heads, matcher, registry, taxonomy, color, config), Next.js web in `apps/web`.
- Pipeline: YOLO26n-seg (Ultralytics, **AGPL-3.0**) → crop/mask → DINOv2-small (identity embedding, CLS) + SigLIP2-base (zero-shot attributes with prompt ensembles) → small MLP heads on SigLIP features (`training/train_heads.py`, fused with zero-shot via alpha/temperature) → matcher (logit fusion of identity cosine + attribute agreement).
- Current metrics (v0.4): dogs species 100%, breed top-1 92.9% / top-3 98%; cats breed top-1 72.5%; size 77.8%; re-ID R@1 0.87 / R@5 1.0 (46 ids + 104 look-alike distractors); server p50 ~200 ms. Known weaknesses: small/far animals (52% detected), no "no animal" gate (a car can come back as "dog"), YOLO class errors (husky → "cat"), no multi-animal handling.
- Existing tools: `training/auto_train.py`, `training/evaluate.py` (re-ID, `--loo`, writes calibration.json), `training/fetch_identity_set.py`, browser bench `apps/web/public/test/bench.js`, `data/eval/bench_exclude.txt`, dev job API (`/dev/jobs`, `/dev/reload`), `Registry.reembed_stale`, `/analyze` + `/search` return an `explain` payload used by the UI (`apps/web/components/VisionLab.tsx`).

## Hard constraints
1. **Hardware: one Apple M2 Mac (MPS).** No paid GPUs. Every training job must finish in ≲ 2 h on M2 and fit in ~8 GB. Strategy: precompute and cache frozen backbone features to disk (fp16 .npy, keyed by image hash + model version), train only light heads or LoRA adapters, small batches + gradient accumulation, fp32 on MPS, `PYTORCH_ENABLE_MPS_FALLBACK=1`. Make every long job resumable (checkpoint each epoch).
2. **Free & license-clean in production.** No paid APIs or SaaS. Every weight file shipped/served must be Apache-2.0, MIT or BSD. **Remove the AGPL YOLO dependency from the runtime path.** Record every model and dataset with its license and allowed use in `DATA_LICENSES.md`; a dataset that does not allow commercial use may be used for evaluation only, never for weights that ship.
3. **Must run on a free/cheap CPU host** (target: 2 vCPU, ≤ 4 GB RAM, no GPU). p95 latency ≤ 400 ms per photo on that box; cold start ≤ 30 s.
4. **Do not break contracts:** keep `/analyze`, `/search`, `/pets`, `/health` request/response shapes and the `explain` payload backward compatible; UI text stays Mongolian; `PETTI_FAKE_MODELS=1 python3 -m pytest -q tests` must stay green; add tests for everything new.
5. **No test leakage.** Test sets are frozen, versioned and excluded from all training (extend `bench_exclude.txt` / dedupe by perceptual hash).
6. Small, reviewable commits per phase. Never delete user data without asking.

## Phase 0 — Evaluation harness and regression gate (do this first)
- `training/eval_all.py`: one command → `artifacts/reports/<version>.json` + `.md` with: species acc; breed top-1/top-3 per species; size/age acc (exact and ±1); color/marks F1; ECE (calibration) per task; re-ID R@1, R@5, mAP and false-match rate on look-alikes; detection recall, mask IoU; **no-animal false-positive rate** (build a negative set: cars, people, furniture, toys, drawings); slices: small/far animals, dark/blurry, multi-animal, mixed breeds; CPU latency p50/p95.
- Freeze test sets under `data/eval/` with a manifest (hashes, sources, licenses).
- `training/gate.py`: compares a candidate report with `artifacts/reports/baseline.json`; fails if any key metric drops beyond a tolerance (e.g. 1 pt) or latency regresses > 15%. Wire the fake-model tests into CI (GitHub Actions); the real eval runs locally.
- Deliver: baseline report for the current v0.4.

## Phase 1 — Own detector/segmenter (replace YOLO)
- Build a light segmentation head on **frozen DINOv2-small patch tokens** (full image at 448 px → 32×32 grid): 1×1 conv or tiny conv decoder → foreground logits, upsample, threshold, connected components → instances (box + polygon). Species comes from SigLIP, not the detector.
- Training data: Oxford-IIIT Pet trimaps (CC BY-SA 4.0) + COCO 2017 dog/cat instance masks (annotations CC BY 4.0) + hard negatives (no-animal images). Cache patch features once; head training must take minutes on M2.
- Add a **"no animal" gate** (foreground area + SigLIP animal-vs-not score) → clear Mongolian message instead of a wrong answer; add a **multi-animal warning**.
- Acceptance (vs YOLO on Phase-0 sets): detection recall ≥ YOLO overall and on small animals, mask IoU ≥ 0.80 on Oxford test, no-animal FPR ≤ 2%, total latency lower. If it fails, report numbers and fall back to fine-tuning an Apache-2.0 detector (verify the exact model size's license first); YOLO may remain only as a dev-only flag, never in the shipped image.

## Phase 2 — Identity (re-ID) fine-tuning on M2
- LoRA (PEFT, r≈8) on the last 2–4 blocks of DINOv2-small + projection head (384→256, L2-norm); loss: sub-center ArcFace (or multi-similarity) with PK sampling (e.g. P=16 identities × K=4); light augmentation (crop, flip, small color jitter), 224 px.
- Data: owner registrations (each pet's ≥5 photos = one identity, with consent), identity sets whose license allows it, and self-supervised instance discrimination (two augmented views of one photo = same id) as warm-up. Verify every dataset license before training.
- Merge LoRA into the weights for export; bump `embedding_version`; `reembed_stale` must re-embed old registrations.
- Acceptance: R@1 +≥5 pts on the frozen identity set, look-alike false-match rate not worse, matcher re-calibrated (`evaluate.py` writes calibration.json).

## Phase 3 — Breed/attributes and calibration
- Train heads on cached multi-crop features (e.g. 5 crops/image) with class-balanced sampling, label smoothing, feature-space mixup, early stopping on val; temperature-scale every head on a held-out split; refit head/zero-shot fusion (alpha, T) per task.
- Improve cats (only 12 Oxford breeds): add license-clean cat data or fall back to zero-shot for unseen breeds; mixed-breed handling stays honest ("resembles A, B").
- Acceptance: cat top-1 +≥5 pts, dogs not worse, ECE ≤ 0.05 per task, "high" confidence ≥ 95% correct.

## Phase 4 — Production serving
- Export the vision parts (DINOv2 + merged LoRA + seg head, SigLIP2 vision tower) to ONNX (opset 17); INT8 dynamic quantization with onnxruntime; check output cosine ≥ 0.99 vs PyTorch and Phase-0 metrics within tolerance. Text embeddings stay precomputed, so the text tower is not needed at runtime.
- Runtime image: onnxruntime + numpy only (no torch); pinned dependencies; Dockerfile for CPU; models in `artifacts/models/<version>/` with `manifest.json` (sha256, metrics, licenses, data versions); `/health` reports model and embedding versions.
- Vector search: hnswlib (Apache-2.0) or FAISS (MIT) index persisted alongside the registry; exact search below ~10k pets.
- Reliability: input validation (size, type, HEIC), per-request timeout, max photos, graceful errors, structured JSON logs, a simple `/metrics` endpoint, load test (e.g. 20 concurrent users) on a 2 vCPU container.

## Phase 5 — Continuous improvement (data flywheel)
- With explicit user consent, log predictions + owner corrections (hashes, not raw PII) to `data/feedback/`; dedupe by perceptual hash; queue low-confidence cases for labeling.
- `make retrain`: build the dataset → cache features → train → `eval_all` → `gate` → write a new versioned model only if the gate passes; one-command rollback to the previous version.
- Support deleting a user's photos/embeddings on request.

## Deliverables and reporting
- After each phase: a short report (metrics before/after on every slice, latency, what changed, risks) and an updated `MODEL_CARD.md` (intended use, metrics, limitations, licenses) and `DATA_LICENSES.md`.
- Do not start the next phase until the current one passes its acceptance criteria; if it does not, stop and present the numbers and options.
- Ask before any step that would download > 5 GB, run > 2 h, or change public API shapes.
