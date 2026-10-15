# Petti Vision 🐾

Тэжээвэр амьтныг **зургаар таних** ба **Lost & Found** хайлтын demo.
Зураг оруулахад амьтны төрөл, ойролцоо үүлдэр (эрлийз бол аль үүлдрүүдтэй төстэй), биеийн хэмжээ,
ойролцоо нас, өнгө зүс, үсний урт, онцгой шинж тэмдгийг гаргаж, бүртгэлтэй / алга болсон амьтдаас
хамгийн төстэйг нь хувь, шалтгаантай нь эрэмбэлж өгнө.

```
apps/
  model/   Python · FastAPI · PyTorch — ML pipeline, registry, matcher, training
  web/     Next.js 15 · React 19 · Tailwind 4 — демо UI (монгол хэлээр)
```

## Архитектур

```
 зураг(ууд) ──► YOLO26n detector ──► амьтны crop (square pad, EXIF засна)
                  (dog/cat/bird/horse)      │
                                            ├─► DINOv2  ─► identity embedding  ── "яг энэ амьтан мөн үү?"
                                            └─► SigLIP2 ─► attribute embedding
                                                   │
           ┌───────────────────────────────────────┼─────────────────────────────┐
           ▼                                       ▼                             ▼
   zero-shot (prompt ensemble,        манай сургасан MLP heads         Lab өнгөний pixel estimator
   species-тэй нөхцөлтэй prompt)      (species/breed/size/age/          (background хасна)
                                       colour/pattern/coat/marks)
           └──────────────── product-of-experts fusion ───────────────┘
                     + breed→size prior, олон зургийн нэгтгэл, consistency check
                                            │
                                            ▼
                    AnimalProfile  →  JSON + монгол/англи тайлбар текст
                                            │
                     Matcher: 0.50·identity + 0.15·appearance + 0.35·attributes
                     (species hard filter, multi-view max, онцгой шинжийн bonus)
                                            │
                                            ▼
                         эрэмбэлсэн жагсаалт + "яагаад" гэдэг шалтгаан
```

| Хэсэг | Модел | Яагаад |
|---|---|---|
| Detector | Ultralytics **YOLO26n** (fallback yolo11n) | NMS-free, CPU дээр ч хурдан; COCO-д dog/cat/bird/horse бий |
| Identity | **DINOv2** ViT-S/14 (сонголт: ViT-B, DINOv3) | Instance-level retrieval-д нээлттэй feature-үүдээс хамгийн хүчтэй |
| Attributes | **SigLIP2** base (сонголт: so400m) | Zero-shot ангилал маш сайн → 0 өгөгдөлтэй эхний өдрөөс ажиллана |
| Манай ML | MLP heads + re-ID projection | Frozen backbone дээр → минутын дотор дахин сургана, numpy-аар inference |

**Өгөгдлийн flywheel:** эзэн бүртгүүлэхдээ 5+ зураг оруулж, системийн ангиллыг баталж/засна →
энэ нь (1) attribute heads-ийн ground truth, (2) re-ID-ийн identity label болно → `train_heads.py`,
`train_reid.py`-г дахин ажиллуулах бүрт нарийвчлал өснө.

## Mac дээр ажиллуулах

```bash
# 1) Model server (Python 3.11+)
cd apps/model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 download_models.py          # ~1 GB жин татна + zero-shot text cache үүсгэнэ (нэг удаа)
python3 server.py                   # http://localhost:8765  (Swagger: /docs)

# 2) Web
cd apps/web
npm install
npm run dev                         # http://localhost:3000
```

Apple Silicon дээр автоматаар `mps` (fp32) ашиглана. Хурд шалгах: `python3 benchmark.py dog.jpg`.
Үр дүн буруу бол: `python3 diagnose.py dog.jpg` — шат бүр юу харж байгааг хэвлэнэ.
Жин татахгүйгээр зөвхөн UI хөгжүүлэх: `PETTI_FAKE_MODELS=1 python3 server.py` — ⚠ энэ горимд таамаг санамсаргүй, вэб дээр улаан анхааруулга гарна.

Демо өгөгдөл: `python3 seed_demo.py demo_pets --status lost` (`demo_pets/<нэр>/*.jpg`).

## Үнэгүй deploy (Hugging Face Gradio)

Vercel дээр model багтахгүй. Hugging Face дээр **Docker SDK төлбөртэй**, **Gradio үнэгүй**.

Space: [Gankhulug/petti-vision](https://huggingface.co/spaces/Gankhulug/petti-vision). Орох файл нь root `app.py` — Gradio хуудас болон `/health`, `/analyze` API хамт. Эхний ачаалалт жин татдаг тул хэдэн минут болно.

## API

| Method | Path | Тайлбар |
|---|---|---|
| `POST` | `/analyze` | `images[]` (1..12) → бүрэн profile |
| `POST` | `/pets` | Бүртгэл: `images[]` (home/lost ≥5, found ≥1), `status`, `name`, `contact`, `location`, `owner_labels` (JSON) → lost/found бол шууд match |
| `GET` | `/pets?status=lost` | Жагсаалт |
| `GET` | `/pets/{id}/matches` | Хадгалсан амьтныг эсрэг сантай (lost↔found) тулгана |
| `PATCH` | `/pets/{id}/status` | `home→lost`, `→reunited` |
| `POST` | `/search` | `images[]` + `statuses=lost` → эрэмбэлсэн жагсаалт |
| `GET` | `/taxonomy` | Үүлдэр, өнгө, онцгой шинжийн жагсаалт (UI form-д) |

Жишээ хариу (`/analyze`):

```json
{
  "species": {"key": "dog", "mn": "Нохой", "confidence": 0.98},
  "breed": {"is_mixed": true, "looks_like": [{"key": "siberian_husky", "mn": "Хаски", "confidence": 0.41}, ...]},
  "size": {"key": "medium", "mn": "Дунд (10-25кг)", "confidence": 0.71},
  "age": {"key": "adult", "mn": "Насанд хүрсэн (2-7 нас)", "confidence": 0.64},
  "colors": [{"key": "black", "mn": "Хар", "share": 0.52}, {"key": "white", "mn": "Цагаан", "share": 0.38}],
  "pattern": {"key": "bicolor", "mn": "Хоёр өнгө"},
  "marks": [{"key": "blue_eyes", "mn": "Цэнхэр нүдтэй", "confidence": 0.81}],
  "category": "dog/medium/adult/black+white/bicolor",
  "description_mn": "Төрөл: Нохой (98%). Үүлдэр: эрлийз байж магадгүй — Хаски 41%, ...",
  "timing_ms": {"detect": 18, "embed": 95, "reason": 3, "total": 116, "photos": 1}
}
```
*(тоонууд жишээ — бодит утгыг өөрийн машин дээр `benchmark.py`-аар хэмжинэ)*

## Нарийвчлал (benchmark)

Тогтмол тест багц: dog.ceo-оос нохойн 33 үүлдрийн 99 зураг, Wikimedia Commons-оос муурны 15 үүлдрийн 40 зураг.
Browser console дээр `const m = await import('/test/bench.js'); await m.run(3)` гэж ажиллуулна.

| | v0.2 (deploy) | v0.3 zero-shot | v0.3 + Oxford heads |
|---|---|---|---|
| Нохой — төрөл | 98.0% | **100%** | **100%** |
| Нохой — үүлдэр top-1 / top-3 | 88.9 / 96.0% | 89.9 / 97.0% | **90.9 / 97.0%** |
| Нохой — хэмжээ (±1 ангилал) | 92.9% | 91.9% | 91.9% |
| Муур — төрөл | 95.0% | **100%** | **100%** |
| Муур — үүлдэр top-1 / top-3 | 70.0 / 80.0% | **75.0 / 85.0%** | 72.5 / 82.5% |
| Онцгой шинж / зураг (ихэнх нь худал) | 0.55 | **0.18** | 0.18 |
| Нохойд муурын зүс ("колор-поинт" г.м.) | 18 | **0** | 0 |

"Итгэлтэй" гэж гарсан үүлдрийн 97% нь зөв, "Магадгүй" нь 77%, "Тодорхойгүй" нь 36% (confidence calibration).

**Re-ID (яг тэр амьтан мөн үү):** dog.ceo дээрх эзэдийн өөрсдийн оруулсан зургаас 46 нохой (нэрээр нь бүлэглэсэн, 168 зураг),
мөн ижил үүлдрийн 104 өөр нохойг төөрөгдүүлэх зорилгоор нэмсэн. Нэг зургаар хайхад жинхэнэ нохой **1-рт 82.6%, эхний 5-д 97.8%** гарсан.
`python3 training/fetch_identity_set.py && python3 training/evaluate.py --manifest data/eval/dogceo_ids.csv --distractors data/eval/dogceo_distractors.csv`

## Сургалт (үргэлжлүүлж сургах)

Дэлгэрэнгүй: [`apps/model/training/README.md`](apps/model/training/README.md) · Colab: [`colab_train.ipynb`](apps/model/training/colab_train.ipynb)

```bash
cd apps/model
python3 training/auto_train.py                               # НЭГ КОМАНД: dataset татах → feature → heads → тайлан
python3 training/prepare_data.py --oxford --stanford-dogs   # public dataset → manifest
python3 training/export_registry.py                          # бүртгэлийн өгөгдөл → manifest
python3 training/train_heads.py [--resume]                   # attribute heads
python3 training/train_reid.py                               # re-ID projection (baseline-аас сайжирвал л хадгална)
python3 training/reindex.py                                  # хуучин бүртгэлийг шинэ embedding руу
python3 training/evaluate.py --manifest data/manifests/registry.csv   # Recall@k + calibration.json
```

## Тест

```bash
cd apps/model && PETTI_FAKE_MODELS=1 pytest -q      # numpy fake backbone-оор бүх урсгалыг шалгана
cd apps/web && npm run build
```

## Лиценз анхааруулга

* Ultralytics YOLO — **AGPL-3.0**. Хаалттай арилжааны бүтээгдэхүүнд Ultralytics Enterprise license авах,
  эсвэл detector-ийг Apache-2.0 модел (жишээ нь RT-DETR / D-FINE, transformers-оор) болгож солих хэрэгтэй
  (`backbones.YoloDetector`-ийг л солино).
* DINOv2, SigLIP2 — Apache-2.0. DINOv3 — Meta-гийн тусгай лиценз (HF дээр gated).
* Stanford Dogs, Oxford-IIIT Pet — судалгааны dataset; арилжааны сургалтад ашиглахаас өмнө нөхцөлийг шалгана уу.
