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

## Vercel дээр deploy (вэб)

Vercel зөвхөн **Next.js** (`apps/web`) — ML сервер (`apps/model`) тусад нь GPU/CPU сервер дээр асаана (жишээ нь [Render](https://render.com) `render.yaml`).

1. [GitHub repo](https://github.com/GankhulugD/petti-vision-test) → Vercel **Import**
2. **Root Directory:** `apps/web`
3. **Environment Variables:** `PETTI_API_URL` = model API-ийн public URL (trailing slash байхгүй), жишээ нь `https://petti-vision-api.onrender.com`
4. Deploy — браузер `/api/*` хүсэлтийг `PETTI_API_URL` руу proxy хийнэ (`next.config.ts` rewrites)

Локал `.env` жишээ: `apps/web/.env.example`. Model server Render дээр: root-ийн `render.yaml` (Starter+ санах ой шаардлагатай, эхний build удаан).

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

## Сургалт (үргэлжлүүлж сургах)

Дэлгэрэнгүй: [`apps/model/training/README.md`](apps/model/training/README.md) · Colab: [`colab_train.ipynb`](apps/model/training/colab_train.ipynb)

```bash
cd apps/model
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
