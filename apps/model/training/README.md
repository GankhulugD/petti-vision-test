# Petti Vision — ML сургалтын pipeline

## Хурдан эхлэх (нэг команд)

```bash
python3 training/auto_train.py            # Oxford Pets + Stanford Dogs (dog.ceo CDN-ээр) → heads → тайлан
python3 training/fetch_identity_set.py    # re-ID тест багц (46 нохой, эзэдийн утасны зураг + 104 төстэй нохой)
python3 training/evaluate.py --manifest data/eval/dogceo_ids.csv --distractors data/eval/dogceo_distractors.csv
python3 training/evaluate.py ... --loo    # зөвхөн identity embedding-ийн хурдан харьцуулалт (168 query)
```

`data/eval/bench_exclude.txt`-д байгаа зургууд (browser benchmark-ийн) болон эзэдийн зургууд (re-ID тест)
сургалтад хэзээ ч орохгүй — тэгэхгүй бол тоо хуурамчаар өндөр гарна.

### Server дээрээс сургах (dev горим)

`apps/model/.dev` файл байвал (эсвэл `PETTI_DEV=1`) server нь зөвхөн localhost-оос хандах `/dev/*` endpoint нээнэ:

| Endpoint | Үйлдэл |
|---|---|
| `POST /dev/jobs {"script":"auto_train","args":[...],"env":{"PETTI_REID_POOL":"masked"}}` | training/ доторх script-ийг тусдаа process-оор ажиллуулна |
| `GET /dev/jobs/{id}?tail=40` | log-ийн сүүл, ажиллаж байгаа эсэх |
| `DELETE /dev/jobs/{id}` | зогсооно |
| `POST /dev/reload` | шинэ heads / calibration-ийг server-ийг унтраалгүй ачаална |

Deploy (HF Space) дээр `.dev` байхгүй тул эдгээр нээгдэхгүй (`.gitignore`-д орсон).

### Сургалтаас сурсан зүйлс
* **Size head сургахгүй.** Public dataset-д хэмжээний шошго байхгүй, үүлдрээс гаргасан "сул" шошгоор сургахад
  benchmark дээр хэмжээний нарийвчлал 76%→50% унасан. Үүлдрийн prior-оор тооцох нь илүү.
* **Head-ийн итгэлийг хязгаарлана** (alpha ≤ 0.5, temperature ≥ 1). Oxford/Stanford зураг цэвэрхэн, хэрэглэгчийн
  утасны зураг тийм биш — in-domain val дээр сонгосон alpha=1 out-of-domain дээр zero-shot-оос муу ажилласан.
* **Heads нь SigLIP feature дээр л** (`--feature vlm`), тиймээс re-ID descriptor солиход heads хүчингүй болохгүй.
* **Re-ID descriptor = DINOv2 CLS token** (LOO R@1: CLS 0.893, CLS‖animal-mask mean 0.875, CLS‖mean 0.869).


Backbone-ууд (YOLO, DINOv2, SigLIP2) **frozen**, бид дээр нь жижиг модел сургадаг. Тиймээс:

* feature-ийг нэг удаа гаргаад cache хийнэ (`artifacts/cache/*.npz`) → дараагийн сургалт 1-2 минут
* inference дээр heads numpy-аар ажиллана (<1 ms)
* `--resume`-ээр өмнөх жингээс үргэлжлүүлж сургана

```
data/manifests/*.csv ──► extract_features (detect→crop→SigLIP2‖DINOv2 + zero-shot probs) ──► cache
                                                    │
               ┌────────────────────────────────────┼─────────────────────────────┐
               ▼                                    ▼                             ▼
        train_heads.py                       train_reid.py                  evaluate.py
  species/breed/size/age/color/        SupCon P×K, 768→256 projection   Recall@1/5/10, MRR,
  pattern/coat/marks MLP +             (residual), baseline-аас сайн     cosine→prob calibration,
  temperature + zero-shot fusion α     бол л хадгална                    score weight grid search
               │                                    │                             │
               ▼                                    ▼                             ▼
     artifacts/heads.npz|json           artifacts/reid_projection.npz     artifacts/calibration.json
```

Server эхлэхдээ `artifacts/`-д байгаа бүхнийг автоматаар ачаална.

## 1. Өгөгдөл — manifest CSV

```
path,species,breed,size,age,colors,pattern,coat,marks,pet_id,source
data/my/1.jpg,dog,siberian_husky,medium,adult,black|white,bicolor,medium,blue_eyes,pet_42,owner
```

* Хоосон нүд = мэдэгдэхгүй (тухайн task-ийн loss-д оролцохгүй)
* `colors`, `marks` — `|`-ээр тусгаарласан multi-label. `marks=none` → "онцгой шинжгүй гэж баталгаажсан"
* `pet_id` — нэг амьтны зургуудыг холбоно (re-ID + leak-гүй train/val split)
* Label key-үүд: `petti_vision/taxonomy.py` (шинэ үүлдэр/өнгө/шинж нэмэхэд zero-shot шууд ажиллана)

Эх үүсвэрүүд:

```bash
python3 training/prepare_data.py --oxford                # 37 үүлдэр, ~7.4k зураг
python3 training/prepare_data.py --stanford-dogs         # 120 нохойн үүлдэр, ~20k
python3 training/prepare_data.py --folder data/my_photos # <breed_key|species>/[<pet_id>/]*.jpg
python3 training/export_registry.py                      # live бүртгэл → registry.csv
```

**Монголд хамгийн чухал өгөгдөл:** банхар, эрлийз нохой, гудамжны муур. Эдгээрийг `--folder`-оор
(`data/my_photos/bankhar/<pet_id>/*.jpg`) нэмж, сургах бүрт үүлдрийн нарийвчлал Монголын нөхцөлд өснө.
Re-ID-д PetFace (ECCV 2024) зэрэг identity dataset-ийг
`<species>/<identity>/*.jpg` хэлбэрээр нэмж болно (лицензийг нь шалгаарай).

## 2. Attribute heads

```bash
python3 training/train_heads.py                 # data/manifests/*.csv бүгд
python3 training/train_heads.py --resume --epochs 30
```

Task бүрийн үр дүн `artifacts/heads_report.txt`-д: `val_acc` (head), `zs_val_acc` (зөвхөн zero-shot),
`fused_val_acc` (хоёулаа), `alpha` (head-д итгэх жин), `temperature` (confidence calibration).
Шошготой train жишээ 30-аас бага task алгасагдана (өмнөх head хадгалагдана).

## 3. Re-ID

```bash
python3 training/train_reid.py --epochs 30 --P 32 --K 4
python3 training/reindex.py      # ⚠ projection хадгалагдсан бол заавал
```

## 4. Evaluate & calibrate

```bash
python3 training/evaluate.py --manifest data/manifests/registry.csv --limit 300
```

Бодит хэрэглээг дуурайна: identity бүрийн сүүлийн зураг = "олдсон амьтны зураг", бусад нь = эзний
бүртгэл. Recall@1/5/10 гаргаад `calibration.json` бичнэ (server дараагийн эхлэлд ашиглана).

## Нарийвчлалыг өсгөх дараалал

1. `PETTI_VLM_MODEL=google/siglip2-so400m-patch14-384`, `PETTI_REID_MODEL=facebook/dinov2-base` — том backbone (удаан, илүү нарийвчлалтай)
2. Монгол өгөгдлөөр heads сургах (банхар, эрлийз, гудамжны муур)
3. Бүртгэлийн identity-үүдээр re-ID сургах (хэдэн зуун амьтнаас эхлэн мэдэгдэхүйц өснө)
4. evaluate.py-аар weight/calibration тааруулах
5. Цаашид: backbone-ийн сүүлийн давхаргуудыг LoRA-аар fine-tune хийх (GPU шаардлагатай)
