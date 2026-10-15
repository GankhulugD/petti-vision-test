#!/usr/bin/env python3
"""Petti Vision inference + lost&found API (FastAPI).

    uvicorn server:app --host 0.0.0.0 --port 8765
    # or: python3 server.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from petti_vision import Config, PettiVision
from petti_vision.matcher import Matcher
from petti_vision.pipeline import apply_owner_labels
from petti_vision.registry import STATUSES, Registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("petti.server")

PORT = int(os.environ.get("PORT", "8765"))
MAX_FILES = 12
MAX_BYTES = 15 * 1024 * 1024

cfg = Config()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    if cfg.fake_models:
        log.warning("!!! PETTI_FAKE_MODELS=1 — random test models, predictions are meaningless !!!")
    vision = PettiVision(cfg)
    vision.warmup()  # first call compiles kernels; do it before traffic
    state["vision"] = vision
    state["registry"] = Registry(cfg.db_path, cfg.images_dir)
    state["matcher"] = Matcher(cfg)
    log.info("ready in %.1fs (%d pets in registry)", time.perf_counter() - t0, len(state["registry"].pets))
    yield


app = FastAPI(title="Petti Vision", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
cfg.images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(cfg.images_dir)), name="images")


def _read(files: list[UploadFile]) -> list[bytes]:
    if not files:
        raise HTTPException(400, "Дор хаяж 1 зураг оруулна уу")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Хамгийн ихдээ {MAX_FILES} зураг")
    out = []
    for f in files:
        raw = f.file.read()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: файл хэт том (15MB хүртэл)")
        out.append(raw)
    return out


def _analyze(raw: list[bytes]):
    try:
        return state["vision"].analyze(raw)
    except Exception as exc:  # unreadable image etc.
        log.exception("analyze failed")
        raise HTTPException(422, f"Зургийг уншиж чадсангүй: {exc}") from exc


# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    v: PettiVision = state["vision"]
    heads = v.heads
    return {
        "status": "ok",
        "models": {
            "detector": cfg.detector_weights,
            "reid": cfg.reid_model,
            "vlm": cfg.vlm_model,
            "trained_heads": {k: t.get("val_acc") for k, t in heads.tasks.items()} if heads else None,
            "reid_projection": cfg.reid_proj_path.exists(),
            "fake": cfg.fake_models,
        },
        "load_seconds": round(v.load_seconds, 2),
        "pets": {s: len(state["registry"].list(status=s)) for s in STATUSES},
        "min_photos": cfg.min_photos,
    }


@app.get("/taxonomy")
def taxonomy():
    """Label lists for UI forms (owner confirmation of breed/size/age/colours/marks)."""
    from petti_vision import taxonomy as T

    lab = lambda xs: [{"key": x.key, "mn": x.mn, "en": x.en} for x in xs]  # noqa: E731
    return {
        "species": lab(T.SPECIES),
        "breeds": [{"key": b.key, "mn": b.mn, "en": b.en, "species": b.species, "size": b.size} for b in T.BREEDS],
        "size": lab(T.SIZES), "age": lab(T.AGES), "colors": lab(T.COLORS), "pattern": lab(T.PATTERNS),
        "coat": lab(T.COAT_LENGTH),
        "marks": [{"key": m.key, "mn": m.mn, "en": m.en, "species": list(m.species)} for m in T.MARKS],
    }


@app.post("/analyze")
def analyze(images: list[UploadFile] = File(...)):
    """1..N photos of one animal → full profile (species, breed, size, age, colours, marks, text)."""
    return _analyze(_read(images)).to_json()


@app.post("/pets")
def create_pet(
    images: list[UploadFile] = File(...),
    status: str = Form("home"),
    name: str | None = Form(None),
    owner_name: str | None = Form(None),
    contact: str | None = Form(None),
    location: str | None = Form(None),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    event_at: str | None = Form(None),
    notes: str | None = Form(None),
    owner_labels: str | None = Form(None, description='JSON: {"breed":..,"size":..,"age":..,"colors":[..],"marks":[..]}'),
):
    """Register a pet (owner: ≥5 photos) or a found-animal report (finder: ≥1 photo)."""
    if status not in STATUSES:
        raise HTTPException(400, f"status: {STATUSES}")
    raw = _read(images)
    if status in ("home", "lost") and len(raw) < cfg.min_photos:
        raise HTTPException(400, f"Бүртгэлд дор хаяж {cfg.min_photos} зураг (өөр өөр өнцгөөс) шаардлагатай")
    profile = _analyze(raw)
    if owner_labels:
        try:
            profile.data = apply_owner_labels(profile.data, json.loads(owner_labels))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, f"owner_labels буруу: {exc}") from exc
    rec = state["registry"].add(
        profile=profile, images=raw, status=status, name=name, owner_name=owner_name, contact=contact,
        location=location, lat=lat, lng=lng, event_at=event_at, notes=notes,
    )
    resp = {"pet": rec.public()}
    # a new found-report is immediately matched against lost pets (and vice versa)
    if status in ("lost", "found"):
        resp["matches"] = _matches_for(rec, top_k=5)
    return resp


@app.get("/pets")
def list_pets(status: str | None = None, species: str | None = None):
    return [p.public() for p in state["registry"].list(status=status, species=species)]


@app.get("/pets/{pid}")
def get_pet(pid: str):
    rec = state["registry"].get(pid)
    if not rec:
        raise HTTPException(404, "Олдсонгүй")
    return rec.public()


class StatusIn(BaseModel):
    status: str
    event_at: str | None = None
    location: str | None = None


@app.patch("/pets/{pid}/status")
def set_status(pid: str, body: StatusIn):
    reg: Registry = state["registry"]
    if not reg.get(pid):
        raise HTTPException(404, "Олдсонгүй")
    try:
        reg.set_status(pid, body.status, body.event_at, body.location)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = reg.get(pid)
    resp = {"pet": rec.public()}
    if body.status in ("lost", "found"):
        resp["matches"] = _matches_for(rec, top_k=5)
    return resp


@app.delete("/pets/{pid}")
def delete_pet(pid: str):
    state["registry"].delete(pid)
    return {"ok": True}


def _matches_for(rec, top_k: int = 10):
    """Match a stored pet against the opposite pool (lost ↔ found)."""
    from petti_vision.pipeline import AnimalProfile

    reg: Registry = state["registry"]
    pool = "found" if rec.status in ("lost", "home") else "lost"
    cands = [p for p in reg.list(status=pool) if p.id != rec.id]
    q = AnimalProfile(data=rec.profile, reid=rec.reid, vlm=rec.vlm)
    return [m.to_json() for m in state["matcher"].search(q, cands, top_k=top_k)]


@app.get("/pets/{pid}/matches")
def pet_matches(pid: str, top_k: int = 10):
    rec = state["registry"].get(pid)
    if not rec:
        raise HTTPException(404, "Олдсонгүй")
    return {"pet": rec.public(), "results": _matches_for(rec, top_k)}


@app.post("/search")
def search(
    images: list[UploadFile] = File(...),
    statuses: str = Form("lost"),
    top_k: int = Form(10),
):
    """Photo(s) of an animal → ranked list of registered pets it could be.

    statuses: comma separated pools to search, e.g. "lost" (finder searching
    owners), "found" (owner searching reports) or "lost,home" (whole registry).
    """
    t0 = time.perf_counter()
    profile = _analyze(_read(images))
    reg: Registry = state["registry"]
    pools = [s.strip() for s in statuses.split(",") if s.strip() in STATUSES]
    cands = [p for s in pools for p in reg.list(status=s)]
    t1 = time.perf_counter()
    matches = state["matcher"].search(profile, cands, top_k=top_k)
    t2 = time.perf_counter()
    q = profile.to_json()
    q["timing_ms"]["match"] = round((t2 - t1) * 1000, 1)
    q["timing_ms"]["total"] = round((t2 - t0) * 1000, 1)
    return {"query": q, "searched": len(cands), "results": [m.to_json() for m in matches]}


if __name__ == "__main__":
    import uvicorn

    # PETTI_RELOAD=1 → auto-restart when code changes (development)
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=os.environ.get("PETTI_RELOAD") == "1")
