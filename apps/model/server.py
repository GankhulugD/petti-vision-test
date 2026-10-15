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

# PETTI_PORT wins so a combined container can give $PORT to Next.js
# and keep the model on 8765. Alone, $PORT (Render/HF) or 8765 is used.
PORT = int(os.environ.get("PETTI_PORT") or os.environ.get("PORT") or "8765")
MAX_FILES = 12
MAX_BYTES = 15 * 1024 * 1024

cfg = Config()
state: dict = {}
MODEL_VERSION = "0.4.0"  # bump when inference behaviour changes


def _code_hash() -> str:
    import hashlib
    import pathlib

    h = hashlib.md5()
    for f in sorted(pathlib.Path(__file__).resolve().parent.joinpath("petti_vision").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


CODE_HASH = _code_hash()  # which inference code is actually loaded
# enables /dev/* (training jobs, hot reload) – localhost only. Turn on with
# PETTI_DEV=1 or by creating an (untracked) file apps/model/.dev
DEV = os.environ.get("PETTI_DEV") == "1" or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dev"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    if cfg.fake_models:
        log.warning("!!! PETTI_FAKE_MODELS=1 — random test models, predictions are meaningless !!!")
    vision = PettiVision(cfg)
    vision.warmup()  # first call compiles kernels; do it before traffic
    state["vision"] = vision
    state["registry"] = Registry(cfg.db_path, cfg.images_dir)
    n_stale = state["registry"].reembed_stale(state["vision"])  # model changed → refresh old embeddings
    if n_stale:
        log.info("re-embedded %d pets for the current model", n_stale)
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
        "model_version": MODEL_VERSION,
        "code": CODE_HASH,
        "breed_temperature": getattr(v, "breed_temperature", None),
        "dev": DEV,
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


# --------------------------------------------------------------------------- #
# Dev-only: run training jobs on the machine that serves the model and hot-reload
# the result. Enabled with PETTI_DEV=1, accepted only from localhost.
# --------------------------------------------------------------------------- #
if DEV:
    import subprocess
    import sys
    import uuid

    from fastapi import Request

    JOBS: dict[str, dict] = {}
    SCRIPTS = {"auto_train", "train_heads", "train_reid", "evaluate", "prepare_data", "reindex", "fetch_identity_set"}
    JOB_DIR = cfg.data_dir / "jobs"

    def _local(request: Request):
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "dev endpoints are localhost-only")

    class JobIn(BaseModel):
        script: str
        args: list[str] = []
        env: dict[str, str] = {}  # only PETTI_* overrides, e.g. {"PETTI_REID_MODEL": "facebook/dinov2-base"}

    @app.post("/dev/jobs")
    def dev_start(body: JobIn, request: Request):
        _local(request)
        if body.script not in SCRIPTS:
            raise HTTPException(400, f"script must be one of {sorted(SCRIPTS)}")
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        jid = uuid.uuid4().hex[:8]
        log_path = JOB_DIR / f"{jid}.log"
        here = os.path.dirname(os.path.abspath(__file__))
        env = {**os.environ, **{k: v for k, v in body.env.items() if k.startswith("PETTI_")}}
        env.pop("PETTI_DEV", None)
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(here, "training", f"{body.script}.py"), *body.args],
            cwd=here, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
            start_new_session=True,  # survives server reloads
        )
        (JOB_DIR / f"{jid}.pid").write_text(str(proc.pid))
        JOBS[jid] = {"proc": proc, "log": log_path, "script": body.script, "started": time.time()}
        return {"id": jid}

    @app.get("/dev/jobs/{jid}")
    def dev_status(jid: str, request: Request, tail: int = 60):
        _local(request)
        log_path = JOB_DIR / f"{jid}.log"
        if not log_path.exists():
            raise HTTPException(404, "no such job")
        j = JOBS.get(jid)
        if j:
            rc = j["proc"].poll()
        else:  # server reloaded since: is the process still alive?
            try:
                os.kill(int((JOB_DIR / f"{jid}.pid").read_text()), 0)
                rc = None
            except (OSError, ValueError):
                rc = "finished (exit code unknown)"
        lines = log_path.read_text(errors="replace").splitlines()
        # tqdm writes \r progress; keep the last state of each line
        lines = [ln.split("\r")[-1] for ln in lines]
        return {"running": rc is None, "returncode": rc, "lines": len(lines), "tail": lines[-tail:]}

    @app.delete("/dev/jobs/{jid}")
    def dev_kill(jid: str, request: Request):
        _local(request)
        import signal

        try:
            pid = int((JOB_DIR / f"{jid}.pid").read_text())
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # job runs in its own session
            return {"killed": pid}
        except (OSError, ValueError) as exc:
            raise HTTPException(404, f"not running: {exc}") from exc

    @app.post("/dev/reload")
    def dev_reload(request: Request):
        """Reload heads / calibration / re-ID projection without restarting."""
        _local(request)
        v = PettiVision(cfg)
        state["vision"] = v
        state["matcher"] = Matcher(cfg)
        return health()


if __name__ == "__main__":
    import uvicorn

    # PETTI_RELOAD=1 → auto-restart when code changes (development)
    here = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run(
        "server:app", host="0.0.0.0", port=PORT, reload=os.environ.get("PETTI_RELOAD") == "1",
        # watch only code: data/ holds ~13k dataset images, scanning it burns CPU
        reload_dirs=[here], reload_includes=["*.py"],
        reload_excludes=["data/*", "artifacts/*", ".venv/*", "training/*"],
    )
