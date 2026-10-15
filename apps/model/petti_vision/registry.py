"""Pet registry: SQLite metadata + in-memory vector index of photo embeddings.

For the demo SQLite + numpy is plenty (100k photos ≈ 10 ms brute-force on CPU).
In production swap `VectorIndex` for pgvector / Qdrant / FAISS – the interface
(add / search / remove) is all the matcher needs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATUSES = ("home", "lost", "found", "reunited")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pets (
    id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'home',
    species TEXT,
    category TEXT,
    owner_name TEXT,
    contact TEXT,
    location TEXT,
    lat REAL, lng REAL,
    event_at TEXT,
    notes TEXT,
    profile_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pets_status ON pets(status, species);
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    pet_id TEXT NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    path TEXT NOT NULL,
    reid BLOB NOT NULL,
    vlm BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_pet ON photos(pet_id);
"""


@dataclass
class PetRecord:
    id: str
    name: str | None
    status: str
    species: str
    category: str
    owner_name: str | None
    contact: str | None
    location: str | None
    lat: float | None
    lng: float | None
    event_at: str | None
    notes: str | None
    profile: dict
    created_at: float
    photo_paths: list[str]
    reid: np.ndarray  # (P, D)
    vlm: np.ndarray  # (P, D)

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "species": self.species,
            "category": self.category,
            "owner_name": self.owner_name,
            "contact": self.contact,
            "location": self.location,
            "lat": self.lat,
            "lng": self.lng,
            "event_at": self.event_at,
            "notes": self.notes,
            "profile": self.profile,
            "created_at": self.created_at,
            "photos": [f"/images/{p}" for p in self.photo_paths],
        }


class Registry:
    def __init__(self, db_path: Path, images_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = images_dir
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.pets: dict[str, PetRecord] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        rows = self.db.execute("SELECT * FROM pets").fetchall()
        cols = [c[0] for c in self.db.execute("SELECT * FROM pets LIMIT 0").description]
        for row in rows:
            r = dict(zip(cols, row))
            ph = self.db.execute("SELECT path, reid, vlm FROM photos WHERE pet_id=? ORDER BY idx", (r["id"],)).fetchall()
            self.pets[r["id"]] = PetRecord(
                id=r["id"], name=r["name"], status=r["status"], species=r["species"], category=r["category"],
                owner_name=r["owner_name"], contact=r["contact"], location=r["location"], lat=r["lat"], lng=r["lng"],
                event_at=r["event_at"], notes=r["notes"], profile=json.loads(r["profile_json"]), created_at=r["created_at"],
                photo_paths=[p[0] for p in ph],
                reid=np.stack([np.frombuffer(p[1], dtype=np.float32) for p in ph]) if ph else np.zeros((0, 1), np.float32),
                vlm=np.stack([np.frombuffer(p[2], dtype=np.float32) for p in ph]) if ph else np.zeros((0, 1), np.float32),
            )

    # ------------------------------------------------------------------ #
    def add(self, *, profile, images: list[bytes], status: str, name=None, owner_name=None, contact=None,
            location=None, lat=None, lng=None, event_at=None, notes=None) -> PetRecord:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        pid = uuid.uuid4().hex[:12]
        now = time.time()
        paths = []
        pet_dir = self.images_dir / pid
        pet_dir.mkdir(parents=True, exist_ok=True)
        for i, raw in enumerate(images):
            p = pet_dir / f"{i}.jpg"
            _save_jpeg(raw, p)
            paths.append(f"{pid}/{i}.jpg")
        data = profile.to_json()
        with self.lock:
            self.db.execute(
                "INSERT INTO pets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, name, status, data["species"]["key"], data["category"], owner_name, contact, location, lat, lng,
                 event_at, notes, json.dumps(data, ensure_ascii=False), now, now),
            )
            for i, path in enumerate(paths):
                self.db.execute(
                    "INSERT INTO photos VALUES (?,?,?,?,?,?)",
                    (uuid.uuid4().hex[:12], pid, i, path, profile.reid[i].astype(np.float32).tobytes(), profile.vlm[i].astype(np.float32).tobytes()),
                )
            self.db.commit()
            rec = PetRecord(pid, name, status, data["species"]["key"], data["category"], owner_name, contact, location, lat, lng,
                            event_at, notes, data, now, paths, profile.reid.astype(np.float32), profile.vlm.astype(np.float32))
            self.pets[pid] = rec
        return rec

    def set_status(self, pid: str, status: str, event_at: str | None = None, location: str | None = None):
        if status not in STATUSES:
            raise ValueError(status)
        with self.lock:
            self.db.execute(
                "UPDATE pets SET status=?, event_at=COALESCE(?, event_at), location=COALESCE(?, location), updated_at=? WHERE id=?",
                (status, event_at, location, time.time(), pid),
            )
            self.db.commit()
            rec = self.pets[pid]
            rec.status = status
            rec.event_at = event_at or rec.event_at
            rec.location = location or rec.location

    def delete(self, pid: str):
        with self.lock:
            self.db.execute("DELETE FROM photos WHERE pet_id=?", (pid,))
            self.db.execute("DELETE FROM pets WHERE id=?", (pid,))
            self.db.commit()
            self.pets.pop(pid, None)

    def list(self, status: str | None = None, species: str | None = None) -> list[PetRecord]:
        out = [p for p in self.pets.values() if (status is None or p.status == status) and (species is None or p.species == species)]
        return sorted(out, key=lambda p: -p.created_at)

    def get(self, pid: str) -> PetRecord | None:
        return self.pets.get(pid)


def _save_jpeg(raw: bytes, path: Path, max_side: int = 1024):
    from .backbones import load_image

    img = load_image(raw, max_side=max_side)
    img.save(path, "JPEG", quality=88)
