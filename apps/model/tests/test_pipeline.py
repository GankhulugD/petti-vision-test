"""End-to-end tests with fake numpy backbones (no torch / downloads needed).

    PETTI_FAKE_MODELS=1 pytest -q
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ["PETTI_FAKE_MODELS"] = "1"


def make_animal(seed: int, color=(30, 30, 30), jitter: int = 0) -> bytes:
    """Synthetic 'animal': coloured blob with a seed-specific pattern."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (320, 320), (200, 210, 220))
    d = ImageDraw.Draw(img)
    d.ellipse((60 + jitter, 80, 260 + jitter, 260), fill=color)
    for _ in range(6):
        x, y = rng.integers(80, 230, size=2)
        d.ellipse((x + jitter, y, x + 30 + jitter, y + 30), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def cfg(tmp_path):
    from petti_vision.config import Config

    return Config(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "art", fake_models=True)


def test_fuse_is_distribution():
    from petti_vision.pipeline import fuse

    a = np.array([0.7, 0.2, 0.1])
    b = np.array([0.1, 0.8, 0.1])
    p = fuse([a, b], [0.5, 0.5])
    assert abs(p.sum() - 1) < 1e-6 and p.argmax() == 1


def test_color_estimator_black_on_light():
    from petti_vision.color import coat_colors

    img = Image.open(io.BytesIO(make_animal(1, color=(20, 20, 20))))
    shares = coat_colors(img)
    assert max(shares, key=shares.get) == "black"


def test_analyze_profile_shape(cfg):
    from petti_vision import PettiVision

    v = PettiVision(cfg)
    prof = v.analyze([make_animal(i) for i in range(5)])
    d = prof.to_json()
    for k in ("species", "breed", "size", "age", "colors", "pattern", "coat", "marks", "category", "description_mn", "timing_ms"):
        assert k in d, k
    assert prof.reid.shape[0] == 5
    assert abs(sum(d["species_probs"].values()) - 1) < 1e-3
    assert d["description_mn"].startswith("Төрөл")


def test_registry_and_search_finds_same_animal(cfg):
    from petti_vision import PettiVision
    from petti_vision.matcher import Matcher
    from petti_vision.registry import Registry

    v = PettiVision(cfg)
    reg = Registry(cfg.db_path, cfg.images_dir)
    colors = [(20, 20, 20), (230, 230, 225), (170, 110, 60), (120, 120, 125)]
    for s in range(8):
        imgs = [make_animal(100 + s, colors[s % 4], jitter=j) for j in (0, 4, -4, 8, -8)]
        reg.add(profile=v.analyze(imgs), images=imgs, status="lost", name=f"pet{s}")

    target = 100 + 5
    q = v.analyze([make_animal(target, colors[5 % 4], jitter=6)])
    res = Matcher(cfg).search(q, reg.list(status="lost"), top_k=3)
    assert res and res[0].pet.name == "pet5"
    assert 0 <= res[0].score <= 1

    # registry persists and reloads
    reg2 = Registry(cfg.db_path, cfg.images_dir)
    assert len(reg2.list(status="lost")) == 8
    assert reg2.get(res[0].pet.id).reid.shape == res[0].pet.reid.shape


def test_api_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PETTI_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("PETTI_ARTIFACTS_DIR", str(tmp_path / "a"))
    import importlib

    import petti_vision.config as c

    importlib.reload(c)
    import server

    importlib.reload(server)
    from fastapi.testclient import TestClient

    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        files = [("images", (f"{i}.jpg", make_animal(7, jitter=i), "image/jpeg")) for i in range(5)]
        r = client.post("/pets", files=files, data={"status": "lost", "name": "Бантан", "contact": "99112233"})
        assert r.status_code == 200, r.text
        # too few photos rejected
        r = client.post("/pets", files=files[:2], data={"status": "home"})
        assert r.status_code == 400
        # a finder reports with 1 photo → auto-matched against lost
        r = client.post("/pets", files=[("images", ("q.jpg", make_animal(7, jitter=3), "image/jpeg"))], data={"status": "found"})
        assert r.status_code == 200 and r.json()["matches"][0]["pet"]["name"] == "Бантан"
        r = client.post("/search", files=[("images", ("q.jpg", make_animal(7, jitter=2), "image/jpeg"))], data={"statuses": "lost"})
        body = r.json()
        assert body["results"][0]["pet"]["name"] == "Бантан"
        assert "match" in body["query"]["timing_ms"]
        ex = body["query"]["explain"]  # the UI's "how it works" payload
        assert len(ex["fingerprint"]) == 48 and ex["image"]["w"] == 320
        assert "explain" not in body["results"][0]["pet"]  # never stored with registrations
        pid = body["results"][0]["pet"]["id"]
        assert client.patch(f"/pets/{pid}/status", json={"status": "reunited"}).status_code == 200
        assert client.get("/pets?status=lost").json() == []


def test_head_fusion_keeps_zero_shot_for_unseen_classes(cfg):
    """A head trained without Bankhar photos must not wipe out Bankhar."""
    from petti_vision import PettiVision

    v = PettiVision(cfg)

    class FakeHeads:
        tasks = {"breed": {"multi": False, "alpha": 0.8, "seen": ["a", "b"]}}

        def has(self, t):
            return t in self.tasks

        def predict(self, task, feats):
            return ["a", "b", "c"], np.array([[0.9, 0.1, 0.0]])

    v.heads = FakeHeads()
    zs = np.array([0.2, 0.2, 0.6])  # zero-shot thinks "c" (unseen by head)
    out = v._fuse_task("breed", zs, np.zeros(4), ["a", "b", "c"])
    assert abs(out.sum() - 1) < 1e-6
    assert abs(out[2] - 0.6) < 1e-6  # unseen class keeps its zero-shot mass
    assert out[0] > out[1]  # head reorders the seen classes


def test_confidence_levels_and_pattern_species(cfg):
    from petti_vision import PettiVision
    from petti_vision import taxonomy as T

    v = PettiVision(cfg)
    d = v.analyze([make_animal(3)]).to_json()
    assert d["species"]["level"] in ("high", "medium", "low")
    if d["species"]["key"] == "dog":
        assert d["pattern"]["key"] not in [k for k, sp in T.PATTERN_SPECIES.items() if "dog" not in sp]
    assert all(m["key"] != "short_tail" for m in d["marks"])  # disabled for zero-shot


def test_stale_embeddings_are_refreshed(cfg):
    """After a model change old registrations must be re-embedded, not crash search."""
    from petti_vision import PettiVision
    from petti_vision.matcher import Matcher
    from petti_vision.registry import Registry

    v = PettiVision(cfg)
    reg = Registry(cfg.db_path, cfg.images_dir)
    imgs = [make_animal(42, jitter=j) for j in (0, 3, -3, 6, -6)]
    rec = reg.add(profile=v.analyze(imgs), images=imgs, status="lost", name="old")
    rec.reid = rec.reid[:, :16]  # pretend: stored by an older, smaller model
    q = v.analyze([make_animal(42, jitter=2)])
    assert Matcher(cfg).search(q, reg.list(status="lost")) == []  # skipped, no crash
    assert reg.reembed_stale(v) == 1
    assert Matcher(cfg).search(q, reg.list(status="lost"))[0].pet.name == "old"


def test_explain_payload(cfg):
    import json

    from petti_vision import PettiVision

    v = PettiVision(cfg)
    d = v.analyze([make_animal(i) for i in range(3)], explain=True).to_json()
    ex = d["explain"]
    assert 0 <= ex["photo_index"] < 3
    assert ex["attention"] is None or len(ex["attention"]["values"]) == ex["attention"]["grid"] ** 2
    assert max(abs(x) for x in ex["fingerprint"]) <= 1.0 + 1e-6
    json.dumps(d)  # serialisable
    assert "explain" not in v.analyze([make_animal(1)]).to_json()


def test_feedback_consent(tmp_path, monkeypatch):
    monkeypatch.setenv("PETTI_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("PETTI_ARTIFACTS_DIR", str(tmp_path / "a"))
    import importlib
    import json

    import petti_vision.config as c

    importlib.reload(c)
    import server

    importlib.reload(server)
    from fastapi.testclient import TestClient

    with TestClient(server.app) as client:
        img = [("images", ("a.jpg", make_animal(3), "image/jpeg"))]
        r = client.post("/feedback", files=img, data={"corrections": json.dumps({"breed": "pug", "evil": 1}), "consent": "false"})
        assert r.status_code == 200 and r.json()["stored_photos"] is False
        fb = server.cfg.data_dir / "feedback"
        before = set((fb / "images").iterdir())
        assert set((fb / "images").iterdir()) == before
        rec = json.loads((fb / "feedback.jsonl").read_text().splitlines()[-1])
        assert rec["corrections"] == {"breed": "pug"}
        r = client.post("/feedback", files=img, data={"corrections": "{}", "consent": "true"})
        sha = json.loads((fb / "feedback.jsonl").read_text().splitlines()[-1])["images"][0]
        assert (fb / "images" / f"{sha}.jpg").exists()
        assert client.delete(f"/feedback/{sha}").json()["deleted"] is True
        assert client.post("/feedback", files=img, data={"corrections": "[1]"}).status_code == 400
