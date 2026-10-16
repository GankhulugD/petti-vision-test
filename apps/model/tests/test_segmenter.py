"""Segmentation head helpers and the no-animal decision. No weights."""

from __future__ import annotations

import numpy as np

from petti_vision.segmenter import apply_no_animal, decide_no_animal, foreground_prob, instances_from_prob, label_components


def test_components_find_two_blobs():
    m = np.zeros((8, 8), dtype=bool)
    m[1:3, 1:3] = True
    m[5:7, 5:7] = True
    lab = label_components(m)
    assert len(set(lab.ravel()) - {0}) == 2


def test_fragments_of_one_animal_merge_and_distant_ones_stay():
    close = np.zeros((16, 16), dtype=np.float32)
    close[2:5, 2:5] = 0.9
    close[6:9, 6:9] = 0.9
    assert len(instances_from_prob(close, (160, 160), min_frac=0.01)) == 1
    far = np.zeros((16, 16), dtype=np.float32)
    far[0:4, 0:4] = 0.9
    far[12:16, 12:16] = 0.9
    assert len(instances_from_prob(far, (160, 160), min_frac=0.01)) == 2


def test_instances_skip_specks_and_keep_a_blob():
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[2:10, 3:12] = 0.9
    prob[0, 0] = 0.9  # one pixel, below the area floor
    dets = instances_from_prob(prob, (320, 240), thr=0.5, min_frac=0.01)
    assert len(dets) == 1
    assert dets[0].species == "pet"
    mask = dets[0].mask((320, 240))
    assert mask is not None and mask.shape == (240, 320) and mask.any()


def test_foreground_head_is_finite():
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((256, 384)).astype(np.float32)
    head = {
        "w1": rng.standard_normal((8, 384)).astype(np.float32) * 0.02,
        "b1": np.zeros(8, dtype=np.float32),
        "w2": rng.standard_normal((1, 8, 3, 3)).astype(np.float32) * 0.02,
        "b2": np.zeros(1, dtype=np.float32),
    }
    prob = foreground_prob(tokens, head)
    assert prob.shape == (16, 16)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1


def test_no_animal_gate_rejects_a_car_and_keeps_a_dog():
    assert decide_no_animal(0.0, 0.2, 0.8, False)
    assert not decide_no_animal(0.3, 0.9, 0.1, True)
    data = {"quality": {"warnings": [], "photos": 1}, "breed": {"top": []}, "description_mn": "Төрөл: Нохой"}
    apply_no_animal(data)
    assert data["quality"]["no_animal"] is True
    assert data["breed"] is None
    assert "нохой эсвэл муур" in data["description_mn"]
