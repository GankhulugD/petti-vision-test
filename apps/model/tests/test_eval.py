"""Eval metrics and the regression gate. No model weights."""

from __future__ import annotations

import numpy as np

from training.gate import compare
from training.metrics import expected_calibration_error, retrieval


def test_ece_perfect_is_zero():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(conf, correct) == 0.0


def test_retrieval_separates_identities():
    emb = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
        [0.7, 0.7],  # distractor
    ], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.array(["a", "a", "b", "b", "d"])
    dist = np.array([False, False, False, False, True])
    m = retrieval(emb, ids, dist)
    assert m["queries"] == 4
    assert m["R@1"] == 1.0
    assert m["false_match_rate"] == 0.0


def test_gate_fails_on_one_point_drop_and_latency():
    base = {"breed": {"dog": {"top1": 0.90}}, "latency": {"p50_ms": 200, "p95_ms": 300}}
    cand = {"breed": {"dog": {"top1": 0.88}}, "latency": {"p50_ms": 250, "p95_ms": 300}}
    fails = compare(base, cand)
    assert any("breed.dog.top1" in f for f in fails)
    assert any("latency.p50_ms" in f for f in fails)


def test_gate_ignores_unmeasured_and_passes_within_tolerance():
    base = {"breed": {"dog": {"top1": 0.90}, "cat": {"top1": None}}, "reid": {"R@1": 0.87}}
    cand = {"breed": {"dog": {"top1": 0.895}}, "reid": {"R@1": 0.87}}
    assert compare(base, cand) == []


def test_gate_fails_when_candidate_drops_a_measured_metric():
    base = {"reid": {"R@1": 0.87}}
    cand = {"reid": {}}
    fails = compare(base, cand)
    assert fails and "no measurement" in fails[0]
