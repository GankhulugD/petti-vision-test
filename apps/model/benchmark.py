#!/usr/bin/env python3
"""Latency benchmark.

    python3 benchmark.py path/to/dog.jpg [--n 20]
"""

import argparse
import statistics
import time

from petti_vision import Config, PettiVision

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--photos", type=int, default=1, help="photos per call (registration = 5)")
a = ap.parse_args()

v = PettiVision(Config())
v.warmup()
raw = open(a.image, "rb").read()
times, last = [], None
for _ in range(a.n):
    t = time.perf_counter()
    last = v.analyze([raw] * a.photos)
    times.append((time.perf_counter() - t) * 1000)
print(last.data["description_mn"])
print(last.data["timing_ms"])
print(f"{a.photos} photo(s): median {statistics.median(times):.0f} ms, p90 {sorted(times)[int(0.9 * len(times))]:.0f} ms")
