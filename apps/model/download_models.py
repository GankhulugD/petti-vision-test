#!/usr/bin/env python3
"""Pre-download all weights + build the zero-shot text cache (run once, ~1-2 GB).

    python3 download_models.py
"""

from petti_vision import Config, PettiVision

if __name__ == "__main__":
    cfg = Config()
    print(f"detector={cfg.detector_weights} reid={cfg.reid_model} vlm={cfg.vlm_model}")
    v = PettiVision(cfg)
    v.warmup()
    print(f"ok — loaded in {v.load_seconds:.1f}s, text cache at {cfg.text_cache_path}")
