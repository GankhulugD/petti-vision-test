"""Free Hugging Face Gradio Space entry.

The Space SDK already serves `demo` on port 7860. Starting uvicorn as well
makes the process crash with "address already in use".
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "apps" / "model"))

os.environ.setdefault("PETTI_DEVICE", "cpu")
os.environ.setdefault("PETTI_HALF", "0")
os.environ.setdefault("PETTI_DATA_DIR", "/tmp/petti-data")

import gradio as gr
from PIL import Image

from server import state

_load_error: list[str] = []


def _load_models() -> None:
    from petti_vision import Config, PettiVision
    from petti_vision.matcher import Matcher
    from petti_vision.registry import Registry

    try:
        cfg = Config()
        vision = PettiVision(cfg)
        vision.warmup()
        state["vision"] = vision
        state["registry"] = Registry(cfg.db_path, cfg.images_dir)
        state["matcher"] = Matcher(cfg)
    except Exception as exc:
        _load_error.append(str(exc))


threading.Thread(target=_load_models, daemon=True).start()


def recognize(image):
    if _load_error:
        return f"Model ачаалагдсангүй: {_load_error[0]}"
    if "vision" not in state:
        return "Model ачааллаж байна (эхний удаа хэдэн минут). Дахин «Шинжлэх» дарна уу."
    if image is None:
        return "Зураг оруулна уу."
    img = Image.fromarray(image).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    profile = state["vision"].analyze([buf.getvalue()])
    return profile.to_json().get("description_mn") or "Үр дүн гарсангүй."


with gr.Blocks(title="Petti Vision") as demo:
    gr.Markdown("# Petti Vision\nАмьтны зургаар таних. Эхний ачаалалт хэдэн минут болно.")
    with gr.Row():
        image = gr.Image(type="numpy", label="Зураг")
        result = gr.Textbox(label="Үр дүн", lines=14)
    gr.Button("Шинжлэх").click(recognize, inputs=image, outputs=result)


if __name__ == "__main__":
    demo.launch()
