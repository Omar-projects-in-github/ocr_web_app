"""Local Chandra OCR inference service."""

from pathlib import Path
from threading import Event, Lock
from typing import Any
import os

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "/models/chandra",
    )
)

_model: Any | None = None
_model_lock = Lock()
_model_ready = Event()

def _load_model() -> Any:
    """ Load the OCR model once, on the first transcription request. """
    global _model

    if _model is not None:
        return _model
    
    with _model_lock:
        if _model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_kwargs: dict[str, Any] = {"local_files_only": True}
            if torch.cuda.is_available():
                model_kwargs.update({"dtype": torch.bfloat16, "device_map": "auto"})
            else:
                model_kwargs["dtype"] = torch.float32

            model = AutoModelForImageTextToText.from_pretrained(
                str(MODEL_PATH),
                **model_kwargs,
            )
            model.processor = AutoProcessor.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
            )
            model.processor.tokenizer.padding_side = "left"
            model.eval()
            _model = model

    return _model


def is_model_ready() -> bool:
    """Return True once the model is loaded and the warm-up OCR has run."""
    return _model_ready.is_set()


def warm_up() -> None:
    """Load the model and run one OCR on a generated image.

    The first inference also initialises CUDA kernels, so running it at
    startup keeps the cold start away from the first real user.
    """
    from PIL import Image, ImageDraw
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem

    image = Image.new("RGB", (640, 160), "white")
    ImageDraw.Draw(image).text((20, 60), "Warm-up OCR 12345", fill = "black")

    batch = [BatchInputItem(image = image, prompt_type = "ocr_layout")]
    generate_hf(batch, _load_model())

    _model_ready.set()


def transcribe_text(image_path: str | Path) -> str:
	"""Transcribe an image with the local Chandra OCR model."""
	image_file = Path(image_path)
	if not image_file.is_file():
		raise FileNotFoundError(f"Image not found: {image_file}")

	from PIL import Image
	from chandra.model.hf import generate_hf
	from chandra.model.schema import BatchInputItem
	from chandra.output import parse_markdown

	with Image.open(image_file) as image:
		batch = [BatchInputItem(image=image.convert("RGB"), prompt_type="ocr_layout")]
		result = generate_hf(batch, _load_model())[0]

	return parse_markdown(result.raw)


if __name__ == "__main__":
	text = transcribe_text(
		# "/home/omar_polat/Worms_data/real_data/normal_version/images/00031.jpg",
        "/home/omar_polat/projects/ocr_project/data/real_non_handwritten_data/images/0161.jpg",
	)
	print(text)
