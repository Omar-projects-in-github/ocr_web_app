"""Local Chandra OCR inference service."""

from pathlib import Path
from threading import Event

_model_ready = Event()

def is_model_ready() -> bool:
    """Return whether the vLLM warm-up request completed."""
    return _model_ready.is_set()


def warm_up() -> None:
    """Load the model and run one OCR on a generated image.

    The first inference also initialises CUDA kernels, so running it at
    startup keeps the cold start away from the first real user.
    """
    from PIL import Image, ImageDraw
    from chandra.model.schema import BatchInputItem
    from chandra.model.vllm import generate_vllm

    image = Image.new("RGB", (640, 160), "white")
    ImageDraw.Draw(image).text(
        (20, 60),
        "Warm-up OCR 12345",
        fill = "black"
    )

    batch = [
        BatchInputItem(
            image = image,
            prompt_type = "ocr_layout"
        )
    ]
    results = generate_vllm(batch)

    if not results or results[0].error:
        raise RuntimeError("vLLM warm-up request failed.")

    _model_ready.set()


def transcribe_text(image_path: str | Path) -> str:
    """Transcribe an image with the local Chandra OCR model."""
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    from PIL import Image
    from chandra.model.schema import BatchInputItem
    from chandra.model.vllm import generate_vllm
    from chandra.output import parse_markdown

    with Image.open(image_file) as image:
        batch = [
            BatchInputItem(
                image=image.convert("RGB"),
                prompt_type="ocr_layout"
            )
        ]

        result = generate_vllm(batch)[0]

    if result.error:
        raise RuntimeError("vLLM OCR request failed.")

    return parse_markdown(result.raw)


if __name__ == "__main__":
    text = transcribe_text(
        # "/home/omar_polat/Worms_data/real_data/normal_version/images/00031.jpg",
        "/home/omar_polat/projects/ocr_project/data/real_non_handwritten_data/images/0161.jpg",
    )
    print(text)
