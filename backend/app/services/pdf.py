"""PDF-to-text service."""

import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import pymupdf

from app.services.ocr import transcribe_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_PDF_PAGES = 30
RENDER_SCALE = 1.5
MAX_RENDER_PIXELS = 16_000_000

def transcribe_pdf(
    pdf_path: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
    """Render each PDF page as an image and OCR the pages in order."""
    document_path = Path(pdf_path)

    if not document_path.is_file():
        raise FileNotFoundError(f"PDF not found: {document_path}")

    page_texts: list[str] = []

    with pymupdf.open(document_path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF does not contain any pages.")

        if document.page_count > MAX_PDF_PAGES:
            raise ValueError(
                f"The PDF contains {document.page_count} pages. "
                f"The maximum allowed is {MAX_PDF_PAGES}."
            )

        total_pages = document.page_count

        for page_number in range(total_pages):
            current_page = page_number + 1

            if progress_callback is not None:
                progress_callback(current_page, total_pages)

            page = document.load_page(page_number)

            print(
                f"Processing PDF page {current_page}/{total_pages}",
                flush = True,
            )

            logger.info(
                "Processing PDF page %s/%s",
                page_number + 1,
                document.page_count,
            )

            page_width = page.rect.width * RENDER_SCALE
            page_height = page.rect.height * RENDER_SCALE
            render_scale = RENDER_SCALE

            rendered_pixels = page_width * page_height

            if rendered_pixels > MAX_RENDER_PIXELS:
                reduction_factor = (
                    MAX_RENDER_PIXELS / rendered_pixels
                ) ** 0.5
                render_scale *= reduction_factor

                logger.warning(
                    "Reducing render scale for page %s to %.2f",
                    page_number + 1,
                    render_scale,
                )

            # Render the page at 2x resolution to improve OCR readibility.
            pixmap = page.get_pixmap(
                matrix = pymupdf.Matrix(
                    render_scale,
                    render_scale,
                ),
                alpha = False,
            )

            with NamedTemporaryFile(suffix = ".png") as image_file:
                image_file.write(pixmap.tobytes("png"))
                image_file.flush()

                page_text = transcribe_text(image_file.name)

            page_texts.append(page_text)

    return "\n\n".join(page_texts)