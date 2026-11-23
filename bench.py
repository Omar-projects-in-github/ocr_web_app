"""Run a resumable OCR benchmark against a CSV image dataset."""

import argparse
import csv
import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_LANGUAGE = "kk"

logger = logging.getLogger(__name__)


def html_to_text(text: str) -> str:
    """Remove common HTML tags while preserving explicit line breaks."""
    if not text:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(table|tr|td|th|tbody|thead|div|p|span|b|i|strong|em|ul|ol|li)[^>]*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def filter_image_description(text: str) -> str:
    """Remove generated image descriptions from markdown OCR output."""
    if not text:
        return ""

    text = re.sub(r"!\[.*?(?=\n#+|\n\n|$)", "", text, flags=re.DOTALL)
    text = html_to_text(text)

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    common_english_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "with",
    }

    kept_paragraphs: list[str] = []
    for paragraph in paragraphs:
        words = re.findall(r"[A-Za-z]+", paragraph.lower())
        if re.search(r"[^\x00-\x7F]", paragraph):
            kept_paragraphs.append(paragraph)
            continue
        if len(words) < 20:
            kept_paragraphs.append(paragraph)
            continue

        english_ratio = sum(
            word in common_english_words for word in words
        ) / max(len(words), 1)
        if english_ratio <= 0.12:
            kept_paragraphs.append(paragraph)

    return "\n\n".join(kept_paragraphs).strip()


def markdown_to_plain_text(text: str) -> str:
    """Convert the model's markdown response into comparable plain text."""
    if not text:
        return ""

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = text.replace("\\_", "_")
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset_rows(dataset_root: Path) -> list[dict[str, str]]:
    """Load image paths and ground truth from dataset_root/data.csv."""
    csv_path = dataset_root / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"image_path", "text"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset manifest is missing columns: {missing}")

        for row in reader:
            relative_path = (row.get("image_path") or "").strip()
            if not relative_path:
                continue

            image_path = Path(relative_path)
            if image_path.is_absolute():
                resolved_path = image_path
            else:
                resolved_path = dataset_root / image_path

            if resolved_path.suffix.lower() not in IMAGE_EXTENSIONS:
                logger.warning("Skipping unsupported image: %s", resolved_path)
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "resolved_path": str(resolved_path),
                    "ground_truth": row.get("text") or "",
                }
            )

    if not rows:
        raise ValueError(f"No supported images found in {csv_path}")
    return rows


def find_local_api_base() -> str:
    """Return the vLLM API base configured in the project environment."""
    configured_base = os.getenv("VLLM_API_BASE")

    if not configured_base:
        raise RuntimeError(
            "VLLM_API_BASE is not configured. "
            "Add it to the project .env file."
        )
    
    return configured_base.rstrip("/")


def configure_inference_environment(api_base: str) -> None:
    """Configure only non-secret inference settings loaded from .env."""
    api_key = os.getenv("VLLM_API_KEY")
    model_name = os.getenv("VLLM_MODEL_NAME")

    if not api_key:
        raise RuntimeError(
            "VLLM_API_KEY is not configured in the project .env file."
        )

    if not model_name:
        raise RuntimeError(
            "VLLM_MODEL_NAME is not configured in the project .env file."
        )

    os.environ["VLLM_API_BASE"] = api_base
    os.environ["VLLM_API_KEY"] = api_key
    os.environ["VLLM_MODEL_NAME"] = model_name


def infer_image(manager: Any, image_path: Path) -> dict[str, Any]:
    """Run OCR and return predictions with model inference latency."""
    from PIL import Image
    from chandra.model.schema import BatchInputItem

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        started_at = time.perf_counter()
        result = manager.generate(
            [
                BatchInputItem(
                    image=rgb_image,
                    prompt_type="ocr_layout",
                )
            ]
        )[0]
        latency_seconds = time.perf_counter() - started_at

    raw_markdown = getattr(result, "markdown", "") or ""
    filtered_markdown = filter_image_description(raw_markdown)

    return {
        "prediction_raw": raw_markdown,
        "prediction_text": markdown_to_plain_text(filtered_markdown),
        "latency_seconds": latency_seconds,
    }

def build_run_metadata(
    dataset_root: Path,
    dataset_size: int,
    language: str,
    started_at: str,
) -> dict[str, Any]:
    """Build non-sensitive metadata describing one benchmark run."""
    return {
        "dataset": dataset_root.name,
        "language": language,
        "dataset_size": dataset_size,
        "inference_backend": "vllm",
        "prompt_type": "ocr_layout",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started_at,
    }


def load_existing_result(
    output_path: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load metadata and records from a resumable benchmark artifact."""
    if not output_path.is_file():
        return {}, []

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    # Legacy format: a bare list of records with no metadata.
    if isinstance(payload, list):
        return {}, payload

    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        records = payload.get("records", [])

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid metadata in benchmark output: {output_path}"
            )
        if not isinstance(records, list):
            raise ValueError(
                f"Invalid records in benchmark output: {output_path}"
            )
        return metadata, records

    raise ValueError(f"Unexpected benchmark output format: {output_path}")


def write_result(
    output_path: Path, 
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
 ) -> None:
    """Atomically write benchmark metadata and records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    payload = {
        "metadata": metadata,
        "records": records,
    }

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def default_output_path(dataset_root: Path) -> Path:
    """Return a repository-local output path without machine-specific paths."""
    dataset_name = dataset_root.name or "ocr_dataset"
    return Path("artifacts") / "benchmarks" / f"{dataset_name}.json"


def build_benchmark(
    dataset_root: Path,
    output_path: Path,
    language: str,
) -> None:
    """Run a resumable benchmark for one dataset variant."""
    api_base = find_local_api_base()
    configure_inference_environment(api_base)

    from chandra.model import InferenceManager
    from tqdm import tqdm

    dataset_rows = load_dataset_rows(dataset_root)
    started_at = datetime.now(timezone.utc).isoformat()

    existing_metadata, records = load_existing_result(output_path)

    if existing_metadata:
        metadata = existing_metadata
    else:
        metadata = build_run_metadata(
            dataset_root = dataset_root,
            dataset_size = len(dataset_rows),
            language = language,
            started_at = started_at
        )

    processed_paths = {
        record.get("image_path")
        for record in records
        if record.get("image_path")
    }

    manager = InferenceManager(method="vllm")

    for item in tqdm(dataset_rows, desc=dataset_root.name, unit="image"):
        image_path = item["image_path"]
        if image_path in processed_paths:
            continue

        record: dict[str, Any] = {
            "image_path": image_path,
            "ground_truth": item["ground_truth"],
            "prediction_raw": "",
            "prediction_text": "",
            "latency_seconds": None,
            "language": language,
            "dataset": dataset_root.name,
        }
        try:
            record.update(
                infer_image(
                    manager,
                    Path(item["resolved_path"]),
                )
            )
        except Exception as error:
            logger.exception("OCR failed for %s", image_path)
            record["error"] = str(error)

        records.append(record)
        processed_paths.add(image_path)
        write_result(output_path, metadata, records)

    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["completed_records"] = len(records)
    metadata["failed_records"] = sum(
        1 for record in records if record.get("error")
    )

    latencies = [
        record["latency_seconds"]
        for record in records
        if isinstance(record.get("latency_seconds"), (int, float))
    ]
    if latencies:
        metadata["mean_latency_seconds"] = sum(latencies) / len(latencies)
        metadata["measured_records"] = len(latencies)

    write_result(output_path, metadata, records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OCR benchmark for one dataset variant.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset directory containing data.csv and its images directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to artifacts/benchmarks/<dataset>.json.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="BCP-47 language code stored in each record.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    args = parse_args()
    output_path = args.output or default_output_path(args.dataset_root)
    build_benchmark(
        dataset_root=args.dataset_root,
        output_path=output_path,
        language=args.language,
    )


if __name__ == "__main__":
    main()