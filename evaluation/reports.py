"""JSON and CSV report generation for OCR evaluation."""

import csv
import json
from pathlib import Path
from typing import Any

def write_json_report(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    """Atomically write one JSON evaluation report."""
    output_path.parent.mkdir(parents = True, exist_ok = True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding = "utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii = False,
            indent = 2,
        )

    temporary_path.replace(output_path)

def write_csv_report(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write per-image metrics and predictions to CSV."""
    if not records:
        output_path.parent.mkdir(parents = True, exist_ok = True)
        output_path.write_text("", encoding = "utf-8")
        return
    
    fieldnames = [
        "image_path",
        "ground_truth",
        "prediction_text",
        "prediction_raw",
        "latency_seconds",
        "language",
        "dataset",
        "character_edit_distance",
        "reference_character_count",
        "cer",
        "word_edit_distance",
        "reference_word_count",
        "wer",
        "exact_match",
    ]

    output_path.parent.mkdir(parents = True, exist_ok = True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding = "utf-8",
        newline = "",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames = fieldnames,
            extrasaction = "ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    temporary_path.replace(output_path)