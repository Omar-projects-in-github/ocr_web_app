"""Evaluate completed OCR benchmark artifacts and create reports."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.error_analysis import write_error_analysis
from evaluation.metrics import evaluate_records
from evaluation.reports import (
    write_csv_report,
    write_json_report,
)

DEFAULT_BENCHMARK_ROOT = Path("artifacts/benchmarks")
DEFAULT_OUTPUT_ROOT = Path("artifacts/evaluation")
DEFAULT_ERROR_OUTPUT_ROOT = Path("artifacts/error_analysis")


def load_benchmark_artifact(
    benchmark_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate one benchmark artifact."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Benchmark artifact not found: {benchmark_path}"
        )

    with benchmark_path.open("r", encoding = "utf-8") as file:
        artifact = json.load(file)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Benchmark must use metadata/records format: "
            f"{benchmark_path}"
        )

    metadata = artifact.get("metadata")
    records = artifact.get("records")

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Missing or invalid metadata: {benchmark_path}"
        )

    if not isinstance(records, list):
        raise ValueError(
            f"Missing or invalid records: {benchmark_path}"
        )

    return metadata, records

def merge_record_with_metrics(
    benchmark_record: dict[str, Any],
    metric_record: dict[str, Any],
) -> dict[str, Any]:
    """Merge benchmark information with calculated metrics."""
    merged = deepcopy(benchmark_record)

    for key in (
        "character_edit_distance",
        "reference_character_count",
        "cer",
        "word_edit_distance",
        "reference_word_count",
        "wer",
        "exact_match",
    ):
        merged[key] = metric_record.get(key)

    return merged

def build_evaluation_report(
    benchmark_path: Path,
) -> dict[str, Any]:
    """Evaluate one benchmark artifact and build its report."""
    metadata, benchmark_records = load_benchmark_artifact(
        benchmark_path
    )

    evaluated = evaluate_records(benchmark_records)

    metric_records = evaluated["records"]
    metrics_by_image = {
        metric_record["image_path"]: metric_record
        for metric_record in metric_records
        if metric_record.get("image_path")
    }

    report_records: list[dict[str, Any]] = []

    for benchmark_record in benchmark_records:
        if benchmark_record.get("error"):
            continue
        
        image_path = benchmark_record.get("image_path")
        metric_record = metrics_by_image.get(image_path)

        if metric_record is None:
            continue
        
        report_records.append(
            merge_record_with_metrics(
                benchmark_record,
                metric_record,
            )
        )

    return {
        "benchmark": metadata,
        "summary": evaluated["summary"],
        "records": report_records,
    }

def evaluate_one_artifact(
    benchmark_path: Path,
    output_root: Path,
    error_output_root: Path | None = None,
) -> tuple[Path, Path]:
    """Evaluated one artifact and write JSON and CSV reports."""
    benchmark_path = benchmark_path.resolve()
    report = build_evaluation_report(benchmark_path)

    metadata = report["benchmark"]
    dataset_name = str(
        metadata.get("dataset")
        or benchmark_path.stem
    )

    json_output = output_root / f"{dataset_name}.json"
    csv_output = output_root / f"{dataset_name}.csv"

    if json_output.resolve() == benchmark_path:
        raise ValueError(
            "Evaluation JSON output cannot replace the benchmark artifact: "
            f"{benchmark_path}"
        )

    write_json_report(json_output, report)
    write_csv_report(csv_output, report["records"])

    if error_output_root is not None:
        write_error_analysis(
            error_output_root,
            report,
        )

    return json_output, csv_output

def find_benchmark_artifacts(
    benchmark_root: Path,
) -> list[Path]:
    """Find completed benchmark JSON files."""
    if not benchmark_root.is_dir():
        raise FileNotFoundError(
            f"Benchmark directory not found: {benchmark_root}"
        )

    artifacts = sorted(benchmark_root.glob("*.json"))

    if not artifacts:
        raise ValueError(
            f"No benchmark JSON files found in {benchmark_root}"
        )

    return artifacts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Evaluate OCR benchmark artifacts.",
    )
    parser.add_argument(
        "--benchmark",
        type = Path,
        help = (
            "Evaluate one benchmark JSON file. "
            "If omitted, evaluate every JSON file in --benchmark-root."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        type = Path,
        default = DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type = Path,
        default = DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--error-output-root",
        type = Path,
        default = DEFAULT_ERROR_OUTPUT_ROOT,
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.benchmark:
        benchmark_paths = [args.benchmark]
    else:
        benchmark_paths = find_benchmark_artifacts(
            args.benchmark_root
        )
    
    for benchmark_path in benchmark_paths:
        json_output, csv_output = evaluate_one_artifact(
            benchmark_path,
            args.output_root,
            args.error_output_root,
        )
        print(f"JSON report: {json_output}")
        print(f"CSV report: {csv_output}")

if __name__ == "__main__":
    main()