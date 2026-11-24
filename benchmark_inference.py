"""Run reproducible sequential or concurrent OCR benchmarks."""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarking.gpu import read_gpu_metrics
from benchmarking.latency import (
    run_concurrent,
    run_sequential,
    summarize_latencies,
)
from benchmarking.report import write_report

DEFAULT_OUTPUT_ROOT = Path("artifacts/benchmarks/performance")

def load_image_paths(
    benchmark_path: Path,
    dataset_root: Path,
    limit: int | None,
) -> list[Path]:
    """Load image paths from a completed benchmark artifact."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Benchmark artifact not found: {benchmark_path}"
        )

    with benchmark_path.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"Benchmark artifact has no valid records: {benchmark_path}"
        )

    image_paths: list[Path] = []

    for record in records:
        if record.get("error"):
            continue

        image_path = record.get("image_path")
        if not isinstance(image_path, str):
            continue

        resolved_path = dataset_root / image_path
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Image from benchmark was not found: {resolved_path}"
            )

        image_paths.append(resolved_path)

    if limit is not None:
        image_paths = image_paths[:limit]

    return image_paths

def build_metadata(
    benchmark_path: Path,
    image_count: int,
    mode: str,
    concurrency: int,
    warmup_count: int,
    started_at: str,
) -> dict[str, Any]:
    """Build non-sensitive benchmark metadata."""
    return {
        "source_benchmark": benchmark_path.name,
        "image_count": image_count,
        "language": "kk",
        "mode": mode,
        "concurrency": concurrency,
        "warmup_count": warmup_count,
        "inference_backend": "vllm",
        "prompt_type": "ocr_layout",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started_at,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR inference performance.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Completed benchmark JSON artifact.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing the images directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "concurrent"),
        default="sequential",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent workers.",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Benchmark only the first N images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.warmup_count < 0:
        raise ValueError("Warm-up count cannot be negative.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be at least 1.")

    if args.mode == "sequential" and args.concurrency != 1:
        raise ValueError(
            "Sequential mode requires --concurrency 1."
        )

    image_paths = load_image_paths(
        benchmark_path=args.benchmark,
        dataset_root=args.dataset_root,
        limit=args.limit,
    )

    if not image_paths:
        raise ValueError("No benchmark images are available.")

    started_at = datetime.now(timezone.utc).isoformat()
    wall_started_at = time.perf_counter()

    if args.mode == "sequential":
        request_results = run_sequential(
            image_paths=image_paths,
            warmup_count=args.warmup_count,
        )
    else:
        request_results = run_concurrent(
            image_paths=image_paths,
            warmup_count=args.warmup_count,
            concurrency=args.concurrency,
        )

    wall_time_seconds = time.perf_counter() - wall_started_at

    report = {
        "metadata": build_metadata(
            benchmark_path=args.benchmark,
            image_count=len(image_paths),
            mode=args.mode,
            concurrency=args.concurrency,
            warmup_count=args.warmup_count,
            started_at=started_at,
        ),
        "summary": summarize_latencies(
            request_results=request_results,
            wall_time_seconds=wall_time_seconds,
        ),
        "gpu": read_gpu_metrics(),
        "requests": request_results,
    }

    report["metadata"]["completed_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    dataset_name = args.benchmark.stem
    output_name = (
        f"{dataset_name}_{args.mode}_"
        f"concurrency_{args.concurrency}.json"
    )
    output_path = args.output_root / output_name

    write_report(output_path, report)

    print(f"Report: {output_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()