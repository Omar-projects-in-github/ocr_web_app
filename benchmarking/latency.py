"""Latency and throughput benchmarking for OCR inference."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from typing import Any

_thread_state = local()

def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return a percentile using linear interpolation."""
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )

def get_manager() -> Any:
    """Create one inference manager per worker thread."""
    if not hasattr(_thread_state, "manager"):
        from chandra.model import InferenceManager

        _thread_state.manager = InferenceManager(method = "vllm")

    return _thread_state.manager

def infer_one(image_path: Path) -> dict[str, Any]:
    """Run one OCR request and measure model inference latency."""
    from PIL import Image
    from chandra.model.schema import BatchInputItem

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        started_at = time.perf_counter()

        result = get_manager().generate(
            [
                BatchInputItem(
                    image=rgb_image,
                    prompt_type="ocr_layout",
                )
            ]
        )[0]

        latency_seconds = time.perf_counter() - started_at

    return {
        "image_path": str(image_path),
        "latency_seconds": latency_seconds,
        "error": getattr(result, "error", None),
    }

def run_sequential(
    image_paths: list[Path],
    warmup_count: int,
) -> list[dict[str, Any]]:
    """Run warm-up requests, then benchmark images sequentially."""
    if not image_paths:
        return []

    for image_path in image_paths[:warmup_count]:
        infer_one(image_path)

    return [
        infer_one(image_path)
        for image_path in image_paths
    ]

def run_concurrent(
    image_paths: list[Path],
    warmup_count: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run benchmark images using a configurable worker count."""
    if concurrency < 1:
        raise ValueError("Concurrency must be at least 1.")

    if not image_paths:
        return []

    for image_path in image_paths[:warmup_count]:
        infer_one(image_path)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(infer_one, image_paths))

def summarize_latencies(
    request_results: list[dict[str, Any]],
    wall_time_seconds: float,
) -> dict[str, Any]:
    """Calculate latency and throughput statistics."""
    successful_latencies = [
        result["latency_seconds"]
        for result in request_results
        if not result.get("error")
        and isinstance(result.get("latency_seconds"), (int, float))
    ]

    failed_requests = len(request_results) - len(successful_latencies)

    return {
        "requested_requests": len(request_results),
        "successful_requests": len(successful_latencies),
        "failed_requests": failed_requests,
        "mean_latency_seconds": (
            statistics.mean(successful_latencies)
            if successful_latencies
            else None
        ),
        "median_latency_seconds": (
            statistics.median(successful_latencies)
            if successful_latencies
            else None
        ),
        "p50_latency_seconds": percentile(
            successful_latencies,
            0.50,
        ),
        "p95_latency_seconds": percentile(
            successful_latencies,
            0.95,
        ),
        "min_latency_seconds": (
            min(successful_latencies)
            if successful_latencies
            else None
        ),
        "max_latency_seconds": (
            max(successful_latencies)
            if successful_latencies
            else None
        ),
        "wall_time_seconds": wall_time_seconds,
        "throughput_images_per_second": (
            len(successful_latencies) / wall_time_seconds
            if successful_latencies and wall_time_seconds > 0
            else None
        ),
    }