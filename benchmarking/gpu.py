"""GPU telemetry collection using nvidia-smi."""

import shutil
import subprocess
from typing import Any

def read_gpu_metrics() -> dict[str, Any]:
    """Read GPU utilization and memory for GPU index zero."""
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "reason": "nvidia-smi was not found.",
        }

    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "available": False,
            "reason": f"nvidia-smi timed out: {error}",
        }
    except subprocess.CalledProcessError as error:
        return {
            "available": False,
            "reason": (
                "nvidia-smi failed: "
                f"{error.stderr.strip() or error}"
            ),
        }

    values = [
        value.strip()
        for value in completed.stdout.strip().split(",")
    ]

    if len(values) != 4:
        return {
            "available": False,
            "reason": "Unexpected nvidia-smi output.",
        }

    name, utilization, memory_used, memory_total = values

    return {
        "available": True,
        "name": name,
        "utilization_percent": float(utilization),
        "memory_used_mb": float(memory_used),
        "memory_total_mb": float(memory_total),
    }