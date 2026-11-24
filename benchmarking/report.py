"""Benchmark report serialization."""

import json
from pathlib import Path
from typing import Any

def write_report(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    """Atomically write a benchmark report."""
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