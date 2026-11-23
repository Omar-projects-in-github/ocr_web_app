"""Character, word, and sample-level OCR error analysis."""

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from evaluation.metrics import normalize_text
from evaluation.reports import (
    write_csv_report,
    write_json_report,
)

CHARACTER_MATCH = "match"
CHARACTER_SUBSTITUTION = "substitution"
CHARACTER_DELETION = "deletion"
CHARACTER_INSERTION = "insertion"

def build_alignment_matrix(
    reference: list[str],
    prediction: list[str],
) -> list[list[int]]:
    """Build the Levenshtein dynamic-programming matrix."""
    matrix = [
        [0] * (len(prediction) + 1)
        for _ in range(len(reference) + 1)
    ]

    for reference_index in range(len(reference) + 1):
        matrix[reference_index][0] = reference_index

    for prediction_index in range(len(prediction) + 1):
        matrix[0][prediction_index] = prediction_index

    for reference_index in range(1, len(reference) + 1):
        for prediction_index in range(1, len(prediction) + 1):
            insertion_cost = (
                matrix[reference_index][prediction_index - 1] + 1
            )
            deletion_cost = (
                matrix[reference_index - 1][prediction_index] + 1
            )
            substitution_cost = (
                matrix[reference_index - 1][prediction_index - 1]
                + (
                    reference[reference_index - 1]
                    != prediction[prediction_index - 1]
                )
            )

            matrix[reference_index][prediction_index] = min(
                insertion_cost,
                deletion_cost,
                substitution_cost,
            )

    return matrix

def align_sequences(
    reference: list[str],
    prediction: list[str],
) -> list[dict[str, str | None]]:
    """Align two sequences and identify edit operations."""
    matrix = build_alignment_matrix(reference, prediction)

    reference_index = len(reference)
    prediction_index = len(prediction)
    operations: list[dict[str, str | None]] = []

    while reference_index > 0 or prediction_index > 0:
        if (
            reference_index > 0
            and prediction_index > 0
            and reference[reference_index - 1]
            == prediction[prediction_index - 1]
        ):
            operations.append(
                {
                    "operation": CHARACTER_MATCH,
                    "reference": reference[reference_index - 1],
                    "prediction": prediction[prediction_index - 1],
                }
            )
            reference_index -= 1
            prediction_index -= 1
            continue

        if reference_index > 0 and prediction_index > 0:
            diagonal_cost = (
                matrix[reference_index - 1][prediction_index - 1]
            )
            current_cost = matrix[reference_index][prediction_index]

            if diagonal_cost + 1 == current_cost:
                operations.append(
                    {
                        "operation": CHARACTER_SUBSTITUTION,
                        "reference": reference[reference_index - 1],
                        "prediction": prediction[prediction_index - 1],
                    }
                )
                reference_index -= 1
                prediction_index -= 1
                continue

        if reference_index > 0:
            deletion_cost = (
                matrix[reference_index - 1][prediction_index] + 1
            )
            if deletion_cost == matrix[reference_index][prediction_index]:
                operations.append(
                    {
                        "operation": CHARACTER_DELETION,
                        "reference": reference[reference_index - 1],
                        "prediction": None,
                    }
                )
                reference_index -= 1
                continue

        if prediction_index > 0:
            operations.append(
                {
                    "operation": CHARACTER_INSERTION,
                    "reference": None,
                    "prediction": prediction[prediction_index - 1],
                }
            )
            prediction_index -= 1

    operations.reverse()
    return operations

def analyze_character_errors(
    reference_text: str,
    prediction_text: str,
) -> list[dict[str, str | None]]:
    """Align characters and return character edit operations."""
    reference = list(normalize_text(reference_text))
    prediction = list(normalize_text(prediction_text))
    return align_sequences(reference, prediction)

def analyze_word_errors(
    reference_text: str,
    prediction_text: str,
) -> list[dict[str, str | None]]:
    """Align words and return word edit operations."""
    reference = normalize_text(reference_text).split()
    prediction = normalize_text(prediction_text).split()
    return align_sequences(reference, prediction)

def display_operation_key(
    operation: dict[str, str | None],
) -> str:
    """Create a stable, human-readable operation key."""
    reference = operation["reference"]
    prediction = operation["prediction"]

    if operation["operation"] == CHARACTER_SUBSTITUTION:
        return f"{reference or '∅'}→{prediction or '∅'}"
    if operation["operation"] == CHARACTER_DELETION:
        return f"{reference or '∅'}→∅"
    if operation["operation"] == CHARACTER_INSERTION:
        return f"∅→{prediction or '∅'}"
    return ""

def aggregate_operations(
    operation_records: Iterable[dict[str, str | None]],
) -> dict[str, int]:
    """Convert operation records into frequency counters."""
    counter: Counter[str] = Counter()

    for operation in operation_records:
        if operation["operation"] == CHARACTER_MATCH:
            continue

        key = display_operation_key(operation)
        if key:
            counter[key] += 1

    return dict(
        sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

def rank_difficult_records(
    records: Iterable[Mapping[str, Any]],
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Rank the most difficult OCR records."""
    ranked_records = [
        dict(record)
        for record in records
        if record.get("exact_match") is False
    ]

    ranked_records.sort(
        key=lambda record: (
            record.get("cer") is None,
            -(record.get("cer") or 0.0),
            record.get("wer") is None,
            -(record.get("wer") or 0.0),
            -(
                record.get("character_edit_distance")
                or 0
            ),
        )
    )

    return ranked_records[:limit]

def build_error_analysis_report(
    evaluation_report: Mapping[str, Any],
    difficult_limit: int = 50,
) -> dict[str, Any]:
    """Build character, word, and difficult-example reports."""
    benchmark = evaluation_report.get("benchmark", {})
    records = evaluation_report.get("records", [])

    character_operations: list[dict[str, str | None]] = []
    word_operations: list[dict[str, str | None]] = []

    for record in records:
        reference = str(record.get("ground_truth", ""))
        prediction = str(record.get("prediction_text", ""))

        character_operations.extend(
            analyze_character_errors(reference, prediction)
        )
        word_operations.extend(
            analyze_word_errors(reference, prediction)
        )

    character_substitutions: Counter[str] = Counter()
    character_deletions: Counter[str] = Counter()
    character_insertions: Counter[str] = Counter()

    for operation in character_operations:
        key = display_operation_key(operation)

        if operation["operation"] == CHARACTER_SUBSTITUTION:
            character_substitutions[key] += 1
        elif operation["operation"] == CHARACTER_DELETION:
            character_deletions[key] += 1
        elif operation["operation"] == CHARACTER_INSERTION:
            character_insertions[key] += 1

    word_substitutions: Counter[str] = Counter()
    word_deletions: Counter[str] = Counter()
    word_insertions: Counter[str] = Counter()

    for operation in word_operations:
        key = display_operation_key(operation)

        if operation["operation"] == CHARACTER_SUBSTITUTION:
            word_substitutions[key] += 1
        elif operation["operation"] == CHARACTER_DELETION:
            word_deletions[key] += 1
        elif operation["operation"] == CHARACTER_INSERTION:
            word_insertions[key] += 1

    return {
        "benchmark": dict(benchmark),
        "character_errors": {
            "substitutions": dict(
                sorted(
                    character_substitutions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "deletions": dict(
                sorted(
                    character_deletions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "insertions": dict(
                sorted(
                    character_insertions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "word_errors": {
            "substitutions": dict(
                sorted(
                    word_substitutions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "deletions": dict(
                sorted(
                    word_deletions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "insertions": dict(
                sorted(
                    word_insertions.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "difficult_records": rank_difficult_records(
            records,
            limit=difficult_limit,
        ),
    }

def write_error_analysis(
    output_root: Path,
    evaluation_report: Mapping[str, Any],
    difficult_limit: int = 50,
) -> dict[str, Path]:
    """Write character, word, and difficult-image reports."""
    report = build_error_analysis_report(
        evaluation_report,
        difficult_limit=difficult_limit,
    )

    benchmark = report["benchmark"]
    dataset_name = str(
        benchmark.get("dataset")
        or "ocr_dataset"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    character_output = (
        output_root
        / f"{dataset_name}_character_errors.json"
    )
    word_output = (
        output_root
        / f"{dataset_name}_word_errors.json"
    )
    difficult_json_output = (
        output_root
        / f"{dataset_name}_difficult_images.json"
    )
    difficult_csv_output = (
        output_root
        / f"{dataset_name}_difficult_images.csv"
    )

    write_json_report(
        character_output,
        {
            "benchmark": benchmark,
            "errors": report["character_errors"],
        },
    )
    write_json_report(
        word_output,
        {
            "benchmark": benchmark,
            "errors": report["word_errors"],
        },
    )
    write_json_report(
        difficult_json_output,
        {
            "benchmark": benchmark,
            "records": report["difficult_records"],
        },
    )
    write_csv_report(
        difficult_csv_output,
        report["difficult_records"],
    )

    return {
        "character_errors": character_output,
        "word_errors": word_output,
        "difficult_images_json": difficult_json_output,
        "difficult_images_csv": difficult_csv_output,
    }