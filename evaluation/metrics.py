"""Metrics for evaluating OCR predictions."""

from collections.abc import Iterable, Mapping
from typing import Any

def normalize_text(text: str) -> str:
    """Normalize whitespace without changing OCR character or punctuation."""
    return " ".join(text.split())

def levenshtein_distance(reference: list[str], prediction: list[str]) -> int:
    """Return the minimum edit distance between two token sequences."""
    previous_row = list(range(len(prediction) + 1))

    for reference_index, reference_token in enumerate(reference, start = 1):
        current_row = [reference_index]

        for prediction_index, prediction_token in enumerate(
            prediction,
            start = 1,
        ):
            insertion_cost = current_row[prediction_index - 1] + 1
            deletion_cost = previous_row[prediction_index] + 1
            substitution_cost = previous_row[prediction_index - 1] + (
                reference_token != prediction_token
            )

            current_row.append(
                min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row
    
    return previous_row[-1]

def character_error_metrics(
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    """Calculate character-level edit distance and CER."""
    reference_text = normalize_text(reference)
    prediction_text = normalize_text(prediction)

    reference_characters = list(reference_text)
    prediction_characters = list(prediction_text)

    edit_distance = levenshtein_distance(
        reference_characters,
        prediction_characters,
    )

    reference_count = len(reference_characters)

    return {
        "character_edit_distance": edit_distance,
        "reference_character_count": reference_count,
        "cer": (
            edit_distance / reference_count
            if reference_count > 0
            else None
        ),
    }

def word_error_metrics(
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    """Calculate word-level edit distance and WER."""
    reference_words = normalize_text(reference).split()
    prediction_words = normalize_text(prediction).split()

    edit_distance = levenshtein_distance(
        reference_words,
        prediction_words,
    )

    reference_count = len(reference_words)

    return {
        "word_edit_distance": edit_distance,
        "reference_word_count": reference_count,
        "wer": (
            edit_distance / reference_count
            if reference_count > 0
            else None
        ),
    }

def exact_match(reference: str, prediction: str) -> bool:
    """Return whether normalized reference and prediction are identical."""
    return normalize_text(reference) == normalize_text(prediction)

def evaluate_pair(
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    """Calculate all supported metrics for one OCR prediction."""
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)

    character_metrics = character_error_metrics(
        normalized_reference,
        normalized_prediction,
    )
    word_metrics = word_error_metrics(
        normalized_reference,
        normalized_prediction,
    )

    return {
        **character_metrics,
        **word_metrics,
        "exact_match": normalized_reference == normalized_prediction,
    }

def evaluate_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate per-record metrics and corpus-level aggregate metrics."""
    evaluated_records: list[dict[str, Any]] = []
    failed_records = 0

    total_character_edit_distance = 0
    total_reference_characters = 0
    total_word_edit_distance = 0
    total_reference_words = 0
    exact_match_count = 0

    for record in records:
        if record.get("error"):
            failed_records += 1
            continue

        reference = str(record.get("ground_truth", ""))
        prediction = str(record.get("prediction_text", ""))

        metrics = evaluate_pair(reference, prediction)

        evaluated_record = {
            "image_path": record.get("image_path"),
            **metrics,
        }
        evaluated_records.append(evaluated_record)

        total_character_edit_distance += metrics[
            "character_edit_distance"
        ]
        total_reference_characters += metrics[
            "reference_character_count"
        ]
        total_word_edit_distance += metrics["word_edit_distance"]
        total_reference_words += metrics["reference_word_count"]

        if metrics["exact_match"]:
            exact_match_count += 1

    evaluated_count = len(evaluated_records)

    return {
        "records": evaluated_records,
        "summary": {
            "evaluated_records": evaluated_count,
            "failed_records": failed_records,
            "corpus_cer": (
                total_character_edit_distance
                / total_reference_characters
                if total_reference_characters > 0
                else None
            ),
            "corpus_wer": (
                total_word_edit_distance / total_reference_words
                if total_reference_words > 0
                else None
            ),
            "exact_match_accuracy": (
                exact_match_count / evaluated_count
                if evaluated_count > 0
                else None
            ),
            "exact_match_count": exact_match_count,
            "total_character_edit_distance": (
                total_character_edit_distance
            ),
            "total_reference_characters": total_reference_characters,
            "total_word_edit_distance": total_word_edit_distance,
            "total_reference_words": total_reference_words,
        },
    }