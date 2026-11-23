import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run import (
    build_evaluation_report,
    evaluate_one_artifact,
)

class TestEvlauationPipeline(unittest.TestCase):
    def test_build_evaluation_report(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark_path = Path(directory) / "benchmark.json"

            payload = {
                "metadata": {
                    "dataset": "test_dataset",
                    "language": "kk",
                },
                "records": [
                    {
                        "image_path": "images/0001.jpg",
                        "ground_truth": "Сәлем әлем",
                        "prediction_raw": "Сәлем әлем",
                        "prediction_text": "Сәлем әлем",
                        "latency_seconds": 1.25,
                        "language": "kk",
                        "dataset": "test_dataset",
                    },
                    {
                        "image_path": "images/0002.jpg",
                        "ground_truth": "Қалада жаңбыр",
                        "prediction_raw": "Қалада қар",
                        "prediction_text": "Қалада қар",
                        "latency_seconds": 2.50,
                        "language": "kk",
                        "dataset": "test_dataset",
                    },
                ],
            }

            with benchmark_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(payload, file)

            report = build_evaluation_report(benchmark_path)

            self.assertEqual(
                report["benchmark"]["dataset"],
                "test_dataset",
            )
            self.assertEqual(
                len(report["records"]),
                2,
            )
            self.assertEqual(
                report["summary"]["exact_match_count"],
                1,
            )
            self.assertEqual(
                report["summary"]["exact_match_accuracy"],
                0.5,
            )

    def test_writes_json_and_csv_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path = root / "test_dataset.json"
            output_root = root / "evaluation"

            payload = {
                "metadata": {
                    "dataset": "test_dataset",
                    "language": "kk",
                },
                "records": [
                    {
                        "image_path": "images/0001.jpg",
                        "ground_truth": "Қала",
                        "prediction_raw": "Қала",
                        "prediction_text": "Қала",
                        "latency_seconds": 1.0,
                        "language": "kk",
                        "dataset": "test_dataset",
                    }
                ],
            }

            with benchmark_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(payload, file)

            json_path, csv_path = evaluate_one_artifact(
                benchmark_path,
                output_root,
            )

            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                report = json.load(file)

            self.assertEqual(
                report["summary"]["exact_match_accuracy"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()