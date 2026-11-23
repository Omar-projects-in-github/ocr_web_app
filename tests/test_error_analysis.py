import unittest

from evaluation.error_analysis import (
    align_sequences,
    analyze_character_errors,
    analyze_word_errors,
    build_error_analysis_report,
    rank_difficult_records,
)

class TestCharacterAlignment(unittest.TestCase):
    def test_perfect_match(self):
        operations = align_sequences(
            list("қала"),
            list("қала"),
        )

        self.assertEqual(
            [
                operation["operation"]
                for operation in operations
            ],
            ["match", "match", "match", "match"],
        )

    def test_substitution(self):
        operations = analyze_character_errors(
            "қала",
            "қана",
        )

        substitutions = [
            operation
            for operation in operations
            if operation["operation"] == "substitution"
        ]

        self.assertEqual(
            substitutions,
            [
                {
                    "operation": "substitution",
                    "reference": "л",
                    "prediction": "н",
                }
            ],
        )

    def test_deletion(self):
        operations = analyze_character_errors(
            "мәтін",
            "мтин",
        )

        deletions = [
            operation
            for operation in operations
            if operation["operation"] == "deletion"
        ]

        self.assertGreaterEqual(
            len(deletions),
            1,
        )

    def test_insertion(self):
        operations = analyze_character_errors(
            "қала",
            "қалуа",
        )

        insertions = [
            operation
            for operation in operations
            if operation["operation"] == "insertion"
        ]

        self.assertGreaterEqual(
            len(insertions),
            1,
        )

    def test_empty_strings(self):
        operations = analyze_character_errors(
            "",
            "",
        )

        self.assertEqual(operations, [])

class TestWordAlignment(unittest.TestCase):
    def test_word_deletion(self):
        operations = analyze_word_errors(
            "Бұл жақсы мәтін",
            "Бұл мәтін",
        )

        deletions = [
            operation
            for operation in operations
            if operation["operation"] == "deletion"
        ]

        self.assertEqual(
            deletions,
            [
                {
                    "operation": "deletion",
                    "reference": "жақсы",
                    "prediction": None,
                }
            ],
        )

    def test_word_insertion(self):
        operations = analyze_word_errors(
            "Бұл мәтін",
            "Бұл жақсы мәтін",
        )

        insertions = [
            operation
            for operation in operations
            if operation["operation"] == "insertion"
        ]

        self.assertEqual(
            insertions,
            [
                {
                    "operation": "insertion",
                    "reference": None,
                    "prediction": "жақсы",
                }
            ],
        )

class TestErrorAggregation(unittest.TestCase):
    def test_build_report(self):
        evaluation_report = {
            "benchmark": {
                "dataset": "test_dataset",
                "language": "kk",
            },
            "records": [
                {
                    "image_path": "images/0001.jpg",
                    "ground_truth": "қала",
                    "prediction_text": "қана",
                    "cer": 0.25,
                    "wer": 1.0,
                    "character_edit_distance": 1,
                    "exact_match": False,
                },
            ],
        }

        report = build_error_analysis_report(
            evaluation_report
        )

        self.assertEqual(
            report["character_errors"]["substitutions"]["л→н"],
            1,
        )
        self.assertEqual(
            len(report["difficult_records"]),
            1,
        )
    
class TestDifficultRanking(unittest.TestCase):
    def test_sort_by_cer(self):
        records = [
            {
                "image_path": "images/0001.jpg",
                "cer": 0.20,
                "wer": 0.25,
                "character_edit_distance": 2,
                "exact_match": False,
            },
            {
                "image_path": "images/0002.jpg",
                "cer": 0.80,
                "wer": 0.90,
                "character_edit_distance": 8,
                "exact_match": False,
            },
        ]

        ranked = rank_difficult_records(records)

        self.assertEqual(
            ranked[0]["image_path"],
            "images/0002.jpg",
        )

    def test_respect_limit(self):
        records = [
            {
                "image_path": f"images/{index}.jpg",
                "cer": index / 100,
                "wer": index / 100,
                "character_edit_distance": index,
                "exact_match": False,
            }
            for index in range(1, 10)
        ]

        ranked = rank_difficult_records(
            records,
            limit=3,
        )

        self.assertEqual(len(ranked), 3)


if __name__ == "__main__":
    unittest.main()