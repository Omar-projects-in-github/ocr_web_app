import unittest

from evaluation.metrics import (
    character_error_metrics,
    evaluate_pair,
    evaluate_records,
    exact_match,
    levenshtein_distance,
    normalize_text,
    word_error_metrics,
)

class TestTextNormalization(unittest.TestCase):
    def test_collapses_whitespace(self):
        text = "  Сәлем\n\nәлем\t!"

        self.assertEqual(
            normalize_text(text),
            "Сәлем әлем !",
        )

    def test_preserves_kazakh_characters(self):
        text = "ә ғ қ ң ө ұ ү һ і"

        self.assertEqual(
            normalize_text(text),
            text,
        )

class TestLevenshteinDistance(unittest.TestCase):
    def test_identical_sequences(self):
        self.assertEqual(
            levenshtein_distance(
                list("мәтін"),
                list("мәтін"),
            ),
            0,
        )

    def test_substitution(self):
        self.assertEqual(
            levenshtein_distance(
                list("қала"),
                list("қана"),
            ),
            1,
        )

    def test_insertion(self):
        self.assertEqual(
            levenshtein_distance(
                list("мәтін"),
                list("мәтіндер"),
            ),
            3,
        )

    def test_deletion(self):
        self.assertEqual(
            levenshtein_distance(
                ["Бұл", "жақсы", "мәтін"],
                ["Бұл", "мәтін"],
            ),
            1,
        )

class TestCharacterMetrics(unittest.TestCase):
    def test_perfect_character_match(self):
        result = character_error_metrics(
            "Сәлем әлем",
            "Сәлем әлем",
        )

        self.assertEqual(result["character_edit_distance"], 0)
        self.assertEqual(result["cer"], 0.0)

    def test_character_substitution(self):
        result = character_error_metrics(
            "қала",
            "қана",
        )

        self.assertEqual(result["character_edit_distance"], 1)
        self.assertEqual(result["reference_character_count"], 4)
        self.assertEqual(result["cer"], 0.25)

    def test_empty_reference(self):
        result = character_error_metrics(
            "",
            "мәтін",
        )

        self.assertIsNone(result["cer"])

class TestWordMetrics(unittest.TestCase):
    def test_perfect_word_match(self):
        result = word_error_metrics(
            "Бұл жақсы мәтін",
            "Бұл жақсы мәтін",
        )

        self.assertEqual(result["word_edit_distance"], 0)
        self.assertEqual(result["wer"], 0.0)

    def test_word_deletion(self):
        result = word_error_metrics(
            "Бұл жақсы мәтін",
            "Бұл мәтін",
        )

        self.assertEqual(result["word_edit_distance"], 1)
        self.assertEqual(result["reference_word_count"], 3)
        self.assertAlmostEqual(result["wer"], 1 / 3)

class TestExactMatch(unittest.TestCase):
    def test_whitespace_difference_is_ignored(self):
        self.assertTrue(
            exact_match(
                "Сәлем\nәлем",
                "Сәлем әлем",
            )
        )

    def test_content_difference_is_not_ignored(self):
        self.assertFalse(
            exact_match(
                "Сәлем әлем",
                "Сәлем басқа",
            )
        )

class TestPairEvaluation(unittest.TestCase):
    def test_returns_all_metrics(self):
        result = evaluate_pair(
            "Бұл жақсы мәтін",
            "Бұл мәтін",
        )

        self.assertIn("cer", result)
        self.assertIn("wer", result)
        self.assertIn("exact_match", result)
        self.assertFalse(result["exact_match"])

class TestRecordEvaluation(unittest.TestCase):
    def test_calculates_per_record_and_corpus_metrics(self):
        records = [
            {
                "image_path": "images/0001.jpg",
                "ground_truth": "Бұл жақсы мәтін",
                "prediction_text": "Бұл жақсы мәтін",
            },
            {
                "image_path": "images/0002.jpg",
                "ground_truth": "Сәлем әлем",
                "prediction_text": "Сәлем басқа",
            },
        ]

        result = evaluate_records(records)

        self.assertEqual(
            len(result["records"]),
            2,
        )
        self.assertEqual(
            result["summary"]["evaluated_records"],
            2,
        )
        self.assertEqual(
            result["summary"]["exact_match_count"],
            1,
        )
        self.assertEqual(
            result["summary"]["exact_match_accuracy"],
            0.5,
        )

    def test_excludes_failed_records(self):
        records = [
            {
                "image_path": "images/0001.jpg",
                "ground_truth": "мәтін",
                "prediction_text": "",
                "error": "OCR failed",
            },
        ]

        result = evaluate_records(records)

        self.assertEqual(
            result["summary"]["evaluated_records"],
            0,
        )
        self.assertEqual(
            result["summary"]["failed_records"],
            1,
        )

if __name__ == "__main__":
    unittest.main()