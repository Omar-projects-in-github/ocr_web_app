import unittest

from benchmarking.latency import (
    percentile,
    summarize_latencies,
)

class TestLatencyStatistics(unittest.TestCase):
    def test_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertEqual(
            percentile(values, 0.50),
            3.0,
        )
        self.assertEqual(
            percentile(values, 0.95),
            4.8,
        )

    def test_empty_percentile(self):
        self.assertIsNone(percentile([], 0.50))

    def test_summary(self):
        results = [
            {
                "image_path": "one.jpg",
                "latency_seconds": 1.0,
                "error": None,
            },
            {
                "image_path": "two.jpg",
                "latency_seconds": 3.0,
                "error": None,
            },
            {
                "image_path": "three.jpg",
                "latency_seconds": None,
                "error": "failed",
            },
        ]

        summary = summarize_latencies(
            results,
            wall_time_seconds=4.0,
        )

        self.assertEqual(summary["requested_requests"], 3)
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["mean_latency_seconds"], 2.0)
        self.assertEqual(summary["throughput_images_per_second"], 0.5)

if __name__ == "__main__":
    unittest.main()