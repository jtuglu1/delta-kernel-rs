"""Tests for report-only heap comparisons; no external Python dependencies."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import compare_heap


def report(name="snapshot", values=(100, 110, 120, 130, 140)):
    return {
        "schema_version": 1,
        "runtime_threads": 2,
        "workloads": [{
            "name": name,
            "samples": [
                {
                    "allocated_bytes": value,
                    "allocation_calls": value,
                    "peak_extra_live_bytes": value,
                    "baseline_live_bytes": 1000,
                    "end_live_bytes": 1000,
                }
                for value in values
            ],
        }],
    }


class CompareHeapTests(unittest.TestCase):
    def test_comparison_uses_median_and_reports_sample_range(self):
        text = compare_heap.render(report(), report(values=(90, 90, 180, 200, 1000)))
        self.assertIn("120 [100, 140]", text)
        self.assertIn("180 [90, 1,000]", text)
        self.assertIn("+50.0%", text)
        self.assertIn("report-only", text)
        self.assertIn("do not fail", text)

    def test_zero_baselines_have_explicit_changes(self):
        for base_values, pr_values, expected in [
            ((0,), (0,), "0.0%"),
            ((0,), (10,), "+10 (from zero)"),
            ((10,), (0,), "-100.0%"),
        ]:
            with self.subTest(base=base_values, pr=pr_values):
                text = compare_heap.render(
                    report(values=base_values), report(values=pr_values)
                )
                self.assertIn(expected, text)

    def test_added_removed_and_unavailable_baselines_are_not_zero(self):
        text = compare_heap.render(report("removed"), report("added"))
        self.assertIn("removed | Allocated bytes/op | 120 [100, 140] | N/A | N/A", text)
        self.assertIn("added | Allocated bytes/op | N/A | 120 [100, 140] | N/A", text)
        text = compare_heap.render(None, report())
        self.assertIn("only PR values", text)
        self.assertIn("| N/A | 120 [100, 140] | N/A |", text)
        self.assertIsNone(compare_heap.load_report(None))

    def test_workload_names_cannot_break_the_table(self):
        text = compare_heap.render(None, report("<script>|`name`\nnext"))
        self.assertIn("&lt;script&gt;&#124;&#96;name&#96; next", text)
        self.assertNotIn("<script>", text)

    def test_different_runtime_thread_counts_are_rejected(self):
        other = report()
        other["runtime_threads"] = 4
        with self.assertRaises(ValueError):
            compare_heap.render(report(), other)

    def test_report_validation_rejects_malformed_or_ambiguous_samples(self):
        valid = report()
        invalid = []
        for field, value in [
            ("allocated_bytes", -1),
            ("allocation_calls", True),
            ("peak_extra_live_bytes", 1.2),
            ("baseline_live_bytes", 2**64),
            ("end_live_bytes", None),
        ]:
            candidate = copy.deepcopy(valid)
            candidate["workloads"][0]["samples"][0][field] = value
            invalid.append(candidate)
        for field, value in [
            ("schema_version", 2),
            ("schema_version", True),
            ("runtime_threads", 0),
            ("workloads", []),
        ]:
            candidate = copy.deepcopy(valid)
            candidate[field] = value
            invalid.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["workloads"].append(copy.deepcopy(candidate["workloads"][0]))
        invalid.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["workloads"][0]["samples"] = []
        invalid.append(candidate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(valid))
            self.assertEqual(compare_heap.load_report(path), valid)
            for candidate in invalid:
                with self.subTest(candidate=candidate):
                    path.write_text(json.dumps(candidate))
                    with self.assertRaises(ValueError):
                        compare_heap.load_report(path)


if __name__ == "__main__":
    unittest.main()
