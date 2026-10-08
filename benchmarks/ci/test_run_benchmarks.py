"""Exercise the CI script with mocked builds, including bases without heap tracking."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MOCK_TOOL = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ["MOCK_ROOT"])
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
phase_file = root / "phase"
phase = phase_file.read_text() if phase_file.exists() else "changes"
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps({"tool": tool, "args": args, "phase": phase}) + "\\n")
if tool == "git":
    if args[0] == "rev-parse":
        print("a" * 40)
    elif args[0] == "checkout":
        phase_file.write_text("base" if args[1] == "FETCH_HEAD" else "changes")
elif tool == "cargo":
    assert "--locked" in args, args
    if args[0] == "metadata":
        features = {"heap-tracking": []} if (
            phase == "changes" or os.environ["MOCK_BASE_HEAP"] == "true"
        ) else {}
        print(json.dumps({"packages": [{
            "name": "delta_kernel_benchmarks", "features": features,
        }]}))
    elif "--features" in args:
        assert "--test" in args, args
        assert "checkpoint.*" in args, args
        assert os.environ["BENCH_TAGS"] == "base,v2-checkpoint"
        value = 100 if phase == "base" else 150
        sample = {
            "allocated_bytes": value, "allocation_calls": value,
            "peak_extra_live_bytes": value,
            "baseline_live_bytes": 1000, "end_live_bytes": 1000,
        }
        Path(os.environ["BENCH_HEAP_OUTPUT"]).write_text(json.dumps({
            "schema_version": 1, "runtime_threads": 2,
            "workloads": [{"name": "checkpoint/test", "samples": [sample] * 5}],
        }))
    else:
        assert "--features" not in args
elif tool == "critcmp":
    print("header\\nheader\\ncheckpoint/test  1.00  1.0±0.1ms  1.00  1.0±0.1ms")
elif tool == "free":
    print("Mem: 1000 200 0 0 0 800")
"""


class RunBenchmarksTests(unittest.TestCase):
    def test_timing_and_heap_passes_with_and_without_baseline_support(self):
        for base_supported in (False, True):
            with self.subTest(base_supported=base_supported):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    tools = root / "bin"
                    tools.mkdir()
                    for name in ("git", "cargo", "critcmp", "lscpu", "free"):
                        tool = tools / name
                        tool.write_text(MOCK_TOOL)
                        tool.chmod(0o755)
                    scripts = root / "benchmarks/ci"
                    scripts.mkdir(parents=True)
                    for name in ("compare_heap.py", "parse_critcmp.py"):
                        (scripts / name).write_text((SCRIPT_DIR / name).read_text())
                    script = root / "run-benchmarks.sh"
                    script.write_text(
                        (SCRIPT_DIR / "run-benchmarks.sh").read_text()
                        .replace("/tmp/bench-comment.md", str(root / "comment.md"))
                        .replace("/tmp/bench-regression.txt", str(root / "regression.txt"))
                    )
                    env = {
                        **os.environ,
                        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                        "MOCK_ROOT": str(root),
                        "MOCK_BASE_HEAP": str(base_supported).lower(),
                        "BASE_REF": "main",
                        "HEAD_SHA": "b" * 40,
                        "COMMENT": "/bench --tags base,v2-checkpoint --filter checkpoint.*",
                        "GITHUB_OUTPUT": str(root / "outputs"),
                        "TMPDIR": str(root),
                    }
                    result = subprocess.run(
                        ["bash", str(script)], cwd=root, env=env,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    calls = [
                        json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()
                    ]
                    builds = [
                        call for call in calls
                        if call["tool"] == "cargo" and call["args"][0] == "bench"
                    ]
                    heap_builds = [call for call in builds if "--features" in call["args"]]
                    self.assertEqual(len(builds) - len(heap_builds), 2)
                    self.assertEqual(len(heap_builds), 2 if base_supported else 1)
                    comment = (root / "comment.md").read_text()
                    self.assertIn("delta-kernel-bench-comment", comment)
                    self.assertIn("Heap allocation profiles (report-only)", comment)
                    self.assertIn(
                        "+50.0%" if base_supported else "only PR values", comment
                    )
                    self.assertEqual((root / "regression.txt").read_text(), "false")
                    output = (root / "outputs").read_text().strip()
                    self.assertTrue(output.startswith("heap_results_dir="))
                    self.assertTrue(Path(output.split("=", 1)[1], "changes.json").is_file())


if __name__ == "__main__":
    unittest.main()
