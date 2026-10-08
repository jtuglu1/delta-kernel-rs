#!/usr/bin/env python3
"""Compare heap-profile sidecars and print report-only Markdown. No regression gate."""

import argparse
import html
import json
import statistics
from pathlib import Path


METRICS = (
    ("allocated_bytes", "Allocated bytes/op"),
    ("allocation_calls", "Allocation calls/op"),
    ("peak_extra_live_bytes", "Peak extra live bytes/op"),
)
SAMPLE_FIELDS = tuple(key for key, _ in METRICS) + (
    "baseline_live_bytes", "end_live_bytes",
)


def load_report(path):
    """Read and validate a report; return None for an unavailable baseline."""
    if path is None:
        return None
    report = json.loads(Path(path).read_text())
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported heap report schema")
    threads = report.get("runtime_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("runtime_threads must be a positive integer")
    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("heap report must contain workloads")
    names = set()
    for workload in workloads:
        name = workload.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("workload names must be unique nonempty strings")
        names.add(name)
        samples = workload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{name}: no heap samples")
        for sample in samples:
            for field in SAMPLE_FIELDS:
                value = sample.get(field)
                if type(value) is not int or not 0 <= value < 2**64:
                    raise ValueError(f"{name}: invalid {field}")
    return report


def number(value):
    """Render exact integer counts, retaining fractions for even-sized sample sets."""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"


def summary(samples, metric):
    """Render the median and observed min/max; missing workloads are not zero usage."""
    if samples is None:
        return "N/A"
    values = [sample[metric] for sample in samples]
    return (
        f"{number(statistics.median(values))} "
        f"[{number(min(values))}, {number(max(values))}]"
    )


def difference(base_samples, pr_samples, metric):
    """Compare medians, including meaningful zero baselines without dividing by zero."""
    if base_samples is None or pr_samples is None:
        return "N/A"
    base = statistics.median(sample[metric] for sample in base_samples)
    pr = statistics.median(sample[metric] for sample in pr_samples)
    if base == 0:
        return "0.0%" if pr == 0 else f"+{number(pr)} (from zero)"
    return f"{(pr / base - 1) * 100:+.1f}%"


def safe_name(name):
    """Keep workload-controlled text inside a Markdown table cell."""
    return html.escape(name).replace("|", "&#124;").replace("`", "&#96;").replace(
        "\n", " "
    ).replace("\r", " ")


def render(base, pr):
    """Render each workload's metrics without modifying the timing regression verdict."""
    if base is not None and base["runtime_threads"] != pr["runtime_threads"]:
        raise ValueError("heap profiles use different runtime thread counts")
    base_workloads = {
        workload["name"]: workload["samples"] for workload in base["workloads"]
    } if base is not None else {}
    pr_workloads = {workload["name"]: workload["samples"] for workload in pr["workloads"]}
    lines = [
        "### Heap allocation profiles (report-only)",
        "",
        "Values are medians [min, max] over individual warmed-up operations. "
        "Positive changes mean more allocation or live memory.",
        "",
    ]
    if base is None:
        lines.extend([
            "The base branch does not support heap profiling; only PR values are available.",
            "",
        ])
    lines.extend([
        "<details>",
        "<summary>Per-workload heap usage</summary>",
        "",
        "| Workload | Metric | Base | PR | Change |",
        "|----------|--------|------|----|--------|",
    ])
    for name in sorted(base_workloads.keys() | pr_workloads.keys()):
        base_samples = base_workloads.get(name)
        pr_samples = pr_workloads.get(name)
        for key, label in METRICS:
            lines.append(
                f"| {safe_name(name)} | {label} | {summary(base_samples, key)} "
                f"| {summary(pr_samples, key)} | {difference(base_samples, pr_samples, key)} |"
            )
    lines.extend([
        "",
        "</details>",
        "",
        "Requested Rust heap sizes only, not RSS or C/mmap memory. Allocated bytes include "
        "positive realloc growth; calls include successful reallocations. "
        "Peak is advisory and process-wide; background activity can affect every metric. "
        "Heap changes do not fail the benchmark gate.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--pr", type=Path, required=True)
    args = parser.parse_args()
    print(render(load_report(args.base), load_report(args.pr)))


if __name__ == "__main__":
    main()
