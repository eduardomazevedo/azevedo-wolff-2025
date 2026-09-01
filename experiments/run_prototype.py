"""CLI for the small FOA experiment prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.storage import run_manifest_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/foa_experiments.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", default="smoke")
    selection.add_argument("--all", action="store_true", help="Run every manifest case")
    parser.add_argument("--output", default="output/foa-prototype")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    run = run_manifest_atomic(
        args.manifest,
        args.output,
        suite=None if args.all else args.suite,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    payload = run["payload"]

    completed = sum(row["execution_status"] == "completed" for row in run["task_index"])
    failed = len(run["task_index"]) - completed
    cached = sum(bool(row["cached"]) for row in run["task_index"])
    print(f"Completed {completed} case(s); failed={failed}; resumed={cached}.")
    for case in payload["results"]:
        print(f"\n{case['case_id']}")
        support = case.get("support_validation", {})
        if support:
            print(
                f"  support: {support['status']}; "
                f"expansions={support['expansions']}"
            )
        for name, benchmark in case["monopsony"].items():
            if not isinstance(benchmark, dict) or "status" not in benchmark:
                continue
            selected = benchmark.get("selected", {})
            print(
                f"  {name}: {benchmark['status']}; "
                f"CE={selected.get('delivered_ce_wage', float('nan')):.4f}, "
                f"a={selected.get('action', float('nan')):.4f}"
            )
        for exercise, result in case["exercises"].items():
            summary = result["summary"]
            validation_counts = {}
            for validation in result.get("validation", []):
                status = validation["status"]
                validation_counts[status] = validation_counts.get(status, 0) + 1
            print(
                f"  {exercise}: persistent grid threshold="
                f"{summary['persistent_threshold_on_grid']}; "
                f"transitions={len(summary['transitions'])}; "
                f"reversals={len(summary['reversals'])}; "
                f"crosschecks={validation_counts or 'none'}"
            )

    print(f"\nAtomic results: {Path(args.output) / 'atomic'}")
    print(f"Task index: {Path(args.output) / 'task_index.json'}")
    print(f"Compatibility summary: {Path(args.output) / 'prototype_results.json'}")


if __name__ == "__main__":
    main()
