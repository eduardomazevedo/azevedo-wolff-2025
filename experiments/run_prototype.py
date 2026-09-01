"""CLI for the small FOA experiment prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.prototype import run_manifest, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/foa_experiments.yaml")
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--output", default="output/foa-prototype")
    args = parser.parse_args()

    payload = run_manifest(args.manifest, args.suite)
    write_outputs(payload, args.output)

    print(f"Completed {len(payload['results'])} case(s).")
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

    print(f"\nWrote {Path(args.output) / 'prototype_results.json'}")
    print(f"Wrote {Path(args.output) / 'prototype_points.csv'}")


if __name__ == "__main__":
    main()
