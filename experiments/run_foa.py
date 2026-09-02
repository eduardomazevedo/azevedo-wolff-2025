"""Run the exact specifications used by the two FOA paper summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.storage import run_paper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/foa_paper.yaml")
    parser.add_argument("--output", default="output/foa-paper")
    args = parser.parse_args()

    records = run_paper(args.manifest, args.output)
    completed = sum(row["execution_status"] == "completed" for row in records)
    failed = len(records) - completed
    print(f"Completed {completed} paper specification(s); failed={failed}.")
    print(f"Saved results under {Path(args.output)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
