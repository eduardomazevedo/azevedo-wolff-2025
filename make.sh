#!/bin/bash
set -euo pipefail

if ! command -v uv &> /dev/null
then
    echo "Error: 'uv' is not installed. Please install it to continue."
    exit 1
fi

# Existing paper assets and controlled timing-benchmark presentation.
uv run py/benchmark_assets.py
uv run py/main_figures.py
uv run py/solver_comparison.py

# Two FOA-validity paper summaries.
uv run python -m experiments.run_foa
uv run python -m experiments.compute_competitive
uv run python -m experiments.compute_fixed_action_benchmarks
uv run python -m experiments.report_principal_figure
uv run python -m experiments.report_fixed_action_figure
