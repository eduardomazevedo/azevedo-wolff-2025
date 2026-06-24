#!/bin/bash

if ! command -v uv &> /dev/null
then
    echo "Error: 'uv' is not installed. Please install it to continue."
    exit 1
fi

uv run py/benchmark_assets.py
uv run py/main_figures.py
uv run py/solver_comparison.py