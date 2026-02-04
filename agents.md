# Reproduction Code for Azevedo-Wolff (2025)

This repository contains the reproduction code for the academic paper "Azevedo-Wolff (2025)" and the paper manuscript. The primary purpose of this repository is to generate all figures, tables, and numerical results presented in the paper, ensuring turnkey reproducibility.

## Repository Structure

- `.git/`: Git version control metadata.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `.venv/`: Python virtual environment managed by `uv`.
- `.vscode/`: VS Code specific settings.
- `figures/`: Contains generated figures for the paper, organized by experiment.
- `output/`: Stores numerical outputs, such as machine specifications and timing results.
- `py/`: Contains Python scripts used for:
    - `benchmark_assets.py`: Benchmarking asset-related computations.
    - `benchmarks.py`: General benchmarking scripts.
    - `figure_maker.py`: Defines the consistent style and utilities for generating all figures.
    - `main_figures.py`: Script to generate the main figures for the paper.
    - `solver_comparison.py`: Scripts for comparing different numerical solvers.
- `tex/`: Contains the LaTeX source code for the paper manuscript, including chapters, appendices, and figures/tables subdirectories.
- `make.sh`: A shell script to automate the entire reproduction process (running all Python scripts and compiling the LaTeX manuscript).
- `pyproject.toml`: Project configuration file for Python, including dependencies managed by `uv`.
- `readme.md`: This file.
- `uv.lock`: Lock file for Python dependencies, ensuring reproducible environments.

## Basic Instructions for Reproduction

All numerical algorithms are implemented in the external `moralhazard` Python package. This repository uses that package to generate the results for the paper.

To reproduce the results and compile the manuscript, follow these steps:

1.  **Install `uv`:** If you don't have `uv` installed, you can install it using pip:
    ```bash
    pip install uv
    ```
    Or refer to the official `uv` documentation for other installation methods.

2.  **Navigate to the project root:** Ensure you are in the root directory of this repository:
    ```bash
    cd /Users/eduaze/projects/azevedo-wolff-2025
    ```

3.  **Set up the Python environment and install dependencies:**
    ```bash
    uv sync
    ```
    This command will create a virtual environment in `.venv/` and install all necessary Python packages specified in `pyproject.toml` and `uv.lock`.

4.  **Run the reproduction script:**
    ```bash
    ./make.sh
    ```
    This script will:
    - Activate the `uv` virtual environment.
    - Execute all Python scripts in the `py/` directory to generate numerical results, figures, and tables.
    - Compile the LaTeX manuscript located in `tex/` to produce the final PDF.

    All generated figures will adhere to the consistent style defined in `py/figure_maker.py`.

## Key Technologies

- **Python:** Used for all numerical computations and data generation.
- **`uv`:** A fast Python package installer and dependency resolver used to manage the project's virtual environment and dependencies.
- **LaTeX:** Used for typesetting the academic manuscript.
- **`make.sh`:** A simple shell script orchestrating the entire reproduction workflow.