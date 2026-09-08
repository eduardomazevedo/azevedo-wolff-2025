#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root/tex"
mkdir -p pdf

latexmk_args=(
    -xelatex
    -synctex=1
    -interaction=nonstopmode
    -file-line-error
    -outdir=pdf
)

# Build in dependency order. Each document imports the other's auxiliary file,
# so rebuild the manuscript after generating the SI auxiliary file.
latexmk "${latexmk_args[@]}" manuscript.tex
latexmk "${latexmk_args[@]}" si.tex
latexmk "${latexmk_args[@]}" manuscript.tex
