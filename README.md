# Replication code for Azevedo and Wolff (2025) "Broad Validity of the First-Order Approach in Moral Hazard"

**Paper:** [arXiv:2506.18873](https://arxiv.org/abs/2506.18873)

## Requirements

- **Python** (3.12)
- **uv** (Python package manager)

## Reproduction instructions

From the repository root, install the pinned dependencies and run the reproduction script:

```bash
uv sync
./make.sh
```

This regenerates the illustrative and Pareto figures, benchmark presentation, solver comparison, and both FOA-validity summary figures used by the paper. Timing benchmarks are not rerun: the build uses the controlled standard-server results committed under `output/`. The script does not compile LaTeX; build the manuscript separately from `tex/manuscript.tex`.

## Algorithm 1 Implementation

https://github.com/eduardomazevedo/moralhazard

[**Quickstart Colab Demo** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduardomazevedo/moralhazard/blob/main/examples/colab_demo.ipynb)
