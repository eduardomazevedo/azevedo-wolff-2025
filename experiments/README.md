# FOA paper summaries

This directory contains the calculations for the two large FOA-validity summary
figures used by the paper.

## Reproduce from a fresh clone

From the repository root:

```bash
# Run exactly the 27 specifications selected by either paper summary.
uv run python -m experiments.run_foa

# Compute the saved full-GIC benchmark markers.
uv run python -m experiments.compute_competitive
uv run python -m experiments.compute_fixed_action_benchmarks

# Render from saved results only; these commands never solve models.
uv run python -m experiments.report_principal_figure
uv run python -m experiments.report_fixed_action_figure
```

Generated results are written under `output/foa-paper/`. Final assets are:

- `figures/foa-principal-summary.pdf` and `.png`/`.csv` companions;
- `figures/foa-fixed-action-summary.pdf` and `.png`/`.csv` companions.

All generated results and figures are ignored by Git.

## Files

- `foa_paper.yaml`: exact paper specifications, numerical settings, figure
  order, and labels;
- `foa.py`: model construction, FOA calculations, validation, and result schema;
- `storage.py`: deterministic execution and readable saved results;
- `run_foa.py`: paper-only runner;
- `compute_competitive.py`: principal monopsony/competitive markers;
- `compute_fixed_action_benchmarks.py`: separate fixed-action markers;
- `report_common.py`: small shared style and row helper;
- `report_principal_figure.py` and `report_fixed_action_figure.py`: saved-results
  renderers;
- `tests/test_foa.py`: numerical and configuration tests.

Principal and fixed-action exercises remain separate throughout. The common YAML
selection does not pool their results or benchmarks.

The timing benchmark is unrelated to this directory. Its controlled CSV and
machine description remain committed because timings are refreshed only on the
designated standard server.
