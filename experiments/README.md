# FOA Experiment Prototype

Run the unit tests and small smoke suite from the repository root:

```bash
uv run python -m unittest discover -s experiments/tests -v
uv run python -m experiments.run_prototype --suite smoke
uv run python -m experiments.run_prototype --suite first_atlas --output output/foa-first-atlas
```

The smoke suite contains Gaussian/log and Poisson/log positive cases plus a fixed-action Student-`t`/log negative control that is expected to remain FOA-invalid throughout its reservation-wage grid.

Outputs are written to `output/foa-prototype/`:

- `prototype_results.json`: complete benchmarks, point results, transition refinement, and warnings;
- `prototype_points.csv`: compact initial-grid results.

The declarative manifest is `experiments/foa_experiments.yaml`. The current implementation is intentionally a prototype. It provides CE conversion, a multistart global-deviation search, an adaptive true full-GIC slack-IR monopsony scan, separate principal/fixed-action exercises, transition refinement, reversal detection, and grid-based distribution diagnostics (mass, score mean, and action derivatives). These diagnostics are numerical truncation checks, not analytic tail certificates.

Selected points bracketing each transition are also checked against the discretized CVXPY/Clarabel formulation. A cross-check compares relaxed and full objectives, constraint residuals, independent deviation classifications, and contract differences. Disagreement is retained as `unresolved`; it does not trigger general solver debugging.

Outcome supports are expanded adaptively until grid mass and score-mean tolerances pass, and the final expanded grid is the effective grid used by model solves. Relaxed contracts at transition endpoints are additionally re-solved on baseline, denser, and expanded-support grids; classification or expected-wage instability is retained as unresolved.

The prototype computes numerical safe-region diagnostics over the full configured action grid: cutoff supremum, safe mass, incentive capacity, curvature, score width, normalized width, cost-curvature floor, and the log curvature-width condition. Metrics are saved at the reference fixed action, the full-GIC monopsony action, and transition-endpoint actions, and are checked over action-grid and derivative-step refinements. Each result is explicitly labeled as a grid diagnostic rather than a proof.

The relaxed `lambda=0` monopsony scan remains available as an optional debugging routine but is disabled in the manifest because the pilot showed it is not an economically useful benchmark.
