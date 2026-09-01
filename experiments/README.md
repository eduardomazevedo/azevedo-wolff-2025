# FOA Experiment Prototype

Run the unit tests and small smoke suite from the repository root:

```bash
uv run python -m unittest discover -s experiments/tests -v
uv run python -m experiments.run_prototype --suite smoke
uv run python -m experiments.run_prototype --suite first_atlas --output output/foa-first-atlas
uv run python -m experiments.run_prototype --suite common_distributions --output output/foa-common-distributions --resume
uv run python -m experiments.run_prototype --all --output output/foa-internal-atlas --resume
uv run python -m experiments.summarize_internal --input output/foa-internal-atlas
uv run python -m experiments.diagnose_problematic
```

The smoke suite contains Gaussian/log and Poisson/log positive cases plus a fixed-action Student-`t`/log negative control that is expected to remain FOA-invalid throughout its reservation-wage grid. The common-distribution suite additionally contains exponential, gamma, geometric, Bernoulli, and binomial cases. These are internal experiments, not a paper-selected subset.

Outputs are written to `output/foa-prototype/`:

- `manifest.snapshot.yaml`, `environment.json`, and (for dirty trees) `source.diff.patch` plus `source.untracked.json`: exact run configuration and Git/package provenance;
- `task_index.json`/`.csv`: deterministic task index and completion status;
- `atomic/<task_hash>.json`: authoritative, crash-safe case result with complete economic and numerical configurations;
- `arrays/`: reserved for task-keyed numerical arrays;
- `prototype_results.json` and `prototype_points.csv`: temporary compatibility summaries.

Task hashes exclude labels and suite membership but include the complete economic and numerical configuration. Explicit cases and declarative Cartesian `case_families` expand in deterministic order. `--resume` reuses completed and failed atomic records; add `--retry-failed` to rerun failures. Nonfinite values are replaced by JSON `null`, their exact paths are recorded, and strict status remains unresolved. Human review records from the manifest are attached without replacing strict numerical statuses or warnings.

`experiments.summarize_internal` only reads atomic records; it never solves a model. It writes internal threshold, failure, and warning tables under `summary_tables/`. These are diagnostic atlas outputs, not paper tables.

The declarative manifest is `experiments/foa_experiments.yaml`. The current implementation is intentionally a prototype. It provides CE conversion, a multistart global-deviation search, an adaptive true full-GIC slack-IR monopsony scan, separate principal/fixed-action exercises, transition refinement, reversal detection, and grid-based distribution diagnostics (mass, score mean, and action derivatives). These diagnostics are numerical truncation checks, not analytic tail certificates.

Selected points bracketing each transition are also checked against the discretized CVXPY/Clarabel formulation. A cross-check compares relaxed and full objectives, constraint residuals, independent deviation classifications, and contract differences. Disagreement is retained as `unresolved`; it does not trigger general solver debugging.

Outcome supports are expanded adaptively until grid mass and score-mean tolerances pass, and the final expanded grid is the effective grid used by model solves. Relaxed contracts at transition endpoints are additionally re-solved on baseline, denser, and expanded-support grids; classification or expected-wage instability is retained as unresolved.

The prototype computes numerical safe-region diagnostics over the full configured action grid: cutoff supremum, safe mass, incentive capacity, curvature, score width, normalized width, cost-curvature floor, and the log curvature-width condition. Metrics are saved at the reference fixed action, the full-GIC monopsony action, and transition-endpoint actions, and are checked over action-grid and derivative-step refinements. Each result is explicitly labeled as a grid diagnostic rather than a proof.

For utility bounded above, a fast limited-liability incentive-capacity precheck runs before optimization. Infeasible fixed actions are recorded as `infeasible_local_incentives`. The principal's intended-action search is capped just inside the highest locally feasible action, but its global-IC deviation domain remains the original configured action interval.

The relaxed `lambda=0` monopsony scan remains available as an optional debugging routine but is disabled in the manifest because the pilot showed it is not an economically useful benchmark.

## Problem diagnostics

`experiments.diagnose_problematic` re-solves only the suspicious cells declared in the manifest's `problem_diagnostics` section. It writes an organized ignored directory at `output/foa-problem-diagnostics/`, with one subdirectory per cell containing:

- `contracts_and_agent_objective.png` or `support_convergence.png`;
- `metadata.json` with configurations, warnings, statuses, and deviation diagnostics;
- `diagnostic_arrays.npz` for contract solves;
- a root `README.md` and `index.json` linking all diagnostics.

These are internal diagnostic figures, not paper figures.
