# Simple Paper-Reproduction Cleanup Plan

## Goal

Keep the repository simple: a fresh clone plus the committed code/configuration
regenerates the paper figures and numerical results. Generated model output and
figures remain ignored. The timing benchmark is the one deliberate exception:
its CSV and machine description are produced occasionally on the standard
server and committed.

There are four asset types:

1. illustrative contract figures already produced by `py/main_figures.py` and
   `py/figure_maker.py`;
2. the Pareto-frontier figure produced by the same existing code;
3. benchmark and solver-comparison assets;
4. the two new large FOA-validity summary figures.

Types 1--3 are already reasonably organized and should remain substantially as
they are. Cleanup should concentrate on type 4, where prototype development and
diagnostics left unnecessary machinery.

## 1. Existing illustrative and Pareto figures

Keep `py/main_figures.py` and `py/figure_maker.py` for now. Do not rewrite them
into a general asset framework.

A later, small improvement may replace the repeated hard-coded specification
blocks with a YAML list such as:

```yaml
illustrative_figures:
  - id: gaussian_log
    utility: {kind: log, initial_wealth: 50}
    distribution: {kind: gaussian, sigma: 50}
    outputs: [principal_stacked, fixed_action_stacked, pareto_frontier]
  - id: poisson_log
    # ...
    outputs: [principal_stacked]
```

`main_figures.py` would loop over this list and call the existing figure maker.
Only manuscript-used outputs should be requested. This is optional and should
not block the FOA cleanup.

## 2. Benchmarks and solver comparison

Keep the benchmark workflow separate.

- `output/timing_results.csv` and `output/machine_specs.txt` remain tracked.
- They are refreshed only on the designated standard server.
- Ordinary reproduction consumes those committed inputs and regenerates the TeX
  table/description; it does not rerun timing benchmarks.
- `py/solver_comparison.py` and `py/benchmark_assets.py` can remain as they are
  unless a small config-driven cleanup is obviously useful.

## 3. FOA summary pipeline: target structure

This is the main cleanup target. Aim for a small directory like:

```text
experiments/
  foa_paper.yaml              # exact paper specifications and figure rows
  foa.py                      # economic construction and FOA calculations
  run_foa.py                  # run only paper-selected specifications
  report_common.py            # shared rows/style helpers
  report_principal_figure.py
  report_fixed_action_figure.py
  tests/
```

The exact filenames can change, but there should be no separate general
replication package, asset graph, dependency graph, or broad experiment
framework.

### 3.1 One FOA configuration

The FOA YAML should contain:

- the exact economic specifications shown in either summary figure;
- shared numerical settings required to calculate those results;
- principal/fixed-action exercise selection;
- the exact row order and labels for the two figures;
- final output filenames.

The fixed-action figure should reuse the common row list and add only its two
extra intended-action specifications. Economic specifications should not be
copied merely because both exercises use them.

### 3.2 Keep the current numerical core

Retain the tested calculations that are needed for paper values:

- model and cost construction;
- utility/CE conversion;
- relaxed and full-GIC principal and fixed-action solves;
- global-deviation checks;
- threshold/refinement and reversal checks;
- monopsony and competitive benchmarks;
- support checks needed for the selected cases.

Organize these functions inside one focused module or a few small modules. Do
not rewrite working numerical code solely to obtain a new abstraction.

### 3.3 Strip prototype and diagnostic fat

Remove from the final paper pipeline:

- smoke, first-atlas, internal-atlas, preflight, and adverse-control suite
  machinery;
- specifications not shown in either final summary;
- action-bound variants, risk-neutral controls, CRRA gamma 3, and other excluded
  rows;
- diagnostic re-solves and diagnostic image galleries;
- historical review records embedded in the experiment manifest;
- internal atlas summary tables and warning atlases;
- compatibility CSV/JSON products retained only from prototype development;
- resume/task-hash machinery if it is not needed for the final paper run;
- `mock`, `prototype`, and `atlas` terminology in final commands and filenames.

Keep normal solver warnings and concise validation statuses in the saved paper
results. Removing diagnostic products does not mean suppressing failures.

### 3.4 Simple solve/render separation

Use two straightforward steps:

1. the runner reads the FOA YAML and writes one ignored results file/directory;
2. the two report scripts read those saved results and produce the final figures.

Report scripts must not solve models. A normal full reproduction runs the solve
step first and then both reports.

Final names should be semantic, for example:

- `figures/foa-principal-summary.pdf`;
- `figures/foa-fixed-action-summary.pdf`.

## 4. Concrete migration sequence

1. Put the approved common row list and fixed-action additions in the existing
   FOA YAML. **Started.**
2. Create a paper-only FOA manifest by copying only the 27 specifications used
   by at least one of the two summaries.
3. Give every retained case a single `paper` run selection and remove all suite,
   review, and diagnostic sections.
4. Point the current runner at that paper-only manifest and verify that it
   reproduces the approved numbers from a fresh output directory.
5. Merge duplicate benchmark-solving code where principal and fixed-action logic
   genuinely shares an implementation, while keeping the exercises distinct.
6. Remove diagnostic commands, internal summarizers, storage compatibility
   outputs, and unselected cases once the paper-only run passes.
7. Rename report outputs from `mock.*` to final semantic names.
8. Add both FOA figures and captions to the manuscript.
9. Optionally make `main_figures.py` read a short YAML list for asset types 1 and
   2; otherwise leave types 1--3 alone.
10. Update `make.sh` and the README with the simple final sequence.

## 5. Tests to retain

Keep focused tests for:

- utility/CE transformations;
- cost calibration and `C(lower action)=0`;
- Gaussian and Student-t action-zero invariants;
- global-deviation and transition logic;
- selected manifest expansion;
- principal/fixed-action separation;
- exact summary row counts/order;
- a few headline numerical regression intervals;
- report scripts operating from saved results only.

Delete tests whose only purpose is obsolete suite expansion, diagnostic output,
or compatibility formats after those features are removed.

## 6. Completion criteria

The cleanup is complete when:

1. `uv sync --frozen` works on a fresh clone;
2. one documented FOA command calculates exactly the paper-selected cases;
3. two report commands generate the approved summary PDFs with final names;
4. existing scripts generate the illustrative, Pareto, benchmark, and
   solver-comparison assets;
5. the manuscript compiles with both new summaries;
6. all tests pass;
7. generated model output and figures remain ignored;
8. only the controlled standard-server timing CSV and machine description are
   committed as generated numerical outputs;
9. `git status --short` is empty after a clean reproduction.
