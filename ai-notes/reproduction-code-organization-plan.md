# Paper Reproduction Code Organization Plan

## 1. Goal

Turn the repository into a small, declarative reproduction package that, from a
fresh clone, regenerates every numerical result, figure, and table used by the
paper.

The final repository should have these properties:

- one documented command builds all paper assets;
- one YAML manifest is the source of truth for economic specifications,
  numerical settings, paper selections, labels, and output filenames;
- each economic specification is defined once and reused across figures;
- solvers and renderers are separate: rendering never silently solves models;
- principal and fixed-action exercises remain distinct;
- only calculations required by the paper are run;
- generated model results, figures, tables, logs, and provenance are ignored by
  Git and reproducible from committed code, manifest, and `uv.lock`; the sole
  exception is the controlled timing-benchmark dataset, which is deliberately
  produced on the standard server and committed;
- filenames are final and semantic, with no `mock`, `prototype`, `atlas`, or
  debugging terminology in the paper pipeline;
- historical diagnostics and abandoned experiments are deleted from the working
  tree rather than maintained indefinitely; Git history remains the archive.

## 2. Scope of the final reproduction

Before moving code, construct an explicit asset inventory from the LaTeX source.
Every `\includegraphics` and generated `\input` must map to one declared asset.
At present this includes:

- Gaussian/log principal example;
- Gaussian/log fixed-action example;
- Poisson/log principal example;
- Gaussian/CARA principal example;
- Student-t/log principal example;
- solver-comparison figure;
- principal FOA-validity summary;
- fixed-action FOA-validity summary;
- benchmark table and machine description.

The inventory, not the current scripts, determines what survives. In particular,
the old example runner currently generates many wage, utility, stacked, and
Pareto files that the manuscript never includes. Those undeclared files should
no longer be produced.

## 3. Proposed repository structure

```text
replication/
  __init__.py
  __main__.py                 # single CLI
  config.py                   # load and validate YAML
  models.py                   # shared model/cost/spec construction
  results.py                  # typed result records and JSON/NPZ I/O
  solve/
    examples.py               # contracts/Pareto calculations used in figures
    foa.py                    # FOA scans and global-deviation checks
    benchmarks.py             # solver comparison; controlled timing refresh
  plot/
    style.py                  # paper colors, fonts, line/marker conventions
    examples.py
    foa_summary.py
    solver_comparison.py
  tables.py                   # TeX generated from model or benchmark inputs
  tests/
    test_config.py
    test_models.py
    test_foa.py
    test_assets.py
    test_reproduction_smoke.py

paper.yaml                    # sole paper calculation/asset manifest
build/                        # all generated intermediate results; ignored
figures/                      # generated final paper figures; ignored
output/                       # generated TeX ignored; controlled timing inputs tracked
make.sh                       # thin wrapper around the single CLI
```

The exact package name can change, but there should be one package rather than
parallel `py/` and `experiments/` implementations.

## 4. One declarative paper manifest

Replace the hard-coded loops in `py/main_figures.py`, the suite-oriented
`experiments/foa_experiments.yaml`, and the hard-coded summary `ROWS` lists with
one `paper.yaml` containing four sections.

### 4.1 Shared specifications

Each model is defined once with a stable ID. Variations explicitly extend a base
specification rather than copying all fields.

```yaml
specifications:
  gaussian_log:
    utility: {kind: log, initial_wealth: 50}
    distribution: {kind: gaussian, sigma: 50}
    action_bounds: [0, 180]
    target_action: 100
    revenue: {slope: 1}
    cost: {kind: quadratic, zero_at_lower_bound: true, calibration: paper_log}

  gaussian_crra_025:
    extends: gaussian_log
    utility: {kind: crra, gamma: 0.25, initial_wealth: 50}
```

The loader should resolve `extends`, reject unknown fields, detect cycles, and
save the fully resolved specification in generated result metadata.

### 4.2 Numerical profiles

Define reusable numerical profiles separately from economics, for example
`continuous_standard`, `student_t`, `discrete`, `foa_summary`, and
`solver_comparison`. A specification or calculation references a profile and
only declares genuine overrides.

### 4.3 Calculations

Declare only calculations consumed by an asset. Examples are:

- selected principal contracts at stated reservation wages;
- selected fixed-action contracts;
- a Pareto frontier only where the manuscript displays one;
- principal FOA scan and its monopsony/competitive benchmarks;
- fixed-action FOA scan and its separate benchmarks;
- solver timings and solver comparison.

The CLI builds a dependency graph from paper assets to calculations. There
should be no broad `internal_atlas`, `smoke`, `first_atlas`, diagnostic, or
preflight suite in the production manifest.

### 4.4 Paper assets

Figure order, labels, grouping, exercise, calculation dependencies, and final
filenames belong in the manifest. For example:

```yaml
assets:
  - id: foa_principal_summary
    kind: foa_summary
    exercise: principal
    output: figures/foa-principal-summary.pdf
    rows:
      - {header: Gaussian}
      - {spec: gaussian_log, label: "Baseline (log, sigma = 50)"}
      # ... exact paper rows ...
      - {header: "Student-t (empty safe region)"}

  - id: foa_fixed_action_summary
    kind: foa_summary
    exercise: fixed_action
    output: figures/foa-fixed-action-summary.pdf
```

This removes duplicated row lists and ensures both summaries use the same
specification IDs and style definitions while preserving their different
benchmarks and interpretation.

## 5. Shared model and solver code

### 5.1 Extract the durable core

Move the genuinely reusable parts of `experiments/prototype.py` into focused
modules:

- utility and certainty-equivalent transformations;
- economic model construction and cost normalization;
- support construction and required support checks;
- global-deviation oracle;
- principal and fixed-action solver wrappers;
- FOA transition calculation;
- monopsony and competitive benchmarks;
- serialization of the paper result schema.

Do not move historical control flow, suite expansion, diagnostic plotting,
warning atlases, or compatibility output merely to preserve it.

### 5.2 One constructor for all paper figures

The old figure scripts and FOA experiments currently construct overlapping
Gaussian, Poisson, CARA, and Student-t models independently. All assets must use
the same resolved YAML specification and the same model constructor. This is
especially important for revenue scaling, cost normalization, action domains,
and support grids.

### 5.3 Keep important validation, remove diagnostic products

Removing the diagnostic atlas does not mean removing numerical safeguards. A
paper solve should still perform and save concise checks needed to trust it:

- support mass and score diagnostics;
- global-deviation validation;
- action-bound checks;
- transition/reversal checks;
- relevant full-GIC checks;
- warnings and dependency/code provenance.

Failures should stop the paper build or produce a clearly failed asset. The
pipeline should not generate diagnostic folders, contract-comparison galleries,
internal warning tables, or broad robustness summaries that are absent from the
paper.

## 6. Result and rendering pipeline

Use a strict two-stage design for model calculations:

1. **Solve stage** writes declared paper calculation results under
   `build/results/` as JSON plus NPZ only where arrays are needed.
2. **Render stage** reads those saved results and writes PDFs/PNGs/TeX under
   `figures/` and `output/`. It must never call a solver.

Timing benchmarks are a deliberate exception to the ordinary fresh-build path.
The renderer reads their committed standard-server results and creates the
paper's benchmark table; it does not rerun timings during a normal build.

Every result should include:

- calculation and specification IDs;
- fully resolved economic and numerical configuration;
- package versions and Git commit/dirty status;
- classification tolerances;
- concise validation status and warnings;
- numerical values and only the arrays needed by a declared paper asset.

Default builds should start clean rather than resume stale calculations. If a
cache is later useful, its key must include the resolved configuration,
implementation version or Git tree hash, and dependency lock—not configuration
alone.

## 7. Commands and fresh-clone behavior

Expose one CLI with a small command surface:

```bash
uv sync --frozen
uv run python -m replication build        # solve and render every paper asset
uv run python -m replication solve        # calculations only
uv run python -m replication render       # saved results only
uv run python -m replication check        # manifest/asset consistency
uv run python -m unittest discover -s replication/tests -v
```

`./make.sh` should be a strict shell wrapper (`set -euo pipefail`) around the
frozen install, tests/checks, and `build`. Optionally add `./make.sh paper` to
compile LaTeX after assets exist.

The build should create missing directories and work when `build/`, `figures/`,
and generated `output/` files do not exist.

## 8. Output and benchmark policy

Apply the no-generated-output rule to model results and paper assets:

- ignore all of `build/` and `figures/`;
- ignore generated TeX and ordinary model outputs under `output/`;
- do not commit FOA results, contracts, arrays, rendered figures, diagnostic
  data, or manuscript build products.

Timing benchmarks are the explicit exception. Keep
`output/timing_results.csv` and `output/machine_specs.txt` tracked because they
are deliberately refreshed only on the standard server. This prevents ordinary
fresh-clone builds on heterogeneous hardware from changing the paper's timing
numbers. The normal paper build consumes these committed inputs and regenerates
`benchmarks_table.tex` and `specs_description.tex` without rerunning timings.

Keep benchmark execution outside the normal `replication build` command. Give it
a clearly named manual command such as:

```bash
uv run python -m replication.benchmark refresh
```

Document that this command must be run only on the designated standard server,
that it overwrites the two controlled tracked files, and that their diff must be
reviewed before commit. The benchmark configuration and implementation should
still be organized and declarative, but benchmark refresh is an occasional
maintainer workflow rather than part of fresh-clone paper reproduction.

## 9. Stable asset names and LaTeX integration

Rename generated files to semantic final names and update LaTeX once:

- `figures/foa-principal-summary.pdf`;
- `figures/foa-fixed-action-summary.pdf`;
- similarly concise names for the five example figures and solver comparison.

Do not retain `mock.pdf`, parameter-encoded directory names, or multiple unused
variants. PNG may be generated for review, but LaTeX should reference the final
PDF name.

Add an automated asset check that parses or maintains an explicit inventory of
all manuscript `\includegraphics` and generated `\input` paths and verifies:

- every referenced generated asset is declared in `paper.yaml`;
- every declared paper asset is referenced by the manuscript, unless explicitly
  marked supplementary;
- no manuscript path contains `mock` or a retired output location.

## 10. Code to retire after migration

After replacement assets agree with the approved figures and paper numbers,
delete rather than preserve redundant pipelines:

- `py/main_figures.py`;
- `py/figure_maker.py` after its needed plotting functions are migrated;
- standalone solver-comparison scripts after migration into the package;
- the standalone timing script after its logic moves to the separate,
  standard-server-only benchmark refresh command;
- `experiments/run_prototype.py`;
- `experiments/storage.py`;
- `experiments/summarize_internal.py`;
- `experiments/diagnose_problematic.py`;
- the prototype/atlas YAML and historical review sections;
- old report and benchmark scripts after their functionality is unified;
- `interactive_dashboard/` if it is not part of the paper replication package;
- stale generated LaTeX auxiliaries already covered by `.gitignore`.

Do not delete old code until its declared paper asset has been reproduced by the
new path. No compatibility wrapper is needed after cutover; Git history is the
archive.

## 11. Tests and acceptance criteria

### 11.1 Unit and schema tests

Retain focused tests for economic and numerical invariants, including:

- utility/CE round trips;
- `C(lower action)=0` and calibration values;
- Gaussian and Student-t paper action sets beginning at zero;
- distribution/support checks;
- global-deviation examples;
- principal/fixed-action separation;
- transition and reversal logic;
- manifest inheritance and validation.

### 11.2 Paper regression tests

Use small numerical assertions rather than committed generated output:

- selected headline thresholds and benchmarks lie in approved tight intervals;
- expected paper specification and asset IDs are present;
- expected row counts and ordering are fixed by the manifest;
- all final filenames are semantic and unique;
- renderers can operate with solver calls disabled;
- the manuscript asset inventory is complete.

### 11.3 Fresh-clone acceptance test

The migration is complete only when this succeeds from a clean checkout with no
generated directories:

1. `uv sync --frozen`;
2. run all tests;
3. run the complete paper build without `--resume`; this consumes the committed
   standard-server benchmark inputs and does not rerun timings;
4. verify every declared PDF/PNG/TeX/CSV exists;
5. compile the manuscript and supplement;
6. inspect the compiled figures at final scale;
7. confirm `git status --short` remains empty.

## 12. Migration sequence

### Phase 1: Inventory and freeze decisions

1. Build the manuscript asset inventory.
2. Mark each current script output as paper-used or delete-after-migration.
3. Record the approved final FOA row selections, labels, colors, and benchmark
   interpretations in the new manifest.
4. Decide whether the interactive dashboard is outside replication scope
   (recommended: remove it).

### Phase 2: Manifest and package skeleton

1. Add `paper.yaml` and schema validation.
2. Add the package/CLI skeleton and shared style module.
3. Add manifest and asset-inventory tests.
4. Keep current scripts working during this phase.

### Phase 3: Migrate the FOA summaries

1. Extract the model constructor, CE utilities, global-deviation oracle, and FOA
   calculations from `prototype.py`.
2. Define only the specifications displayed by the principal and fixed-action
   paper summaries.
3. Unify principal/fixed benchmark calculation behind one shared API with an
   explicit exercise type.
4. Render both approved figures with final filenames from one summary renderer.
5. Compare values and PDFs against the approved local versions.

### Phase 4: Migrate existing example figures

1. Replace hard-coded model blocks in `py/main_figures.py` with manifest jobs.
2. Extract only plotting functions needed by manuscript-referenced assets.
3. Harmonize overlapping model definitions with the FOA specifications.
4. Stop producing unused `cm_*`, `pp_*`, and Pareto variants.

### Phase 5: Migrate solver comparison and benchmarks

1. Move solver-comparison and timing specifications into the manifest.
2. Migrate solver-comparison calculations into the ordinary solve/render
   pipeline.
3. Add the separate standard-server-only timing refresh command.
4. Keep the controlled timing CSV and machine specification tracked, and have
   the normal render stage generate their TeX presentation without rerunning
   timings.

### Phase 6: Cut over the paper

1. Rename outputs to final semantic paths.
2. Update LaTeX references and captions, including the harmonized Poisson
   calibration and both FOA summary notes.
3. Run the manuscript asset check and compile at final scale.

### Phase 7: Delete old infrastructure

1. Delete superseded scripts, diagnostic commands, internal suites, dashboard,
   and prototype terminology.
2. Simplify `.gitignore`, root README, and `make.sh` around the final workflow.
3. Ensure no tracked generated output remains except the two explicitly
   controlled standard-server timing files.

### Phase 8: Clean-room reproduction

Run the fresh-clone acceptance test, compare paper numbers and appearance one
last time, and only then declare the old pipeline retired.

## 13. Non-goals

- Rebuilding a general-purpose experiment platform inside this paper repo.
- Preserving every internal robustness run in the current working tree.
- Maintaining compatibility with old ignored result directories.
- Automatically generating diagnostics not included in the paper.
- Adding new parameter sweeps during cleanup.
- Changing economic results while moving code; any substantive recalibration
  must be a separate, explicit change.
