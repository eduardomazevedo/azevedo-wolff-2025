# FOA Numerical Experiments: Implementation Plan

## 1. Objective

Build a reproducible numerical experiment suite testing the paper's practical conclusion:

> With moderate risk aversion and a sufficiently large safe low-score region, the first-order approach (FOA) is typically valid beginning at a very low reservation utility, often close to the utility delivered by a monopsonist when participation is slack.

The implementation must support two distinct exercises:

1. **Principal's problem:** the intended action is chosen endogenously. This is the main economic exercise and the source of the headline dollar results.
2. **Fixed-action cost minimization:** the intended action is fixed. This is closest to the theorem and is the cleanest diagnostic of its mechanism.

We expect similar patterns, but must never pool or conflate the two exercises.

The project will generate a large internal atlas. The paper will report a smaller, preselected set of economically interpretable results and robustness summaries. Internal outputs must retain every specification, warning, unresolved result, and adverse case.

## 2. Core estimands and benchmarks

All utility thresholds should also be represented as **certainty-equivalent reservation wages** in the model's dollar units. If total wealth utility is `u(w0 + w)`, define

\[
\operatorname{CEw}(U)=u^{-1}(U)-w_0.
\]

Here `U` is the agent's total expected utility net of effort cost. This is the wage in a deterministic outside option that gives utility `U`.

### 2.1 True monopsony benchmark

Define the **true monopsony utility** \(U_M^G\) as the utility delivered to the agent in the true principal optimum when the participation constraint is slack. The contract must satisfy global incentive compatibility (GIC).

Numerically, solve the full principal's problem at successively lower reservation utilities until:

- IR is slack by more than tolerance;
- the optimal action, contract cost, profit, and delivered utility are stable as the reservation requirement is lowered further; and
- the contract passes the independent global-deviation check.

Record

\[
\bar w_M^G=\operatorname{CEw}(U_M^G).
\]

This is the primary economic reference point.

### 2.2 Relaxed zero-IR-multiplier benchmark (dropped from the atlas)

The prototype computed the principal's relaxed optimum with the IR multiplier fixed at \(\lambda=0\). In both smoke cases this benchmark was far below the true monopsony utility and selected the upper action boundary. It therefore does not measure the economically relevant statement that FOA validity begins near the true monopsony outcome.

Do **not** compute this benchmark in the main internal atlas or reporting pipeline. Retain the prototype routine only as an optional debugging tool. The sole primary monopsony benchmark is the utility delivered by the true GIC principal optimum with slack IR, \(U_M^G\).

### 2.3 FOA deviation diagnostic

For a relaxed contract \(w^R\) and intended action \(a_0\), define

\[
G_U=\max_{a\in\mathcal A}U(w^R,a)-U(w^R,a_0).
\]

FOA validity should be diagnosed from this global deviation calculation, not solely from the solver's Boolean flag. Also report a certainty-equivalent deviation gain:

\[
G_{CE}=\operatorname{CEw}\!\left(\max_a U(w^R,a)\right)
       -\operatorname{CEw}\!\left(U(w^R,a_0)\right).
\]

Classifications:

- **valid:** `G_CE <= valid_tolerance` and all numerical checks pass;
- **invalid:** `G_CE >= invalid_tolerance` and a profitable deviation is located robustly;
- **unresolved:** the gain lies in the numerical gray zone, solvers disagree, a boundary is active unexpectedly, or validation fails.

Use separate valid and invalid tolerances so numerical noise is not forced into a binary result.

### 2.4 FOA threshold and reversal statistics

For each specification define the **persistent FOA threshold**

\[
\bar w^*=\inf\{\bar w:\text{FOA is valid for every tested }\bar w'\geq\bar w\}.
\]

This, rather than the first isolated valid point, is the primary threshold. Report:

\[
\Delta_M=\bar w^*-\bar w_M^G.
\]

Report gaps in dollars and normalized by:

- output standard deviation when meaningful;
- expected output at the relevant optimum;
- expected compensation at the monopsony optimum.

Also record:

- first valid reservation wage;
- all valid/invalid transitions;
- number and location of reversals;
- whether validity is monotone on the tested range;
- whether persistent validity is reached before the upper end of the range.

A `valid -> invalid` transition must trigger denser local evaluation and stronger numerical validation. Never silently impose monotonicity.

## 3. Implementation order

Implementation should proceed in four stages. Do not begin the large atlas before Stage 1 tests pass.

### Stage 1: Validate basic numerical functions

Create a small, well-tested experiment library rather than placing logic directly in plotting scripts. At minimum implement and test:

1. **Utility and CE transformations**
   - `utility_from_reservation_wage`
   - `reservation_wage_from_utility`
   - round-trip tests for log, CRRA, and CARA;
   - explicit tests that effort cost is handled consistently when converting delivered utility.

2. **Problem construction**
   - a common constructor for distribution, utility, cost, action bounds, and computational grids;
   - consistent cost normalization across utility specifications;
   - no hidden global state or closure-capture bugs in cost functions.

3. **Distribution diagnostics**
   - density/mass normalization;
   - omitted tail mass;
   - score mean approximately zero at the intended action;
   - analytic versus finite-difference checks for \(f_a\), \(f_{aa}\), and the score;
   - discrete-support truncation checks.

4. **Global-deviation oracle**
   - coarse action scan;
   - detection of all candidate local maxima;
   - bounded local optimization from multiple starts;
   - explicit endpoint checks;
   - return best action, utility gain, CE gain, local maxima, and diagnostics;
   - tests on hand-constructed one- and two-maximum examples.

5. **Relaxed and full problem wrappers**
   - fixed-action relaxed cost minimization;
   - fixed-action full cost minimization;
   - relaxed principal's problem;
   - full principal's problem;
   - extract a common result schema independent of solver internals.

6. **Monopsony routines**
   - true full-GIC slack-IR benchmark;
   - plateau/stability tests for the benchmark;
   - IR slackness and multiplier checks;
   - keep the relaxed \(\lambda=0\) routine disabled and available only for debugging.

7. **Threshold and reversal finder**
   - broad initial scan;
   - adaptive refinement around every classification transition;
   - persistent-threshold calculation;
   - explicit handling of unresolved intervals;
   - synthetic tests with monotone and nonmonotone validity sequences.

8. **Safe-region metrics**
   - safe cutoff or set of safe cutoffs;
   - safe mass under \(a_0\);
   - safe incentive capacity;
   - safe curvature;
   - safe width and normalized width where finite;
   - theorem-relevant condition values;
   - numerical certification grid over the full action set, with clear indication that this is a numerical diagnostic rather than a proof.

9. **Independent validation path**
   - compare selected active-set solutions with the discretized CVXPY/Clarabel formulation;
   - compare full and relaxed objectives when FOA is classified as valid;
   - convergence checks over outcome grids, action grids, tail ranges, and optimizer starts.

Initial regression tests should reproduce the qualitative paper examples:

- Gaussian/log: failure at sufficiently low reservation wage and validity shortly above it;
- Poisson/log: analogous low threshold;
- Gaussian/CARA: analogous pattern;
- Student-\(t\)/log: common nonlocal deviations and lack of the safe-tail mechanism.

### Stage 2: Declarative YAML experiment manifest

After Stage 1, create a tracked manifest, proposed path:

`experiments/foa_experiments.yaml`

The YAML file is the source of truth for every run. The runner must expand it deterministically into atomic tasks. Avoid hand-coded case loops in figure scripts.

Suggested top-level schema:

```yaml
schema_version: 1
experiment_id: foa-atlas-v1
seed: 2025
units: USD_1000

numerics:
  deviation:
    coarse_action_points: 241
    multistart_count: 12
    valid_tolerance_ce: 1.0e-4
    invalid_tolerance_ce: 1.0e-3
  threshold:
    initial_points: 21
    refinement_tolerance_wage: 0.01
    test_reversals: true
  validation:
    outcome_grids: [201, 401, 801]
    action_grids: [241, 481]
    convex_crosscheck: selected

calibrations:
  paper_baseline:
    initial_wealth: 50
    target_action: 100
    revenue: linear
    cost: quadratic
    cost_normalization: local_consumption_equivalent

suites:
  smoke:
    enabled: true
    report_group: internal
    # Minimal cases used by CI and development.

  common_distributions:
    enabled: true
    report_group: main_candidate
    # Gaussian, Poisson, exponential, gamma, geometric,
    # binomial/Bernoulli where computationally appropriate.

  safe_region_variation:
    enabled: true
    report_group: mechanism
    # Noise, action-space width, tail shape, and other parameters.

  risk_aversion:
    enabled: true
    report_group: robustness
    # CRRA gamma grid and CARA calibrations.

  adverse_controls:
    enabled: true
    report_group: negative_control
    # Student-t, near-risk-neutral cases, finite-state counterexample.

tasks:
  principal_problem: true
  fixed_action: true
  monopsony_full: true
  threshold_search: true
  reversal_search: true
  safe_metrics: true
  validation: true
```

Each expanded task should receive a stable hash from its economic and numerical configuration. This allows resume, caching, deduplication, and exact provenance.

The runner should support commands conceptually like:

```bash
uv run python -m experiments.run --manifest experiments/foa_experiments.yaml --suite smoke
uv run python -m experiments.run --manifest experiments/foa_experiments.yaml --suite common_distributions
uv run python -m experiments.run --manifest experiments/foa_experiments.yaml --all --resume
uv run python -m experiments.validate --experiment-id foa-atlas-v1
uv run python -m experiments.report --experiment-id foa-atlas-v1
```

Exact module names may change, but one command must be able to run all enabled tasks from the manifest without editing Python.

### Stage 3: Internal atlas

Run a larger internal design than will appear in the paper.

#### Common distribution collection

Include, subject to implementation support and numerical validation:

- Gaussian;
- Poisson;
- exponential;
- gamma;
- geometric;
- Bernoulli/binomial;
- Student-\(t\) as an adverse comparison.

The common-family collection establishes breadth. Do not treat all families as equally informative about the mechanism.

#### Risk-aversion variation

At minimum:

- log;
- CRRA \(\gamma\in\{0.25,0.5,1,1.5,2,3\}\), treating \(\gamma=1\) as log;
- CARA calibrated to relative risk aversion approximately \(0.25,0.5,1,2\) at a declared reference wealth.

Use the same local consumption-equivalent effort-cost normalization across utilities. Save both the primitive cost coefficient and its economic interpretation.

#### Smart safe-region variation

Use controlled variation in addition to family comparisons:

- Gaussian \(\sigma\), including the paper's 10, 20, and 50 thousand dollar values;
- width and lower endpoint of the action set relative to the intended action;
- initial wealth, e.g. 25, 50, 100, and 200 thousand dollars;
- effort-cost curvature;
- tail-shape or degrees-of-freedom parameters;
- if feasible, a smoothly parameterized location family that moves from light-tailed/log-concave behavior toward heavy-tailed/nonmonotone-score behavior.

For Gaussian output, action-space variation is especially useful: uniform safety over the full action set changes as the lower action bound moves relative to \(a_0\), even though Gaussian safe width is formally infinite.

#### Fixed-action design

For each selected economic environment, evaluate several interior intended actions, including:

- below the monopsony action;
- near the monopsony action;
- near the paper's \$100k reference action;
- above it where locally implementable.

Do not use intended actions too close to computational boundaries. If a solution approaches a boundary, automatically extend the action interval when economically permissible or classify the case as boundary-contaminated.

#### Principal-problem design

For each environment:

1. compute the true full-GIC slack-IR monopsony benchmark;
2. scan reservation wages below and above that benchmark;
3. solve the relaxed principal problem at each wage;
4. run the independent deviation oracle;
5. refine every validity transition;
6. compute the persistent threshold;
7. cross-check selected points with the full problem.

The reservation range should be defined relative to the monopsony benchmarks where possible, while retaining dollar-valued coordinates.

### Stage 4: Reporting layer

Reporting code must consume saved atomic results; it must never rerun models implicitly.

## 4. Numerical safeguards and acceptance rules

### 4.1 Around every transition

For points bracketing an FOA transition:

- increase action-grid density;
- use multistart local maximization of agent utility;
- increase the outcome grid and tail range;
- check endpoints and all detected local maxima;
- compare the relaxed contract with the full convex solution where feasible;
- verify objective and constraint residuals;
- save plots of the agent objective for inspection.

### 4.2 Tail and support checks

For continuous distributions, save omitted probability mass and relevant omitted moments. For discrete distributions, increase the support until results and thresholds stabilize.

Do not classify a result as valid merely because no deviation appears on a coarse action grid.

### 4.3 Solver warnings

Save warnings verbatim. A task with serious warnings is unresolved until independently validated. In particular:

- solver fallback is not itself a failure, but requires logging;
- action-boundary optima require review;
- disagreement between active-set and convex solvers requires review;
- nonfinite values, overflow, or failed reparametrizations invalidate automatic classification.

### 4.4 Existing atlas status

The current ignored `figures/atlas` outputs are useful pilot evidence but should not be treated as final data. They use a coarse 17-point reservation grid and several cells contain extensive solver warnings or boundary actions. They suggest:

- Gaussian transitions near \$0--1k across the utility cases tried;
- Poisson transitions near \(-\$0.25k\)--\$0.125k;
- broad validity in several exponential cases;
- frequent failures in Student-\(t\) cases.

The generation source for these atlas artifacts is not currently tracked. The new implementation must be tracked and manifest-driven.

## 5. Output and provenance layout

Suggested layout:

```text
experiments/
  foa_experiments.yaml
  run.py
  validate.py
  report.py
  schema.py
  problem_factory.py
  deviation.py
  thresholds.py
  safe_region.py
  monopsony.py
  tests/

output/foa-atlas-v1/
  manifest.snapshot.yaml
  environment.json
  task_index.parquet
  atomic/<task_hash>.json
  arrays/<task_hash>.npz
  diagnostics/<task_hash>/
  combined_results.parquet
  validation_results.parquet
  failures.parquet
  warnings.parquet
  summary_tables/

figures/foa-atlas-v1/
  internal/
  paper/
```

Every atomic result should contain:

- experiment and task IDs;
- git commit;
- moralhazard package commit/version;
- complete economic configuration;
- complete numerical configuration;
- timestamps and runtime;
- solver status and warnings;
- constraint residuals;
- monopsony benchmarks;
- deviation diagnostics;
- FOA classification;
- validation status.

Use tabular formats for summaries and JSON/NPZ for detailed per-task outputs. Generated figures are not the authoritative data.

## 6. Internal versus paper reporting

### 6.1 Internal atlas

The internal report should include:

- every specification and threshold;
- every transition and reversal;
- all unresolved and failed cells;
- solver-warning and boundary maps;
- safe-region diagnostics;
- convergence comparisons;
- both principal and fixed-action results.

This atlas is for diagnosis, robustness, and avoiding selection based on attractive outcomes.

### 6.2 Main paper candidates

#### Threshold table

For a concise set of realistic baseline cases, report:

| Distribution | Utility | Risk aversion | True monopsony CE | Persistent FOA threshold | Gap from true monopsony | Safe-region metric |
|---|---:|---:|---:|---:|---:|---:|

Dollar values should remain prominent.

#### Threshold/deviation figure

Use small multiples with:

- x-axis: reservation wage minus true monopsony CE, in \$1,000s;
- y-axis: CE gain from the best deviation;
- zero line: FOA boundary;
- marker for true monopsony;
- vertical line or interval for the persistent threshold;
- unresolved numerical region shown explicitly.

#### Mechanism figure

For selected cases plot against reservation wage:

- limited-liability strike/kink;
- safe-region boundary;
- probability or mass where limited liability binds;
- intended action and best deviating action;
- CE deviation gain;
- curvature/concavity diagnostic where available.

This connects the threshold finding to the theorem's mechanism.

#### Broad appendix atlas

Report heatmaps or compact tables over:

- distributions;
- risk aversion;
- noise/tail parameters;
- initial wealth;
- safe-region size/capacity;
- fixed intended action versus endogenous action.

Useful descriptive summaries include the fraction of predeclared, numerically resolved specifications for which:

- FOA holds at the true monopsony point;
- the threshold lies within \$1k, \$5k, or \$10k of true monopsony;
- FOA remains valid throughout the rest of the tested reservation range;
- reversals are detected.

Always give the denominator and separately count unresolved tasks. These are numerical frequencies, not statistical hypothesis tests.

### 6.3 Adverse cases

Report at least one Student-\(t\) case and the risk-neutral counterexample. The goal is to show that the experiment can detect failure when risk aversion or the safe-region structure is absent, not to claim universal validity.

## 7. Interpretation discipline

The numerical experiments can support statements such as:

- FOA thresholds are close to monopsony utility in a broad, predeclared set of calibrated environments;
- threshold gaps tend to be smaller when safe mass/capacity is larger;
- failures and delayed validity are more common when scores are nonmonotone or safe regions disappear;
- fixed-action and principal-problem experiments display similar or different patterns.

They should not be described as proving necessity, monotonicity, or universal validity. Correlations between safe-region metrics and thresholds are descriptive mechanism evidence.

## 8. Immediate implementation checklist

1. Create the experiment package and smoke-test manifest.
2. Implement CE conversion and common result schemas.
3. Implement and unit-test the global-deviation oracle.
4. Implement and validate the full-GIC slack-IR monopsony routine.
5. Implement adaptive transition and reversal detection.
6. Reproduce the four qualitative regression cases.
7. Implement safe-region diagnostics for Gaussian, Poisson, exponential, and Student-\(t\).
8. Add independent convex-solver checks around transitions.
9. Finalize the YAML schema and expand it deterministically into tasks.
10. Run the smoke suite and inspect every diagnostic.
11. Run the common-distribution and smart-variation suites.
12. Freeze an internal atlas before selecting the smaller paper presentation.

## 9. Prototype status (initial smoke run)

A tracked prototype now lives in `experiments/`, driven by
`experiments/foa_experiments.yaml`. It implements CE transformations, a
grid-plus-local-refinement deviation oracle, separate principal and fixed-action
runs, a true-monopsony scan, transition refinement, and reversal detection.
Seventeen unit tests pass. Results are written to
`output/foa-prototype/`. The prototype now also saves grid-based mass,
score-mean, and action-derivative diagnostics, uses multistart deviation
refinement, extends the monopsony scan downward adaptively when needed,
cross-checks transition brackets with the discretized CVXPY/Clarabel solver,
performs adaptive support plus transition-contract grid convergence checks,
and computes theorem-specific safe-region diagnostics on the configured grids.

The first smoke run produced the following preliminary transition brackets (all
wages in thousands of dollars):

- Gaussian/log principal problem: FOA changes from invalid to valid around
  `[0.625, 0.688]`; the interval is wider because a midpoint falls in the
  predeclared numerical gray zone.
- Gaussian/log fixed action `a0=100`: transition around `[1.156, 1.172]`.
- Poisson/log principal problem and fixed action `a0=7`: transition around
  `[-0.0156, 0]`.
- Student-`t`/log fixed action `a0=100` is now a negative smoke control. FOA
  is invalid at every tested reservation wage (`-1`, `20`, `50`, and `100`),
  with CE deviation gains increasing from about `$12.7k` to `$37.8k`; the best
  deviation is `a=0` throughout. There were no solver warnings.
- No reversals appeared on these small grids.

The full-GIC slack-IR monopsony CE wages were approximately `0.553` for the
Gaussian case and `0` for Poisson. Thus the preliminary principal-problem gaps
from true monopsony are very small. The prototype also showed that the relaxed
`lambda=0` points were far lower and selected upper-bound actions. We therefore
dropped that benchmark from the planned atlas and retained only the true
full-GIC slack-IR monopsony benchmark.

These are prototype diagnostics, not paper-ready estimates. Safe-region
metrics, adaptive support checks, and convex-solver cross-validation remain to
be implemented.

## 10. Handoff brief

### Current working implementation

- Main plan: `ai-notes/foa-experiments-implementation-plan.md`.
- Declarative prototype manifest: `experiments/foa_experiments.yaml`.
- Prototype library: `experiments/prototype.py`.
- CLI: `experiments/run_prototype.py`.
- Tests: `experiments/tests/test_prototype.py`.
- Usage notes: `experiments/README.md`.
- Latest smoke outputs: `output/foa-prototype/`.

Run the current baseline before making changes:

```bash
uv sync
uv run python -m unittest discover -s experiments/tests -v
uv run python -m experiments.run_prototype --suite smoke
```

Expected baseline: five tests pass; Gaussian/log and Poisson/log transition
from invalid to valid at low reservation wages; Student-`t`/log remains invalid
at all four tested wages; no smoke case emits solver warnings.

### Decisions already made

1. Use the true full-GIC principal optimum with slack IR as the only monopsony
   benchmark.
2. Do not spend atlas runtime on the relaxed `lambda=0` principal benchmark. Its
   code may remain as a disabled debugging path.
3. Keep principal and fixed-action experiments separate.
4. Include both common distributions and controlled safe-region variation.
5. Search explicitly for validity reversals rather than imposing monotonicity.
6. Include adverse controls, especially Student-`t` and risk-neutral cases.
7. Build a larger internal atlas than the subset eventually reported.

### Known limitations of the prototype

- Transition refinement stops and reports an interval when a midpoint falls in
  the valid/invalid gray zone. It does not yet perform higher-precision
  validation to resolve that interval.
- The deviation oracle now uses a dense grid, top-value and uniform multistarts,
  endpoint checks, and bounded refinement, with synthetic narrow/tied-maximum
  tests. It still needs convergence comparisons across action grids and starts.
- The true-monopsony routine now searches downward adaptively, requires two
  slack-IR points, and checks action, profit, expected wage, delivered CE, and
  independent GIC stability. It still needs an independent full-solver
  cross-check and grid/support convergence tests.
- Continuous and discrete grid mass, score means, and action derivatives are
  now diagnosed, and supports expand adaptively against mass and score-mean
  tolerances. Gaussian and Poisson pass on their baseline supports. Student-t
  remains explicitly `not_converged`: four expansions reduce omitted mass from
  about 3 percent to about 0.47 percent, still well above tolerance. Relevant
  omitted tail moments are not yet certified.
- Relaxed transition classifications are stable over baseline, dense, and
  expanded-support grids in Gaussian and Poisson. The remaining unresolved
  cells arise from the independent CVXPY comparison, not support/grid changes.
- Safe-region cutoff, mass, capacity, curvature, width, normalized width, and
  the log curvature-width condition are implemented as explicitly numerical
  full-action-grid diagnostics. They still need action-grid and derivative-step
  convergence checks before atlas expansion.
- CVXPY/Clarabel cross-checks are integrated at transition endpoints. The
  latest run passes both Gaussian fixed-action endpoints and the invalid-side
  Poisson endpoints. It leaves the invalid-side Gaussian principal full solve
  unresolved (active-set full GIC is in the CE gray zone) and the valid-side
  Poisson relaxed solves unresolved (the active-set and CVXPY contracts have
  nearly identical costs but disagree at the tight CE classification boundary).
  These cells should remain unresolved pending grid/support convergence rather
  than prompting general solver debugging.
- Results are JSON/CSV prototype files rather than hashed atomic tasks with
  resume and provenance.
- The current YAML is a smoke manifest, not yet the complete atlas todo grid.

### Next agent's recommended sequence

1. Add dedicated tests for monopsony plateau selection and GIC status handling.
2. Add outcome-support, density normalization, score-mean, and derivative tests.
3. Strengthen the deviation oracle and test it on synthetic narrow and tied
   maxima.
4. Integrate selected CVXPY/Clarabel cross-checks, beginning with points around
   the Gaussian and Poisson transition brackets.
5. Make monopsony search adaptive and verify the Gaussian and Poisson benchmark
   values under grid/support refinement.
6. Implement safe-region diagnostics first for Gaussian, Poisson, exponential,
   and Student-`t`.
7. Only after those checks pass, expand the YAML into the first small atlas:
   log utility crossed with Gaussian, Poisson, exponential, and Student-`t`,
   with principal and fixed-action tasks where appropriate.
8. Inspect all warnings and unresolved cells before adding CRRA/CARA and larger
   parameter sweeps.

Do not begin paper figures or broad summary statistics until the validation
steps above pass. Reporting code should consume saved experiment outputs and
must not rerun models implicitly.
