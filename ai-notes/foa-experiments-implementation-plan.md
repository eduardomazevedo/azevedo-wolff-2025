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

## 8. Implementation checklist and current state

Completed:

1. Experiment package and declarative smoke manifest.
2. CE transformations and common result schemas.
3. Multistart global-deviation oracle with endpoint checks and synthetic tests.
4. Adaptive full-GIC slack-IR monopsony routine with plateau checks.
5. Adaptive transition refinement and reversal detection.
6. Gaussian/log, Poisson/log, and Student-`t`/log regression cases.
7. Adaptive support diagnostics and use of the certified expanded support in actual solves.
8. Density normalization, score-mean, and action-derivative diagnostics.
9. Safe-region cutoff, mass, capacity, curvature, width, normalized width, and theorem-condition diagnostics.
10. Safe-metric convergence over action grids and derivative steps.
11. Selected CVXPY/Clarabel cross-checks around transitions.
12. First small internal atlas: log utility crossed with Gaussian, Poisson, exponential, and Student-`t`.
13. Manifest-driven diagnostic figures for all human-reviewed suspicious cells.
14. Stable SHA-256 task hashes, atomic JSON records, task indexes, manifest/environment snapshots, resume, failure retention, and nonfinite-value path recording.
15. Deterministic Cartesian YAML expansion for controlled case families.
16. Separate strict numerical and human/economic review records, seeded with the five existing human assessments.
17. Analytic-versus-grid omitted first- and second-moment diagnostics where the moments exist.
18. Remaining common log-utility families: gamma, geometric, Bernoulli, and binomial.
19. Controlled Gaussian variation in sigma, action bounds, wealth, and effort-cost scale.
20. CRRA and CARA grids with local consumption-equivalent cost normalization, plus explicit cost-coefficient metadata.
21. Multiple fixed intended actions and a Gaussian risk-neutral adverse control.
22. First expanded internal run with 29 predeclared tasks and an internal-only saved-results summarizer.

Still to do before freezing the internal atlas:

1. Review the expanded atlas's warnings, boundary-contaminated cases, and strict unresolved cells without erasing provenance.
2. Resolve or retain as failed the two high-risk-aversion tasks (CARA `alpha=0.04` and CRRA `gamma=3`) whose agent-utility scans became nonfinite. Do not let these failures block inspection of the other tasks.
3. Decide whether the four boundary monopsony actions in low-risk-aversion/low-cost cases require wider action ranges; explicit boundary-contamination diagnostics are now implemented for future solves and derived in the saved-results summary.
4. Add any final predeclared gamma/binomial parameter variation only if it is needed for internal mechanism coverage.
5. Freeze the manifest and complete internal atlas before doing any paper selection or paper reporting.

## 9. Current numerical results and human review

A tracked implementation lives in `experiments/`, driven by `experiments/foa_experiments.yaml`. Twenty-two unit tests pass. Generated output is ignored by Git.

### 9.1 Smoke and first-atlas results

All wages and CE gains below are in the manifest's `USD_1000` units unless a dollar sign is stated explicitly.

- **Gaussian/log principal:** invalid-to-valid transition bracket `[0.625, 0.688]`.
- **Gaussian/log fixed action `a0=100`:** transition bracket `[1.156, 1.172]`.
- **Poisson/log principal and fixed action `a0=7`:** transition bracket `[-0.0156, 0]`.
- **Exponential/log principal:** transition bracket approximately `[-0.171875, -0.15625]`.
- **Exponential/log fixed action `a0=100`:** valid at every tested reservation wage beginning at `-1`.
- **Student-`t`/log fixed action `a0=100`:** robustly invalid at every tested wage. On the support-expanded first-atlas run, CE deviation gains range from about `$13.7k` to `$42.0k`, with best deviation `a=0` throughout.
- No reversals have appeared in the smoke or first-atlas grids.

The full-GIC slack-IR monopsony CE wages are approximately `0.553` for Gaussian and `0` for Poisson. The exponential candidate is approximately `-0.167` at action `82.97`; it retains a strict numerical gray-zone flag discussed below. The relaxed `lambda=0` benchmark remains disabled because it is economically irrelevant and boundary-driven.

### 9.2 Reconciliation of strict numerical flags with human review

The diagnostic figures under `output/foa-problem-diagnostics/` were inspected by a human. The contracts and agent-objective curves look economically stable. The remaining strict flags represent minor numerical instability in difficult or extremely low-probability regions, not evidence of economically meaningful solver failure.

- **Gaussian principal, invalid side:** active-set and CVXPY relaxed solutions both locate the economically relevant deviation, with CE gains about `$27.58` and `$33.21`. After imposing full GIC, residual gains are only about `$0.21` for the active solver and `$0.08` for CVXPY; expected wages differ by about `$0.10`. This is harmless gray-zone noise after the deviation is corrected.
- **Poisson valid boundary:** active relaxed/full solutions coincide. CVXPY relaxed reports a `$1.16` endpoint gain even though expected costs differ by less than one cent. Its utility contract differs at `y=0` by only about `2.8e-5` utility units, while visible pointwise irregularity is concentrated around `y>=24`, whose intended-action probability is about `7.3e-7`. This is tolerance sensitivity, not a meaningful economic disagreement.
- **Exponential principal transition:** the active relaxed solution finds a small but genuine `$5.70` deviation gain at `a=10`; CVXPY full removes it for about `$5.42` additional expected cost. Their agent-objective curves nearly coincide. The active full-contract construction can become nonfinite on the long support, but the relaxed active and full CVXPY solutions give a coherent transition result.
- **Exponential monopsony:** the active contract has only a `$0.50` residual CE gain and its agent-objective curve essentially coincides with CVXPY relaxed. Expected cost differs by about `$5.56`. Warnings mostly arise during outer search and extreme-tail calculations. Keep the strict numerical flag for provenance, but treat the point as human-reviewed and economically stable to a few dollars, not as a qualitatively suspect optimum.
- **Student-`t`:** after five support expansions, omitted mass remains about `0.30%`. This prevents formal support certification, but the `$13.7k--$42.0k` deviations are far too large to plausibly be numerical noise. It remains a robust adverse control.

Operational rule for the next agent: retain strict fields such as `unresolved` and all warnings, but add a separate review/materiality field. Do not erase strict provenance, and do not let sub-dollar or few-dollar numerical noise block internal atlas expansion. Do not begin broad solver debugging based on these reviewed plots.

### 9.3 Expanded internal atlas run

The first expanded manifest run is saved under ignored `output/foa-internal-atlas/`. It contains 29 deterministic atomic tasks: 27 completed and two retained failures. The saved-results-only internal summarizer produced 50 case-exercise threshold rows, two failure rows, and 255 warning occurrences. No reversals were detected.

Preliminary internal findings include:

- New common-family principal transition brackets: geometric `[-0.0781, -0.0625]`, gamma shape 2 `[-0.2969, -0.2813]`, and binomial(10) `[-0.0156, 0]`. Bernoulli is valid throughout the tested range beginning at `-1`.
- Gaussian sigma 10 and 20 principal thresholds both bracket `[-0.0156, 0]`; the sigma-50 baseline remains `[0.625, 0.6875]`, with the existing gray-zone midpoint retained.
- The completed moderate-risk-aversion principal cases generally have thresholds near their computed monopsony CE. Examples are CARA alpha `0.01`: monopsony `0.339`, threshold `[0.359, 0.375]`; CARA alpha `0.02`: monopsony `0.908`, threshold `[0.906, 0.922]`; and CRRA gamma `1.5`: monopsony `0.948`, threshold `[0.984, 1.000]`.
- The added fixed Gaussian actions have brackets `[1.5625, 1.5781]` at intended action 70 and `[0.6719, 0.6875]` at intended action 130.
- The risk-neutral Gaussian and Student-`t` adverse controls never reach persistent validity on their tested ranges.
- CARA alpha `0.04` and CRRA gamma `3` fail with nonfinite agent-utility scans and remain explicit failures. Low-risk-aversion CARA/CRRA cases and the low-cost Gaussian case place the monopsony action at or effectively at the upper action boundary and require boundary-contamination review.

These are internal diagnostic results, not a selected paper sample. Strict statuses remain unresolved in many completed cells because transition cross-checks, monopsony checks, or warnings are conservative; the internal tables preserve these separately from human review.

## 10. Handoff brief

### 10.1 Files and commands

- Plan: `ai-notes/foa-experiments-implementation-plan.md`
- Manifest: `experiments/foa_experiments.yaml`
- Library: `experiments/prototype.py`
- Main CLI: `experiments/run_prototype.py`
- Atomic storage/expansion: `experiments/storage.py`
- Internal saved-results summarizer: `experiments/summarize_internal.py`
- Problem diagnostics CLI: `experiments/diagnose_problematic.py`
- Tests: `experiments/tests/test_prototype.py`
- Usage: `experiments/README.md`

Baseline commands:

```bash
uv sync
uv run python -m unittest discover -s experiments/tests -v
uv run python -m experiments.run_prototype --suite smoke
uv run python -m experiments.run_prototype --suite first_atlas --output output/foa-first-atlas
uv run python -m experiments.run_prototype --suite internal_atlas --output output/foa-internal-atlas --resume
uv run python -m experiments.summarize_internal --input output/foa-internal-atlas
uv run python -m experiments.diagnose_problematic
```

Expected test baseline: **22 tests pass**.

Generated, ignored output locations:

- `output/foa-prototype/`
- `output/foa-first-atlas/`
- `output/foa-internal-atlas/`
- `output/foa-problem-diagnostics/`

The diagnostic root contains an index and one folder per reviewed cell, with figures, JSON metadata, and NPZ arrays. These are internal diagnostics, not paper figures.

### 10.2 Decisions that must be preserved

1. True full-GIC slack-IR monopsony is the only economic monopsony benchmark.
2. Relaxed `lambda=0` monopsony stays disabled except as an optional debugging routine.
3. Principal and fixed-action exercises remain separate.
4. Reversals are searched for; monotonicity is never imposed.
5. Strict numerical status, warnings, and human/economic review status must be stored separately.
6. Reviewed Gaussian, Poisson, and exponential cells do not justify general solver debugging and should not block the next internal atlas stage.
7. Student-`t` remains a robust adverse control despite incomplete support certification.
8. Build and freeze a larger internal atlas before selecting any paper subset.
9. Do not create paper figures or paper summaries yet.
10. Generated experiment output remains ignored. The controlled benchmark files `output/machine_specs.txt` and `output/timing_results.csv` remain tracked.

### 10.3 Known remaining implementation limitations

- Atomic tasks are case-level bundles because support certification, monopsony, and threshold scans share expensive state; they are not yet split into one file per individual reservation-wage solve.
- Detailed solver arrays are not yet copied into task-keyed NPZ files; `arrays/` is reserved, while the complete scalar/nested result is atomic JSON.
- Transition refinement stops at a gray-zone midpoint rather than launching an automatic precision ladder. Intervals and gray-zone points remain explicit.
- Analytic omitted first/second moment errors are saved when moments exist, but they are diagnostics rather than family-specific rigorous tail certificates.
- The Student-`t` support is intentionally unresolved.
- Safe-region diagnostics are numerical grid certifications, not proofs.
- The current run has two explicit high-risk-aversion failures and several boundary-contaminated monopsony candidates requiring internal review.
- Runs made from a dirty working tree record the commit and dirty flag but do not yet snapshot the source diff inside each atomic result.

### 10.4 Recommended sequence for the next agent

1. Inspect `summary_tables/failures.csv`, `warnings.csv`, and all boundary monopsony cells; add review/materiality records rather than weakening strict status.
2. Add explicit automatic boundary-contamination fields and, where economically permissible, rerun only boundary cases with wider action bounds.
3. Diagnose the two nonfinite high-risk-aversion scans narrowly. Retain failures if robust evaluation is not feasible; do not launch broad solver debugging.
4. Decide whether the internal design needs a small additional gamma/binomial parameter grid. If not, freeze the current manifest.
5. Snapshot source-diff provenance for dirty runs, then freeze and inspect the complete internal atlas.
6. Stop before paper tables, paper figures, or selecting attractive cases.

The reporting layer must eventually consume saved atomic results and must never rerun models implicitly.
