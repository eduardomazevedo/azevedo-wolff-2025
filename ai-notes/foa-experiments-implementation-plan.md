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
23. Fast bounded-utility incentive-capacity checks: economically infeasible fixed actions are classified before solving, and principal intended-action searches are capped just inside the highest locally feasible action while retaining the full deviation domain.
24. Harmonized economic scaling across distribution families: baseline target expected revenue is $100k or $105k, with round raw-output values, and every principal solve uses the case's declared revenue slope.
25. Uniform zero-cost convention: quadratic effort cost is zero at the configured lower action bound in every case, including Gaussian action-bound variations.
26. Figure-1-based cost calibration generalized to non-dollar action parameters using `theta = R'(a0) / [a0 (w0 + R(a0))]`, with regression tests for revenue, cost levels, and the unchanged Gaussian benchmark.
27. Paper-example inventory and proposed harmonization recorded in `ai-notes/parametric-examples.md`; the Student-t principal exercise is included for provenance if its paper figure is retained.
28. Harmonized common-distribution and smoke preflights completed from fresh directories: no task failures, all common-family supports passed, target effort costs matched the declared 0.33--0.34 range, and principal actions were economically reasonable. Student-t principal and fixed-action adverse exercises both completed while retaining unresolved support certification.
29. Final internal atlas regenerated from clean commit `fa339bb` without resume: 29 of 29 tasks completed, no atomic failures occurred, and all bounded-utility cases contain the capacity precheck.
30. Boundary and narrow adverse-case review completed. Retain the four low-risk/low-cost Gaussian boundary cases and bounded-probability binomial case as explicitly contaminated internal cells; do not widen them merely to remove warnings. CRRA gamma 3 is economically infeasible at fixed action 100 and globally invalid throughout its principal grid, so it remains unresolved and outside headline claims.

Still to do before generating paper assets:

1. Obtain human plot review of the harmonized Poisson diagnostic if the Poisson figure remains in the paper. Its old review is inactive; the manifest correctly records the new review as pending.
2. Declare the concise paper-candidate subset/report groups before writing the reporting code. Keep principal and fixed-action results separate and retain adverse cases.
3. Implement saved-results-only paper tables and figures using `output/foa-internal-atlas-final/`; reporting must not solve models implicitly.
4. Update the paper's Poisson caption and numerical discussion to the $15k-per-success calibration, and regenerate all retained paper examples from the harmonized pipeline.
5. Preserve strict unresolved statuses, warnings, four failed transition cross-check records, and boundary flags in internal outputs. They do not justify broad solver debugging because the failures occur in non-headline robustness cells and independent active/support checks remain informative.
6. Do not add broad parameter sweeps unless paper reporting exposes a specific missing comparison.

## 9. Current numerical results and human review

A tracked implementation lives in `experiments/`, driven by `experiments/foa_experiments.yaml`. Twenty-eight unit tests pass after the harmonized-calibration changes. Generated output is ignored by Git.

### 9.1 Historical smoke and first-atlas results (superseded where calibrations changed)

All wages and CE gains below are in the manifest's `USD_1000` units unless a dollar sign is stated explicitly.

- **Gaussian/log principal:** invalid-to-valid transition bracket `[0.625, 0.688]`.
- **Gaussian/log fixed action `a0=100`:** transition bracket `[1.156, 1.172]`.
- **Poisson/log principal and fixed action `a0=7`:** transition bracket `[-0.0156, 0]`.
- **Exponential/log principal:** after normalizing cost to `C(10)=0`, invalid at `-0.05`, unresolved at `0` with a CE gain of about `$0.22`, and valid at `0.05`; record the threshold as the unresolved interval `[-0.05, 0.05]`.
- **Exponential/log fixed action `a0=100`:** valid at every tested reservation wage beginning at `-1`.
- **Student-`t`/log fixed action `a0=100`:** robustly invalid at every tested wage. On the support-expanded first-atlas run, CE deviation gains range from about `$13.7k` to `$42.0k`, with best deviation `a=0` throughout.
- No reversals have appeared in the smoke or first-atlas grids.

The full-GIC slack-IR monopsony CE wages are approximately `0.553` for Gaussian and `0` for Poisson. After setting `C(10)=0`, the exponential candidate is numerically zero (`-0.00033`, about `-$0.33`) at action `82.97` and retains a strict global-IC gray-zone flag. The relaxed `lambda=0` benchmark remains disabled because it is economically irrelevant and boundary-driven.

### 9.2 Historical reconciliation of strict numerical flags with human review

The diagnostic figures under `output/foa-problem-diagnostics/` were inspected by a human. The contracts and agent-objective curves look economically stable. The remaining strict flags represent minor numerical instability in difficult or extremely low-probability regions, not evidence of economically meaningful solver failure.

- **Gaussian principal, invalid side:** active-set and CVXPY relaxed solutions both locate the economically relevant deviation, with CE gains about `$27.58` and `$33.21`. After imposing full GIC, residual gains are only about `$0.21` for the active solver and `$0.08` for CVXPY; expected wages differ by about `$0.10`. This is harmless gray-zone noise after the deviation is corrected.
- **Poisson valid boundary:** active relaxed/full solutions coincide. CVXPY relaxed reports a `$1.16` endpoint gain even though expected costs differ by less than one cent. Its utility contract differs at `y=0` by only about `2.8e-5` utility units, while visible pointwise irregularity is concentrated around `y>=24`, whose intended-action probability is about `7.3e-7`. This is tolerance sensitivity, not a meaningful economic disagreement.
- **Exponential diagnostics under the old cost level:** the active relaxed solution found a small but genuine `$5.70` deviation gain at `a=10`; CVXPY full removed it for about `$5.42` additional expected cost. The old monopsony diagnostic had only a `$0.50` residual CE gain. These reviews are retained as historical provenance but are inactive because shifting cost to `C(10)=0` changes delivered utility and reservation-wage coordinates. The revised exponential cells require fresh review if used beyond the internal atlas.
- **Student-`t`:** after five support expansions, omitted mass remains about `0.30%`. This prevents formal support certification, but the `$13.7k--$42.0k` deviations are far too large to plausibly be numerical noise. It remains a robust adverse control.

Operational rule for the next agent: retain strict fields such as `unresolved` and all warnings, but add a separate review/materiality field. Do not erase strict provenance, and do not let sub-dollar or few-dollar numerical noise block internal atlas expansion. Do not begin broad solver debugging based on these reviewed plots.

### 9.3 Historical expanded internal atlas run (superseded calibration)

The expanded manifest run saved under ignored `output/foa-internal-atlas/` used the pre-harmonization revenue and cost calibration and is retained only as historical diagnostic provenance. It must not be used for paper quantities. In that run, all 29 deterministic atomic tasks completed and the two formerly crashing fixed-action cells were recorded as economically infeasible rather than numerical failures. The saved-results-only internal summarizer produced 54 case-exercise rows and no task-failure rows. No reversals were detected.

Preliminary internal findings include:

- With positive-domain effort costs normalized to zero at the lowest action, the gamma shape-2 monopsony CE is `0.0011` and its principal threshold is `[0, 0.0156]`; the exponential monopsony CE is numerically zero and its principal threshold lies in the unresolved interval `[-0.05, 0.05]`. Other common-family principal brackets include geometric `[-0.0781, -0.0625]` and binomial(10) `[-0.0156, 0]`. Bernoulli is valid throughout the tested range beginning at `-1`.
- Gaussian sigma 10 and 20 principal thresholds both bracket `[-0.0156, 0]`; the sigma-50 baseline remains `[0.625, 0.6875]`, with the existing gray-zone midpoint retained.
- The completed moderate-risk-aversion principal cases generally have thresholds near their computed monopsony CE. Examples are CARA alpha `0.01`: monopsony `0.339`, threshold `[0.359, 0.375]`; CARA alpha `0.02`: monopsony `0.908`, threshold `[0.906, 0.922]`; and CRRA gamma `1.5`: monopsony `0.948`, threshold `[0.984, 1.000]`.
- The added fixed Gaussian actions have brackets `[1.5625, 1.5781]` at intended action 70 and `[0.6719, 0.6875]` at intended action 130.
- The risk-neutral Gaussian and Student-`t` adverse controls never reach persistent validity on their tested ranges.
- CARA alpha `0.04` and CRRA gamma `3` have bounded-utility incentive capacity sufficient only up to action about `59.86`, so fixed action 100 is economically infeasible. After clipping only the principal's intended-action search, CARA alpha `0.04` has a valid full-GIC monopsony candidate at action `48.18`, CE `4.23`, and valid relaxed principal contracts throughout the tested reservation range. CRRA gamma `3` no longer crashes, but its relaxed principal contracts remain globally invalid and its full-GIC monopsony candidate is unverified. Low-risk-aversion CARA/CRRA cases and the low-cost Gaussian case place the monopsony action at or effectively at the upper configured boundary and require boundary-contamination review.

These are internal diagnostic results, not a selected paper sample. Strict statuses remain unresolved in many completed cells because transition cross-checks, monopsony checks, or warnings are conservative; the internal tables preserve these separately from human review.

The current generated atlas is superseded: it predates both the harmonized revenue/cost calibration and complete capacity-precheck records. Use fresh output directories for the calibration preflight and final validation run. Do not commit either directory.

### 9.4 Harmonized calibration preflight

Fresh ignored runs under `output/foa-calibration-preflight/` and
`output/foa-smoke-preflight/` verified the revised specification before the
full atlas. All seven common-distribution tasks and all three smoke tasks
completed without hard failures. Every common-family support check passed; the
Student-t support remained intentionally unresolved after five expansions.
No reversals appeared.

At the slack-IR monopsony candidates, intended actions were approximately:
Gaussian 132.95 (target 100), exponential 82.97 (target 100), gamma shape 2
48.82 (target 50), Poisson 9.29 (target 7), geometric 5.89 (target 7),
Bernoulli 0.578 (target 0.7), and binomial 0.95 (target 0.7). These are
reasonable for the deliberately stylized calibration. The binomial optimum is
at its declared maximum success probability and remains explicitly
boundary-contaminated; it is an internal breadth case, not a reason to alter
the economic action set. Poisson is interior, though its monopsony plateau
retains a strict unverified flag. Gamma's corrected principal optimum is
interior and near its target.

Target effort costs are 0.330--0.339 utility units in all seven common
families, as specified. The Student-t principal and fixed-action exercises both
remain FOA-invalid throughout their tested grids, so adding principal
provenance for the paper counterexample did not weaken the adverse control.
These preflight outputs are diagnostic and ignored; they are not substitutes
for the final clean atlas.

### 9.5 Final clean harmonized atlas

The final ignored run is `output/foa-internal-atlas-final/`, generated without
resume from clean commit `fa339bb32e977f9622cb5cf59ba4818bf0a161bc`. All 29
atomic tasks completed, none were cached, and there were no task failures. The
saved-results summarizer produced 55 exercise rows: 53 completed exercises and
two high-risk-aversion fixed actions classified as economically infeasible.
No reversals were detected.

The conservative strict layer marks 51 rows unresolved and four passed. This
is expected because strict status aggregates support, monopsony, transition,
and cross-solver checks. There are 174 retained warning records. Student-t is
the only strictly support-uncertified case, contributing its principal and
fixed-action rows; human review accepts the small residual truncation as
numerically immaterial while preserving the strict flag. Five economic cases
are boundary-contaminated: CARA alpha 0.005, CRRA
gamma 0.25, CRRA gamma 0.5, the half-cost Gaussian, and binomial(10). Four
transition cross-check records failed: two Clarabel failures for the Gaussian
lower-bound-50 fixed-action case and two inaccurate CVXPY full solves for the
CRRA gamma-0.5 fixed-action case. These are internal non-headline robustness
cells; their strict failures remain recorded.

Selected harmonized results are:

- Gaussian/log: monopsony CE 0.553; principal threshold `[0.625, 0.6875]` and
  fixed-action-100 threshold `[1.15625, 1.171875]`.
- Poisson/log at $15k per success: monopsony candidate CE 0.00086 with an
  unverified plateau flag; principal and fixed-action thresholds both
  `[-0.015625, 0]`.
- Gamma shape 2: monopsony CE -0.00016; principal and fixed-action thresholds
  both `[-0.015625, 0]`.
- CARA alpha 0.02: monopsony CE 0.908; principal threshold
  `[0.90625, 0.921875]` and fixed-action threshold `[0.828125, 0.84375]`; this
  case passes the aggregate strict layer.
- CRRA gamma 1.5: monopsony candidate CE 0.948 with a global-IC strict flag;
  principal threshold `[0.984375, 1]` and fixed-action threshold
  `[0.953125, 0.96875]`.
- Student-t principal and fixed-action exercises, the risk-neutral Gaussian
  fixed-action exercise, and the CRRA gamma-3 principal exercise never reach
  persistent validity on their tested ranges. Student-t is accepted as a
  robust adverse control despite the retained strict support flag.

Fresh harmonized Poisson diagnostics show coincident active relaxed and full
solutions at reservation wage zero. CVXPY relaxed finds an endpoint CE gain of
about $1.21, while CVXPY full removes it for less than one dollar of additional
expected cost and is globally valid. This is likely immaterial tolerance
sensitivity, but the manifest correctly leaves human review pending because
the previous human assessment used the old calibration.

## 10. Handoff brief

### 10.1 Files and commands

- Plan: `ai-notes/foa-experiments-implementation-plan.md`
- Parametric example inventory: `ai-notes/parametric-examples.md`
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

# Final validation (completed from clean commit fa339bb, without resume):
uv run python -m experiments.run_prototype --suite internal_atlas --output output/foa-internal-atlas-final
uv run python -m experiments.summarize_internal --input output/foa-internal-atlas-final
```

Do not use `--resume` for the final validation run, and do not commit either output directory.

Expected test baseline: **28 tests pass**.

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
6. Reviewed Gaussian and Poisson cells do not justify general solver debugging. The earlier exponential reviews used the old cost level and are inactive; revised exponential results require fresh review only if selected for reporting.
7. Student-`t` remains a robust adverse control despite incomplete support certification.
8. Freeze the tracked manifest and experiment code before selecting any paper subset; validate them by regenerating the internal atlas from scratch.
9. Do not create paper figures or paper summaries until that clean validation run is inspected.
10. Generated experiment output remains ignored and is never committed. The reproducible code/manifest pipeline is the deliverable. The controlled benchmark files `output/machine_specs.txt` and `output/timing_results.csv` remain tracked.

### 10.3 Known remaining implementation limitations

- Atomic tasks are case-level bundles because support certification, monopsony, and threshold scans share expensive state; they are not yet split into one file per individual reservation-wage solve.
- Detailed solver arrays are not yet copied into task-keyed NPZ files; `arrays/` is reserved, while the complete scalar/nested result is atomic JSON.
- Transition refinement stops at a gray-zone midpoint rather than launching an automatic precision ladder. Intervals and gray-zone points remain explicit.
- Analytic omitted first/second moment errors are saved when moments exist, but they are diagnostics rather than family-specific rigorous tail certificates.
- The Student-`t` support is intentionally unresolved.
- Safe-region diagnostics are numerical grid certifications, not proofs.
- The current run has no hard task failures. Two high-risk-aversion fixed actions are economically infeasible by the capacity bound, and the CRRA gamma-3 principal/full-GIC result remains unresolved.
- Runs made from a dirty working tree record root-level source snapshots, but the final validation should be run from a clean committed tree.
- Task hashes currently cover economic and numerical configuration but not implementation code. Therefore `--resume` can reuse stale completed records after code changes; use a fresh output directory for the final validation run.
- Existing generated principal results use the superseded economic calibration and must not be reused. The implementation now applies each case's declared revenue slope consistently, but requires a fresh run.

### 10.4 Recommended sequence for the next agent

1. Treat `output/foa-internal-atlas-final/` as the final ignored internal run associated with clean commit `fa339bb`; do not rerun it merely because documentation changes afterward.
2. Have a human inspect `output/foa-problem-diagnostics/poisson-principal-valid-boundary/`. If accepted, add a new review record or change the harmonized pending record without erasing the inactive old-calibration review.
3. Predeclare paper candidates and implement reporting that reads atomic records or saved summary tables only. Never call model solvers from reporting code.
4. Keep fixed-action local infeasibility, Student-t support failure, CRRA gamma-3 invalidity, boundary contamination, and failed convex cross-checks visible in internal reporting.
5. Do **not** add generated output to Git. Reproduction consists of the committed manifest, code, lockfile, and documented commands.
6. Regenerate the retained Gaussian, CARA, Poisson, and Student-t paper examples from harmonized saved results. The Poisson paper figure and numerical text must use the $15k-per-success calibration.

The reporting layer must eventually consume saved atomic results and must never rerun models implicitly.
