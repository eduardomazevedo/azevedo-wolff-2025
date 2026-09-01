# Parametric Examples in the FOA Internal Atlas

This document inventories the economic examples declared in
`experiments/foa_experiments.yaml`. The manifest is the executable source of
truth; this note records their interpretation. All monetary quantities are in
units of USD 1,000.

## Common conventions

- Initial wealth is normally `w0=50`.
- The principal receives output linearly and pays the outcome-contingent wage.
- Wages satisfy limited liability.
- Baseline effort cost is quadratic,
  \[
  C(a)=\frac{\theta}{2}(a^2-a_{\min,0}^2),\qquad C'(a)=\theta a.
  \]
  Usually `a_min,0=0`. For positive-domain exponential and gamma examples it
  equals the lowest feasible action, as described below.
- `paper_log` uses
  \[
  \theta=\frac{1}{a_0(a_0+w_0)},
  \]
  where `a0` is the calibration target. `local_consumption_equivalent`
  rescales this coefficient using marginal utility at zero wage so that effort
  costs have the same local consumption-equivalent interpretation across
  utility functions.
- `cost_output_slope` adjusts cost when one unit of the distribution's action
  parameter changes expected output by more than one unit. `cost_scale` is an
  explicit multiplicative effort-cost variation.
- Each environment is evaluated as a principal problem, a fixed-action cost
  minimization problem, or both. These exercises are stored and reported
  separately.

## Baseline distribution examples

| Case | Output distribution and action | Action bounds | Calibration/fixed action | Cost details | Exercises |
|---|---|---:|---:|---|---|
| `gaussian_log_paper` | Normal location, mean `a`, sigma 50 | `[0,180]` | 100 | Baseline quadratic, zero at 0 | Principal and fixed action |
| `poisson_log_paper` | Poisson mean `a` | `[0.05,10]` | 7 | Baseline quadratic | Principal and fixed action |
| `exponential_log_baseline` | Exponential mean/scale `a` | `[10,180]` | 100 | **Zero at `a=10`;** same `C'(a)` as the original calibration | Principal and fixed action |
| `gamma2_log_baseline` | Gamma shape 2 and scale `a`; mean `2a` | `[5,90]` | 50 | **Zero at `a=5`;** output-slope factor 2; same `C'(a)` as the original calibration | Principal and fixed action |
| `geometric_log_baseline` | Geometric mean parameter `a`, support starting at 1 | `[1.05,10]` | 7 | Baseline quadratic; interpretation of the minimum-mean state requires care | Principal and fixed action |
| `bernoulli_log_baseline` | Success probability `a` | `[0.05,0.95]` | 0.7 | Baseline quadratic | Principal and fixed action |
| `binomial10_log_baseline` | 10 trials with success probability `a`; mean `10a` | `[0.05,0.95]` | 0.7 | Output-slope factor 10 | Principal and fixed action |
| `student_t_log_adverse` | Student-t location `a`, sigma 20, df 1.15 | `[0,180]` | 100 | Baseline quadratic | Fixed-action adverse control |

### Positive-domain cost normalization

The exponential distribution at scale zero and the gamma distribution at scale
zero are degenerate. Their numerical action domains therefore start at 10 and
5 rather than literally at zero. We interpret the lowest feasible action as
the zero-effort baseline and use

\[
C_{\mathrm{exp}}(a)=\frac{\theta}{2}(a^2-10^2),
\qquad
C_{\mathrm{gamma}}(a)=\frac{\theta}{2}(a^2-5^2).
\]

This is a constant shift from the original quadratic costs. It makes effort
cost zero at the lowest action without changing marginal cost, incentive
constraints, the calibration's local scale, or the relative cost of moving
between any two actions. It does change delivered utility and therefore
correctly shifts CE reservation-wage and monopsony benchmarks.

## Gaussian mechanism variations

All use Gaussian location output and log utility unless stated otherwise.

- **Noise:** sigma 10 and 20, compared with the sigma-50 baseline.
- **Action-set lower endpoint:** action bounds `[20,180]` and `[50,180]`.
- **Initial wealth:** 25, 100, and 200, compared with baseline 50.
- **Effort cost:** multiplicative cost scales 0.5 and 2.
- **Fixed intended actions:** 70 and 130 in the baseline environment, compared
  with the fixed action 100 in `gaussian_log_paper`.

These variations diagnose safe-region size, boundary effects, wealth/risk
aversion, effort-cost curvature, and differences between endogenous and fixed
intended actions.

## Risk-aversion examples

The Gaussian sigma-50 environment is crossed with:

- log utility;
- CRRA coefficients `gamma = 0.25, 0.5, 1.5, 2, 3` (gamma 1 is represented by
  the log case);
- CARA coefficients `alpha = 0.005, 0.01, 0.02, 0.04`.

For CARA, reference total wealth is 50, so these alpha values imply relative
risk aversion 0.25, 0.5, 1, and 2 at the reference wealth. All non-log utility
cases use local consumption-equivalent effort-cost normalization.

The high-risk-aversion CARA `alpha=0.04` and CRRA `gamma=3` tasks are retained
as explicit numerical failures in the current internal atlas rather than being
silently dropped.

## Adverse controls

- `student_t_log_adverse` tests a heavy-tailed, nonmonotone-score environment
  and remains robustly FOA-invalid over its tested fixed-action range.
- `gaussian_risk_neutral_adverse` uses linear utility in the Gaussian sigma-50
  environment and remains FOA-invalid over its tested fixed-action range.

These controls test whether the experiment suite detects failure when risk
aversion or the safe-tail mechanism is absent. They are not pooled with the
positive examples.

## Interpretation cautions

1. Principal and fixed-action thresholds are distinct estimands.
2. A threshold below the lowest tested reservation wage is left-censored; it
   should be reported as "valid throughout the tested range," not as an exact
   threshold.
3. Positive-domain and geometric action parameters require explicit economic
   interpretation of their lowest feasible states.
4. Boundary monopsony actions, support failures, solver warnings, and human
   review statuses remain separate fields in the internal atlas.
5. Safe-region diagnostics are numerical certifications on configured grids,
   not analytic proofs.
