# Parametric Examples in the FOA Internal Atlas

This document specifies the economic examples for the FOA atlas. The tracked
manifest `experiments/foa_paper.yaml` is the executable source of truth
once it has been synchronized with this specification. Until then, existing
manifest entries and generated results are provisional. All monetary
quantities are in units of USD 1,000.

## Common economic scale and conventions

All monetary quantities are measured in thousands of dollars. The examples
represent employment relationships in which expected output and the
principal's preferred action are normally on the order of $100,000. The
Gaussian example in Figure 1 is the calibration guide: output has mean `a`, one
unit of output is $1,000, the reference action is `a0=100`, initial wealth is
`w0=50`, and

\[
\operatorname{Revenue}(a)=a,\qquad c(a)=\frac{a^2}{30000}.
\]

The other distributional examples use the same economic scale even when their
primitive action parameter is not itself expected output in thousands of
dollars.

For every case, let

\[
\operatorname{Revenue}(a)=p_Y\mathbb E[Y\mid a]=\beta a,
\]

where `p_Y` is the dollar value of one unit of the distribution's raw outcome
and `beta` is expected revenue's slope with respect to the action parameter. The
principal's problem must use this declared revenue function, rather than
assuming `Revenue(a)=a`. The calibration target `a0` is chosen so that
`Revenue(a0)` is approximately
$100,000 in each baseline case. Thus the action parameter may be 100 units of
output, 7 completed projects, a success probability of 0.7, or a gamma scale of
50, while associated expected revenue is either 100 or 105 in the model's
units.

Wages satisfy limited liability. Effort cost is quadratic and is **always zero
at the smallest action in the economic action set being considered**. If that
smallest action is `a_min`, then

\[
c(a)=\frac{\kappa}{2}(a^2-a_{\min}^2),\qquad c'(a)=\kappa a.
\]

This convention applies uniformly: an action set beginning at 0.05 has
`c(0.05)=0`, one beginning at 1.05 has `c(1.05)=0`, and Gaussian action-set
variations beginning at 20 or 50 have `c(20)=0` or `c(50)=0`. We do not attach
a different cost normalization to a lower bound based on why that bound was
chosen. Changing the lower endpoint changes only the level of effort cost; it
does not change marginal cost or the relative cost of any two actions that
remain in the set.

For log utility, let `Revenue0=Revenue(a0)`. The Figure 1 calibration
generalizes to

\[
\kappa=\frac{\beta}{a_0(w_0+\operatorname{Revenue}_0)}.
\]

Equivalently,

\[
\frac{c'(a_0)}{u'(w_0+\operatorname{Revenue}_0)}
=\beta=\operatorname{Revenue}'(a_0).
\]

Thus, at total wealth `w0+Revenue0`, the local consumption-equivalent marginal
cost of raising action equals marginal revenue. In Figure 1,
`Revenue0=a0=100`, so this reduces exactly to
`kappa=1/[100(100+50)]` and `c(a)=a^2/30000`. Unlike a formula
with `a0+w0` in every case, this calibration does not mistakenly treat a count,
probability, or scale parameter as a dollar amount. Before explicit `cost_scale`
variations, it keeps effort cost at the target near one-third of a log-utility
unit, which makes compensation requirements comparable. The binomial case has
a documented scale of 1.5 to prevent its unusually informative signal from
forcing the principal to the probability boundary.

The target is a scale calibration, not a claim that every constrained
moral-hazard optimum equals `a0` exactly; endogenous principal actions may move
around it. `local_consumption_equivalent` additionally rescales the coefficient
using marginal utility at the declared reference wealth so that effort costs
have the same local consumption-equivalent interpretation across utility
functions. `cost_scale` is an explicit multiplicative effort-cost variation
applied after these calibrations.

The executable manifest should therefore declare `revenue_slope` (or an
equivalent expected-output value) separately from `cost_scale`. The same
revenue function must be used in relaxed principal solves, full-GIC monopsony
solves, and principal diagnostics. Fixed-action cost minimization does not use
principal revenue, but it uses the same cost calibration. Principal and
fixed-action exercises remain stored and reported separately.

## Exact baseline distribution examples

The following values define the examples to be implemented. We deliberately
use round economic values: a raw count unit in the Poisson, geometric, and
binomial cases is worth $15,000, and a Bernoulli success is worth $150,000.
Expected revenue at the declared target is then $105,000. This is close enough
to the $100,000 Gaussian target while being easier to interpret than values
chosen to force exact equality.

| Case | Raw outcome and conditional mean | Value per raw outcome unit `p_Y` | Principal revenue | Action set; `a0` | Zero-cost point | Exercises |
|---|---|---:|---:|---:|---:|---|
| `gaussian_log_paper` | Normal location, `E[Y|a]=a`, sigma 50 | 1 | `Revenue(a)=a` | `[0,180]`; 100 | 0 | Principal and fixed action |
| `poisson_log_paper` | Poisson count, `E[Y|a]=a` | 15 | `Revenue(a)=15a` | `[0.05,10]`; 7 | 0.05 | Principal and fixed action |
| `exponential_log_baseline` | Exponential output, `E[Y|a]=a` | 1 | `Revenue(a)=a` | `[10,180]`; 100 | 10 | Principal and fixed action |
| `gamma2_log_baseline` | Gamma shape 2 and scale `a`, `E[Y|a]=2a` | 1 | `Revenue(a)=2a` | `[5,90]`; 50 | 5 | Principal and fixed action |
| `geometric_log_baseline` | Geometric count on `{1,2,...}`, `E[Y|a]=a` | 15 | `Revenue(a)=15a` | `[1.05,10]`; 7 | 1.05 | Principal and fixed action |
| `bernoulli_log_baseline` | Project success indicator, `E[Y|a]=a` | 150 | `Revenue(a)=150a` | `[0.05,0.95]`; 0.7 | 0.05 | Principal and fixed action |
| `binomial10_log_baseline` | Number of successes in 10 trials, `E[Y|a]=10a` | 15 | `Revenue(a)=150a` | `[0.05,0.99]`; 0.7 | 0.05 | Principal and fixed action; cost scale 1.5 |
| `student_t_log_adverse` | Student-t location, `E[Y|a]=a`, sigma 20, df 1.15 | 1 | `Revenue(a)=a` | `[0,180]`; 100 | 0 | Principal and fixed-action adverse control |

At the target, `Revenue(a0)=100` for Gaussian, exponential, gamma, and Student-t,
and `Revenue(a0)=105` for Poisson, geometric, Bernoulli, and binomial. For the
log baseline the cost coefficient is

\[
\kappa=\frac{\operatorname{Revenue}'(a_0)}
{a_0(50+\operatorname{Revenue}(a_0))},
\]

and the cost level is shifted so that `c(a_min)=0`. In particular,

\[
\begin{aligned}
c_{\mathrm{Poisson}}(a)
  &=\frac{1}{2}\frac{15}{7(50+105)}(a^2-0.05^2),\\
c_{\mathrm{exponential}}(a)
  &=\frac{1}{2}\frac{1}{100(100+50)}(a^2-10^2),\\
c_{\mathrm{gamma}}(a)
  &=\frac{1}{2}\frac{2}{50(50+100)}(a^2-5^2),\\
c_{\mathrm{geometric}}(a)
  &=\frac{1}{2}\frac{15}{7(50+105)}(a^2-1.05^2),\\
c_{\mathrm{Bernoulli}}(a)
  &=\frac{1}{2}\frac{150}{0.7(50+105)}(a^2-0.05^2),\\
c_{\mathrm{binomial}}(a)
  &=\frac{1}{2}(1.5)\frac{150}{0.7(50+105)}(a^2-0.05^2).
\end{aligned}
\]

The unscaled Bernoulli and binomial calibrations would have the same cost
because they use the same action parameter, target, and marginal revenue. A
quick widened-probability check showed that this made the binomial principal
choose the maximum probability even at 0.99: ten output signals make incentives
much cheaper than one Bernoulli signal. We therefore predeclare a binomial cost
scale of 1.5. It raises target effort cost from 0.337 to 0.505 and produces an
interior slack-IR action near 0.77 rather than manufacturing an optimum with an
arbitrary probability ceiling. All constant cost shifts preserve `c'(a)`,
local incentive constraints, and relative effort costs. They change delivered
utility and therefore must be included when computing reservation-wage and
monopsony CE benchmarks.

### Scale reasonableness check

The calibration gives the following target magnitudes:

| Family | Target expected revenue | Effort cost `c(a0)` in utility units | Economic reading |
|---|---:|---:|---|
| Gaussian | 100 | 0.333 | $100,000 expected output with $50,000 initial wealth |
| Poisson | 105 | 0.339 | Seven projects worth $15,000 each on average |
| Exponential | 100 | 0.330 | $100,000 expected output above a $10,000 lowest-action mean |
| Gamma shape 2 | 100 | 0.330 | $100,000 expected output at scale 50 |
| Geometric | 105 | 0.331 | Mean of seven units worth $15,000 each |
| Bernoulli | 105 | 0.337 | A project worth $150,000 with success probability 0.7 |
| Binomial(10) | 105 | 0.505 | Seven target successes worth $15,000 each; steeper cost keeps the information-rich principal problem interior |
| Student-t | 100 | 0.333 | Same location and dollar scale as the Gaussian adverse comparison |

These are intentionally stylized rather than industry-specific calibrations.
They put revenue, wealth, and the utility cost of the target action on similar
scales. They also leave room for endogenous optimal actions to differ from the
target because of incentive and risk costs. Common-family and targeted-boundary
preflights verify the calibration before the final atlas; any future adjustment
must change a declared target or cost scale openly rather than silently
changing the value of output.

## Existing paper examples and proposed harmonization

The current paper has four substantive numerical environments, plus Gaussian
solver and cost-minimization variants. The common specification above should
be used when they are regenerated.

| Paper use | Existing specification | Harmonized specification and decision |
|---|---|---|
| Main Gaussian/log principal figure (Figure 1) | `w0=50`, `Y|a ~ N(a,50^2)`, action set `[0,180]`, `Revenue(a)=a`, and `c(a)=a^2/30000` | **Keep unchanged.** This is the reference calibration. |
| Gaussian/log fixed-action and Pareto-frontier figures | Intended action 100 with the Figure 1 environment | **Keep unchanged.** The atlas baseline fixed-action exercise is the same economic case. |
| Gaussian solver comparison | Figure 1 environment with sigma 10 and fixed action 100 | **Keep unchanged.** It corresponds to the sigma-10 mechanism variation. |
| Gaussian/CARA principal figure | Figure 1 output environment; `alpha=1/50=0.02`, giving relative risk aversion 1 at wealth 50; the legacy script multiplies the log cost coefficient by 10 | **Retain the economic case, but regenerate it from the atlas CARA `alpha=0.02` specification using the declared local consumption-equivalent normalization.** The paper figure should not keep a separate hand-coded cost rule. |
| Poisson/log principal figure | Mean action 7, one thousand dollars per success, an action set starting at 0 in the legacy script, and a caption reporting `c(a)=a^2/399` | **Change to the harmonized Poisson case:** $15,000 per success, `Revenue(a)=15a`, action set `[0.05,10]`, `c(0.05)=0`, and the cost formula above. The old script actually implements `a^2/798`, so the existing caption and code already disagree; both should be replaced from the common specification. |
| Student-t/log counterexample figure | Location action in `[0,180]`, sigma 20, df 1.15, `Revenue(a)=a`, and `c(a)=a^2/30000` | **Keep the economic calibration.** To harmonize provenance, add the principal exercise to the atlas Student-t case if this figure remains in the paper. Continue to label support certification unresolved and do not use it for a precise threshold claim. |

The supplementary appendix's analytic three-state risk-neutral counterexample
is not a calibrated employment example and does not assign dollar values to
its three output labels. Keep it as a self-contained theoretical
counterexample; it does not need to be forced onto the $100,000 revenue scale.
The separate Gaussian risk-neutral atlas control remains useful as the
economically scaled numerical adverse case.

The solver benchmarks use Gaussian cases from Figure 1 and the Student-t case
from the counterexample figure. When benchmarks are next regenerated, their
economic configurations should be loaded from the same harmonized case
specifications even though timing grids and solver settings remain
benchmark-specific.

The legacy figure script also generates Gaussian sigma 20 and exponential
figures that are not currently included in the manuscript. They need not be
preserved as paper examples. Their corresponding atlas cases may still be used
for mechanism analysis or an appendix selected from saved results.

Several sentences in the current paper describe the old generated outcomes,
for example that the Poisson figure uses $1,000 per success and that FOA is
valid at every displayed positive reservation wage. Those are results of the
old calibration, not assumptions. After harmonization, captions and numerical
claims must be regenerated from saved atlas results. The unchanged Gaussian
claims can be expected to survive, but the revised Poisson claims must be
rechecked rather than copied.

## Gaussian mechanism variations

All use Gaussian location output and log utility unless stated otherwise.

- **Noise:** sigma 10 and 20, compared with the sigma-50 baseline.
- **Action-set lower endpoint:** action bounds `[20,180]` and `[50,180]`, with
  effort cost shifted consistently so that `c(20)=0` and `c(50)=0`,
  respectively.
- **Initial wealth:** 25, 100, and 200, compared with baseline 50.
- **Effort cost:** multiplicative cost scales 0.5 and 2. The half-cost case
  uses action bounds `[0,250]` and outcome grid `[-300,550]`; its slack-IR
  action stabilizes near 225 once the former 180 ceiling is removed. The
  double-cost case retains the baseline bounds.
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

The low-risk-aversion cases CARA `alpha=0.005` and CRRA `gamma=0.25,0.5` use
Gaussian action bounds `[0,250]` and outcome grid `[-300,550]`. Quick full-GIC
checks with bounds 240, 300, and 400 showed stable slack-IR actions near 189.8,
226.1, and 183.7, respectively. The common bound 250 therefore removes the
artificial 180 ceiling with meaningful slack while preserving a plausible
employment-output scale. Other utility cases retain the Figure 1 action set
`[0,180]`.

For CARA `alpha=0.04` and CRRA `gamma=3`, bounded utility and limited liability
imply a highest locally implementable Gaussian action of about 59.86. Fixed
action 100 is therefore recorded as economically infeasible without calling
the solver. The principal's intended-action search is capped just inside this
boundary, while global deviations are still checked over the original full
action interval. The CARA principal case then solves cleanly; the CRRA gamma-3
principal/full-GIC result remains numerically unresolved rather than crashing.

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
3. Revenue is determined by the declared dollar value of raw output and must
   not be inferred from the name or numerical scale of the action parameter.
4. Effort cost is zero at the declared lower endpoint of each economic action
   set, including all action-bound variations.
5. Boundary monopsony actions, support failures, solver warnings, and human
   review statuses remain separate fields in the internal atlas.
6. Safe-region diagnostics are numerical certifications on configured grids,
   not analytic proofs.
