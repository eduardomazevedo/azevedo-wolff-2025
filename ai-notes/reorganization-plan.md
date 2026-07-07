## 1. Model

This section should be concise and use economically natural notation.

### Definition: Actions and output

Introduce:

* action set $\mathcal A$;
* intended action $a_0$;
* output space $\mathcal Y\subseteq\mathbb R$;
* common dominating measure $\nu$;
* density or probability mass function $f(y|a)$;
* common support.

Add convention:

> We write integrals throughout; in discrete-output cases, these are sums with respect to counting measure.

### Definition: Utility, wealth, and contracts

Introduce:

* initial wealth $w_0$;
* payment contract $x(y)\ge0$;
* final wealth contract $W(y)=w_0+x(y)$;
* utility over final wealth $u(W)$;
* effort cost $c(a)$.

### Definition: Utility contracts

Introduce the analytical representation:

$$
v(y)=u(W(y)).
$$

Define:

* $\underline u=u(w_0)$;
* $\bar u=\lim_{W\to\infty}u(W)$;
* $\Delta u=\bar u-\underline u$;
* $k=u^{-1}$;
* payment induced by utility contract: $k(v(y))-w_0$.

### Definition: Agent utility and compensation cost

Define:

* $U(v,a)$: agent expected utility;
* $C(v,a)$: expected compensation cost.

Use $C$, not $W$, for expected compensation cost to avoid conflict with final wealth $W(y)$.

### Definition: Cost-minimization problem

State the full problem:

* minimize $C(v,a_0)$;
* subject to IR;
* subject to GIC.

### Definition: Relaxed cost-minimization problem

State the relaxed problem:

* minimize $C(v,a_0)$;
* subject to IR;
* subject to LIC.

### Definition: Score

Define:

$$
S(y|a)=\partial_a\log f(y|a).
$$

Explain that $S(y|a_0)$ measures how strongly output $y$ supports higher effort at the intended action.

---

## 4. Maintained Assumptions

This section should contain the assumptions that are needed to state the main results, but avoid introducing proof-only constants such as $A_q$ and $M_q$ unless absolutely necessary.

### Assumption 1: Regularity

Include:

* compact action set;
* interior intended action when needed;
* smooth utility and cost;
* smooth density in action;
* common support;
* dominated convergence / Leibniz conditions for differentiating under the integral sign;
* finite expected compensation cost for canonical contracts.

### Assumption 2: Local implementability

State in economic language:

* the intended action can be locally implemented by some feasible utility contract.

Technical version can be placed in appendix or footnote:

$$
\Delta u\,E_{a_0}[S_+]>c'(a_0)
$$

is a sufficient condition.

### Assumption 3: Low-score curvature

State:

* sufficiently low-score outcomes have $f_{aa}(y|a)>0$ uniformly in $a$.

Informal wording:

> Very bad news about effort becomes less likely at an increasing rate as effort falls, so zero-payment states in that region contribute negative curvature to the agent's objective.

The formal score-cutoff version can be defined later in the proof section.

### Assumption 4: Low-score capacity

State:

* the low-score curvature region contains enough incentive capacity to satisfy LIC while placing the limited-liability kink there.

In the main text this can be named and explained. The formal expression with $A_q$ can be introduced in the proof section.

### Assumption 5: Paid-region curvature control

State that one of the following routes holds:

1. flattening route; or
2. asymptotic-affinity route.

Do not put the full $M_q$ and residual conditions here. They will be defined in the proof section.

---

## 5. Relaxed Optimal Contracts

This is the first analytical section. It solves the relaxed problem before discussing global IC.

### Definition: Marginal cost maps

Define:

* $m_0=1/u'(w_0)$;
* $h(z)=(k')^{-1}(z)$;
* $g(z)=h(z\vee m_0)$.

Explain:

* $h$ is the unconstrained inverse marginal-cost map;
* $g$ adds the limited-liability floor.

### Definition: Canonical contract

Define:

$$
V(y|\lambda,\mu)=g(\lambda+\mu S(y|a_0)).
$$

Also define associated final wealth and payment:

$$
W(y|\lambda,\mu)=k(V(y|\lambda,\mu)),
$$

$$
x(y|\lambda,\mu)=k(V(y|\lambda,\mu))-w_0.
$$

### Proposition 1: Relaxed optimal contract

Main-text result.

Content:

1. relaxed problem has an a.e. unique solution for every feasible reservation utility below the upper relaxed frontier;
2. solution is canonical;
3. as reservation utility approaches the upper relaxed frontier, $\lambda^*(\bar U)\to\infty$;
4. if utility is unbounded, the upper relaxed frontier is $\infty$.

Keep the statement shorter than the current Proposition 1 if possible. Move detailed comparative statics to proof or appendix.

### Definition: Upper relaxed frontier

Define after or inside Proposition 1:

* $\tilde\mu(\lambda)$: LIC multiplier for fixed $\lambda$;
* $\tilde V(y|\lambda)=V(y|\lambda,\tilde\mu(\lambda))$;
* $\tilde U(\lambda)=U(\tilde V(\cdot|\lambda),a_0)$;
* $\bar U_R=\lim_{\lambda\to\infty}\tilde U(\lambda)$.

### Remark: High feasible reservation utility

Explain:

* high feasible reservation utility means $\bar U\uparrow\bar U_R$;
* if $\bar U_R=\infty$, this means $\bar U\to\infty$.

---

## 6. From Local to Global Incentive Compatibility

This section begins the chronological proof of the main result.

### Lemma 1: Concavity implies FOA validity

State:

If the relaxed solution satisfies

$$
U_{aa}(v^*,a)\le0\quad\text{for all }a\in\mathcal A,
$$

then LIC implies GIC and the relaxed solution solves the full problem.

This lemma is simple and should be in the main text.

### Roadmap paragraph

Explain that the rest of the analysis proves concavity of the agent's objective for high feasible reservation utility.

---

## 7. The Limited-Liability Kink

### Proof notation introduced here

Fix $a_0$ and write:

$$
s(y)=S(y|a_0).
$$

### Definition: Kink / threshold score

Define:

$$
\bar s(\lambda,\mu)=\frac{m_0-\lambda}{\mu}.
$$

For the LIC path:

$$
\bar s(\lambda)=\frac{m_0-\lambda}{\tilde\mu(\lambda)}.
$$

Zero-payment region:

$$
\{s\le \bar s(\lambda)\}.
$$

### Explanation

High feasible reservation utility corresponds to large $\lambda$. The key question is where the kink lies as $\lambda$ becomes large.

---

## 8. Low-Score Capacity and the Kink Lemma

This is where the formal score-tail objects enter.

### Definition: Tail-safe cutoff

Define $q<0$ such that

$$
f_{aa}(y|a)>0
$$

for all $a$ and all $y$ with $s(y)\le q$.

### Definition: Low-score set

$$
L_q=\{y:s(y)\le q\}.
$$

### Definition: Low-score incentive capacity

$$
A_q=-\int_{L_q}s(y)f(y|a_0)\,d\nu(y).
$$

### Assumption: Tail-safe capacity

$$
\Delta u\,A_q>c'(a_0).
$$

This is the formal version of low-score capacity.

### Definition: Trial multiplier placing kink at $q$

$$
\mu_q(\lambda)=\frac{m_0-\lambda}{q}.
$$

### Definition: Incentive functional

$$
I(\lambda,\mu)=\int g(\lambda+\mu s(y))s(y)f(y|a_0)\,d\nu(y).
$$

### Lemma 2: Kink lemma

Content:

If the contract whose kink is at $q$ provides at least enough incentive, then the true LIC contract has kink weakly below $q$.

### Corollary: Capacity implies kink eventually below $q$

Content:

Tail-safe capacity implies that, for all sufficiently high feasible reservation utilities, the relaxed optimal contract's kink lies in the tail-safe low-score region.

---

## 9. Curvature Decomposition

### Definition: Secant slope term

For the LIC path, define

$$
\Delta g(y|\lambda)=
\frac{g(\lambda+\tilde\mu(\lambda)s(y))-g(\lambda)}
{\tilde\mu(\lambda)s(y)}.
$$

### Lemma 3: Curvature formula

State:

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
=
\tilde\mu(\lambda)
\int \Delta g(y|\lambda)s(y)f_{aa}(y|a)\,d\nu(y)
-c''(a).
$$

### Definition: Positive-part bound

Define

$$
M_q=
\sup_{a\in\mathcal A}
\int_{\{s>q\}}[s(y)f_{aa}(y|a)]_+\,d\nu(y).
$$

### Proposition 2: Finite-$\lambda$ concavity criterion

Content:

If:

1. the kink is below a tail-safe cutoff $q$;
2. $M_q<\infty$;
3. the positive paid-region bound satisfies

$$
\tilde\mu(\lambda)g'(\lambda+\tilde\mu(\lambda)q)M_q\le \underline c'',
$$

then

$$
U_{aa}\le0
$$

for all actions.

This proposition can be in the proof section. Full proof can be in appendix.

---

## 10. High-Reservation Concavity: Flattening Route

### Proposition 3: Flattening theorem

Content:

If:

1. tail-safe capacity holds;
2. $M_q<\infty$;
3. along the LIC path,

$$
\tilde\mu(\lambda)g'(\lambda+\tilde\mu(\lambda)q)\to0,
$$

then the agent's problem is concave for all sufficiently high feasible reservation utilities.

### Lemma 4: Bounded-utility step-contract argument

Content:

Under tail-safe capacity, for bounded utility,

$$
z_q(\lambda):=\lambda+\tilde\mu(\lambda)q\to\infty.
$$

With a stricter cutoff, $z_q(\lambda)$ grows at least linearly in $\lambda$.

This lemma may go in the appendix if too technical.

### Corollary: Log utility

Content:

Log utility satisfies flattening; because utility is unbounded, capacity is automatic when $A_q>0$.

### Corollary: CARA utility

Content:

CARA satisfies flattening under the capacity condition

$$
\frac{e^{-\alpha w_0}}{\alpha}A_q>c'(a_0).
$$

### Corollary: CRRA with $\gamma>1$

Content:

Bounded CRRA satisfies flattening under the capacity condition

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_q>c'(a_0).
$$

---

## 11. High-Reservation Concavity: Asymptotic-Affinity Route

This section handles cases where flattening may fail, especially CRRA with $\gamma\in(1/2,1)$.

### Definition: Affine-score curvature

Define condition:

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)\le0
\quad\text{for all }a\in\mathcal A.
$$

Equivalent interpretation:

$$
a\mapsto E_a[S(Y|a_0)]
$$

is concave.

### Definition: Residual condition

State the weighted residual condition:

$$
\int [g(\lambda+\tilde\mu s)-g(\lambda)-\tilde\mu g'(\lambda)s]f_{aa}\,d\nu\to0
$$

uniformly in $a$.

### Proposition 4: Asymptotic-affinity theorem

Content:

If affine-score curvature and the residual condition hold, then the agent's problem is concave for all sufficiently high feasible reservation utilities.

### Corollary: CRRA with $\gamma\in(1/2,1)$

Content:

CRRA with $\gamma\in(1/2,1)$ satisfies the residual condition under suitable moment/uniform-integrability assumptions and $\tilde\mu/\lambda\to0$.

### Corollary: Gaussian location family

Content:

For Gaussian location families,

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)=0,
$$

so affine-score curvature holds with equality. Therefore Gaussian CRRA is covered for $\gamma>1/2$.

---

## 12. Main Validity Theorem

After the chronological analysis, state the formal main theorem.

### Theorem 1: Validity with high feasible reservation utility

Content:

Under the maintained assumptions, including:

1. local implementability;
2. low-score capacity;
3. either flattening or asymptotic-affinity paid-region curvature control;

there exists $U^*<\bar U_R$ such that the first-order approach is valid for every

$$
\bar U\in[U^*,\bar U_R).
$$

If utility is unbounded above, then $\bar U_R=\infty$, so the first-order approach is valid for all sufficiently high reservation utilities.

### Proof of Theorem 1

One paragraph using previous results:

1. Proposition 1 gives the relaxed solution and $\lambda^*(\bar U)\to\infty$ as $\bar U\uparrow\bar U_R$.
2. Kink lemma and capacity place the kink in the tail-safe region.
3. Flattening or asymptotic affinity gives concavity for large $\lambda$.
4. Concavity implies GIC.
5. Therefore relaxed solution solves the full problem.

---

## 13. Applications and Closed Forms

This section can follow the main theorem.

### Utility table

Update table to use utility over final wealth $u(W)$ and initial wealth $w_0$ separately.

Include:

* log;
* CARA;
* CRRA.

For each utility, list:

1. $u(W)$;
2. $k(v)$;
3. $h(z)=(k')^{-1}(z)$;
4. $g(z)=h(z\vee m_0)$;
5. induced payment $x(y)=k(g(\lambda+\mu S))-w_0$;
6. utility spread $\Delta u$.

### Distribution table

Keep or update distribution table.

For each distribution, list:

1. output support $\mathcal Y$;
2. density/mass $f(y|a)$;
3. score $S(y|a)$;
4. notes on low-score curvature / tail-safe cutoff.

### Gaussian corollary

State cleanly:

* Gaussian location families satisfy low-score curvature;
* affine-score curvature holds with equality;
* log, CARA under capacity, and CRRA $\gamma>1/2$ under the appropriate branch are covered.

### Poisson discussion or corollary

Depending on how much we can prove cleanly:

* show how the common-measure/score-cutoff notation covers Poisson;
* either state a finite-$\lambda$ criterion for Poisson or present it as an example of the computational method;
* avoid overclaiming if bounded-below score requires model-specific limiting curvature.

---

## 14. Numerical Methods

Keep after theoretical applications.

Definitions/results to include:

1. algorithm for solving relaxed problem;
2. algorithm for checking GIC / global deviations;
3. algorithm for full problem when FOA fails;
4. computational complexity discussion.

Notation should use payment $x(y)$ and final wealth $W(y)$ in exposition, with utility contracts $v(y)$ in formulas where useful.

---

## 15. Counterexamples and Limitations

Use this section to explain what the assumptions rule out.

Topics:

1. low reservation utility failures;
2. failure of low-score curvature;
3. failure of capacity with bounded utility;
4. cases where paid-region curvature does not vanish and affine-score curvature fails;
5. relation to existing restrictive FOA sufficient conditions.

---

## 16. Literature Discussion

Can remain late in the paper.

Emphasize:

1. difference from Jewitt: limited liability and high feasible reservation utility;
2. relation to standard first-order approach;
3. relation to failures of FOA at low reservation utility;
4. why existing global sufficient conditions are restrictive.

If desired, include a historical note that Jewitt denotes $(k')^{-1}$ by $\omega$, but the paper uses $h$ for transparency.

---

## 17. Conclusion

Restate the chronological insight:

1. relaxed contracts are canonical;
2. high feasible reservation utility sends the IR multiplier to infinity;
3. the limited-liability kink moves into low-score states;
4. curvature becomes negative;
5. FOA becomes valid.

---

# Appendix Plan

The appendix should contain full technical proofs and more general statements.

---

## Appendix A. Regularity and Differentiation

### Remark/Lemma: Differentiation under the integral sign

Generalize current Remark \ref{rem:leibniz} to common dominating measure.

### Lemma: Score has mean zero

Show

$$
\int S(y|a)f(y|a)\,d\nu(y)=0.
$$

### Lemma: Derivatives of expected utility

For feasible utility contracts,

$$
U_a(v,a)=\int v(y)f_a(y|a)\,d\nu(y)-c'(a),
$$

and

$$
U_{aa}(v,a)=\int v(y)f_{aa}(y|a)\,d\nu(y)-c''(a).
$$

---

## Appendix B. Proof of Relaxed Optimal Contract Proposition

Full proof of Proposition 1.

Definitions/results:

1. Lagrangian.
2. Lemma: canonical contracts minimize Lagrangian.
3. Lemma: existence and uniqueness of $\tilde\mu(\lambda)$ under local implementability.
4. Lemma: relaxed Pareto problem solved by $\tilde V(\cdot|\lambda)$.
5. Lemma: $\tilde U(\lambda)$ is continuous and strictly increasing.
6. Lemma: definition and properties of $\bar U_R$.
7. Proposition proof: existence, uniqueness, canonical form, and $\lambda^*(\bar U)\to\infty$ as $\bar U\uparrow\bar U_R$.

---

## Appendix C. Kink Lemma and Finite-$\lambda$ Concavity Criterion

Full technical proofs of Sections 8 and 9.

Definitions/results:

1. $L_q$, $A_q$, $I(\lambda,\mu)$, $\mu_q(\lambda)$.
2. Lemma: monotonicity of $I$ in $\mu$.
3. Lemma: kink comparison.
4. Lemma: capacity implies eventual kink placement.
5. Lemma: curvature formula with secant slope.
6. Lemma: secant slope bound from concavity of $g$.
7. Proposition: finite-$\lambda$ concavity criterion.

---

## Appendix D. Flattening Route

Full proof of Proposition 3.

Definitions/results:

1. bounded-utility step-contract lemma;
2. proof that $z_q(\lambda)\to\infty$;
3. proof of linear lower bound on $z_q(\lambda)$ using stricter cutoff;
4. verification for log;
5. verification for CARA;
6. verification for CRRA $\gamma>1$.

---

## Appendix E. Asymptotic-Affinity Route

Full proof of Proposition 4.

Definitions/results:

1. affine-score curvature condition;
2. residual condition;
3. primitive sufficient conditions for residual;
4. Taylor expansion and central-region bound;
5. uniform-integrability tail argument;
6. verification for CRRA $\gamma\in(1/2,1)$;
7. Gaussian location calculation.

---

## Appendix F. Closed-Form Derivations

Move or keep current supplementary derivations here, depending on journal length constraints.

Include:

1. utility functions with separated $w_0$;
2. link functions $h$ and $g$;
3. payment formulas;
4. distribution scores;
5. Gaussian calculations;
6. Poisson calculations.

---

## Appendix G. Additional Numerical and Counterexample Details

Technical details supporting numerical-methods and counterexamples sections.
