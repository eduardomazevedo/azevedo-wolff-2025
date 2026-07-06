Below is a detailed roadmap for revising **Proposition 1**, parallel in style to the concavity-proof plan, but focused only on the **one-dimensional action case**.

---

# Summary of the Proposition 1 refactor plan

## 1. Separate the relaxed problem from the FOA-validity theorem

The first organizational change is to make Proposition 1 a theorem about the **relaxed cost-minimization problem only**.

The current Proposition 1 does three things at once:

1. establishes existence and uniqueness of the relaxed solution;
2. derives the canonical first-order formula;
3. gives comparative statics in reservation utility, including $(\lambda^*(\bar U)\to\infty)$ as $(\bar U\to\infty)$. 

This should remain the core, but the assumptions should be weakened and clarified. In particular, Proposition 1 should not rely on the assumptions used later for global IC or concavity.

The refactored Proposition 1 should not require:

$$
\operatorname{supp} f(\cdot|a)=\mathbb R,
$$

nor

$$
S(\cdot|a_0)\text{ strictly increasing with image }\mathbb R,
$$

nor

$$
f_{aa}(y|a)>0
\quad\text{in the left tail}.
$$

Those are concavity / FOA-validity assumptions. They are not needed to solve the relaxed problem.

The relaxed problem only needs enough structure to make the Lagrangian well-defined, guarantee existence, and ensure that the local IC constraint can be satisfied by a canonical contract.

---

## 2. Work on a general one-dimensional outcome space

For Proposition 1, we should allow output to have support different from (\mathbb R). The outcome space can be written as a measurable space

$$
(Y,\mathcal Y,\nu),
$$

where $(\nu)$ is a dominating measure and $(f(y|a))$ is a density or probability mass function with respect to $(\nu)$.

This covers:

$$
Y=\mathbb R,\quad Y=[0,\infty),\quad Y=[\underline y,\bar y],\quad Y={0,1,\dots,n},\quad Y=\mathbb N,
$$

and mixtures can be handled later if useful.

The score at the intended action is

$$
S(y|a_0):=\frac{f_a(y|a_0)}{f(y|a_0)}
$$

on the support of (f(\cdot|a_0)).

The relaxed problem is still:

$$
\min_{v\in C} W(v,a_0)
$$

subject to

$$
U(v,a_0)\ge \bar U,
$$

and

$$
U_a(v,a_0)=0.
$$

Equivalently,

$$
\int v(y)f(y|a_0),d\nu(y)-c(a_0)\ge \bar U,
$$

and

$$
\int v(y)f_a(y|a_0),d\nu(y)=c'(a_0).
$$

The existing paper already defines the relaxed problem in this way for continuous real-valued output, replacing global IC with local IC.  The refactor should make clear that nothing in this part depends on real-line support.

---

## 3. Clarify utility, compensation cost, and bounded utility

Let

$$
\bar u:=u(\infty):=\sup_{x\ge0}u(x),
$$

allowing either

$$
\bar u<\infty
$$

or

$$
\bar u=\infty.
$$

Contracts are utility contracts:

$$
v:Y\to [u(0),\bar u).
$$

If $(\bar u<\infty)$, one can either allow the endpoint $(v=\bar u)$ as a formal closure point or keep contracts strictly below $(\bar u)$ and treat endpoint feasibility separately. The cleanest approach is probably:

$$
C\subseteq {v:Y\to [u(0),\bar u]}
$$

with the convention that $(k(\bar u)=+\infty)$ if $(\bar u)$ is not attainable by finite wages. Then finite expected wage automatically prevents paying $(\bar u)$ on a positive-probability set unless $(\bar u)$ is actually attainable.

The compensation cost function is

$$
k=u^{-1},
$$

defined on ([u(0),\bar u)), with

$$
k(v)\to\infty
\quad\text{as}\quad
v\uparrow \bar u
$$

when $\bar u<\infty$ is a utility supremum generated only by infinite wages.

This gives a unified treatment:

* unbounded utility: $\bar u=\infty$;
* bounded utility: $\bar u<\infty$;
* endpoint utility may or may not be attainable, depending on whether finite wages can deliver it.

The current paper explicitly notes that unbounded utility is used to avoid the bounded-feasibility endpoint and that bounded utility would require handling the case where (\bar U) exceeds feasible utility.  The refactor should make this endpoint part of the proposition instead of treating it as a complication.

---

## 4. Define the generalized link function

The canonical formula should be written using a generalized link function.

Let

$$
m:=k'(u(0))=\frac{1}{u'(0)}.
$$

Define

$$
g(z):=\arg\min_{v\in [u(0),\bar u]}
{k(v)-zv}.
$$

Equivalently, when the inverse marginal cost exists,

$$
g(z)=k'^{-1}(\max\{m,z\}).
$$

This automatically incorporates limited liability. If (z<m), the minimum is at the lower bound:

$$
g(z)=u(0).
$$

If (z\ge m), then

$$
k'(g(z))=z.
$$

This is the same object as the current link function, except that the current draft defines it only under unbounded utility assumptions.  The refactored definition should be stated in a way that works for bounded utility as well.

For bounded utility, as

$$
z\to\infty,
$$

we have

$$
g(z)\uparrow \bar u.
$$

This will be essential for showing that $(\lambda^*(\bar U)\to\infty)$ as $(\bar U)$ approaches the top feasible relaxed utility.

---

## 5. State the canonical contract

For scalar action, the canonical contract is

$$
V(y|\lambda,\mu)
=
g\left(\lambda+\mu S(y|a_0)\right).
$$

Here:

* $(\lambda\ge0)$ is the multiplier on IR;
* $(\mu\in\mathbb R)$ is the multiplier on local IC;
* in the usual increasing-score case with $(a_0>0)$, we expect $(\mu>0)$, but the relaxed-problem theorem itself can state the sign using a separate incentive-orientation condition.

The current paper’s canonical contract is exactly this:

$$
V(y|\lambda,\mu)
=
g\left(\lambda+\mu S(y|a_0)\right).
$$



But the refactored proposition should emphasize that this formula is valid on arbitrary support, not only on (\mathbb R).

---

## 6. Replace strong score assumptions with a local incentive feasibility condition

The current proof uses strict monotonicity and unbounded image of the score to prove that, for each (\lambda), there is a unique (\mu>0) satisfying local IC. 

For Proposition 1, this can be weakened.

Define

$$
I(\lambda,\mu)
:=
\int g(\lambda+\mu S(y|a_0)),\, f_a(y|a_0),d\nu(y).
$$

Using the score,

$$
f_a(y|a_0)=S(y|a_0)f(y|a_0),
$$

so equivalently

$$
I(\lambda,\mu)
=
\int g(\lambda+\mu S(y|a_0))S(y|a_0)f(y|a_0),d\nu(y).
$$

The local IC condition is

$$
I(\lambda,\mu)=c'(a_0).
$$

A clean assumption is:

**Local incentive feasibility.** For every relevant (\lambda), the equation

$$
I(\lambda,\mu)=c'(a_0)
$$

has a unique solution (\tilde\mu(\lambda)).

A more primitive sufficient condition is:

1. (S(Y|a_0)) is not almost surely constant;
2. (S) takes both positive and negative values with positive probability, or at least enough positive-score variation exists to generate incentives;
3. the map (\mu\mapsto I(\lambda,\mu)) is continuous and strictly increasing;
4. the attainable range of (I(\lambda,\cdot)) contains (c'(a_0)).

Strict monotonicity follows from

$$
I_\mu(\lambda,\mu)
=
\int g'(\lambda+\mu S(y|a_0))S(y|a_0)^2f(y|a_0),d\nu(y)\ge0,
$$

with strict inequality when the paid region contains nontrivial score variation.

This is the right general replacement for the stronger increasing-score assumption.

---

## 7. Define the relaxed Pareto frontier

The cleanest proof should first solve a Pareto problem.

For each (\lambda), define

$$
\tilde\mu(\lambda)
$$

as the unique multiplier satisfying LIC, and define

$$
\tilde V_\lambda(y)
:=
V(y|\lambda,\tilde\mu(\lambda)).
$$

Then define

$$
\tilde U(\lambda)
:=
U(\tilde V_\lambda,a_0),
$$

and

$$
\tilde W(\lambda)
:=
W(\tilde V_\lambda,a_0).
$$

The current proof already does this in substance by defining a one-dimensional family of canonical contracts indexed by (\lambda), then showing that these contracts solve the relaxed Pareto problem. 

The refactor should make this structure more explicit.

---

## 8. Define the maximum relaxed utility

This is the main bounded-utility improvement.

Define

$$
\bar U^R
:=
\sup\left\{
U(v,a_0):
v\in C,\ U_a(v,a_0)=0
\right\}.
$$

This is the highest utility that can be delivered to the agent while satisfying the local IC constraint at (a_0).

Then:

* if (u) is unbounded above, often $\bar U^R=\infty$;
* if (u) is bounded above, then necessarily
$$
  \bar U^R\le \bar u-c(a_0)<\infty.
$$

The relaxed problem is feasible only for

$$
\bar U\le \bar U^R
$$

if the supremum is attained, and for

$$
\bar U<\bar U^R
$$

otherwise.

This is the precise replacement for the current statement that $(\lambda^*(\bar U)\to\infty)$ as $(\bar U\to\infty)$. The general statement is:

$$
\lambda^*(\bar U)\to\infty
\quad\text{as}\quad
\bar U\uparrow \bar U^R.
$$

When $\bar U^R=\infty$, this reduces to the old unbounded-utility statement.

---

## 9. Low reservation utility region

As in the current Proposition 1, there may be a low reservation utility region where IR is slack.

Define

$$
\bar U^L:=\tilde U(0).
$$

Then for

$$
\bar U\le \bar U^L,
$$

the IR constraint does not bind. The solution is the canonical contract corresponding to (\lambda=0):

$$
v^*(y|\bar U)=\tilde V_0(y).
$$

In this region:

$$
\lambda^*(\bar U)=0,
$$

and

$$
\omega(\bar U)
$$

is constant.

This is exactly the current low-reservation conclusion. 

The refactor should preserve it.

---

## 10. Binding reservation utility region

For

$$
\bar U\in(\bar U^L,\bar U^R),
$$

the IR constraint binds.

Then (\lambda^*(\bar U)) is defined by

$$
\tilde U(\lambda^*(\bar U))=\bar U.
$$

The optimal contract is

$$
v^*(y|\bar U)
=
   \tilde V_{\lambda^*(\bar U)}(y)
=
g\left(\lambda^*(\bar U)+\tilde\mu(\lambda^*(\bar U))S(y|a_0)\right).
$$

The relaxed expected wage is

$$
\omega(\bar U)
=
\tilde W(\lambda^*(\bar U)).
$$

The value function should satisfy:

$$
\omega'(\bar U)=\lambda^*(\bar U)
$$

where differentiable, and more generally (\lambda^*(\bar U)) is a subgradient.

Because (k) is strictly convex, (\omega) is strictly convex on the binding range, except possibly in degenerate cases where canonical contracts fail to differ on a positive-probability set. The current proof already uses strict convexity of (k) to establish strict monotonicity of (\tilde U). 

---

## 11. Endpoint behavior and $\lambda\to\infty$

The key new bounded-utility result is the endpoint statement.

The proposition should show:

$$
\lim_{\bar U\uparrow \bar U^R}
\lambda^*(\bar U)
=
\infty.
$$

Equivalently,

$$
\lim_{\lambda\to\infty}\tilde U(\lambda)=\bar U^R.
$$

The proof idea is convex-analytic.

If (\lambda) stayed bounded along a sequence with

$$
\bar U_n\uparrow \bar U^R,
$$

then the corresponding canonical contracts would remain away from the endpoint in marginal-cost space. Under compactness/integrability, a subsequence would converge to a feasible relaxed contract that attains (\bar U^R) with finite shadow price. But if (\bar U^R) is the top of the attainable frontier, the supporting hyperplane must become vertical; hence the shadow value of utility must diverge.

More concretely, along canonical contracts,

$$
g(\lambda+\tilde\mu(\lambda)S)
$$

approaches the utility-maximizing feasible extreme as $(\lambda\to\infty)$. If $(u)$ is bounded, this means the contract increasingly uses values close to

$$
\bar u
$$

on high-incentive states, while respecting local IC. If (u) is unbounded, the same statement becomes utility going to infinity.

The bounded-utility concavity notes already use this type of logic: as $(\lambda\to\infty)$, canonical contracts converge toward step-like contracts using the maximum utility spread $(u(\infty)-u(0))$. 

For Proposition 1, we do not need the step-contract details; we only need the frontier statement.

---

## 12. Attainment at the top endpoint

The proposition should distinguish three cases.

### Case 1: (\bar U<\bar U^R)

The relaxed problem has a unique canonical solution with finite multipliers.

### Case 2: (\bar U>\bar U^R)

The relaxed problem is infeasible.

### Case 3: (\bar U=\bar U^R)

There are two subcases.

If the supremum is attained by some feasible (v^R), then the relaxed problem is feasible at the endpoint. But the solution may not be represented by finite multipliers. It may instead be the limit of canonical contracts:

$$
v^R=\lim_{\lambda\to\infty}\tilde V_\lambda.
$$

If the supremum is not attained, then the relaxed problem is infeasible at the endpoint.

This is especially important for bounded utility. If (\bar u) is only attainable with infinite wage, the top utility may be a supremum but not feasible at finite expected wage.

The proposition should avoid forcing endpoint existence unless we add a separate compactness condition guaranteeing it.

---

## 13. Almost-everywhere uniqueness

The relaxed objective is

$$
W(v,a_0)=\int k(v(y))f(y|a_0),d\nu(y).
$$

Since (k) is strictly convex, the objective is strictly convex in (v) on the set where

$$
f(y|a_0)>0.
$$

The constraints are linear in (v). Therefore, if a solution exists, it is unique up to (f(\cdot|a_0))-null sets.

This should be stated explicitly:

$$
v^*_1(y)=v^*_2(y)
\quad f(\cdot|a_0)\text{-a.e.}
$$

The current proposition already states almost-everywhere uniqueness. 

---

## 14. Proposed formal statement

A possible final statement is:

**Proposition 1. Relaxed cost-minimization problem.**
Fix (a_0>0). Suppose regularity, integrability, limited liability, strict convexity of (k), and local incentive feasibility hold. Define

$$
\bar U^R
=
\sup{U(v,a_0):v\in C,\ U_a(v,a_0)=0}.
$$

Then there exists

$$
\bar U^L<\bar U^R
$$

such that:

1. **Infeasibility above the relaxed frontier.**
   If
$$
   \bar U>\bar U^R,
$$
   the relaxed problem is infeasible.

2. **Existence and uniqueness below the frontier.**
   If
$$
   \bar U<\bar U^R,
$$
   the relaxed problem has an almost-everywhere unique solution.

3. **Canonical formula.**
   The solution is
$$
   v^*(y|\bar U)
   =
   g\left(\lambda^*(\bar U)+\mu^*(\bar U)S(y|a_0)\right),
$$
   with (\lambda^*(\bar U)\ge0), and (\mu^*(\bar U)) satisfying local IC.

4. **Low reservation region.**
   If
$$
   \bar U\le \bar U^L,
$$
   then
$$
   \lambda^*(\bar U)=0,
$$
   and the optimal contract and expected wage do not vary with (\bar U).

5. **Binding region.**
   If
$$
   \bar U\in(\bar U^L,\bar U^R),
$$
   then IR binds,
$$
   \lambda^*(\bar U)>0,
$$
$$
   \lambda^*(\bar U)\text{ is strictly increasing},
$$
   and
$$
   \omega(\bar U)
$$
   is strictly increasing and strictly convex.

6. **Endpoint behavior.**
   As
$$
   \bar U\uparrow\bar U^R,
$$
   we have
$$
   \lambda^*(\bar U)\to\infty.
$$

7. **Endpoint feasibility.**
   At
$$
   \bar U=\bar U^R,
$$
   the relaxed problem has a solution if and only if the supremum defining (\bar U^R) is attained at finite expected wage. In that case, the solution is the limit of canonical contracts as $\lambda\to\infty$, though it need not have finite multipliers.

This proposition includes the current unbounded case by setting

$$
\bar U^R=\infty.
$$

Then endpoint behavior becomes the old conclusion:

$$
\lambda^*(\bar U)\to\infty
\quad\text{as}\quad
\bar U\to\infty.
$$

---

## 15. Proof structure

The proof should be modular.

### A. Pointwise Lagrangian minimization

Define the Lagrangian:

$$
L(v,\lambda,\mu)
=
W(v,a_0)
+
\lambda(\bar U-U(v,a_0))
+
\mu(-U_a(v,a_0)).
$$

Ignoring constants, this is

$$
\int
\left[
k(v(y))
- (\lambda+\mu S(y|a_0))v(y)
\right]
f(y|a_0),d\nu(y)
+
\text{constants}.
$$

Pointwise minimization gives

$$
v(y)=g(\lambda+\mu S(y|a_0)).
$$

This is the formal source of the FOC formula. The current proof already does this in Lemma A.1. 

### B. Solve local IC for (\mu)

For each (\lambda), solve

$$
I(\lambda,\mu)=c'(a_0).
$$

Under local incentive feasibility, this gives a unique

$$
\tilde\mu(\lambda).
$$

This defines the one-dimensional family

$$
	\tilde V_\lambda(y)=g(\lambda+\tilde\mu(\lambda)S(y|a_0)).
$$

### C. Pareto problem

For fixed (\lambda), consider the Pareto problem:

$$
\min_v W(v,a_0)-\lambda U(v,a_0)
$$

subject to

$$
U_a(v,a_0)=0.
$$

The pointwise Lagrangian result implies that

$$
\tilde V_\lambda
$$

solves this Pareto problem.

This is the clean version of the current Lemma A.3. 

### D. Monotonicity of (\tilde U(\lambda))

Show that

$$
\tilde U(\lambda)
$$

is weakly increasing, and strictly increasing under nondegeneracy.

The standard proof compares optimality at (\lambda_1<\lambda_2):

$$
W(\tilde V_{\lambda_1})-\lambda_1U(\tilde V_{\lambda_1})
\le
W(\tilde V_{\lambda_2})-\lambda_1U(\tilde V_{\lambda_2}),
$$

and

$$
W(\tilde V_{\lambda_2})-\lambda_2U(\tilde V_{\lambda_2})
\le
W(\tilde V_{\lambda_1})-\lambda_2U(\tilde V_{\lambda_1}).
$$

Adding gives

$$
(\lambda_2-\lambda_1)
\left[
	\tilde U(\lambda_2)-\tilde U(\lambda_1)
\right]\ge0.
$$

Strictness follows from strict convexity of (k) and nondegeneracy. This is already the logic of Lemma A.4. 

### E. Identify (\bar U^L) and (\bar U^R)

Define

$$
\bar U^L=\tilde U(0),
$$

and

$$
\bar U^R=\lim_{\lambda\to\infty}\tilde U(\lambda).
$$

Then show that this (\bar U^R) equals the supremum over all locally incentive-compatible contracts:

$$
\bar U^R
=
\sup\left\{U(v,a_0): U_a(v,a_0)=0\right\}.
$$

This is the key step for bounded utility.

### F. Solve the relaxed cost-minimization problem

For $(\bar U\le\bar U^L)$, use $(\lambda=0)$.

For $\bar U\in(\bar U^L,\bar U^R)$, invert

$$
\tilde U(\lambda)=\bar U.
$$

This gives

$$
\lambda^*(\bar U)=\tilde U^{-1}(\bar U),
$$

and the canonical solution.

### G. Endpoint behavior

Because

$$
\bar U^R=\lim_{\lambda\to\infty}\tilde U(\lambda),
$$

we immediately get

$$
\lambda^*(\bar U)\to\infty
\quad\text{as}\quad
\bar U\uparrow\bar U^R.
$$

This is the bounded-utility replacement for the old $\bar U\to\infty$ statement.

---

## 16. How this improves the paper

The refactor makes Proposition 1 more general and cleaner.

It will cover:

$$
\text{bounded utility, unbounded utility, bounded support, unbounded support, continuous output, and discrete output.}
$$

It will also isolate the real source of the FOC formula:

$$
k'(v^*(y))=\lambda^*+\mu^*S(y|a_0),
$$

with limited liability implemented by the link function (g).

Then the later concavity theorem can impose additional assumptions only when they are actually needed. The current paper already recognizes that assumptions like real-line support and left-tail curvature are tied to proving high-reservation FOA validity, not to deriving the canonical relaxed solution. 

---

## 17. Main conceptual improvement

The old Proposition 1 is best understood as the unbounded-utility version of a more general frontier result:

$$
\boxed{
\text{The relaxed optimum is canonical along the entire feasible frontier.}
}
$$

The right high-reservation statement is not always

$$
\bar U\to\infty.
$$

It is

$$
\boxed{
\bar U\uparrow\bar U^R.
}
$$

And the right multiplier statement is

$$
\boxed{
\lambda^*(\bar U)\to\infty
\quad\text{as the relaxed utility frontier is approached.}
}
$$

This makes bounded utility a first-class case rather than an exception.
