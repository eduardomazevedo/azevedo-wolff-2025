# Plan to Generalize the Concavity Result

Throughout, output lies in a fixed set $\mathcal Y\subseteq\mathbb R$, and $f(\cdot|a)$ is a density with respect to a common dominating measure $\nu$. Integrals are with respect to $d\nu(y)$. This covers both continuous and discrete output.

---

## 1. Score notation

Let

$$
s(y):=S(y|a_0)=\partial_a\log f(y|a)\big|_{a=a_0}.
$$

The canonical contract is

$$
V(y|\lambda,\mu)=\ell(\lambda+\mu s(y)),
\qquad
W(y|\lambda,\mu)=L(\lambda+\mu s(y)).
$$

Let

$$
m_0:=\frac1{u'(w_0)}.
$$

Maintain throughout this note that the unconstrained utility link $\ell_0$ is nondecreasing and concave on the nonbinding region $(m_0,\infty)$. The limited-liability link $\ell(z)=\ell_0(z\vee m_0)$ is generally not globally concave because its slope jumps upward at the kink. At the kink, interpret derivatives as the relevant one-sided derivatives.

Every uniform concavity result below also assumes uniform strong convexity of effort cost:

$$
\underline c'':=\inf_{a\in\mathcal A}c''(a)>0.
\tag{CostCurv}
$$

This is stronger than strict convexity alone and is imposed only for the concavity results, not for the characterization of the relaxed solution in generalized Proposition 1.

The limited-liability floor binds exactly when

$$
\lambda+\mu s(y)\le m_0.
$$

For $\mu>0$, define the threshold score

$$
\bar s(\lambda,\mu):=\frac{m_0-\lambda}{\mu}.
$$

Then the zero-payment region is the score lower contour set

$$
\{y:s(y)\le \bar s(\lambda,\mu)\}.
$$

This notation avoids assuming that the output support is an interval or that the score map is invertible. It includes continuous examples and discrete examples such as Poisson.

---

## 2. Tail-safe score interval, boundary, and capacity

### Definitions

A **tail-safe cutoff** is a number $q\le0$ such that

$$
f_{aa}(y|a)\ge0
\quad\text{for all }a\in\mathcal A
\quad\text{and all }y\in\mathcal Y\text{ with }s(y)\le q.
\tag{TS}
$$

Let

$$
\mathcal Q_{\mathrm{safe}}
:=
\{q\le0:q\text{ is tail-safe}\}.
$$

For any cutoff $q\le0$, define the low-score set

$$
B_q:=\{y\in\mathcal Y:s(y)\le q\}.
$$

Define the incentive capacity up to cutoff $q$ by

$$
A_q:=-\int_{B_q}s(y)f(y|a_0)\,d\nu(y)\ge0.
$$

Define the paid-region positive-curvature bound by

$$
M_q:=
\sup_{a\in\mathcal A}
\int_{\{s>q\}}
[s(y)f_{aa}(y|a)]_+\,d\nu(y).
$$

Finally, define the right edge and capacity of the safe score region by

$$
\bar q_{\mathrm{safe}}
:=
\sup \mathcal Q_{\mathrm{safe}},
$$

with the convention that $\bar q_{\mathrm{safe}}=-\infty$ if $\mathcal Q_{\mathrm{safe}}=\varnothing$, and

$$
A_{\mathrm{safe}}
:=
\sup_{q\in\mathcal Q_{\mathrm{safe}}}A_q.
$$

### Conventions and monotonicity

The weak inequality $s(y)\le q$ matches the limited-liability no-payment region exactly. The paid-region set in $M_q$ is the strict complement $\{s>q\}$, so a score atom at $q$ is assigned to the safe side when the kink is at $q$.

The set $\mathcal Q_{\mathrm{safe}}$ is a lower interval in score space: if $q$ is tail-safe, then every $q'<q$ is tail-safe. It may be open or closed at its right edge. This distinction is immaterial for continuous score models and important but harmless for discrete models.

Capacity $A_q$ is increasing in $q$ because increasing the cutoff adds only nonpositive scores to $B_q$. With the weak lower contour convention, $A_q$ is right-continuous as a function of the real cutoff and can jump upward at score atoms. The curvature bound $M_q$ is decreasing in $q$ because increasing the cutoff removes scores from the paid region. For each fixed action, the integral inside $M_q$ is right-continuous when finite and can jump downward at score atoms; since $M_q$ is a supremum over actions, any needed continuity of $M_q$ itself should be imposed as regularity rather than built into the definition.

The supremum definition of $A_{\mathrm{safe}}$ is the clean way to handle both continuous endpoints and discrete jumps. In continuous score models with the relevant continuity, it equals capacity at the right endpoint; in discrete models it avoids having to decide separately whether $\bar q_{\mathrm{safe}}$ itself belongs to $\mathcal Q_{\mathrm{safe}}$.

---

## 3. Kink placement lemma

### Definitions

For fixed $\lambda>m_0$ and a negative tail-safe cutoff $q<0$, define

$$
\mu_q(\lambda):=\frac{m_0-\lambda}{q}>0.
$$

This is the value of $\mu$ that places the limited-liability threshold exactly at score $q$:

$$
\lambda+\mu_q(\lambda)q=m_0.
$$

Define the local-incentive functional

$$
I(\lambda,\mu):=
\int \ell(\lambda+\mu s(y))s(y)f(y|a_0)\,d\nu(y).
$$

The LIC multiplier solves

$$
I(\lambda,\tilde\mu(\lambda))=c'(a_0).
$$

Because $\ell$ is nondecreasing, $I(\lambda,\mu)$ is weakly increasing in $\mu$: increasing $\mu$ raises delivered utility in positive-score states and lowers it in negative-score states, and both changes raise the score-weighted integral. Where differentiation under the integral sign is justified,

$$
I_\mu(\lambda,\mu)=
\int \ell'(\lambda+\mu s(y))s(y)^2 f(y|a_0)\,d\nu(y)\ge0.
$$

Assume the standard score nondegeneracy needed for strict monotonicity at an LIC solution and hence uniqueness of $\tilde\mu(\lambda)$. This condition belongs with generalized Proposition 1.

**Kink placement lemma.** Fix a negative tail-safe cutoff $q<0$. If

$$
[\bar u-\underline u]A_q>c'(a_0),
\tag{Cap-q}
$$

then, for all sufficiently large $\lambda$, the kink score is below $q$:

$$
\bar s(\lambda,\tilde\mu(\lambda))\le q.
$$

Proof logic: since $\ell(\lambda)\to\bar u$, strict capacity implies that, for all sufficiently large $\lambda$,

$$
[\ell(\lambda)-\underline u]A_q
\ge c'(a_0).
$$

At $\mu=\mu_q(\lambda)$, states with $s\le q$ receive utility $\underline u$, while states with $s>q$ receive at least $\ell(\lambda)$ if $s\ge0$. Using $E_{a_0}[s(Y)]=0$ gives the crude lower bound

$$
I(\lambda,\mu_q(\lambda))
\ge
[\ell(\lambda)-\underline u]A_q
\ge c'(a_0).
$$

Since $I(\lambda,\mu)$ is increasing in $\mu$, the LIC solution satisfies $\tilde\mu(\lambda)\le\mu_q(\lambda)$. Because $m_0-\lambda<0$,

$$
\bar s(\lambda,\tilde\mu(\lambda))
=
\frac{m_0-\lambda}{\tilde\mu(\lambda)}
\le
\frac{m_0-\lambda}{\mu_q(\lambda)}
=q.
$$

If $\bar u=\infty$, this condition is automatic for any negative cutoff $q$ with $A_q>0$.

**Multiplier bound from kink placement.** For $\lambda>m_0$ and any cutoff $r<0$, if

$$
\bar s(\lambda,\tilde\mu(\lambda))\le r,
$$

then

$$
\tilde\mu(\lambda)
\le
\mu_r(\lambda)
=
\frac{m_0-\lambda}{r}
=O(\lambda).
$$

This bound uses only kink placement below $r$, not the capacity argument that produced it.

---

## 4. Curvature split lemma

For a canonical LIC-satisfying contract, write $\tilde\mu=\tilde\mu(\lambda)$.
Let $\ell_0$ denote the unconstrained, nonbinding utility link, so that $\ell(z)=\ell_0(z)$ for $z\ge m_0$.
If $\lambda+\tilde\mu q=m_0$, interpret $\ell_0'(\lambda+\tilde\mu q)$ as the right derivative at the kink.

**Curvature split lemma.** Fix $q\in\mathcal Q_{\mathrm{safe}}$ with $M_q<\infty$. Suppose the kink is below $q$:

$$
\bar s(\lambda,\tilde\mu(\lambda))\le q.
$$

Then, for every $a\in\mathcal A$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu),a)
\le
\tilde\mu\,\ell_0'(\lambda+\tilde\mu q)M_q
-c''(a).
\tag{Split}
$$

Proof idea. Define the secant slope term

$$
\Delta \ell(y|\lambda):=
\frac{\ell(\lambda+\tilde\mu s(y))-\ell(\lambda)}{\tilde\mu s(y)}
$$

for $s(y)\ne0$, with the continuous extension at $s(y)=0$. Under the differentiation identity $\int f_{aa}(y|a)\,d\nu(y)=0$, the curvature identity is

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu),a)
=
\tilde\mu
\int \Delta \ell(y|\lambda)s(y)f_{aa}(y|a)\,d\nu(y)
-c''(a).
\tag{Curv}
$$

On $B_q=\{s\le q\}$, tail safety gives $s\le0$ and $f_{aa}\ge0$, while monotonicity of $\ell$ gives $\Delta\ell\ge0$. Thus this region contributes weakly negatively and can be discarded for an upper bound.

On $\{s>q\}$, kink placement below $q$ implies the contract is nonbinding, so $\ell=\ell_0$. Concavity of $\ell_0$ in its scalar argument gives

$$
\Delta \ell(y|\lambda)
\le
\ell_0'(\lambda+\tilde\mu q).
$$

Bounding only the positive part of $s(y)f_{aa}(y|a)$ gives (Split). Thus the remaining object to control is

$$
\tilde\mu\,\ell_0'(\lambda+\tilde\mu q)M_q.
$$

---

## 5. Unified asymptotic concavity criterion

This theorem isolates the common mechanism:

$$
\boxed{\text{the kink is in a safe low-score tail}}
\qquad + \qquad
\boxed{\text{paid-region positive curvature is asymptotically dominated by }c''.}
$$

**Unified asymptotic criterion.** Assume $\ell_0$ is nondecreasing and concave on the nonbinding region and effort cost satisfies (CostCurv). Fix cutoffs $t<q$ with $t<0$ and $t,q\in\mathcal Q_{\mathrm{safe}}$. Suppose:

1. there is enough incentive capacity at the kink cutoff,

$$
[\bar u-\underline u]A_t>c'(a_0);
\tag{Cap-t}
$$

2. the deterministic sufficient bound obtained from kink placement is asymptotically below cost curvature,

$$
\limsup_{\lambda\to\infty}
\mu_t(\lambda)\ell_0'(\lambda+\mu_t(\lambda)q)M_q
<
\underline c'',
\qquad
\underline c'':=\inf_{a\in\mathcal A}c''(a).
\tag{DC}
$$

Then there exists $\lambda_0$ such that, for all $\lambda\ge\lambda_0$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

The kink placement lemma gives

$$
\bar s(\lambda,\tilde\mu(\lambda))\le t<q
$$

for all sufficiently large $\lambda$. The curvature split lemma, combined with the deterministic multiplier bound from $t$, then gives $U_{aa}\le0$ uniformly in $a$ for all sufficiently large $\lambda$.

Using generalized Proposition 1, this implies FOA validity for the corresponding high-reservation range.

To pass from the endogenous curvature term to (DC), use the following elementary monotonicity directly in the proof. For $q\le0$,

$$
\mu\longmapsto \mu\ell_0'(\lambda+\mu q)
$$

is nondecreasing on the relevant nonbinding range: both $\mu$ and $\ell_0'(\lambda+\mu q)$ weakly increase with $\mu$. Hence $\tilde\mu\le\mu_t$ implies

$$
\tilde\mu\ell_0'(\lambda+\tilde\mu q)
\le
\mu_t\ell_0'(\lambda+\mu_t q).
$$

The explicit pair $t<q$ is the safest theorem-level formulation. The condition

$$
[\bar u-\underline u]A_{\mathrm{safe}}>c'(a_0)
$$

does **not** by itself always produce such a pair: capacity can jump at a rightmost safe endpoint with no strictly larger safe cutoff. A primitive corollary should therefore use the interior safe capacity $A_{\mathrm{safe}}^\circ$ defined below, or impose an explicit endpoint/score-gap condition under which $A_{\mathrm{safe}}^\circ=A_{\mathrm{safe}}$.

The remaining results can be presented as verification corollaries:

* safe-tail capacity verifies kink placement;
* strong flattening verifies (DC) with a zero limit, covering CARA and CRRA $\gamma>1$;
* log utility verifies (DC) through safe score width, with the unbounded-left-score case as the infinite-width limit.

CRRA $\gamma\in(1/2,1)$ generally should not be forced into this theorem: its positive-curvature term need not vanish or be small by this route. That case uses the appendix-only asymptotic-affinity mechanism.

---

## 6. High-reservation theorem via strong flattening

The first high-reservation theorem uses the curvature split lemma and verifies the paid-region curvature bound from primitive utility curvature. This route covers bounded utilities such as CARA and CRRA with $\gamma>1$.

### Definitions and assumptions

Let

$$
\underline c'':=\inf_{a\in\mathcal A}c''(a).
$$

For the strong-flattening theorem, the capacity used in the proof must leave room for a strictly larger safe paid-region cutoff. Define the interior safe capacity

$$
A_{\mathrm{safe}}^\circ
:=
\sup\{A_t:t<q,\ t<0,\ t,q\in\mathcal Q_{\mathrm{safe}}\}.
$$

If the set is empty, take $A_{\mathrm{safe}}^\circ=0$.

The capacity assumption is

$$
[\bar u-\underline u]A_{\mathrm{safe}}^\circ>
c'(a_0).
\tag{SafeCap}
$$

The proof only needs an interior pair $t<q$ for which capacity holds at $t$ and

$$
M_q<\infty.
\tag{PairReg}
$$

For a convenient primitive corollary, one may impose the stronger global condition

$$
M_q<\infty
\quad\text{for every }q\in\mathcal Q_{\mathrm{safe}}.
\tag{SafeReg}
$$

The strong-flattening assumption is

$$
\frac{[u'(x)]^2}{-u''(x)}\to0
\quad\text{as }x\to\infty.
\tag{StrongFlat-u}
$$

Equivalently, for the nonbinding link $\ell_0=k'^{-1}$,

$$
z\ell_0'(z)\to0.
\tag{StrongFlat}
$$

### Theorem A: strong-flattening utilities

Assume:

1. $\ell_0$ is nondecreasing and concave on the nonbinding region;
2. effort cost satisfies (CostCurv);
3. there exist $t<q$ with $t<0$ and $t,q\in\mathcal Q_{\mathrm{safe}}$ such that
   $$
   [\bar u-\underline u]A_t>c'(a_0)
   \quad\text{and}\quad M_q<\infty;
   $$
4. utility satisfies strong flattening (StrongFlat-u), equivalently (StrongFlat).

The more primitive pair (SafeCap) and (SafeReg) implies item 3, but is stronger than necessary.

Then, for all sufficiently large $\lambda$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

Using generalized Proposition 1, this implies FOA validity:

* if $\bar u=\infty$, for all sufficiently high $\bar U$;
* if $\bar u<\infty$, for all feasible $\bar U$ sufficiently close to the upper reservation-utility bound from below.

This theorem is the main theorem for CARA and CRRA with $\gamma>1$.

### Proof idea

By item 3, fix $t,q\in\mathcal Q_{\mathrm{safe}}$ with $t<q$ and $t<0$ such that

$$
[\bar u-\underline u]A_t>c'(a_0).
\tag{Cap-t}
$$

Capacity at $t$ and the kink placement lemma imply that, for all large $\lambda$,

$$
\bar s(\lambda,\tilde\mu(\lambda))\le t.
$$

Equivalently, $\tilde\mu(\lambda)\le\mu_t(\lambda)$, where

$$
\mu_t(\lambda)
=
\frac{m_0-\lambda}{t}
=O(\lambda).
$$

Since $t<q\le0$,

$$
z_{tq}(\lambda)
:=
\lambda+\frac{m_0-\lambda}{t}q
\;=\;
\lambda+\mu_t(\lambda)q
\sim
\lambda\left(1-\frac{q}{t}\right).
$$

The coefficient is positive, so $z_{tq}(\lambda)$ grows linearly. The curvature split lemma and the multiplier bound give

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le
\mu_t(\lambda)\ell_0'(z_{tq}(\lambda))M_q
-c''(a).
$$

By item 3, $M_q<\infty$. Since $\mu_t(\lambda)=O(\lambda)$, $z_{tq}(\lambda)$ grows linearly, and $z\ell_0'(z)\to0$, the positive term converges to zero. Hence it is eventually smaller than $\underline c''$, which proves uniform concavity.

### Interpretation of the assumptions

The condition (SafeCap) says that the safe low-score tail has enough incentive capacity to satisfy LIC while placing the limited-liability kink strictly inside the safe region. For bounded utility, this is the substantive capacity condition. For unbounded utility it is automatic whenever $A_{\mathrm{safe}}^\circ>0$.

The regularity condition (SafeReg) is paid-tail integrability. It is automatic in Gaussian examples and should be straightforward for Poisson and exponential examples when the feasible action set is bounded away from zero.

The interior capacity $A_{\mathrm{safe}}^\circ$ equals $A_{\mathrm{safe}}$ in the usual continuous cases and in discrete cases with a positive score gap above the capacity-generating safe atom. It only differs in endpoint cases where all capacity comes from the rightmost safe cutoff and there is no larger safe paid-region cutoff.

### Utility verification: CARA and CRRA $\gamma>1$

The primitive condition is easy to verify because

$$
z\ell_0'(z)
=
\frac{k'(v)}{k''(v)}
=
\frac{[u'(x)]^2}{-u''(x)},
\qquad
v=u(x),\ z=k'(v).
$$

* **CARA:** $u'(x)=e^{-\alpha x}$ and $-u''(x)=\alpha e^{-\alpha x}$, so

$$
\frac{[u'(x)]^2}{-u''(x)}
=
\frac{e^{-\alpha x}}{\alpha}\to0.
$$

* **CRRA with $\gamma>1$:** $u'(x)=(w_0+x)^{-\gamma}$ and $-u''(x)=\gamma(w_0+x)^{-\gamma-1}$, so

$$
\frac{[u'(x)]^2}{-u''(x)}
=
\frac1\gamma(w_0+x)^{1-\gamma}\to0.
$$

Log utility is not a strongly flattening case. For log,

$$
\frac{[u'(x)]^2}{-u''(x)}=1,
$$

or equivalently $\ell_0'(z)=1/z$ and $z\ell_0'(z)=1$. Therefore log should not be folded into Theorem A. Use a separate borderline-log theorem or corollary, still based on the same curvature split lemma but using a different asymptotic argument.

### Definitions and assumptions for Theorem B

For log utility,

$$
\ell_0'(z)=\frac1z.
$$

Define the bottom of the score support

$$
\underline s:=\inf\{s(y):y\in\mathcal Y\}.
$$

The safe score width is

$$
D_{\mathrm{safe}}
:=
\bar q_{\mathrm{safe}}-\underline s
\in[0,\infty].
$$

The proof-facing log-width assumption is that cost satisfies (CostCurv), $\ell_0$ is nondecreasing and concave on the nonbinding region, and there exist cutoffs

$$
\underline s<t<q,
\qquad
t,q\in\mathcal Q_{\mathrm{safe}},
\qquad
t<0,
$$

such that

$$
A_t>0,
\qquad
M_q<\infty,
\qquad
\frac{M_q}{q-t}<\underline c''.
\tag{LogWidth}
$$

### Theorem B: borderline log utility via safe score width

Assume log utility and (LogWidth). Then, for all sufficiently large $\lambda$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

Using generalized Proposition 1, this implies FOA validity for all sufficiently high reservation utility.

### Proof idea

Since log utility is unbounded above, $A_t>0$ implies the capacity condition

$$
[\bar u-\underline u]A_t>c'(a_0).
$$

The kink placement lemma gives $\bar s(\lambda,\tilde\mu(\lambda))\le t$ for all large $\lambda$, hence $\tilde\mu(\lambda)\le\mu_t(\lambda)$. The curvature split lemma gives

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le
\mu_t(\lambda)\ell_0'(\lambda+\mu_t(\lambda)q)M_q
-c''(a).
$$

For log utility,

$$
\mu_t(\lambda)\ell_0'(\lambda+\mu_t(\lambda)q)
=
\frac{\mu_t(\lambda)}{\lambda+\mu_t(\lambda)q}
\to
\frac1{q-t}.
$$

Therefore (LogWidth) implies that the positive curvature term is eventually below $\underline c''$, proving uniform concavity.

### Interpretation of the assumptions

The condition (LogWidth) is the strict-cutoff form of the log theorem. It uses the same curvature split lemma as Theorem A, but log is the borderline case $z\ell_0'(z)=1$, so the positive term does not vanish. The available safe score width must be large enough relative to paid-region curvature.

A boundary-based corollary can define

$$
M_{\mathrm{safe}}
:=
\liminf_{\substack{q\uparrow\bar q_{\mathrm{safe}}\\ q\in\mathcal Q_{\mathrm{safe}}}}M_q
$$

and require

$$
\frac{M_{\mathrm{safe}}}{D_{\mathrm{safe}}}<\underline c''.
\tag{LogWidth'}
$$

This shorthand is valid only with explicit nondegeneracy and endpoint conditions: $D_{\mathrm{safe}}>0$, some interior $t$ has $A_t>0$, the safe edge is approachable by safe cutoffs, and the relevant limiting curvature bound is finite. Use the convention that the ratio is zero when $D_{\mathrm{safe}}=\infty$ and $M_{\mathrm{safe}}<\infty$; do not assign a value to ambiguous cases such as $0/0$ or $\infty/\infty$. The formal theorem should remain the strict-cutoff condition (LogWidth), which handles atoms and endpoints without additional conventions.

This covers:

* **Unbounded-left-score models.** If $\underline s=-\infty$, fix any safe paid-region cutoff $q\in\mathcal Q_{\mathrm{safe}}$ with $M_q<\infty$ and take $t\to-\infty$. Then $q-t\to\infty$, so (LogWidth) is automatic.
* **Bounded-left-score models.** If $\underline s> -\infty$, the condition says that the safe score width must be large enough relative to the positive paid-region curvature. Since log utility is unbounded above, capacity is automatic for any fixed $t$ with $A_t>0$.

For Poisson and exponential examples, impose the natural nondegeneracy condition that the feasible action set is bounded away from zero.

### Verifying the safe-tail conditions in key examples

These examples illustrate how to verify the proof-facing pair conditions and, when convenient, the stronger primitive conditions (SafeCap) and (SafeReg). They also make clear when a model is degenerate.

#### Gaussian location

For $Y\sim N(a,\sigma^2)$,

$$
s(y)=\frac{y-a_0}{\sigma^2}.
$$

The low-tail condition is

$$
f_{aa}(y|a)\ge0
\quad\Longleftrightarrow\quad
|y-a|\ge\sigma.
$$

Thus, if $\mathcal A=[\underline a,\bar a]$,

$$
\bar q_{\mathrm{safe}}
=
\min\left\{0,\frac{\underline a-a_0-\sigma}{\sigma^2}\right\}.
$$

Under $a_0$, $s\sim N(0,1/\sigma^2)$, so

$$
A_q=\frac1\sigma\varphi(\sigma q).
$$

$M_q<\infty$ is automatic by Gaussian tails. Gaussian is the clean benchmark case.

#### Poisson

For $Y\sim \mathrm{Pois}(a)$,

$$
s(y)=\frac{y-a_0}{a_0}.
$$

The second derivative is

$$
p_{aa}(y|a)
=
p(y|a)\frac{(y-a)^2-y}{a^2}.
$$

Thus a low count $y$ is uniformly tail-safe when

$$
y+\sqrt y\le\underline a,
\qquad
\underline a:=\min\mathcal A.
$$

This is why the Poisson theorem should assume the natural nondegeneracy condition $\underline a>0$. Let $K$ be a safe integer cutoff. Then

$$
A_K
=
-\sum_{y\le K}\frac{y-a_0}{a_0}p(y|a_0)
=
P_{a_0}(Y=K),
$$

using $E[Y1\{Y\le K\}]=a_0P(Y\le K-1)$. Hence capacity is very transparent: the maximum utility spread times a safe Poisson mass must exceed $c'(a_0)$. $M_q<\infty$ is regularity on compact action sets bounded away from zero.

#### Exponential with mean action

For $Y\sim \mathrm{Exp}(\text{mean }a)$,

$$
s(y)=\frac{y-a_0}{a_0^2}.
$$

A calculation gives

$$
f_{aa}(y|a)
=
f(y|a)\frac{y^2-4ay+2a^2}{a^4}.
$$

The low-tail safe region is

$$
y\le(2-\sqrt2)a.
$$

Uniformly over $a\in\mathcal A$, this gives the safe output cutoff

$$
y\le(2-\sqrt2)\underline a.
$$

Again the theorem should assume $\underline a>0$; otherwise the uniform safe low-output region collapses. In score units,

$$
\bar q_{\mathrm{safe}}
=
\frac{(2-\sqrt2)\underline a-a_0}{a_0^2}.
$$

For an output cutoff $r=a_0+a_0^2q$ and $b=r/a_0$, under $a_0$,

$$
A_q
=
- E_{a_0}\left[\frac{Y-a_0}{a_0^2}1\{Y\le r\}\right]
=
\frac{b e^{-b}}{a_0}.
$$

$M_q<\infty$ is regularity on compact action sets bounded away from zero.

---

## 7. Appendix-only extension: asymptotic affinity

This material should be separated from the main proof. The main text should emphasize the clean strong-flattening route, which covers CARA, log as a borderline case, and CRRA with $\gamma\ge1$. We will not state a formal main-text remark for the remaining CRRA cases. Instead, the prose can simply say that an appendix treats lower CRRA coefficients under additional technical conditions.

The appendix extension is useful but has a different mechanism. Flattening can fail for some unbounded utilities, notably CRRA with $\gamma\in(1/2,1)$. In that case the contract may not become flat, but it can become asymptotically affine in the score.

For CRRA with $\gamma\in(1/2,1)$,

$$
\ell'(z)\to0
\quad\text{but}\quad
z\ell'(z)\to\infty.
$$

LIC suggests

$$
\tilde\mu(\lambda)\ell'(\lambda)
\int s(y)^2 f(y|a_0)\,d\nu(y)
\approx c'(a_0),
$$

so

$$
\tilde\mu(\lambda)\ell'(\lambda)
\to
\theta:=
\frac{c'(a_0)}{\int s(y)^2 f(y|a_0)\,d\nu(y)}.
$$

The contract behaves like

$$
\ell(\lambda+\tilde\mu s(y))
=
\ell(\lambda)+\tilde\mu \ell'(\lambda)s(y)+o(1)
$$

in the weighted sense relevant for $f_{aa}$.

The constant term contributes nothing to $U_{aa}$ because

$$
\int f_{aa}(y|a)\,d\nu(y)=0.
$$

The affine term contributes

$$
\tilde\mu \ell'(\lambda)
\int s(y)f_{aa}(y|a)\,d\nu(y).
$$

Thus the appendix theorem needs affine-score curvature:

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0
\quad\text{for all }a\in\mathcal A.
\tag{Affine}
$$

Equivalently, the function

$$
a\mapsto E_a[S(Y|a_0)]
$$

is concave.

The appendix theorem can then assume the residual condition

$$
\sup_{a\in\mathcal A}
\left|
\int
\left[
 \ell(\lambda+\tilde\mu s(y))
 -\ell(\lambda)
 -\tilde\mu \ell'(\lambda)s(y)
\right]
 f_{aa}(y|a)\,d\nu(y)
\right|
\to0.
\tag{R}
$$

Under (Affine) and (R),

$$
\sup_{a\in\mathcal A}U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le
-\underline c''+o(1),
$$

uniformly in $a$, and hence concavity holds for large $\lambda$.

### Primitive sufficient conditions for the residual

A sufficient route for (R) is:

$$
\frac{\tilde\mu(\lambda)}{\lambda}\to0,
\qquad
\tilde\mu(\lambda)\ell'(\lambda)=O(1),
$$

a controlled curvature condition such as

$$
\sup_{z\ge \lambda/2}\frac{z|\ell''(z)|}{\ell'(z)}<\infty,
$$

and uniform integrability of $s(y)^2 |f_{aa}(y|a)|$ over $a\in\mathcal A$.

Taylor's theorem gives, on the central region,

$$
\ell(\lambda+\tilde\mu s)
-\ell(\lambda)
-\tilde\mu \ell'(\lambda)s
=
O\left(\tilde\mu^2 |\ell''(\lambda)|s^2\right).
$$

For CRRA $\gamma\in(1/2,1)$,

$$
\tilde\mu^2 |\ell''(\lambda)|
\asymp
[\tilde\mu \ell'(\lambda)]\frac{\tilde\mu}{\lambda}
\to0.
$$

Tails are handled by uniform integrability.

---

## 8. Discrete output and Poisson

The score-cutoff formulation handles discrete output with little extra notation. For Poisson, take

$$
\mathcal Y=\mathbb N,
$$

and let $\nu$ be counting measure. Integrals become sums.

The tail-safe condition is not an interval condition on output. It is a condition on a lower contour set of the score:

$$
f_{aa}(y|a)\ge0
\quad\text{for all }y\text{ such that }s(y)\le q.
$$

The quantities $A_q$ and $M_q$ are sums over score regions:

$$
A_q=-\sum_{s(y)\le q}s(y)f(y|a_0),
$$

and

$$
M_q=
\sup_a\sum_{s(y)>q}[s(y)f_{aa}(y|a)]_+.
$$

Atoms at the cutoff are harmless if the cutoff is chosen away from score atoms. If a cutoff atom is unavoidable, use weak inequalities consistently and keep capacity strict. A safe capacity-generating atom supports the required pair $t<q$ when there is a positive score gap above it before the first unsafe score. Without such a gap, capacity at the rightmost safe atom need not provide a strictly larger safe paid-region cutoff; this is why the formal theorem uses an explicit pair and primitive corollaries use $A_{\mathrm{safe}}^\circ$.

The bounded-below score issue is conceptually separate from discreteness. If the score has a lower bound, the kink may eventually fall below the lowest score. That case can still be handled by the curvature split lemma plus deterministic control, or by an appropriate limiting-curvature argument. Non-interval support by itself adds little complexity once everything is written in score-set notation.

---

## 9. Gaussian example

For Gaussian original output $X\sim N(a,\sigma^2)$,

$$
s(X)=S(X|a_0)=\frac{X-a_0}{\sigma^2}.
$$

Under action $a$,

$$
s(X)\sim N\left(\frac{a-a_0}{\sigma^2},\frac1{\sigma^2}\right).
$$

The left-tail condition is explicit:

$$
f_{aa}(s|a)\ge0
\quad\Longleftrightarrow\quad
\left|s-\frac{a-a_0}{\sigma^2}\right|\ge\frac1\sigma.
$$

Thus any

$$
q<\min\left\{0,
\inf_{a\in\mathcal A}
\left(\frac{a-a_0}{\sigma^2}-\frac1\sigma\right)
\right\}
$$

is tail-safe.

Also,

$$
A_q=-\int_{s\le q}s f(s|a_0)\,ds
=
\frac1\sigma\varphi(\sigma q).
$$

The positive-part constant is finite automatically:

$$
M_q=
\sup_{a\in\mathcal A}
\int_q^\infty
\left[
s\left(\left(s-\frac{a-a_0}{\sigma^2}\right)^2-\frac1{\sigma^2}\right)
\right]_+
f(s|a)\,ds<\infty.
$$

Finally,

$$
\int s f_{aa}(s|a)\,ds
=
\frac{\partial^2}{\partial a^2}E_a[s]
=0,
$$

because

$$
E_a[s]=\frac{a-a_0}{\sigma^2}
$$

is affine. Therefore Gaussian location families satisfy the affine-score curvature condition with equality.

So in Gaussian-CRRA:

* $\gamma>1$: covered in the main text by strong flattening, with capacity at a stricter cutoff $t<q$;
* $\gamma=1$ (log): covered in the main text either by the unbounded-left-score route or by the finite-bound route;
* $\gamma\in(1/2,1)$: appendix-only extension via asymptotic affinity because the affine-score term contributes zero curvature.

---

## 10. CRRA corollary

For CRRA,

$$
\ell_0'(z)=\frac1\gamma z^{1/\gamma-2}.
$$

The main text should claim CRRA only for $\gamma\ge1$. The possible extension of the asymptotic-affinity argument to $\gamma\le1/2$ is left open in this revision.

### Main-text branch 1: $\gamma>1$

If $\gamma>1$, utility is bounded and capacity at the stricter cutoff $t<q\le0$ becomes

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_t>c'(a_0),
$$

up to the exact normalization of CRRA utility in the paper. Then the strong-flattening theorem applies because $z\ell_0'(z)\to0$.

### Main-text branch 2: $\gamma=1$ / log

If $\gamma=1$, this is log utility and utility is unbounded. Capacity is automatic at any cutoff with $A_t>0$, but log is only borderline flattening because $z\ell_0'(z)=1$. Therefore log requires either the unbounded-left-score route or the finite-bound route based on the curvature split lemma.

### Appendix-only branch: $\gamma\in(1/2,1)$

Do not include this as a main-text corollary or formal remark. If needed, mention in prose that an appendix treats lower CRRA coefficients under extra affine-score and residual/integrability conditions.

Utility is unbounded, so capacity is automatic. Flattening may fail, but asymptotic affinity applies if

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0
\quad\text{for all }a\in\mathcal A,
$$

plus the residual and integrability conditions.

For Gaussian location families, this integral is zero, so the appendix can cover CRRA coefficients

$$
\gamma>1/2.
$$

Do not make a broad claim for $\gamma\le1/2$. For those values, the nonlinear residual need not vanish.

---

## 11. CARA corollary

For CARA,

$$
u(W)=-\frac{e^{-\alpha W}}{\alpha},
$$

so

$$
\bar u-\underline u=\frac{e^{-\alpha w_0}}{\alpha},
$$

and

$$
\ell_0'(z)=\frac1{\alpha z^2}.
$$

CARA is covered by the strong-flattening theorem under the capacity condition at a stricter cutoff $t<q\le0$:

$$
\frac{e^{-\alpha w_0}}{\alpha}A_t>c'(a_0).
$$

For Gaussian output, this becomes

$$
\frac{e^{-\alpha w_0}}{\alpha\sigma}\varphi(\sigma t)>c'(a_0).
$$

This is conservative but clean. It explains why bounded CARA can still work: the maximum utility spread must be large enough to generate incentives, and once it is, the paid-region slope vanishes quickly enough. Since $e^{-\alpha w_0}/\alpha$ increases as $w_0$ falls, the condition holds for sufficiently low initial wealth whenever the remaining primitives are fixed and the admissible wealth domain permits those values. This may include negative $w_0$ for CARA. The corollary is conditional and need not cover every numerical calibration used for illustration.

---

## 11.5 Interface with generalized Proposition 1

The concavity results should import only the following conclusions and regularity facts from generalized Proposition 1:

1. for each relevant $\lambda$, there is a unique LIC multiplier $\tilde\mu(\lambda)>0$;
2. the canonical LIC path is well defined and consists of feasible contracts;
3. score integrability and differentiation under the integral sign justify
   $$
   E_{a_0}[s(Y)]=0,
   \qquad
   \int f_{aa}(y|a)\,d\nu(y)=0;
   $$
4. the relaxed optimum is indexed by $\lambda^*(\bar U)$; and
5. as reservation utility approaches the upper relaxed frontier (a supremum, not necessarily an attained maximum),
   $$
   \lambda^*(\bar U)\to\infty
   \quad\text{as}\quad
   \bar U\uparrow\bar U_R.
   $$

Local implementability, score nondegeneracy, and the integrability needed for these statements belong with Proposition 1. Tail safety, capacity at a safe cutoff, nonbinding link concavity, paid-region integrability, and (CostCurv) belong only to the concavity results.

---

## 12. Proposed organization of the revised proof

The revised argument should be presented in the following order.

### A. Setup and concavity assumptions

Use the common dominating measure, fix $s(y)=S(y|a_0)$, and state only the assumptions needed from this point onward: nonbinding concavity of $\ell_0$, (CostCurv), differentiation identities, and the score-cutoff conventions. Avoid relying on an inverse score map.

### B. Tail-safe cutoffs and kink placement

Define $B_t$, $A_t$, $\mu_t(\lambda)$, and $I(\lambda,\mu)$. Prove

$$
[\bar u-\underline u]A_t>c'(a_0)
\quad\Longrightarrow\quad
\bar s(\lambda,\tilde\mu(\lambda))\le t
\text{ eventually},
$$

and record the resulting bound $\tilde\mu\le\mu_t=O(\lambda)$.

### C. Finite-$\lambda$ curvature criterion

Define $M_q$ and prove the curvature split with the endogenous term

$$
\tilde\mu\,\ell_0'(\lambda+\tilde\mu q)M_q.
$$

This is the common finite-$\lambda$ result. The stricter cutoff $t<q$ enters only when the endogenous multiplier is replaced by the deterministic bound from kink placement.

### D. Unified asymptotic criterion

State the formal theorem with an explicit safe pair $t<q$, capacity at $t$, $M_q<\infty$, and deterministic curvature domination. Explain directly why

$$
\tilde\mu\ell_0'(\lambda+\tilde\mu q)
\le
\mu_t\ell_0'(\lambda+\mu_tq).
$$

### E. Strong-flattening corollary

Use $z\ell_0'(z)\to0$ and the linear growth of $\lambda+\mu_tq$ to cover CARA and CRRA $\gamma>1$. Introduce $A_{\mathrm{safe}}^\circ$ only as a primitive sufficient condition for existence of the required pair.

### F. Borderline log corollary

Treat log separately through the strict-cutoff condition

$$
\frac{M_q}{q-t}<\underline c''.
$$

Present infinite safe width and finite safe width as corollaries, with endpoint and nondegeneracy qualifications made explicit.

### G. Distribution and utility verifications

Collect Gaussian, Poisson, and exponential calculations in one place. Impose action sets bounded away from zero for Poisson and exponential output. State CARA and CRRA conclusions conditionally on their capacity inequalities.

### H. Appendix asymptotic-affinity extension

State conditions under which

$$
V(y)=\ell(\lambda)+\tilde\mu \ell'(\lambda)s(y)+o(1)
$$

in the $f_{aa}$-weighted sense, together with affine-score curvature

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)\le0.
$$

Use this initially for CRRA $\gamma\in(1/2,1)$, especially Gaussian location families. Mention it only briefly in the main text and leave $\gamma\le1/2$ for later investigation.

---

## 13. Main conceptual improvement

The original proof had one asymptotic mechanism:

$$
\text{the no-pay region moves to the far left tail, and the paid-region curvature becomes small.}
$$

The revised proof separates the logic into sharper components:

$$
\boxed{\text{low-score no-pay states contribute negatively}}
$$

once the kink is below a tail-safe cutoff, and

$$
\boxed{\text{the remaining positive part is controlled by flattening.}}
$$

This gives the clean main-text story. It handles bounded utility through capacity plus strong flattening, treats log as the borderline case, and includes discrete output through score cutoff sets. The asymptotic-affinity argument for CRRA $\gamma\in(1/2,1)$ should be kept as an appendix extension because it uses a different mechanism and requires extra affine-score and residual/integrability conditions.
