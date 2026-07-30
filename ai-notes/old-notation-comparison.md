# Old-to-New Notation Comparison

Terse reference for migrating the existing `tex/` manuscript to the notation fixed in `ai-notes/concavity-generalization-plan.tex`.

---

## Contracts, wealth, and payments

* Old payment variable or argument: $x$, often used in $u(x)$.
  * New payment/wage: $w\ge0$, with contract $w(y)$.

* Old utility specification: utility from payment, $u(x)$, often written in examples as $u(x+w_0)$.
  * New utility is over final wealth, $u(W)$.
  * Utility from payment $w$ is $u(w_0+w)$.

* Old implicit final wealth: $x+w_0$ or $w(y)+w_0$.
  * New final wealth:
    $$
    W(y):=w_0+w(y).
    $$

* Old utility contract: $v(y)$.
  * New: keep $v(y)$, with
    $$
    v(y)=u(W(y)).
    $$

* Old payment induced by a utility contract:
  $$
  w(y)=k(v(y)).
  $$
  * New, with $k:=u^{-1}$:
    $$
    W(y)=k(v(y)),
    \qquad
    w(y)=k(v(y))-w_0.
    $$

The main text treats $U$ and $C$ as functionals of payment contracts:
$$
U(w,a):=\int_{\mathcal Y}u(w_0+w(y))f(y|a)\,d\nu(y)-c(a),
\qquad
C(w,a):=\int_{\mathcal Y}w(y)f(y|a)\,d\nu(y).
$$
The proof appendix uses utility-contract coordinates and visibly distinct transformed functionals:
$$
\widehat U(v,a):=\int_{\mathcal Y}v(y)f(y|a)\,d\nu(y)-c(a),
\qquad
\widehat C(v,a):=\int_{\mathcal Y}[k(v(y))-w_0]f(y|a)\,d\nu(y).
$$
Thus $U(w,a)=\widehat U(v,a)$ and $C(w,a)=\widehat C(v,a)$ under $v=u(w_0+w)$. Do not overload $U$ and $C$ across the two contract representations.

The admissible initial-wealth domain is utility-specific: log and CRRA require $w_0>0$, while CARA permits $w_0\in\mathbb R$.

---

## Expected compensation cost

* Old expected wage/cost:
  $$
  W(v,a):=\int k(v(y))f(y|a)\,dy.
  $$
  * New transformed expected compensation cost in utility coordinates:
    $$
    \widehat C(v,a):=\int_{\mathcal Y}[k(v(y))-w_0]f(y|a)\,d\nu(y).
    $$

* Old relaxed-path expected wage: $\widetilde W(\lambda)$.
  * New: $\widetilde C(\lambda)$.

* Old optimal expected wage/value: sometimes $\omega(\bar U)$.
  * New: $C^*(\bar U)$, or simply “optimal expected compensation cost.”

This leaves uppercase $W(y)$ exclusively for final wealth.

---

## Output distributions and integration

* Old output support: usually $\mathbb R$ with Lebesgue density.
  * New: common output space $\mathcal Y\subseteq\mathbb R$, with density or probability mass function $f(\cdot|a)>0$ relative to a common dominating measure $\nu$.

* Old integrals:
  $$
  \int h(y)f(y|a)\,dy.
  $$
  * New:
    $$
    \int_{\mathcal Y}h(y)f(y|a)\,d\nu(y).
    $$
    For discrete output, $\nu$ is counting measure and the integral denotes a sum.

The generalized results do not maintain interval support, MLRP, or invertibility of the score.

---

## Utility bounds and limited liability

* Old limited-liability utility floor: $u(0)$.
  * New:
    $$
    \underline u:=u(w_0).
    $$

* Old upper utility notation: $u(\infty)$ or an implicit infinity.
  * New:
    $$
    \bar u:=\lim_{W\to\infty}u(W)\in\mathbb R\cup\{\infty\}.
    $$

* Old utility spread: $\bar u-u(0)$.
  * New:
    $$
    \Delta u:=\bar u-\underline u=\bar u-u(w_0).
    $$

* Old marginal floor:
  $$
  m=\frac1{u'(0)}.
  $$
  * New:
    $$
    m_0:=\frac1{u'(w_0)}.
    $$

---

## Link functions

Let $k:=u^{-1}$.

* Old utility link:
  $$
  g(z):=(k')^{-1}(z\vee m).
  $$
  * New unconstrained wealth and utility links:
    $$
    L_0(z):=(u')^{-1}(1/z),
    \qquad
    \ell_0(z):=u(L_0(z))=(k')^{-1}(z).
    $$
  * New limited-liability links:
    $$
    L(z):=L_0(z\vee m_0),
    \qquad
    \ell(z):=\ell_0(z\vee m_0)=u(L(z)).
    $$

Thus $L(z)=w_0$ and $\ell(z)=\underline u$ for $z\le m_0$. Technical paid-region curvature bounds use the derivative of the **nonbinding** link, $\ell_0'$, rather than an undifferentiated reference to $g'$ or $\ell'$.

---

## Canonical contracts and parameter notation

* Old canonical utility contract:
  $$
  V(y|\lambda,\mu)=g(\lambda+\mu S(y|a_0)).
  $$

* New canonical contracts:
  $$
  \begin{aligned}
  W_{\lambda,\mu}(y)&:=L(\lambda+\mu s(y)),\\
  w_{\lambda,\mu}(y)&:=L(\lambda+\mu s(y))-w_0,\\
  v_{\lambda,\mu}(y)&:=\ell(\lambda+\mu s(y)).
  \end{aligned}
  $$

Use subscripts for contract parameters so that the vertical bar remains reserved for conditioning in $f(y|a)$.

Along the local-incentive-compatible path:
$$
 v_\lambda:=v_{\lambda,\widetilde\mu(\lambda)},
 \qquad
 \widetilde U(\lambda):=\widehat U(v_\lambda,a_0),
 \qquad
 \widetilde C(\lambda):=\widehat C(v_\lambda,a_0).
$$

---

## Score notation

* Old likelihood ratio:
  $$
  \frac{f_a(y|a_0)}{f(y|a_0)}.
  $$
  * New score:
    $$
    S(y|a):=\partial_a\log f(y|a)=\frac{f_a(y|a)}{f(y|a)}.
    $$

After fixing the intended action, use
$$
 s(y):=S(y|a_0),
 \qquad
 f_0(y):=f(y|a_0).
$$

---

## Reservation-utility frontiers

* Old high-reservation statement: $\bar U\to\infty$.
  * New phrase: **high feasible reservation utility**, meaning
    $$
    \bar U\uparrow U_R.
    $$

* New relaxed frontiers:
  $$
  U_L:=\widetilde U(0),
  \qquad
  U_R:=\lim_{\lambda\to\infty}\widetilde U(\lambda).
  $$

If utility is unbounded, $U_R=\infty$. If utility is bounded, $U_R<\infty$ and high feasible reservation utility means approaching that frontier from below. Use $U_L,U_R$, not $\bar U_L,\bar U_R$.

---

## Limited-liability cutoff

* Old kink: often an output threshold $\bar y$, or previously proposed score notation $\bar s(\lambda,\mu)$.
  * New limited-liability score cutoff:
    $$
    q^{\mathrm{LL}}(\lambda,\mu):=\frac{m_0-\lambda}{\mu}.
    $$

The no-payment region is
$$
 \{y:s(y)\le q^{\mathrm{LL}}(\lambda,\mu)\}.
$$

Along the LIC path use
$$
 \widetilde q^{\mathrm{LL}}(\lambda)
 :=q^{\mathrm{LL}}(\lambda,\widetilde\mu(\lambda)).
$$

For a trial cutoff $q<0$:
$$
 \mu_q(\lambda):=\frac{m_0-\lambda}{q}.
$$

---

## Safe-tail quantities

Replace old low-output sets and the previously proposed notation $B_q,A_q,M_q$ with named operators.

A cutoff $q\le0$ is tail-safe when
$$
 f_{aa}(y|a)\ge0
 \quad\text{for every }a\in\mathcal A
 \text{ and every }y\text{ with }s(y)\le q.
$$
Define
$$
 \mathcal Q_{\mathrm{safe}}:=\{q\le0:q\text{ is tail-safe}\}.
$$

* Incentive capacity:
  $$
  \Capacity(q):=-\int_{\{s(y)\le q\}}s(y)f_0(y)\,d\nu(y).
  $$

* Paid-region positive curvature:
  $$
  \Curv(q):=
  \sup_{a\in\mathcal A}
  \int_{\{s(y)>q\}}[s(y)f_{aa}(y|a)]_+\,d\nu(y).
  $$

* Aggregate safe-tail objects:
  $$
  \begin{aligned}
  \bar q_{\mathrm{safe}}&:=\sup\mathcal Q_{\mathrm{safe}},\\
  \Capacity_{\mathrm{safe}}
    &:=\sup_{q<\bar q_{\mathrm{safe}}}\Capacity(q),\\
  \Curv_{\mathrm{safe}}
    &:=\inf_{q\in\mathcal Q_{\mathrm{safe}}}\Curv(q),\\
  \underline s&:=\operatorname*{ess\,inf}_{Y\sim f_0}s(Y),\\
  \Width_{\mathrm{safe}}&:=\bar q_{\mathrm{safe}}-\underline s.
  \end{aligned}
  $$

The strict restriction $q<\bar q_{\mathrm{safe}}$ in $\Capacity_{\mathrm{safe}}$ deliberately excludes capacity concentrated at a rightmost safe score atom when no larger safe cutoff exists. The weak set $\{s\le q\}$ assigns a cutoff atom to the safe side; $\{s>q\}$ is its strict paid-region complement.

Capacity conditions now read
$$
 \Delta u\,\Capacity(q)>c'(a_0)
$$
or, in aggregate form,
$$
 \Delta u\,\Capacity_{\mathrm{safe}}>c'(a_0).
$$
When $\Delta u=\infty$, this notation means that the relevant capacity is strictly positive.

---

## Incentive and curvature formulas

* Old incentive functional using $g$:
  $$
  I(\lambda,\mu)=\int g(\lambda+\mu s(y))s(y)f(y|a_0)\,dy.
  $$
  * New:
    $$
    I(\lambda,\mu)
    :=\int_{\mathcal Y}\ell(\lambda+\mu s(y))s(y)f_0(y)\,d\nu(y).
    $$

* Old secant slope: $\Delta g(y|\lambda)$; previously proposed replacement: $\Delta\ell(y|\lambda)$.
  * Final notation:
    $$
    D_\lambda\ell(y):=
    \frac{\ell(\lambda+\widetilde\mu(\lambda)s(y))-\ell(\lambda)}
    {\widetilde\mu(\lambda)s(y)},
    $$
    with continuous extension at $s(y)=0$.

The curvature identity is
$$
 \widehat U_{aa}(v_\lambda,a)
 =\widetilde\mu(\lambda)
 \int D_\lambda\ell(y)s(y)f_{aa}(y|a)\,d\nu(y)-c''(a).
$$

The safe-tail curvature bound uses
$$
 \widetilde\mu(\lambda)
 \ell_0'(\lambda+\widetilde\mu(\lambda)q)\Curv(q)-c''(a).
$$

---

## Utility examples

### Log

* Old:
  $$
  u(x)=\log(w_0+x).
  $$
  * New:
    $$
    u(W)=\log W,
    \qquad w_0>0.
    $$

Links and canonical payment:
$$
 m_0=w_0,
 \qquad
 L(z)=z\vee w_0,
 \qquad
 \ell(z)=\log(z\vee w_0),
$$
$$
 w_{\lambda,\mu}(y)
 =[\lambda+\mu s(y)-w_0]^+.
$$

### CARA

* Old:
  $$
  u(x)=-\frac{e^{-\alpha(x+w_0)}}{\alpha}.
  $$
  * New:
    $$
    u(W)=-\frac{e^{-\alpha W}}{\alpha},
    \qquad w_0\in\mathbb R.
    $$

Links:
$$
 m_0=e^{\alpha w_0},
 \qquad
 L(z)=\frac1\alpha\log(z\vee e^{\alpha w_0}),
 \qquad
 \ell(z)=-\frac1{\alpha(z\vee e^{\alpha w_0})}.
$$

Utility spread:
$$
 \Delta u=\frac{e^{-\alpha w_0}}{\alpha}.
$$

### CRRA

* Old:
  $$
  u(x)=\frac{(w_0+x)^{1-\gamma}}{1-\gamma}.
  $$
  * New:
    $$
    u(W)=\frac{W^{1-\gamma}}{1-\gamma},
    \qquad w_0>0.
    $$

Links:
$$
 m_0=w_0^\gamma,
 \qquad
 L(z)=(z\vee w_0^\gamma)^{1/\gamma},
 \qquad
 \ell(z)=\frac{(z\vee w_0^\gamma)^{(1-\gamma)/\gamma}}{1-\gamma}.
$$

For $\gamma>1$:
$$
 \Delta u=\frac{w_0^{1-\gamma}}{\gamma-1}.
$$
For $\gamma<1$, utility is unbounded above. The main theorem covers $\gamma>1$ through strong flattening and $\gamma=1$ through the log branch; the appendix-only asymptotic-affinity extension covers $\gamma\in(1/2,1)$ under its additional assumptions.
