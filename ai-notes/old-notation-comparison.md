# Old-to-New Notation Comparison

Terse reference for updating the existing `tex/` files to the revised notation.

---

## Contracts, wealth, and payments

* Old payment variable: $x$ or payment argument in $u(x)$.
  * New: payment/wage is $w\ge0$.

* Old wage function: $w(y)$.
  * New: keep $w(y)$ for payment/wage.

* Old utility specification: utility from payment, $u(x)$, often written in examples as $u(x+w_0)$.
  * New: utility is over final wealth, $u(W)$.
  * Payment utility is $u(w_0+w)$.

* Old implicit final wealth: $x+w_0$ or $w(y)+w_0$.
  * New: final wealth is explicit:
    $$
    W(y):=w_0+w(y).
    $$

* Old utility contract: $v(y)$.
  * New: keep $v(y)$, with
    $$
    v(y)=u(W(y)).
    $$

* Old wage/payment induced by utility contract: $w(y)=k(v(y))$.
  * New:
    $$
    W(y)=k(v(y)),
    \qquad
    w(y)=k(v(y))-w_0.
    $$

---

## Expected cost

* Old expected wage/cost: $W(v,a)$.
  * New: expected compensation cost:
    $$
    C(v,a):=\int [k(v(y))-w_0]f(y|a)\,d\nu(y).
    $$

* Old relaxed path expected wage: $\tilde W(\lambda)$.
  * New:
    $$
    \tilde C(\lambda).
    $$

* Old optimal expected wage/value: sometimes $\omega(\bar U)$.
  * New:
    $$
    C^*(\bar U)
    $$
    or simply “optimal expected compensation cost.”

---

## Output distributions and integration

* Old output support: usually $\mathbb R$ with Lebesgue density.
  * New: output space $\mathcal Y\subseteq\mathbb R$ with common dominating measure $\nu$.

* Old integrals:
  $$
  \int h(y)f(y|a)\,dy.
  $$
  * New:
  $$
  \int_{\mathcal Y}h(y)f(y|a)\,d\nu(y).
  $$
  In discrete cases this denotes a sum.

* Old support assumption: support of $f(\cdot|a)$ is $\mathbb R$.
  * New: common support $\mathcal Y$, with $f(y|a)>0$ on $\mathcal Y$.

---

## Utility bounds and limited liability floor

* Old limited-liability utility floor: $u(0)$.
  * New:
    $$
    \underline u:=u(w_0).
    $$

* Old upper utility bound/unbounded notation: $u(\infty)$ or implicit infinity.
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

* Old inverse marginal utility/Jewitt map: $(k')^{-1}$, sometimes denoted by $h$ or related to $\omega$.
  * New: use wealth and utility links.

* Old utility link:
  $$
  g(z):=(k')^{-1}(z\vee m).
  $$
  * New wealth link:
  $$
  L(z):=(u')^{-1}\left(\frac1{z\vee m_0}\right).
  $$

* New utility link:
  $$
  \ell(z):=u(L(z)).
  $$

* Old canonical utility contract:
  $$
  V(y|\lambda,\mu)=g(\lambda+\mu S(y|a_0)).
  $$
  * New:
  $$
  V(y|\lambda,\mu)=\ell(\lambda+\mu S(y|a_0)).
  $$

* New canonical final wealth:
  $$
  W(y|\lambda,\mu)=L(\lambda+\mu S(y|a_0)).
  $$

* New canonical payment:
  $$
  w(y|\lambda,\mu)=L(\lambda+\mu S(y|a_0))-w_0.
  $$

---

## Score notation

* Old score/likelihood ratio:
  $$
  \frac{f_a(y|a_0)}{f(y|a_0)}.
  $$
  * New score:
  $$
  S(y|a):=\partial_a\log f(y|a)=\frac{f_a(y|a)}{f(y|a)}.
  $$

* Old proof notation often used output normalized to score.
  * New proof abbreviation:
  $$
  s(y):=S(y|a_0).
  $$

---

## Reservation utility frontier

* Old high-reservation statement: $\bar U\to\infty$.
  * New: high feasible reservation utility:
  $$
  \bar U\uparrow\bar U_R.
  $$

* Old unbounded frontier implicit.
  * New relaxed upper frontier:
  $$
  \bar U_R:=\lim_{\lambda\to\infty}\tilde U(\lambda).
  $$

* If utility is unbounded, $\bar U_R=\infty$.
* If utility is bounded, $\bar U_R<\infty$.

---

## Kink and no-payment region

* Old kink often described by output cutoff $\bar y$.
  * New: use score cutoff.

* Old floor binds when $\lambda+\mu S\le m$.
  * New floor binds when
  $$
  \lambda+\mu s(y)\le m_0.
  $$

* New threshold score:
  $$
  \bar s(\lambda,\mu):=\frac{m_0-\lambda}{\mu}.
  $$

* New LIC-path threshold:
  $$
  \bar s(\lambda):=\frac{m_0-\lambda}{\tilde\mu(\lambda)}.
  $$

* New no-payment region:
  $$
  \{y:s(y)\le \bar s(\lambda)\}.
  $$

---

## Low-score sets and capacity

* Old low-output or left-tail set: often $\{y\le q\}$ after score normalization.
  * New bad-score set:
  $$
  B_q:=\{y\in\mathcal Y:s(y)\le q\}.
  $$

* Avoid: $L_q$ for the low-score set.
  * Reason: $L$ is now the wealth link.

* Old capacity:
  $$
  [\bar u-u(0)]A_q>c'(a_0).
  $$
  * New capacity:
  $$
  \Delta u\,A_q>c'(a_0),
  \qquad
  \Delta u=\bar u-u(w_0).
  $$

* New low-score mass:
  $$
  A_q:=-\int_{B_q}s(y)f(y|a_0)\,d\nu(y).
  $$

* New paid-region positive curvature bound:
  $$
  M_q:=
  \sup_{a\in\mathcal A}
  \int_{\{s>q\}}[s(y)f_{aa}(y|a)]_+\,d\nu(y).
  $$

---

## Incentive and curvature formulas

* Old incentive functional using $g$:
  $$
  I(\lambda,\mu)=\int g(\lambda+\mu s(y))s(y)f(y|a_0)\,dy.
  $$
  * New:
  $$
  I(\lambda,\mu)=\int \ell(\lambda+\mu s(y))s(y)f(y|a_0)\,d\nu(y).
  $$

* Old trial multiplier:
  $$
  \mu_q(\lambda)=\frac{m-\lambda}{q}.
  $$
  * New:
  $$
  \mu_q(\lambda)=\frac{m_0-\lambda}{q}.
  $$

* Old secant slope with $g$:
  $$
  \Delta g(y|\lambda).
  $$
  * New:
  $$
  \Delta\ell(y|\lambda):=
  \frac{\ell(\lambda+\tilde\mu(\lambda)s(y))-\ell(\lambda)}
  {\tilde\mu(\lambda)s(y)}.
  $$

* Old curvature bound using $g'$.
  * New bound uses $\ell'$:
  $$
  \tilde\mu(\lambda)\ell'(\lambda+\tilde\mu(\lambda)q)M_q
  \le \underline c''.
  $$

---

## Utility examples

### Log

* Old: $u(x)=\log(x+w_0)$.
  * New:
  $$
  u(W)=\log W.
  $$

* New links:
  $$
  L(z)=z\vee w_0,
  \qquad
  \ell(z)=\log(z\vee w_0).
  $$

* New payment:
  $$
  w(y|\lambda,\mu)=[\lambda+\mu S(y|a_0)-w_0]^+.
  $$

### CARA

* Old: $u(x)=-e^{-\alpha(x+w_0)}/\alpha$.
  * New:
  $$
  u(W)=-\frac{e^{-\alpha W}}{\alpha}.
  $$

* New links:
  $$
  L(z)=\frac1\alpha\log(z\vee e^{\alpha w_0}),
  \qquad
  \ell(z)=-\frac1{\alpha(z\vee e^{\alpha w_0})}.
  $$

### CRRA

* Old: $u(x)=\frac{(x+w_0)^{1-\gamma}}{1-\gamma}$.
  * New:
  $$
  u(W)=\frac{W^{1-\gamma}}{1-\gamma}.
  $$

* New links:
  $$
  L(z)=(z\vee w_0^\gamma)^{1/\gamma},
  \qquad
  \ell(z)=\frac{(z\vee w_0^\gamma)^{(1-\gamma)/\gamma}}{1-\gamma}.
  $$
