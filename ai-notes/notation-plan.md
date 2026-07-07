# Notation Plan

This note records the proposed notation for the revised paper. The guiding principle is:

* main text should speak in economically natural objects: payments, final wealth, scores, and reservation utility;
* analysis and proofs can use utility contracts because they make convexity and the first-order formula cleaner;
* initial wealth $w_0$ should be separated from the utility function.

---

## 1. Output, actions, and distributions

### Actions

* $a\in\mathcal A$: agent action.
* $a_0$: intended action.
* $\mathcal A\subseteq\mathbb R_+$: compact action set.

### Output

* $y\in\mathcal Y$: output.
* $\mathcal Y\subseteq\mathbb R$: common output support.
* $\nu$: common dominating measure on $\mathcal Y$.
* $f(y|a)$: density or probability mass function of output under action $a$, with respect to $\nu$.

Integrals are written as

$$
\int_{\mathcal Y}h(y)f(y|a)\,d\nu(y).
$$

When output is discrete, this notation denotes a sum with respect to counting measure.

### Distribution derivatives

* $f_a(y|a)$: first derivative of $f$ with respect to action.
* $f_{aa}(y|a)$: second derivative of $f$ with respect to action.

### Score

Use

$$
S(y|a):=\partial_a\log f(y|a)=\frac{f_a(y|a)}{f(y|a)}.
$$

In proof sections, fix $a_0$ and abbreviate

$$
s(y):=S(y|a_0).
$$

Use $S(y|a_0)$ in main text when readability matters; use $s(y)$ in technical sections to reduce clutter.

---

## 2. Utility, initial wealth, and cost

### Initial wealth and final wealth

* $w_0>0$: agent's initial wealth.
* $w(y)\ge0$: payment made by the principal after output $y$.
* $W(y):=w_0+w(y)$: final wealth.

Limited liability is

$$
w(y)\ge0,
$$

or equivalently

$$
W(y)\ge w_0.
$$

### Utility over final wealth

Use $u$ for utility over final wealth, not utility over payments:

$$
u(W).
$$

The agent's utility from payment $w$ is therefore

$$
u(w_0+w).
$$

The assumptions are on $u(W)$:

* strictly increasing;
* strictly concave;
* smooth;
* $u'(W)\to0$ as $W\to\infty$;
* upper bound

$$
\bar u:=\lim_{W\to\infty}u(W)\in\mathbb R\cup\{\infty\}.
$$

The lower utility induced by limited liability is

$$
\underline u:=u(w_0).
$$

The maximum pointwise utility spread is

$$
\Delta u:=\bar u-\underline u.
$$

If utility is unbounded, $\Delta u=\infty$.

### Effort cost

* $c(a)$: effort cost.
* $c'(a_0)>0$ for interior intended action $a_0>0$.
* $c''(a)>0$.
* $\underline c'':=\inf_{a\in\mathcal A}c''(a)$.

---

## 3. Three equivalent contract representations

There are three useful representations of a contract.

### Payment contract

$$
w:\mathcal Y\to\mathbb R_+.
$$

This is the most intuitive object for the main text and examples.

### Final-wealth contract

$$
W:\mathcal Y\to [w_0,\infty),
\qquad
W(y)=w_0+w(y).
$$

This is economically clean because utility is defined over final wealth and limited liability is $W\ge w_0$.

### Utility contract

$$
v:\mathcal Y\to [\underline u,\bar u),
\qquad
v(y)=u(W(y)).
$$

This is the analytically convenient representation. The inverse mapping is

$$
W(y)=k(v(y)),
$$

where

$$
k:=u^{-1}.
$$

The associated payment is

$$
w(y)=k(v(y))-w_0.
$$

### Recommendation

* Use $w(y)$ or $W(y)$ in the model and intuition.
* Introduce $v(y)=u(W(y))$ before solving the relaxed problem.
* State the canonical formula in final-wealth units using the wealth link $L$, then give the equivalent utility-contract formula using the utility link $\ell$.

---

## 4. Agent utility and expected compensation cost

For a utility contract $v$, the agent's expected utility from action $a$ is

$$
U(v,a):=\int v(y)f(y|a)\,d\nu(y)-c(a).
$$

For a final-wealth contract $W$, equivalently,

$$
U(W,a):=\int u(W(y))f(y|a)\,d\nu(y)-c(a).
$$

For a payment contract $w$, equivalently,

$$
U(w,a):=\int u(w_0+w(y))f(y|a)\,d\nu(y)-c(a).
$$

Expected payment cost under a utility contract is

$$
C(v,a):=\int [k(v(y))-w_0]f(y|a)\,d\nu(y).
$$

The current paper uses $W(v,a)$ for expected wage. To avoid conflict with final wealth $W(y)$, rename expected wage/cost to $C(v,a)$.

Recommended notation:

* $C(v,a)$: expected compensation cost.
* Avoid using $W(v,a)$ because $W$ should denote final wealth.

---

## 5. Link functions and marginal cost maps

The main text should use a **wealth link** because it gives the relaxed optimal contract directly in final-wealth units. The proofs should also use a **utility link** because the curvature of the agent's objective is written in utility units.

Use uppercase $L$ for the wealth link and lowercase $\ell$ for the utility link. Use subscripts $0$ for the unconstrained versions before imposing limited liability. This should not clash with the Lagrangian, which will be denoted by calligraphic $\mathcal L$.

### Cost of utility

Let

$$
k:=u^{-1}.
$$

Thus $k(v)$ is the final wealth needed to deliver utility $v$.

The principal's payment cost of utility $v$ is

$$
k(v)-w_0.
$$

The marginal cost of utility is

$$
k'(v)=\frac1{u'(k(v))}.
$$

The marginal cost at the limited-liability floor is

$$
m_0:=k'(\underline u)=\frac1{u'(w_0)}.
$$

### Wealth links

Define the unconstrained wealth link

$$
L_0(z):=(u')^{-1}\left(\frac1z\right).
$$

Define the limited-liability wealth link

$$
L(z):=L_0(z\vee m_0),
\qquad
z\vee m_0:=\max\{z,m_0\}.
$$

Equivalently,

$$
L(z)=(u')^{-1}\left(\frac1{z\vee m_0}\right).
$$

The wealth link maps the marginal-cost index $z$ into final wealth. It automatically incorporates limited liability:

$$
L(z)\ge w_0,
$$

with equality when $z\le m_0$.

### Utility links

Define the unconstrained utility link

$$
\ell_0(z):=u(L_0(z)).
$$

Equivalently,

$$
\ell_0(z)=(k')^{-1}(z).
$$

Define the limited-liability utility link

$$
\ell(z):=u(L(z))=\ell_0(z\vee m_0).
$$

Thus

$$
\ell(z)\ge \underline u,
$$

with equality when $z\le m_0$.

Do not use Jewitt's notation $\omega$ in the main paper unless we want a historical remark. The notation $\ell_0=(k')^{-1}$ is clear enough, and the economically useful object for the main text is $L$.

### Relationship between links

The links satisfy

$$
\ell=u\circ L,
\qquad
L=k\circ \ell.
$$

On the nonbinding region $z>m_0$,

$$
u'(L(z))=\frac1z.
$$

Therefore

$$
\ell'(z)=u'(L(z))L'(z)=\frac{L'(z)}{z}
\quad\text{for }z>m_0.
$$

The technical curvature conditions are naturally stated using $\ell'$ rather than $L'$, because expected utility curvature depends on utility units.

On the binding region $z<m_0$,

$$
L'(z)=\ell'(z)=0.
$$

At $z=m_0$, both $L$ and $\ell$ may have kinks.

---

## 6. Canonical contracts and multipliers

### Canonical final-wealth and payment contracts

For scalars $\lambda$ and $\mu$, define the canonical final-wealth contract

$$
W(y|\lambda,\mu):=L(\lambda+\mu S(y|a_0)).
$$

The associated payment contract is

$$
w(y|\lambda,\mu):=L(\lambda+\mu S(y|a_0))-w_0.
$$

These are the formulas to emphasize in the main text.

### Canonical utility contract

The equivalent utility contract is

$$
V(y|\lambda,\mu):=\ell(\lambda+\mu S(y|a_0))
=u(W(y|\lambda,\mu)).
$$

In proof sections, using $s(y)=S(y|a_0)$,

$$
W(y|\lambda,\mu)=L(\lambda+\mu s(y)),
\qquad
V(y|\lambda,\mu)=\ell(\lambda+\mu s(y)).
$$

### Multipliers

* $\lambda$: multiplier on promised utility / individual rationality.
* $\mu$: multiplier on local incentive compatibility.
* $\tilde\mu(\lambda)$: unique value of $\mu$ satisfying LIC for a given $\lambda$.
* $\lambda^*(\bar U)$: optimal multiplier for reservation utility $\bar U$.
* $\mu^*(\bar U):=\tilde\mu(\lambda^*(\bar U))$.

### LIC path

Define

$$
\tilde V(y|\lambda):=V(y|\lambda,\tilde\mu(\lambda)).
$$

Then $\tilde V(\cdot|\lambda)$ denotes the canonical contract satisfying LIC.

---

## 7. Reservation utility and upper frontier

* $\bar U$: reservation expected utility.
* $\bar U_L$: lower threshold below which the IR multiplier is zero.
* $\bar U_R$: upper relaxed frontier.

Define

$$
\tilde U(\lambda):=U(\tilde V(\cdot|\lambda),a_0).
$$

Then

$$
\bar U_R:=\lim_{\lambda\to\infty}\tilde U(\lambda).
$$

If $\bar u=\infty$, then $\bar U_R=\infty$. If $\bar u<\infty$, then $\bar U_R<\infty$.

Phrase the main theorem using **high feasible reservation utility**:

$$
\bar U\uparrow\bar U_R.
$$

When $\bar U_R=\infty$, this means ordinary high reservation utility.

Important distinction:

* lowercase $\bar u$: upper bound of pointwise utility from final wealth;
* uppercase $\bar U_R$: upper feasible expected utility frontier of the relaxed problem.

---

## 8. Limited-liability kink notation

The limited-liability floor binds when

$$
\lambda+\mu s(y)\le m_0.
$$

For $\mu>0$, define the threshold score

$$
\bar s(\lambda,\mu):=\frac{m_0-\lambda}{\mu}.
$$

The no-payment region is

$$
\{y:s(y)\le \bar s(\lambda,\mu)\}.
$$

For the LIC path, write

$$
\bar s(\lambda):=\bar s(\lambda,\tilde\mu(\lambda))
=
\frac{m_0-\lambda}{\tilde\mu(\lambda)}.
$$

Avoid using $\bar y$ for the threshold unless output itself has been normalized to the score.

---

## 9. Score-tail quantities for concavity

For a score cutoff $q<0$, define the bad-score set

$$
B_q:=\{y\in\mathcal Y:s(y)\le q\}.
$$

Define

$$
A_q:=-\int_{B_q}s(y)f(y|a_0)\,d\nu(y).
$$

This is positive when $B_q$ has negative score mass.

Define

$$
M_q:=
\sup_{a\in\mathcal A}
\int_{\{s>q\}}[s(y)f_{aa}(y|a)]_+\,d\nu(y).
$$

Tail-safe cutoff:

$$
f_{aa}(y|a)>0
\quad\text{for all }a\in\mathcal A
\quad\text{and all }y\in B_q.
$$

Tail-safe capacity:

$$
\Delta u\,A_q>c'(a_0),
\qquad
\Delta u:=\bar u-\underline u.
$$

For unbounded utility, $\Delta u=\infty$, so this is automatic when $A_q>0$.

---

## 10. Incentive functional and kink comparison

For fixed $\lambda$ and $\mu$, define

$$
I(\lambda,\mu):=
\int \ell(\lambda+\mu s(y))s(y)f(y|a_0)\,d\nu(y).
$$

LIC is

$$
I(\lambda,\tilde\mu(\lambda))=c'(a_0).
$$

For a cutoff $q<0$, define

$$
\mu_q(\lambda):=\frac{m_0-\lambda}{q}.
$$

This is the value of $\mu$ that places the kink at score $q$.

---

## 11. Secant slope notation for curvature

For the LIC path, define

$$
\Delta \ell(y|\lambda):=
\frac{\ell(\lambda+\tilde\mu(\lambda)s(y))-\ell(\lambda)}{\tilde\mu(\lambda)s(y)}
$$

for $s(y)\ne0$, with continuous extension at $s(y)=0$.

Then

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
=
\tilde\mu(\lambda)
\int \Delta \ell(y|\lambda)s(y)f_{aa}(y|a)\,d\nu(y)
-c''(a).
$$

This notation belongs in the proof section, not in the main theorem statement.

---

## 12. Utility examples with separated initial wealth

The applications table should emphasize the wealth link $L$, because it gives the optimal final-wealth contract directly:

$$
W(y|\lambda,\mu)=L(\lambda+\mu S(y|a_0)),
\qquad
w(y|\lambda,\mu)=L(\lambda+\mu S(y|a_0))-w_0.
$$

The utility link $\ell=u\circ L$ should also be listed because proof conditions use $\ell'$.

### Log utility

Utility over final wealth:

$$
u(W)=\log W.
$$

Then

$$
m_0=w_0,
\qquad
L(z)=z\vee w_0,
\qquad
\ell(z)=\log(z\vee w_0).
$$

The payment contract is

$$
w(y|\lambda,\mu)=[\lambda+\mu S(y|a_0)-w_0]^+.
$$

Also,

$$
\underline u=\log w_0,
\qquad
\bar u=\infty,
\qquad
\Delta u=\infty.
$$

### CARA utility

Utility over final wealth:

$$
u(W)=-\frac{e^{-\alpha W}}{\alpha}.
$$

Then

$$
m_0=e^{\alpha w_0},
\qquad
L(z)=\frac1\alpha\log(z\vee e^{\alpha w_0}),
\qquad
\ell(z)=-\frac1{\alpha(z\vee e^{\alpha w_0})}.
$$

The payment contract is

$$
w(y|\lambda,\mu)=
\frac1\alpha\log\left((\lambda+\mu S(y|a_0))\vee e^{\alpha w_0}\right)-w_0.
$$

Also,

$$
\bar u=0,
\qquad
\underline u=-\frac{e^{-\alpha w_0}}{\alpha},
\qquad
\Delta u=\frac{e^{-\alpha w_0}}{\alpha}.
$$

Capacity becomes

$$
\frac{e^{-\alpha w_0}}{\alpha}A_q>c'(a_0).
$$

### CRRA utility

For $\gamma\ne1$,

$$
u(W)=\frac{W^{1-\gamma}}{1-\gamma}.
$$

Then

$$
m_0=w_0^\gamma,
\qquad
L(z)=(z\vee w_0^\gamma)^{1/\gamma},
\qquad
\ell(z)=\frac{(z\vee w_0^\gamma)^{(1-\gamma)/\gamma}}{1-\gamma}.
$$

The payment contract is

$$
w(y|\lambda,\mu)=
\left((\lambda+\mu S(y|a_0))\vee w_0^\gamma\right)^{1/\gamma}-w_0.
$$

If $\gamma>1$,

$$
\bar u=0,
\qquad
\underline u=\frac{w_0^{1-\gamma}}{1-\gamma},
\qquad
\Delta u=\frac{w_0^{1-\gamma}}{\gamma-1}.
$$

Capacity becomes

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_q>c'(a_0).
$$

If $\gamma<1$, utility is unbounded above, so $\Delta u=\infty$.

For $\gamma=1$, CRRA is log utility.

---

## 13. Recommended renamings from current paper

Current notation to revise:

* Current $w(y)$: keep $w(y)$ for payment/wage. Use uppercase $W(y)=w_0+w(y)$ for final wealth.
* Current $W(v,a)$: expected wage. Rename to $C(v,a)$ for expected compensation cost.
* Current utility $u(x)$ from payment. Replace with $u(W)$ over final wealth and write payment utility as $u(w_0+w)$.
* Current floor $u(0)$. Replace with $\underline u=u(w_0)$.
* Current marginal floor $m=1/u'(0)$. Replace with

$$
m_0=\frac1{u'(w_0)}.
$$

* Current link function $g=(k')^{-1}\circ\max\{\cdot,m\}$ returns utility. Replace the main-text formula with the wealth link

$$
L(z)=(u')^{-1}\left(\frac1{z\vee m_0}\right),
$$

and define the utility link for proofs as

$$
\ell(z)=u(L(z)).
$$

* Current capacity $[\bar u-u(0)]A_q>c'(a_0)$. Replace with

$$
\Delta u\,A_q>c'(a_0),
\qquad
\Delta u=\bar u-u(w_0).
$$

---

## 14. Main theorem wording

Use the phrase:

**high feasible reservation utility**.

This means

$$
\bar U\uparrow\bar U_R.
$$

If utility is unbounded, then $\bar U_R=\infty$, so this reduces to the familiar phrase ``sufficiently high reservation utility.''

A clean theorem statement can be:

> Fix an interior intended action $a_0>0$. Under the maintained assumptions, the first-order approach is valid for $(a_0,\bar U)$ for every sufficiently high feasible reservation utility $\bar U$.

Then immediately explain:

> Equivalently, there exists $U^*<\bar U_R$ such that the first-order approach is valid for every $\bar U\in[U^*,\bar U_R)$. If utility is unbounded above, then $\bar U_R=\infty$.
