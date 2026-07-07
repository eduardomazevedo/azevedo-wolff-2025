# Plan to Generalize the Concavity Result

Throughout, output lies in a fixed set $\mathcal Y\subseteq\mathbb R$, and $f(\cdot|a)$ is a density with respect to a common dominating measure $\nu$. Integrals are with respect to $d\nu(y)$. This covers both continuous and discrete output.

---

## 1. Score notation instead of score normalization

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

Maintain throughout this note that the utility link $\ell$ is nondecreasing and concave on the nonbinding region. At the limited-liability kink, interpret $\ell'$ as the relevant one-sided upper derivative when needed.

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

When the score is strictly increasing with image $\mathbb R$, this notation specializes to the old score-normalized proof by taking the score itself as the output variable.

---

## 2. Tail-safe score cutoffs

A **tail-safe cutoff** is a number $q<0$ such that

$$
f_{aa}(y|a)>0
\quad\text{for all }a\in\mathcal A
\quad\text{and all }y\in\mathcal Y\text{ with }s(y)\le q.
\tag{TS}
$$

The condition $q<0$ ensures that score is negative on the low-score region. The condition $f_{aa}>0$ on that region ensures that the low-score, zero-payment contribution to $U_{aa}$ is nonpositive.

Define the bad-score set

$$
B_q:=\{y\in\mathcal Y:s(y)\le q\}.
$$

Define

$$
A_q:=-\int_{B_q}s(y)f(y|a_0)\,d\nu(y)>0.
$$

For continuous score-normalized models, this is the old object

$$
A_q=-\int_{y\le q}y f(y|a_0)\,dy.
$$

For discrete models, it is the same expression with sums over the low-score states.

---

## 3. Kink lemma: getting the floor region below a tail-safe cutoff

For fixed $\lambda>m_0$ and tail-safe $q<0$, define

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

Because

$$
I_\mu(\lambda,\mu)=
\int \ell'(\lambda+\mu s(y))s(y)^2 f(y|a_0)\,d\nu(y)\ge0,
$$

$I(\lambda,\mu)$ is increasing in $\mu$.

**Kink lemma.** If

$$
I(\lambda,\mu_q(\lambda))\ge c'(a_0),
\tag{K}
$$

then

$$
\bar s(\lambda,\tilde\mu(\lambda))\le q.
$$

Equivalently, the zero-payment region is contained in the tail-safe low-score region:

$$
\{y:s(y)\le \bar s(\lambda,\tilde\mu(\lambda))\}
\subseteq
B_q.
$$

A useful sufficient condition for (K) is obtained from $E_{a_0}[s(Y)]=0$. At $\mu=\mu_q(\lambda)$, states with $s\le q$ receive utility $\underline u$, while states with $s>q$ receive at least $\ell(\lambda)$ if $s\ge0$. The crude lower bound is

$$
I(\lambda,\mu_q(\lambda))
\ge
[\ell(\lambda)-\underline u]A_q.
$$

Thus it suffices that

$$
[\ell(\lambda)-\underline u]A_q\ge c'(a_0).
\tag{K'}
$$

At the limiting level this becomes the tail-safe capacity condition

$$
[\bar u-\underline u]A_q>c'(a_0),
\tag{Cap-q}
$$

where

$$
\bar u:=\lim_{W\to\infty}u(W)\le\infty,
\qquad \underline u:=u(w_0).
$$

If $\bar u=\infty$, this condition is automatic for any $q$ with $A_q>0$. If $\bar u<\infty$, it says that the maximum utility spread across the low-score cutoff is enough to generate the required local incentive.

This capacity condition is stronger than the implementability condition used for Proposition 1. Proposition 1 only needs enough total utility spread to satisfy LIC; concavity needs enough incentive power while placing the no-pay region inside a tail-safe low-score set.

---

## 4. Curvature formula and useful split

For a canonical LIC-satisfying contract, write $\tilde\mu=\tilde\mu(\lambda)$. Define the secant slope term

$$
\Delta \ell(y|\lambda):=
\frac{\ell(\lambda+\tilde\mu s(y))-\ell(\lambda)}{\tilde\mu s(y)}
$$

for $s(y)\ne0$, with the usual continuous extension at $s(y)=0$.

Then

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu),a)
=
\tilde\mu
\int \Delta \ell(y|\lambda)s(y)f_{aa}(y|a)\,d\nu(y)
-c''(a).
\tag{Curv}
$$

Once the kink is below $q$, the zero-payment region is contained in $B_q$. On this region, $s<0$ and $f_{aa}>0$, so its contribution to the integral in (Curv) is nonpositive. Therefore, for an upper bound we can discard it.

Define the positive-part constant

$$
M_q:=
\sup_{a\in\mathcal A}
\int_{\{s>q\}}
[s(y)f_{aa}(y|a)]_+\,d\nu(y).
$$

Assume $M_q<\infty$.

On the nonbinding region and under concavity of $\ell$ in its scalar argument, the secant slope is bounded by the marginal slope at the lowest relevant score:

$$
\Delta \ell(y|\lambda)\le \ell'(\lambda+\tilde\mu q)
\quad\text{for }s(y)>q.
$$

Hence

$$
\tilde\mu
\int_{\{s>q\}}
\Delta \ell(y|\lambda)[s(y)f_{aa}(y|a)]_+\,d\nu(y)
\le
\tilde\mu \ell'(\lambda+\tilde\mu q)M_q.
$$

The factor $\tilde\mu$ is important. The old proof controlled it through assumptions on $z\ell'(z)$ and through $\tilde\mu/\lambda\to0$. The generalized proof should keep this factor explicit.

---

## 5. Finite-$\lambda$ concavity criterion

The first result should be a finite-$\lambda$ criterion.

**Finite-$\lambda$ criterion.** Fix a tail-safe cutoff $q<0$ with $M_q<\infty$. Let $\lambda>m_0$, and let $\tilde\mu(\lambda)$ solve LIC. If

$$
I(\lambda,\mu_q(\lambda))\ge c'(a_0)
\tag{K}
$$

and

$$
\tilde\mu(\lambda)\ell'(\lambda+\tilde\mu(\lambda)q)M_q
\le
\underline c'',
\qquad
\underline c'':=\inf_{a\in\mathcal A}c''(a),
\tag{P}
$$

then

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

This proposition is useful because it is quantitative and applies to both continuous and discrete output. It is the basic building block for the asymptotic high-reservation results.

---

## 6. High-reservation theorem via flattening

The first high-reservation theorem uses the finite-$\lambda$ criterion and shows that condition (P) eventually holds because the paid-region contract becomes flat enough.

Assumptions:

1. There exists a tail-safe cutoff $q<0$ with $M_q<\infty$.
2. Tail-safe capacity holds:

$$
[\bar u-\underline u]A_q>c'(a_0).
\tag{Cap-q}
$$

3. Along the LIC path,

$$
\tilde\mu(\lambda)\ell'(\lambda+\tilde\mu(\lambda)q)\to0.
\tag{Flat}
$$

Then the finite-$\lambda$ criterion implies that, for all sufficiently large $\lambda$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

Using generalized Proposition 1, this implies FOA validity:

* if $\bar u=\infty$, for all sufficiently high $\bar U$;
* if $\bar u<\infty$, for all $\bar U$ sufficiently close to $\bar U_R$ from below.

### Bounded-utility step-contract argument

For bounded utility, capacity implies that the cutoff marginal-cost index at $q$ diverges:

$$
z_q(\lambda):=\lambda+\tilde\mu(\lambda)q\to\infty.
$$

Proof idea. Fix $\rho>m_0$ and define

$$
\mu_{q,\rho}(\lambda):=\frac{\rho-\lambda}{q}.
$$

This puts score $q$ at scalar marginal-cost index $\rho$:

$$
\lambda+\mu_{q,\rho}(\lambda)q=\rho.
$$

As $\lambda\to\infty$, the contract

$$
\ell(\lambda+\mu_{q,\rho}(\lambda)s(y))
$$

converges pointwise to the step contract that gives $\underline u$ on $\{s<q\}$ and $\bar u$ on $\{s>q\}$, with the boundary $s=q$ receiving $\ell(\rho)$. To avoid boundary clutter, choose $q$ away from score atoms; in discrete applications this means choosing $q$ between adjacent score values. Then the limiting incentive is

$$
[\bar u-\underline u]A_q.
$$

If there is an atom at $s=q$, use a nearby cutoff and the strict slack in capacity. By strict capacity, the limiting incentive exceeds $c'(a_0)$. Hence for large $\lambda$, the trial multiplier $\mu_{q,\rho}(\lambda)$ gives too much incentive. Since $I$ is increasing in $\mu$, the true multiplier satisfies

$$
\tilde\mu(\lambda)
\le
\mu_{q,\rho}(\lambda).
$$

Because $q<0$, this implies

$$
z_q(\lambda)=\lambda+\tilde\mu(\lambda)q
\ge \rho.
$$

Since $\rho$ is arbitrary, $z_q(\lambda)\to\infty$.

To get linear growth of $z_q$ when needed, choose a stricter cutoff $t<q<0$ that also satisfies capacity. Applying the kink lemma at $t$ gives, for large $\lambda$,

$$
\tilde\mu(\lambda)
\le
\frac{m_0-\lambda}{t}=O(\lambda).
$$

Then

$$
z_q(\lambda)
=
\lambda+\tilde\mu(\lambda)q
\ge
\lambda+\frac{m_0-\lambda}{t}q
\sim
\lambda\left(1-\frac{q}{t}\right),
$$

which is positive and linear because $t<q<0$.

### Utilities covered by flattening

* **Log utility:** $\ell'(z)=1/z$. In the unbounded case, one needs a separate vanishing-ratio lemma, $\tilde\mu/\lambda\to0$; capacity and kink placement alone do not imply it. With that lemma,

$$
\tilde\mu \ell'(\lambda+\tilde\mu q)
\sim
\tilde\mu/\lambda\to0.
$$

* **CARA:** $\ell'(z)=1/(\alpha z^2)$. Under capacity, $z_q$ grows at least linearly and $\tilde\mu=O(\lambda)$, so

$$
\tilde\mu \ell'(z_q)=O(\lambda/\lambda^2)\to0.
$$

* **CRRA with $\gamma>1$:** $\ell'(z)=\frac1\gamma z^{1/\gamma-2}$. Under capacity, $z_q\ge\delta\lambda$ and $\tilde\mu=O(\lambda)$, so

$$
\tilde\mu \ell'(z_q)=O(\lambda^{1/\gamma-1})\to0.
$$

Thus the flattening route covers log, CARA, and bounded CRRA with $\gamma>1$.

---

## 7. High-reservation theorem via asymptotic affinity

Flattening can fail for some unbounded utilities, notably CRRA with $\gamma\in(1/2,1)$. In that case the contract may not become flat, but it can become asymptotically affine in the score.

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

Thus the natural additional condition is affine-score curvature:

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

A theorem can then assume the residual condition

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
f_{aa}(y|a)>0
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

Atoms at the cutoff are harmless if the cutoff is chosen away from score atoms. If a cutoff atom is unavoidable, use weak inequalities consistently and keep capacity strict. In applications, one can choose a slightly different cutoff unless the score support is extremely sparse.

The bounded-below score issue is conceptually separate from discreteness. If the score has a lower bound, the kink may eventually fall below the lowest score. That case can still be handled by the finite-$\lambda$ criterion or by an appropriate limiting-curvature argument. Non-interval support by itself adds little complexity once everything is written in score-set notation.

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
f_{aa}(s|a)>0
\quad\Longleftrightarrow\quad
\left|s-\frac{a-a_0}{\sigma^2}\right|>\frac1\sigma.
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

* $\gamma\ge1$: covered by flattening, with capacity if $\gamma>1$;
* $\gamma\in(1/2,1)$: covered by asymptotic affinity because the affine-score term contributes zero curvature.

---

## 10. CRRA corollary

For CRRA,

$$
\ell'(z)=\frac1\gamma z^{1/\gamma-2}.
$$

### Branch 1: $\gamma\ge1$

If $\gamma=1$, this is log utility and utility is unbounded. Tail-safe capacity is automatic.

If $\gamma>1$, utility is bounded and capacity becomes

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_q>c'(a_0),
$$

up to the exact normalization of CRRA utility in the paper. Then the flattening theorem applies.

### Branch 2: $\gamma\in(1/2,1)$

Utility is unbounded, so capacity is automatic. Flattening may fail, but asymptotic affinity applies if

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0
\quad\text{for all }a\in\mathcal A,
$$

plus the residual and integrability conditions.

For Gaussian location families, this integral is zero, so all CRRA coefficients

$$
\gamma>1/2
$$

are covered.

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
\ell'(z)=\frac1{\alpha z^2}.
$$

CARA is covered by the flattening theorem under the capacity condition

$$
\frac{e^{-\alpha w_0}}{\alpha}A_q>c'(a_0).
$$

For Gaussian output, this becomes

$$
\frac{e^{-\alpha w_0}}{\alpha\sigma}\varphi(\sigma q)>c'(a_0).
$$

This is conservative but clean. It explains why bounded CARA can still work: the maximum utility spread must be large enough to generate incentives, and once it is, the paid-region slope vanishes quickly enough.

---

## 12. Proposed organization of the revised proof

A good revised proof section could be structured as follows.

### A. Common measure and score notation

Use $s(y)=S(y|a_0)$ and score cutoff sets $\{s\le q\}$. Avoid relying on an inverse score map.

### B. Tail-safe cutoffs and kink lemma

Define $q$, $A_q$, $\mu_q(\lambda)$, and $I(\lambda,\mu)$. Prove the kink lemma and the sufficient condition

$$
[\ell(\lambda)-\underline u]A_q\ge c'(a_0).
$$

### C. Finite-$\lambda$ concavity criterion

Define $M_q$. Prove the finite-$\lambda$ criterion with conditions (K) and (P).

### D. High-reservation theorem I: flattening

Use capacity and $\tilde\mu \ell'(z_q)\to0$ to prove concavity. Provide primitive sufficient conditions covering log, CARA, and CRRA $\gamma>1$.

### E. High-reservation theorem II: asymptotic affinity

State conditions under which

$$
V(y)=\ell(\lambda)+\tilde\mu \ell'(\lambda)s(y)+o(1)
$$

in the $f_{aa}$-weighted sense. Add affine-score curvature:

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0.
$$

Conclude concavity. Use this to cover CRRA $\gamma\in(1/2,1)$, especially Gaussian location families.

### F. Corollaries

State clean corollaries for:

1. log utility;
2. CARA;
3. CRRA $\gamma\ge1$;
4. CRRA $\gamma\in(1/2,1)$ under affine-score concavity;
5. Gaussian location families.

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
\boxed{\text{the remaining positive part is controlled by flattening or asymptotic affinity.}}
$$

This is more general and more informative. It handles bounded utility through a capacity condition, includes discrete output through score cutoff sets, and explains why CRRA $\gamma\in(1/2,1)$ can work in Gaussian examples: the contract is not flat, but it is asymptotically affine, and affine score contracts are harmless when expected score is concave in action.
