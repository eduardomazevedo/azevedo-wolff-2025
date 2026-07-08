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

for $s(y)\ne0$, with the continuous extension at $s(y)=0$.

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

This proposition is the basic building block for the asymptotic high-reservation results.

---

## 5.5 Unified asymptotic concavity criterion

It may be useful to insert one theorem between the finite-$\lambda$ criterion and the utility-specific high-reservation results. This theorem is just the asymptotic version of the finite-$\lambda$ criterion. It isolates the common mechanism:

$$
\boxed{\text{the kink is in a safe low-score tail}}
\qquad + \qquad
\boxed{\text{paid-region positive curvature is asymptotically dominated by }c''.}
$$

**Unified asymptotic criterion.** Fix a tail-safe paid-region cutoff $q<0$ with $M_q<\infty$. Define the interior safe-capacity below $q$ by

$$
C_q:=\sup_{r<q}A_r.
$$

Let $\tilde\mu(\lambda)$ solve LIC. Suppose:

1. there is enough incentive capacity strictly below the paid-region cutoff,

$$
[\bar u-\underline u]C_q>c'(a_0),
\tag{ICap-q}
$$

with the convention that the left side is infinite if $\bar u=\infty$ and $C_q>0$;

2. the asymptotic paid-region curvature bound is strictly below cost curvature,

$$
\limsup_{\lambda\to\infty}
\tilde\mu(\lambda)
\ell'(\lambda+\tilde\mu(\lambda)q)M_q
<
\underline c'',
\qquad
\underline c'':=\inf_{a\in\mathcal A}c''(a).
\tag{AP}
$$

Then there exists $\lambda_0$ such that, for all $\lambda\ge\lambda_0$,

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le0
\quad\text{for all }a\in\mathcal A.
$$

Why condition (ICap-q) puts the kink in the safe tail: since the inequality is strict, there is some $t<q$ such that

$$
[\bar u-\underline u]A_t>c'(a_0).
$$

Equivalently, for all sufficiently large $\lambda$,

$$
[\ell(\lambda)-\underline u]A_t>c'(a_0).
$$

The kink lemma then gives

$$
\bar s(\lambda,\tilde\mu(\lambda))\le t<q
$$

for all sufficiently large $\lambda$. Hence the zero-payment region is safely inside the low-score region while the curvature of the paid region is bounded using the larger cutoff $q$.

Using generalized Proposition 1, this implies FOA validity for the corresponding high-reservation range.

This statement is often the most elegant theorem-level object. The finite-$\lambda$ criterion proves it immediately. The remaining results can be presented as verification corollaries:

* safe-tail capacity verifies (AK);
* strong flattening verifies (AP) with a zero limit, covering CARA and CRRA $\gamma>1$;
* log utility verifies (AP) through safe score width, with the unbounded-left-score case as the infinite-width limit.

CRRA $\gamma\in(1/2,1)$ generally should not be forced into this theorem: its positive-curvature term need not vanish or be small by this route. That case uses the appendix-only asymptotic-affinity mechanism.

---

## 5.6 Note on discrete output and score cutoffs

The score-cutoff formulation is also the right way to handle discrete output. A cutoff is a real number in score space, and the associated low-score set is

$$
B_q=\{y:s(y)\le q\}.
$$

The set of tail-safe cutoffs is automatically a lower interval in score space: if $q$ is tail-safe, then every $q'<q$ is tail-safe. The output set $B_q$ may jump as $q$ crosses score atoms, but that causes no conceptual problem.

This is why the interior-capacity condition in the unified criterion is written as

$$
C_q=\sup_{r<q}A_r
$$

rather than requiring a named cutoff $t$ in the theorem statement. In a discrete model, the proof can choose an interior capacity cutoff that includes the desired safe atom and choose $q$ between that atom and the first unsafe score. If only one low-score atom is safe, this still gives a positive safe width in score space whenever the next score is strictly larger.

For example, in Poisson output,

$$
s(y)=\frac{y-a_0}{a_0},
$$

so adjacent scores are separated by $1/a_0$. If only $y=0$ is safe, one can use a capacity cutoff that includes the atom $y=0$ and a paid-region cutoff just below the score of $y=1$. The available width is approximately $1/a_0$. Atoms at endpoints only affect whether inequalities are written as strict or weak; using strict slack in the theorem avoids relying on equality cases.

---

## 6. High-reservation theorem via strong flattening

The first high-reservation theorem uses the finite-$\lambda$ criterion and verifies condition (P) from primitive utility curvature. This route covers bounded utilities such as CARA and CRRA with $\gamma>1$.

The theorem statement should not be phrased as "choose two cutoffs." The primitive distributional object is the largest low-score region in which zero-payment states are known to help concavity.

Define the safe-tail boundary

$$
\bar q_{\mathrm{safe}}
:=
\sup\left\{
q<0:
 f_{aa}(y|a)>0
 \text{ for all }a\in\mathcal A
 \text{ and all }y\text{ with }s(y)\le q
\right\}.
$$

If the set is empty, set $\bar q_{\mathrm{safe}}=-\infty$. Because tail-safety is monotone in the cutoff, every $q<\bar q_{\mathrm{safe}}$ is tail-safe. In examples this object is easy to compute: it is the largest uniformly safe low-score cutoff.

The primitive capacity condition is

$$
[\bar u-\underline u]
\sup_{q<\bar q_{\mathrm{safe}}} A_q
>
c'(a_0).
\tag{SafeCap}
$$

This says that the safe low-score tail has enough incentive capacity to satisfy LIC while placing the limited-liability kink inside the region where the zero-payment contribution to $U_{aa}$ is nonpositive. For bounded utility, this is the substantive capacity condition. For unbounded utility it is automatic whenever $A_q>0$ for some safe cutoff.

Assume also the regularity condition

$$
M_q<\infty
\quad\text{for every }q<\bar q_{\mathrm{safe}}.
\tag{SafeReg}
$$

This is usually just tail regularity. It is automatic in Gaussian examples and should be straightforward for Poisson and exponential examples when the feasible action set is bounded away from zero.

### Theorem A: strong-flattening utilities

Assume:

1. $\bar q_{\mathrm{safe}}> -\infty$;
2. safe-tail capacity (SafeCap);
3. safe-tail regularity (SafeReg);
4. utility satisfies the primitive strong-flattening condition

$$
\frac{[u'(x)]^2}{-u''(x)}\to0
\quad\text{as }x\to\infty.
\tag{StrongFlat-u}
$$

Equivalently, for the nonbinding link $\ell=k'^{-1}$,

$$
z\ell'(z)\to0.
\tag{StrongFlat}
$$

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

### From primitive capacity to the two-cutoff proof

The two cutoffs are proof devices, not primitive assumptions. By (SafeCap), choose a stricter cutoff $t<\bar q_{\mathrm{safe}}$ such that

$$
[\bar u-\underline u]A_t>c'(a_0).
\tag{Cap-t}
$$

Then choose $q$ with

$$
t<q<\bar q_{\mathrm{safe}}\le0.
$$

The cutoff $t$ supplies capacity and controls the growth of $\tilde\mu(\lambda)$; the cutoff $q$ defines the paid-region curvature bound $M_q$, which is finite by (SafeReg).

### Linear-growth argument from the stricter cutoff

Capacity at $t$ and the kink lemma imply that, for all large $\lambda$,

$$
\bar s(\lambda,\tilde\mu(\lambda))\le t.
$$

Equivalently,

$$
\tilde\mu(\lambda)
\le
\frac{m_0-\lambda}{t}=O(\lambda),
$$

because $t<0$. Since $q<0$, this gives

$$
z_q(\lambda)
:=
\lambda+\tilde\mu(\lambda)q
\ge
\lambda+\frac{m_0-\lambda}{t}q
\sim
\lambda\left(1-\frac{q}{t}\right).
$$

The coefficient is positive because $t<q<0$. Thus $z_q(\lambda)$ grows at least linearly and $\tilde\mu(\lambda)=O(\lambda)$. Under the equivalent link condition $z\ell'(z)\to0$,

$$
\tilde\mu(\lambda)\ell'(z_q(\lambda))\to0,
$$

so condition (P) eventually holds.

### Utilities covered by strong flattening

The primitive condition is easy to verify because

$$
z\ell'(z)
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

or equivalently $\ell'(z)=1/z$ and $z\ell'(z)=1$. Therefore log should not be folded into Theorem A. Use a separate borderline-log theorem or corollary, still based on the same finite-$\lambda$ criterion but using a different asymptotic argument.

### Theorem B: borderline log utility via safe score width

Log utility should be stated as one theorem based on the width of the safe score region. This avoids separating the unbounded-left-score and bounded-left-score cases. The unbounded-left-score case is just the limiting case where the safe width is infinite.

For log utility,

$$
\ell'(z)=\frac1z.
$$

The paid-region term in the master criterion is

$$
\tilde\mu(\lambda)\ell'(\lambda+\tilde\mu(\lambda)q)
=
\frac{\tilde\mu(\lambda)}{\lambda+\tilde\mu(\lambda)q}.
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

The formal theorem should avoid endpoint cases and use strict inequalities. Assume there exist cutoffs

$$
\underline s<t<q<\bar q_{\mathrm{safe}}
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

Then, for log utility, concavity holds for all sufficiently large $\lambda$, and hence FOA validity holds for sufficiently high reservation utility.

This is the unified log theorem. It covers:

* **Unbounded-left-score models.** If $\underline s=-\infty$, fix any safe paid-region cutoff $q<\bar q_{\mathrm{safe}}$ with $M_q<\infty$ and take $t\to-\infty$. Then $q-t\to\infty$, so (LogWidth) is automatic. This is the old argument that the kink moves to $-\infty$ and $\tilde\mu/\lambda\to0$.
* **Bounded-left-score models.** If $\underline s> -\infty$, the condition says that the safe score width must be large enough relative to the positive paid-region curvature. Since log utility is unbounded above, capacity is automatic for any fixed $t$ with $A_t>0$.

A cleaner but slightly stronger primitive version is to define

$$
M_{\mathrm{safe}}
:=
\liminf_{q\uparrow\bar q_{\mathrm{safe}}}M_q
$$

and require

$$
\frac{M_{\mathrm{safe}}}{D_{\mathrm{safe}}}<\underline c'',
\tag{LogWidth'}
$$

with the convention that the ratio is zero when $D_{\mathrm{safe}}=\infty$ and $M_{\mathrm{safe}}<\infty$. This is easier to explain: the rightmost safe score minus the bottom score support must be large enough. The proof can still use the strict-cutoff version (LogWidth) to avoid equality and atom issues.

For Poisson and exponential examples, impose the natural nondegeneracy condition that the feasible action set is bounded away from zero.

### Verifying the safe-tail conditions in key examples

These examples suggest that (SafeCap) and (SafeReg) are the right theorem-level assumptions. They are simple to verify and make clear when a model is degenerate.

#### Gaussian location

For $Y\sim N(a,\sigma^2)$,

$$
s(y)=\frac{y-a_0}{\sigma^2}.
$$

The low-tail condition is

$$
f_{aa}(y|a)>0
\quad\Longleftrightarrow\quad
|y-a|>\sigma.
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
y+\sqrt y<\underline a,
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
y<(2-\sqrt2)a.
$$

Uniformly over $a\in\mathcal A$, this gives the safe output cutoff

$$
y<(2-\sqrt2)\underline a.
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

* $\gamma>1$: covered in the main text by strong flattening, with capacity at a stricter cutoff $t<q$;
* $\gamma=1$ (log): covered in the main text either by the unbounded-left-score route or by the finite-bound route;
* $\gamma\in(1/2,1)$: appendix-only extension via asymptotic affinity because the affine-score term contributes zero curvature.

---

## 10. CRRA corollary

For CRRA,

$$
\ell'(z)=\frac1\gamma z^{1/\gamma-2}.
$$

The main text should claim CRRA only for $\gamma\ge1$.

### Main-text branch 1: $\gamma>1$

If $\gamma>1$, utility is bounded and capacity at the stricter cutoff $t<q<0$ becomes

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_t>c'(a_0),
$$

up to the exact normalization of CRRA utility in the paper. Then the strong-flattening theorem applies because $z\ell'(z)\to0$.

### Main-text branch 2: $\gamma=1$ / log

If $\gamma=1$, this is log utility and utility is unbounded. Capacity is automatic at any cutoff with $A_t>0$, but log is only borderline flattening because $z\ell'(z)=1$. Therefore log requires either the unbounded-left-score route, which gives $\tilde\mu/\lambda\to0$, or the finite-bound route based on the master criterion.

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
\ell'(z)=\frac1{\alpha z^2}.
$$

CARA is covered by the strong-flattening theorem under the capacity condition at a stricter cutoff $t<q<0$:

$$
\frac{e^{-\alpha w_0}}{\alpha}A_t>c'(a_0).
$$

For Gaussian output, this becomes

$$
\frac{e^{-\alpha w_0}}{\alpha\sigma}\varphi(\sigma t)>c'(a_0).
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

### D. High-reservation theorem I: strong flattening and log borderline

Use two cutoffs $t<q<0$. Capacity at $t$ gives $\tilde\mu=O(\lambda)$ and $z_q(\lambda)\ge\delta\lambda$. Then $z\ell'(z)\to0$ proves concavity for CARA and CRRA $\gamma>1$. Treat log separately as the borderline case $z\ell'(z)=1$, using either unbounded left score support or the finite-bound condition from the master criterion.

### E. Main-text corollaries

State clean corollaries for:

1. CARA;
2. CRRA $\gamma>1$;
3. log utility as the borderline CRRA $\gamma=1$ case;
4. Gaussian location families for the main-text cases.

The abstract and main theorem discussion should claim CRRA only for $\gamma\ge1$.

### F. Appendix extension: asymptotic affinity

Move the following to an appendix rather than the main proof path. State conditions under which

$$
V(y)=\ell(\lambda)+\tilde\mu \ell'(\lambda)s(y)+o(1)
$$

in the $f_{aa}$-weighted sense. Add affine-score curvature:

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0.
$$

Conclude concavity. Use this to cover CRRA $\gamma\in(1/2,1)$, especially Gaussian location families. In the main text, mention this only in prose as an appendix extension, not as a formal remark or headline result.

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
