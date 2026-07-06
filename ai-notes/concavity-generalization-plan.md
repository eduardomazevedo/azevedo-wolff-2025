Below is the plan as it currently stands. I wrote it as a roadmap for revising the proof, with enough detail that someone can pick up from here and turn it into formal statements.

---

# Summary of the new concavity-proof plan

## 1. Change notation: work in score coordinates

The first simplification is to normalize output so that

$$
S(y|a_0)=y.
$$

This is without loss because the paper assumes the score $S(\cdot|a_0)$ is strictly increasing with image $\mathbb R$. Thus we can use the score as the output variable. After this normalization, the canonical contract is simply

$$
V(y|\lambda,\mu)=g(\lambda+\mu y),
$$

and the limited-liability kink is

$$
\bar y(\lambda,\mu)=\frac{m-\lambda}{\mu},
\qquad
m:=\frac{1}{u'(0)}.
$$

The agent receives zero wage exactly when

$$
y\le \bar y(\lambda,\mu).
$$

This is much cleaner than constantly moving between original output (y) and score (S(y|a_0)).

We also need to include a short invariance argument. If (x) denotes original output and (y=S(x|a_0)), with inverse (x=X(y)), then the transformed density is

$$
\tilde f(y|a)=f(X(y)|a)X'(y).
$$

Because (X'(y)>0) and the transformation is independent of (a),

$$
\tilde f_{aa}(y|a)=f_{aa}(X(y)|a)X'(y),
$$

so the sign of (f_{aa}) is preserved. Thus the condition “(f_{aa}>0) in the left tail” is really a condition on the left tail in score order. The threshold changes according to the score map, but the property is invariant.

---

## 2. Replace “the kink goes to $-\infty$” by “the kink lies left of a tail-safe cutoff”

Define a **tail-safe cutoff** (q<0) by requiring

$$
f_{aa}(y|a)>0
\quad\text{for all }y\le q,\ a\in A.
$$

The condition (q<0) ensures the score is negative on the zero-payment region. The condition (f_{aa}>0) ensures that the no-pay region contributes negatively to (U_{aa}).

The old proof tried to show the kink goes to $-\infty$. That is stronger than necessary. What we actually need is

$$
\bar y(\lambda,\tilde\mu(\lambda))\le q
$$

for some tail-safe (q).

For fixed $\lambda>m$, define

$$
\mu_q(\lambda):=\frac{m-\lambda}{q}.
$$

This is the value of $\mu$ that puts the kink exactly at (q). Let

$$
I(\lambda,\mu):=\int g(\lambda+\mu y)y f(y|a_0),dy.
$$

The local incentive constraint is

$$
I(\lambda,\tilde\mu(\lambda))=c'(a_0).
$$

Since

$$
I_\mu(\lambda,\mu)
=
\int g'$\lambda+\mu y$y^2 f(y|a_0),dy\ge0,
$$

(I) is increasing in $\mu$.

Therefore we get the new kink lemma:

**Kink lemma.** If

$$
I(\lambda,\mu_q(\lambda))\ge c'(a_0),
\tag{K}
$$

then

$$
\bar y(\lambda,\tilde\mu(\lambda))\le q.
$$

This is the clean replacement for the old kink-to-$-\infty$ lemma.

A simple sufficient condition for (K) is obtained by using (E_{a_0}[y]=0). Define

$$
A_q:=-\int_{y\le q}y f(y|a_0),dy>0.
$$

Then

$$
I(\lambda,\mu_q(\lambda))
\ge
[g(\lambda)-u(0)]A_q.
$$

Thus it suffices that

$$
[g(\lambda)-u(0)]A_q\ge c'(a_0).
\tag{K'}
$$

At the limiting level, this becomes the **capacity condition**

$$
[u(\infty)-u(0)]A_q>c'(a_0).
\tag{Cap}
$$

If $u(\infty)=\infty$, this holds automatically. If $u$ is bounded, this is the quantitative condition saying the maximum utility spread across the event $\{y>q\}$ contains enough local incentive power.

---

## 3. Rewrite the concavity proof around the useful split

After normalization, the key curvature expression can be written in the form

$$
U_{aa}(V,a)
=
\mu\int \Delta g(y|\lambda)\,y\,f_{aa}(y|a),dy
c''(a),
$$

where $\Delta g$ is the relevant secant slope term.

Once the kink is to the left of a tail-safe (q), the zero-payment region lies in ({y<0, f_{aa}>0}), so its contribution is negative. Therefore we may discard that region when looking for an upper bound.

The remaining positive part is controlled by

$$
\mu\int_{y>q}\Delta g(y|\lambda)[y f_{aa}(y|a)]_+,dy.
$$

Define

$$
M_q:=
\sup_{a\in A}
\int_{y>q}[y f_{aa}(y|a)]_+,dy.
$$

The first crude bound is

$$
\mu\int_{y>q}\Delta g(y|\lambda)[y f_{aa}(y|a)]_+,dy
\le
\mu g'$\lambda+\mu q$M_q.
$$

Important correction: the factor $\mu$ remains. The positive-part condition is therefore

$$
\mu g'$\lambda+\mu q$M_q
\le
\underline c'',
\qquad
\underline c'':=\inf_{a\in A}c''(a).
\tag{P}
$$

This is the finite-$\lambda$ condition that guarantees

$$
U_{aa}(V,a)\le0
\quad\text{for all }a\in A.
$$

The old condition (\limsup z g'(z)<\infty) was used precisely to control this missing $\mu$. The new proof improves the left-tail part, but the positive part still needs either a flattening argument or an asymptotic-affinity argument.

---

## 4. First main result: finite-$\lambda$ concavity criterion

The first proposition should be a finite-$\lambda$ criterion.

**Proposition, finite-$\lambda$ criterion.** Normalize $S(y|a_0)=y$. Fix a tail-safe cutoff $q<0$, and suppose $M_q<\infty$. Let $\lambda>m$, and let $\tilde\mu(\lambda)$ solve LIC. If

$$
I(\lambda,\mu_q(\lambda))\ge c'(a_0)
$$

and

$$
\tilde\mu(\lambda)g'(\lambda+\tilde\mu(\lambda)q)M_q
\le \underline c'',
$$

then

$$
U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)\le0
\quad\text{for all }a\in A.
$$

This proposition is the basic quantitative replacement for the current asymptotic proof. It is also useful in bounded-utility cases and finite-reservation applications.

---

## 5. Second result: high-reservation theorem via the flattening route

The first high-reservation theorem follows when the positive part vanishes because the contract becomes flat enough.

We need:

1. A tail-safe (q) with $M_q<\infty$.
2. The capacity condition

$$
[u(\infty)-u(0)]A_q>c'(a_0).
$$

3. A condition implying

$$
\tilde\mu(\lambda)g'(\lambda+\tilde\mu(\lambda)q)\to0.
\tag{Flat}
$$

If these hold, then along any sequence with $\lambda\to\infty$,

$$
\sup_{a\in A}U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le
-\underline c''+o(1)<0.
$$

This yields FOA validity for sufficiently high feasible reservation utilities, provided high feasible reservation utility drives (\lambda^*$\bar U$\to\infty).

### How to verify the flattening condition

For log utility, $g'(z)=1/z$. The old proof shows $\tilde\mu/\lambda\to0$, and then

$$
\tilde\mu g'(\lambda+\tilde\mu q)\sim \tilde\mu/\lambda\to0.
$$

For CARA, $g'(z)=1/(\alpha z^2)$. Under capacity, we can show $z_q(\lambda):=\lambda+\tilde\mu q$ grows at least linearly in $\lambda$, while $\tilde\mu=O(\lambda)$. Therefore

$$
\tilde\mu g'(z_q)=O(\lambda/\lambda^2)\to0.
$$

For CRRA with $\gamma>1$, $g'(z)=\frac1\gamma z^{1/\gamma-2}$. Again, under capacity we can get $z_q\ge \delta\lambda$ and $\tilde\mu=O(\lambda)$, so

$$
\tilde\mu g'(z_q)
 =
O(\lambda^{1/\gamma-1})\to0.
$$

Thus the flattening route covers log, CARA, and bounded CRRA $\gamma>1$.

---

## 6. Important bounded-utility step-contract argument

For bounded utility, capacity also implies that

$$
z_q(\lambda):=\lambda+\tilde\mu(\lambda)q\to\infty
$$

and, with a slightly stricter cutoff, (z_q) grows at least linearly in $\lambda$.

The argument is:
\tilde\mu g'(\lambda)\int y f_{aa}(y|a),dy.
Fix (L>m) and define

\text{for all }a\in A,
\mu_{q,L}(\lambda)=\frac{L-\lambda}{q}.
$$

Then the contract satisfying $\lambda+\mu_{q,L}q=L$ converges pointwise to the step contract

$$
u(0)\mathbf 1_{\{y<q\}}+u(\infty)\mathbf 1_{\{y>q\}}.
$$
- \tilde\mu g'(\lambda)y
Its incentive converges to

$$
[u(\infty)-u(0)]A_q.
$$

If capacity holds strictly, this exceeds $c'(a_0)$. Therefore, for large $\lambda$, this contract gives too much incentive. By monotonicity of $I(\lambda,\mu)$, the true LIC multiplier is smaller, which implies $z_q(\lambda)\ge L$. Since $L$ is arbitrary, $z_q(\lambda)\to\infty$.
\tilde\mu g'(\lambda)=O(1),
To get linear growth, choose a stricter cutoff (t<q) that still satisfies capacity. Evaluating the kink at (t) gives

$$
\tilde\mu(\lambda)\le \frac{m-\lambda}{t}=O(\lambda).
- \tilde\mu g'(\lambda)y

Then
O\left(\tilde\mu^2 |g''(\lambda)|y^2\right)
$$
z_q(\lambda)
\tilde\mu^2 |g''(\lambda)|
\lambda+\tilde\mu q
[\tilde\mu g'(\lambda)]\frac{\tilde\mu}{\lambda}
\lambda+\frac{m-\lambda}{t}q
\sim
This route covers CRRA $\gamma\in(1/2,1)$ when affine-score curvature is harmless. It explains the Gaussian numerics.
$$
### Branch 2: $\gamma\in(1/2,1)$
which is linear and positive since (t<q<0).
\tilde\mu^2 |g''(\lambda)|
This is crucial for CARA and CRRA $\gamma>1$.
Normalize $S(y|a_0)=y$. Prove invariance of $f_{aa}>0$ under the transformation.
Define $q$, $A_q$, $\mu_q(\lambda)$, and $I(\lambda,\mu)$. Prove the kink lemma and the sufficient condition $[g(\lambda)-u(0)]A_q\ge c'(a_0)$.
Define $M_q$. Prove the finite-$\lambda$ proposition with conditions $K$ and $P$.
Conclude concavity. Use this to cover CRRA $\gamma\in(1/2,1)$, especially Gaussian location families.
4. CRRA $\gamma\in(1/2,1)$ under affine-score concavity;
This is both more general and more informative. It handles bounded utility through a capacity condition. It also explains why CRRA $\gamma\in(1/2,1)$ can work numerically: the contract is not flat, but it is asymptotically affine, and affine score contracts are harmless in Gaussian location families.

For CRRA $\gamma\in(1/2,1)$,

$$
g'(z)\to0
\quad\text{but}\quad
z g'(z)\to\infty.
$$

LIC suggests

$$
\tilde\mu(\lambda)g'(\lambda)
\int y^2 f(y|a_0),dy
\approx c'(a_0),
$$

so

$$
\tilde\mu(\lambda)g'(\lambda)
\to
\theta:=
\frac{c'(a_0)}
{\int y^2 f(y|a_0),dy},
$$

not zero.

Thus the contract behaves like

$$
g(\lambda+\tilde\mu y)
=
g(\lambda)+\tilde\mu g'(\lambda)y+o(1).
$$

The constant term contributes nothing to (U_{aa}), because

$$
\int f_{aa}(y|a),dy=0.
$$

The affine term contributes

$$
\tilde\mu g'(\lambda)\int y f_{aa}(y|a),dy.
$$

Therefore the right additional condition is:

$$
\int y f_{aa}(y|a),dy\le0
\quad\text{for all }a\in A.
\tag{Affine}
$$

Equivalently, the function

$$
a\mapsto E_a[S(Y|a_0)]
$$

is concave.

If this affine-score curvature condition holds, then the affine part is harmless, and the nonlinear residual vanishes. Hence

$$
U_{aa}(V,a)\le -c''(a)+o(1).
$$

This gives a second high-reservation theorem.

### Residual condition

We need to formalize the asymptotic-affinity residual:

$$
\sup_{a\in A}
\left|
\int
\left[
g(\lambda+\tilde\mu y)
- g(\lambda)
- \tilde\mu g'(\lambda)y
\right]
f_{aa}(y|a),dy
\right|
\to0.
\tag{R}
$$

Primitive sufficient conditions for (R) should include:

$$
\frac{\tilde\mu}{\lambda}\to0,
\qquad
\tilde\mu g'(\lambda)=O(1),
$$

a controlled curvature condition such as

$$
\sup_{z\ge \lambda/2}\frac{z|g''(z)|}{g'(z)}<\infty,
$$

and a uniform integrability condition for (y^2 f_{aa}(y|a)).

Taylor’s theorem gives

$$
g(\lambda+\tilde\mu y)
- g(\lambda)
- \tilde\mu g'(\lambda)y

=
O\left(\tilde\mu^2 |g''(\lambda)|y^2\right)
$$

on the central region, and

$$
\tilde\mu^2 |g''(\lambda)|
\asymp
[\tilde\mu g'(\lambda)]\frac{\tilde\mu}{\lambda}
\to0.
$$

Tails need to be handled by uniform integrability.

This route covers CRRA $\gamma\in(1/2,1)$ when affine-score curvature is harmless. It explains the Gaussian numerics.

---

## 8. Gaussian example

For Gaussian original output (X\sim N(a,\sigma^2)), normalized output is

$$
y=S(X|a_0)=\frac{X-a_0}{\sigma^2}.
$$

Then under action (a),

$$
y\sim N\left(\frac{a-a_0}{\sigma^2},\frac1{\sigma^2}\right).
$$

The left-tail condition is explicit:

$$
f_{aa}(y|a)>0
\quad\Longleftrightarrow\quad
\left|y-\frac{a-a_0}{\sigma^2}\right|>\frac1\sigma.
$$

Thus

$$
y^*(a)=\frac{a-a_0}{\sigma^2}-\frac1\sigma.
$$

Any

$$
q<\min\left\{0,\inf_{a\in A}
\left(\frac{a-a_0}{\sigma^2}-\frac1\sigma\right)\right\}
$$

is tail-safe.

Also,

$$
A_q=-\int_{y\le q}y f(y|a_0),dy
=
\frac1\sigma\varphi(\sigma q).
$$

The positive-part constant is finite automatically:

$$
M_q=
\sup_{a\in A}
\int_q^\infty
\left[
y\left(\left(y-\frac{a-a_0}{\sigma^2}\right)^2-\frac1{\sigma^2}\right)
\right]_+
f(y|a),dy<\infty.
$$

Finally,

$$
\int y f_{aa}(y|a),dy
=
\frac{\partial^2}{\partial a^2}E_a[y]
 =
0,
$$

because

$$
E_a[y]=\frac{a-a_0}{\sigma^2}
$$

is affine. Therefore Gaussian location families satisfy the affine-score curvature condition exactly.

So in Gaussian-CRRA:

* $\gamma\ge1$: covered by flattening, with capacity if $\gamma>1$.
* $\gamma\in(1/2,1)$: covered by asymptotic affinity because the affine-score term contributes zero curvature.

This matches the numerical observation that $\gamma\in(1/2,1)$ does well.

---

## 9. CRRA corollary

The CRRA corollary should probably be stated in two branches.

For CRRA,

$$
g'(z)=\frac1\gamma z^{1/\gamma-2}.
$$

### Branch 1: $\gamma\ge1$

If $\gamma=1$, this is log utility and utility is unbounded. Capacity is automatic.

If $\gamma>1$, utility is bounded and capacity becomes

$$
\frac{w_0^{1-\gamma}}{\gamma-1}A_q>c'(a_0).
$$

Then the flattening theorem applies.

### Branch 2: $\gamma\in(1/2,1)$

Utility is unbounded, so capacity is automatic. Flattening may fail, but asymptotic affinity applies if

$$
\int y f_{aa}(y|a),dy\le0
\quad\text{for all }a\in A,
$$

plus the residual and integrability conditions.

For Gaussian location families,

$$
\int y f_{aa}(y|a),dy=0,
$$

so all CRRA coefficients

$$
\gamma>1/2
$$

are covered.

We should not make a broad claim for $\gamma\le1/2$. For those values, the nonlinear residual need not vanish because

$$
\tilde\mu^2 |g''(\lambda)|
$$

may fail to go to zero.

---

## 10. CARA corollary

For CARA,

$$
u(x)=-\frac{e^{-\alpha(x+w_0)}}{\alpha},
$$

so

$$
u(\infty)-u(0)=\frac{e^{-\alpha w_0}}{\alpha},
$$

and

$$
g'(z)=\frac{1}{\alpha z^2}.
$$

Thus CARA is covered by the flattening theorem under the capacity condition

$$
\frac{e^{-\alpha w_0}}{\alpha}A_q>c'(a_0).
$$

For Gaussian output, this becomes

$$
\frac{e^{-\alpha w_0}}{\alpha\sigma}\varphi(\sigma q)>c'(a_0).
$$

This is conservative but clean. It explains why bounded CARA can still work: the maximum utility spread must be large enough to generate incentives, and once it is, the paid-region slope vanishes quickly enough.

---

## 11. Proposed revised organization of the proof

A good revised proof section could be structured as follows:

### A. Score normalization

Normalize $S(y|a_0)=y$. Prove invariance of $f_{aa}>0$ under the transformation.

### B. Tail-safe cutoffs and kink lemma

Define $q$, $A_q$, $\mu_q(\lambda)$, and $I(\lambda,\mu)$. Prove the kink lemma and the sufficient condition $[g(\lambda)-u(0)]A_q\ge c'(a_0)$.

### C. Finite-$\lambda$ concavity criterion

Define $M_q$. Prove the finite-$\lambda$ proposition with conditions $K$ and $P$.

### D. High-reservation theorem I: flattening

Use capacity and $\tilde\mu g'(z_q)\to0$ to prove concavity. Provide primitive sufficient conditions covering log, CARA, and CRRA $\gamma>1$.

### E. High-reservation theorem II: asymptotic affinity

State conditions under which

$$
V(y)=g(\lambda)+\tilde\mu g'(\lambda)y+o(1)
$$

in the (f_{aa})-weighted sense. Add the affine-score curvature condition

$$
\int y f_{aa}(y|a),dy\le0.
$$

Conclude concavity. Use this to cover CRRA $\gamma\in(1/2,1)$, especially Gaussian location families.

### F. Corollaries

State clean corollaries for:

1. log utility;
2. CARA;
3. CRRA $\gamma\ge1$;
4. CRRA $\gamma\in(1/2,1)$ under affine-score concavity;
5. Gaussian location family, which satisfies the affine-score condition with equality.

---

## 12. Main conceptual improvement

The original proof had one asymptotic mechanism:

$$
\text{the no-pay region vanishes / kink goes to }-\infty,
\quad
\text{and the paid-region curvature becomes small}.
$$

The revised proof separates the logic into sharper components:

$$
\boxed{
\text{left-tail no-pay region contributes negatively}
}
$$

once the kink is left of a tail-safe cutoff;

$$
\boxed{
\text{positive part is controlled either by flattening or asymptotic affinity}
}
$$

depending on the utility function.

This is both more general and more informative. It handles bounded utility through a capacity condition. It also explains why CRRA $\gamma\in(1/2,1)$ can work numerically: the contract is not flat, but it is asymptotically affine, and affine score contracts are harmless in Gaussian location families.
