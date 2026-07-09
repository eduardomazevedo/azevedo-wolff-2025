# Moment Conditions for the Asymptotic-Affinity Residual

This note explains how to replace the nonprimitive residual condition in the asymptotic-affinity route with primitive moment conditions. The target case is CRRA with

$$
\gamma\in(1/2,1).
$$

The residual condition should not appear as a main assumption in the paper. It should be derived from score moments and uniform integrability.

---

## 1. Setup

Fix the intended action $a_0$ and write

$$
s(y):=S(y|a_0).
$$

Let

$$
f_0(y):=f(y|a_0).
$$

The canonical utility contract is

$$
V(y|\lambda,\mu)=\ell(\lambda+\mu s(y)).
$$

For CRRA utility with $\gamma\in(1/2,1)$,

$$
u(W)=\frac{W^{1-\gamma}}{1-\gamma}.
$$

On the nonbinding region, the utility link is

$$
\ell(z)=\frac{z^p}{1-\gamma},
\qquad
p:=\frac{1-\gamma}{\gamma}\in(0,1).
$$

Thus

$$
\ell'(z)=\frac1\gamma z^{p-1},
\qquad
\ell''(z)=\frac{p-1}{\gamma}z^{p-2}.
$$

In particular,

$$
\ell'(z)\to0,
\qquad
z\ell'(z)\to\infty.
$$

So the strong-flattening route fails, but the contract may become asymptotically affine in the score.

---

## 2. Nonprimitive residual we want to derive

The current asymptotic-affinity route uses the condition

$$
\sup_{a\in\mathcal A}
\left|
\int
\left[
 \ell(\lambda+\tilde\mu s(y))
 -\ell(\lambda)
 -\tilde\mu\ell'(\lambda)s(y)
\right]
 f_{aa}(y|a)\,d\nu(y)
\right|
\to0.
\tag{R}
$$

This is not primitive because it depends on the endogenous LIC multiplier $\tilde\mu(\lambda)$.

The goal is to show that (R) follows from:

1. CRRA with $\gamma\in(1/2,1)$;
2. ordinary score moments under $f_0$;
3. uniform weighted integrability of $f_{aa}$;
4. the LIC equation.

---

## 3. Primitive moment assumptions

A clean sufficient set of assumptions is the following.

### Moment under the intended action

There exists $\eta>0$ such that

$$
\int |s(y)|^{2+\eta} f_0(y)\,d\nu(y)<\infty,
$$

and

$$
0<\sigma_s^2:=\int s(y)^2 f_0(y)\,d\nu(y)<\infty.
$$

Also, as usual,

$$
\int s(y)f_0(y)\,d\nu(y)=0.
$$

### Uniform weighted integrability of second action derivatives

There exists the same $\eta>0$ such that

$$
\sup_{a\in\mathcal A}
\int (1+|s(y)|^{2+\eta})|f_{aa}(y|a)|\,d\nu(y)<\infty.
\tag{M2}
$$

This is stronger than needed but easy to state and verify in standard smooth parametric families. It holds in Gaussian location families with compact action set because $f_{aa}$ is a polynomial times a Gaussian density, uniformly over compact action sets.

### Differentiation identities

For all $a\in\mathcal A$,

$$
\int f_{aa}(y|a)\,d\nu(y)=0.
$$

This follows by differentiating $\int f(y|a)d\nu(y)=1$ twice under the integral sign.

---

## 4. LIC multiplier asymptotics

Define

$$
\theta_\lambda:=\tilde\mu(\lambda)\ell'(\lambda).
$$

The LIC equation is

$$
\int \ell(\lambda+\tilde\mu(\lambda)s(y))s(y)f_0(y)\,d\nu(y)
= c'(a_0).
\tag{LIC}
$$

The key asymptotic claim is

$$
\theta_\lambda\to \theta^*:=\frac{c'(a_0)}{\sigma_s^2},
\tag{A1}
$$

and therefore

$$
\frac{\tilde\mu(\lambda)}{\lambda}	\rightarrow 0.
\tag{A2}
$$

Why (A2) follows from (A1): for CRRA with $p\in(0,1)$,

$$
\lambda\ell'(\lambda)=\frac1\gamma\lambda^p\to\infty.
$$

Thus

$$
\frac{\tilde\mu(\lambda)}{\lambda}
=
\frac{\tilde\mu(\lambda)\ell'(\lambda)}{\lambda\ell'(\lambda)}
=
\frac{\theta_\lambda}{\lambda\ell'(\lambda)}
\to0.
$$

### Sketch proof of (A1)

For fixed bounded $\theta$, set

$$
\mu_\lambda(\theta):=\frac{\theta}{\ell'(\lambda)}.
$$

Then

$$
\frac{\mu_\lambda(\theta)}{\lambda}
=
\frac{\theta}{\lambda\ell'(\lambda)}
\to0.
$$

Use the expansion

$$
\ell(\lambda+\mu_\lambda(\theta)s)
=
\ell(\lambda)+\theta s+o(1)
$$

in $L^1(|s|f_0d\nu)$. The constant term drops out because $E_{a_0}[s]=0$. Therefore

$$
\int \ell(\lambda+\mu_\lambda(\theta)s)s f_0\,d\nu
\to
\theta\int s^2 f_0\,d\nu
=
\theta\sigma_s^2.
$$

Since the left side is increasing in $\theta$ and the true LIC multiplier solves the equation with value $c'(a_0)$, the solution satisfies

$$
\theta_\lambda\to \frac{c'(a_0)}{\sigma_s^2}.
$$

The required $L^1(|s|f_0)$ expansion follows from the same central/tail argument used below for the residual, using the moment $E_{a_0}|s|^{2+\eta}<\infty$.

---

## 5. Residual lemma for signed measures

Let $h_\alpha(y)$ be a family of signed densities indexed by $\alpha$ satisfying

$$
\sup_\alpha \int (1+|s(y)|^{2+\eta})|h_\alpha(y)|\,d\nu(y)<\infty.
\tag{H}
$$

Let $\mu_\lambda$ be any positive sequence such that

$$
\theta_\lambda:=\mu_\lambda\ell'(\lambda)=O(1),
\qquad
\frac{\mu_\lambda}{\lambda}\to0.
\tag{B}
$$

Then

$$
\sup_\alpha
\left|
\int
\left[
\ell(\lambda+\mu_\lambda s(y))
-\ell(\lambda)
-\mu_\lambda\ell'(\lambda)s(y)
\right]
 h_\alpha(y)\,d\nu(y)
\right|
\to0.
\tag{RL}
$$

Taking $h_\alpha(y)=f_{aa}(y|a)$ and $\alpha=a$ gives the desired residual (R).

---

## 6. Proof idea for the residual lemma

Write

$$
r_\lambda(y)
:=
\ell(\lambda+\mu_\lambda s(y))
-\ell(\lambda)
-\mu_\lambda\ell'(\lambda)s(y).
$$

Let

$$
R_\lambda:=\frac{\lambda}{2\mu_\lambda}.
$$

Because $\mu_\lambda/\lambda\to0$, we have $R_\lambda\to\infty$.

Split the integral into a central region and a tail region:

$$
\{|s|\le R_\lambda\}
\quad\text{and}\quad
\{|s|>R_\lambda\}.
$$

### Central region

On $|s|\le R_\lambda$,

$$
\lambda+\mu_\lambda s\ge \lambda/2,
$$

so the limited-liability kink is irrelevant for large $\lambda$ and Taylor's theorem applies:

$$
|r_\lambda(y)|
\le
C\mu_\lambda^2 |\ell''(\lambda/2)|s(y)^2.
$$

For CRRA,

$$
|\ell''(\lambda/2)|\le C\lambda^{p-2}.
$$

Using $\theta_\lambda=\mu_\lambda\ell'(\lambda)=O(1)$ and $\ell'(\lambda)\asymp \lambda^{p-1}$,

$$
\mu_\lambda=O(\lambda^{1-p}).
$$

Therefore

$$
\mu_\lambda^2 |\ell''(\lambda/2)|
=
O(\lambda^{2(1-p)}\lambda^{p-2})
=
O(\lambda^{-p})
\to0.
$$

By (H),

$$
\sup_\alpha
\int_{|s|\le R_\lambda}|r_\lambda(y)||h_\alpha(y)|\,d\nu(y)
\le
O(\lambda^{-p})
\sup_\alpha\int s^2|h_\alpha|\,d\nu
\to0.
$$

### Tail region

On $|s|>R_\lambda$, use crude growth bounds. Since $p\in(0,1)$, there is a constant $C$ such that

$$
|\ell(z)|\le C(1+|z|^p)
$$

for the limited-liability link, and

$$
\mu_\lambda\ell'(\lambda)=O(1).
$$

Thus

$$
|r_\lambda(y)|
\le
C\left[1+\lambda^p+\mu_\lambda^p|s(y)|^p+|s(y)|\right].
$$

The weighted moment bound (H) implies the following uniform tail bounds:

$$
\sup_\alpha\int_{|s|>R}|h_\alpha|\,d\nu
=O(R^{-(2+\eta)}),
$$

$$
\sup_\alpha\int_{|s|>R}|s|^p|h_\alpha|\,d\nu
=O(R^{p-(2+\eta)}),
$$

and

$$
\sup_\alpha\int_{|s|>R}|s||h_\alpha|\,d\nu
=O(R^{-(1+\eta)}).
$$

With

$$
R_\lambda=\frac{\lambda}{2\mu_\lambda}
\asymp \lambda^p,
$$

each tail term vanishes:

$$
\lambda^p R_\lambda^{-(2+\eta)}
=O(\lambda^{-p(1+\eta)})\to0,
$$

$$
\mu_\lambda^p R_\lambda^{p-(2+\eta)}
=O(\lambda^{-p(1+\eta)})\to0,
$$

and

$$
R_\lambda^{-(1+\eta)}\to0.
$$

Hence the tail contribution is uniformly $o(1)$.

Combining the central and tail bounds proves (RL).

---

## 7. Result: primitive sufficient conditions for residual

Under the assumptions in Sections 3 and 4, CRRA with $\gamma\in(1/2,1)$ satisfies the residual condition (R):

$$
\sup_{a\in\mathcal A}
\left|
\int
\left[
 \ell(\lambda+\tilde\mu(\lambda)s(y))
 -\ell(\lambda)
 -\tilde\mu(\lambda)\ell'(\lambda)s(y)
\right]
 f_{aa}(y|a)\,d\nu(y)
\right|
\to0.
$$

The proof is:

1. the LIC equation and the $f_0$ moment condition imply
   $$
   \tilde\mu(\lambda)\ell'(\lambda)
   \to
   \frac{c'(a_0)}{\sigma_s^2};
   $$
2. since $\lambda\ell'(\lambda)\to\infty$, this implies
   $$
   \tilde\mu(\lambda)/\lambda\to0;
   $$
3. the residual lemma applies to $h_a=f_{aa}(\cdot|a)$ using the uniform weighted integrability condition (M2).

---

## 8. Completing asymptotic affinity

Once the residual is established, the curvature expansion becomes

$$
\int \ell(\lambda+\tilde\mu s(y))f_{aa}(y|a)\,d\nu(y)
=
\tilde\mu\ell'(\lambda)
\int s(y)f_{aa}(y|a)\,d\nu(y)
+o(1),
$$

uniformly in $a$.

The constant term drops out because

$$
\int f_{aa}(y|a)\,d\nu(y)=0.
$$

Therefore, if affine-score curvature holds,

$$
\int s(y)f_{aa}(y|a)\,d\nu(y)
\le0
\quad\forall a\in\mathcal A,
$$

then

$$
\sup_{a\in\mathcal A}U_{aa}(V(\cdot|\lambda,\tilde\mu(\lambda)),a)
\le
-\underline c''+o(1),
$$

so concavity holds for large $\lambda$.

---

## 9. Gaussian location verification

For Gaussian location output,

$$
Y\sim N(a,\sigma^2),
\qquad
s(Y)=\frac{Y-a_0}{\sigma^2}.
$$

All score moments exist. Moreover, for compact $\mathcal A$, $f_{aa}(y|a)$ is a quadratic polynomial in $s$ times a Gaussian density, uniformly over $a\in\mathcal A$. Therefore

$$
\sup_{a\in\mathcal A}
\int (1+|s|^{2+\eta})|f_{aa}(y|a)|\,dy<\infty
$$

for every finite $\eta$.

Also,

$$
E_a[s(Y)]=\frac{a-a_0}{\sigma^2},
$$

so

$$
\int s(y)f_{aa}(y|a)\,dy
=
\frac{\partial^2}{\partial a^2}E_a[s(Y)]
=0.
$$

Thus Gaussian location families satisfy both:

1. the primitive moment conditions implying the residual; and
2. affine-score curvature, with equality.

Therefore Gaussian CRRA with $\gamma\in(1/2,1)$ is genuinely covered by the asymptotic-affinity route.

---

## 10. How to use this in the paper

Do not state the residual condition as a main assumption. Instead, either:

### Option A: Keep the case as an appendix extension

Main text says:

> CRRA with $\gamma\in(1/2,1)$ can also be handled by an asymptotic-affinity argument under primitive score-moment and affine-score-curvature conditions. Gaussian location families satisfy these conditions.

Appendix proves the result using the argument in this note.

### Option B: Drop the case from the main theorem

If the paper is already too complicated, keep the core theorem focused on:

1. log utility;
2. CARA;
3. CRRA with $\gamma>1$.

Then mention the asymptotic-affinity route as a possible extension or omit it entirely.

The key point is that the residual is not magic: it follows from standard Taylor expansion plus uniform weighted moment bounds. But it is technical enough that it should not be a headline assumption.
