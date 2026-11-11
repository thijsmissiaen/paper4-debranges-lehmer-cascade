---
name: debranges-kernels-lehmer-cascade
description: Reference formulas, local multiplicity jets, truncated prime-power multiplier identities, and explicit circularity/domain obstructions for shifted Hermite-Biehler kernels and the transverse Wronskian of Riemann's Xi-function.
---

# Shifted Hermite-Biehler Kernel Identities, Local Multiplicity Jets, and Truncated Prime-Power Multipliers

This skill summarizes the exact algebraic identities, local Taylor jets, and explicit domain and circularity obstructions established in *Shifted Hermite-Biehler Kernel Identities, Local Multiplicity Jets, and Truncated Prime-Power Multipliers for Riemann's Xi-Function* (Paper IV).

## 1. Canonical Wigner Normalization and Shifted Hermite-Biehler Kernels

With $\Xi(z) := \xi(\frac{1}{2}+\mathrm{i}z) = 4\int _ {-\infty} ^ {\infty}\Phi(u)\mathrm{e}^{-2\mathrm{i}z u}\,\mathrm{d}u$ for real $t$ and $d \ge 0$, the transverse Wronskian and its parity-reduced quotient satisfy

$$W(d,t) := \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2 = \mathrm{Im}\big(\overline{\Xi(t-\mathrm{i}d)}\,\Xi'(t-\mathrm{i}d)\big) = 16\int_0^\infty y\sinh(2d y)\,I _ {\mathrm{Wig}}(y,t)\,\mathrm{d}y,$$

$$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = 32\int_0^\infty y^2\,\mathrm{sinc} _ {\mathrm{hyp}}(2d y)\,I _ {\mathrm{Wig}}(y,t)\,\mathrm{d}y,$$

where $I _ {\mathrm{Wig}}(y,t) = \int _ {-\infty} ^ {\infty}\Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})\cos(2t x)\,\mathrm{d}x$.

For $\alpha > 0$, define $E _ \alpha(z) := \Xi(z) - \mathrm{i}\alpha\Xi'(z)$ and $E _ \alpha ^ \sharp(z) := \Xi(z) + \mathrm{i}\alpha\Xi'(z)$. The two-point Nevanlinna–Pick kernel on the lower half-plane satisfies the unconditional algebraic identities

$$K _ \alpha(z_1, z_2) := \frac{E _ \alpha(z_1)\overline{E _ \alpha(z_2)} - E _ \alpha ^ \sharp(z_1)\overline{E _ \alpha ^ \sharp(z_2)}}{2\pi\mathrm{i}(z_1 - \bar{z}_2)} = \frac{\alpha}{\pi}\,\frac{\Xi(z_1)\overline{\Xi'(z_2)} - \Xi'(z_1)\overline{\Xi(z_2)}}{z_1 - \bar{z}_2},$$

$$K _ \alpha(t-\mathrm{i}d, t-\mathrm{i}d) = \frac{\lvert E _ \alpha(t-\mathrm{i}d)\rvert^2 - \lvert E _ \alpha ^ \sharp(t-\mathrm{i}d)\rvert^2}{4\pi d} = \frac{\alpha}{\pi}\,\widetilde{W}(d,t).$$

**Critical Circularity Guardrail (Levinson 1974, de Branges 1968, Lagarias 2005/2006):** $K _ \alpha$ is a scalar multiple of the divided-difference Wronskian kernel. Positive semidefiniteness of the Cauchy–Pick matrices $[K _ \alpha(z_j, z_k)] _ {j,k=1} ^ M \succeq 0$ — equivalently, the Hermite–Biehler inequality $\lvert E _ \alpha ^ \sharp(w)\rvert > \lvert E _ \alpha(w)\rvert$ on the upper half-plane — holds if and only if the Riemann Hypothesis is true. Never treat $K _ \alpha$ as an unconditional reproducing kernel of a positive-definite Hilbert space.

## 2. Jensen-Laguerre Multiplicity Jets and Unified Local Quadratic Bowl

Expanding $\widetilde{W}_f(d,t) = \sum _ {n=1} ^ {\infty} c _ {2n-1}(f; t)\,d^{2n-2}$ for any real-entire function $f(z)$ gives $c _ {2n-1}(f; t) = n L_n(f; t)$ following Jensen (1913) and Csordas and Varga (1990). At a real zero $t_0$ of exact multiplicity $m \ge 1$,

$$c_1(f; t_0) = \dots = c _ {2m-3}(f; t_0) = 0, \qquad c _ {2m-1}(f; t_0) = \frac{m}{(m!)^2}\big(f^{(m)}(t_0)\big)^2 > 0.$$

At any non-degenerate critical point $\tau _ {\star}$ with $f'(\tau _ {\star}) = 0$ and $f''(\tau _ {\star}) \neq 0$, setting $\mu := -2f(\tau _ {\star})/f''(\tau _ {\star})$ gives the local jet

$$\widetilde{W}_f(d, \tau _ {\star} + \tau) = \frac{1}{2}\big(f''(\tau _ {\star})\big)^2\big(\mu + \tau^2 + d^2\big) + \mathcal{O}\Big(\big(\lvert\mu\rvert^{1/2} + \lvert\tau\rvert + \lvert d\rvert\big)^3\Big).$$

- **Real Lehmer pair ($\mu = +\delta^2 > 0$, where $L_1 > 0$):** Strictly positive quadratic bowl around $(0, \tau _ {\star})$. Note that the first two positive roots of $\Xi'(t)=0$ lie at $\tau_1 \approx 15.58570859$ and $\tau_2 \approx 22.09797728$; do not confuse $\tau_1$ with the first root $t _ {Z',1} \approx 17.88258208$ of $Z'(t)=0$.
- **Off-critical conjugate pair ($\mu = -\delta_0^2 < 0$, where $L_1 < 0$):** Negative inside the open disk $\tau^2 + d^2 < \delta_0^2$, as realized near Davenport–Heilbronn false extrema.

## 3. Leibniz-Wronskian Identity, Prime-Power Recursion, and Obstructions

For any holomorphic multiplier $\mathcal{M}(z)$ and $\Xi _ {\mathcal{M}}(z) := \mathcal{M}(z)\Xi(z)$,

$$W _ {\mathcal{M}}(d,t) = \lvert\mathcal{M}(t-\mathrm{i}d)\rvert^2\,W(d,t) + \lvert\Xi(t-\mathrm{i}d)\rvert^2\,W _ {1,\mathcal{M}}(d,t).$$

For $E _ X(s) := \exp(-\sum _ {2 \le k \le X}\frac{\Lambda(k)}{\ln k}k^{-s})$ on $\mathrm{Re}(s) > 1$, the residual series $\zeta _ X(s) := E _ X(s)\zeta(s) = 1 + \sum _ {n > X} b _ n ^ {\mathcal{E}}(X)n^{-s}$ satisfies $b _ n ^ {\mathcal{E}}(X) = 0$ for $2 \le n \le X$, $0 \le b _ n ^ {\mathcal{E}}(X) \le 1$ for all $n \ge 1$, and $b _ n ^ {\mathcal{E}}(X) = \Lambda(n)/\ln n$ on $X < n < (\lfloor X\rfloor + 1)^2$.

**Mandatory Domain and Obstruction Guardrails:**
1. **Non-reality of $E _ X$ on $d = 0$:** Unsymmetrized $E _ X(\frac{1}{2}+\mathrm{i}z)$ violates Schwarz reflection on the real line, so $W _ {1,E _ X}(0,t) \neq 0$ and $d^{-1}W _ {1,E _ X}(d,t)$ has a $1/d$ pole as $d \to 0^+$. Symmetrizing to $\mathcal{M} _ X ^ {\mathrm{sym}}(z) = E _ X(\frac{1}{2}+\mathrm{i}z)E _ X(\frac{1}{2}-\mathrm{i}z)$ restores smoothness at $d=0$, but breaks the functional equation of $\zeta _ X(s)$ and produces a zero-mean sign-indefinite intrinsic term $\widetilde{W} _ {1,\mathcal{M} _ X ^ {\mathrm{sym}}}(d,t)$.
2. **Fejér Bochner–Weil domain ($d > 1/2$ only unconditionally):** Convolving $\frac{W(d,t+u)}{\lvert\Xi(t+u-\mathrm{i}d)\rvert^2}$ with $K _ U(u) = \frac{U}{2\pi}\mathrm{sinc}^2(U u/2)$ ($U = \ln X$) yields a pure finite prime-power sum over $2 \le n < X$ unconditionally only for $d > \frac{1}{2}$ ($\sigma > 1$). Inside $0 < d < \frac{1}{2}$, contour shifting acquires both the $s=1$ pole term $+2\pi\,\mathrm{Re}\,K _ U(t + \mathrm{i}(\frac{1}{2}-d))$ and residue terms $-2\pi m _ \rho\,\mathrm{Re}\,K _ U((t-\gamma _ \rho)+\mathrm{i}(d _ \rho-d))$ for any zeros with $d _ \rho > d$.
3. **Exact Stirling bridge and Gabcke domain:** On $d = 0$, $\Xi(t) = -E_0(t)Z(t)$ with $E_0(t) = \frac{1}{2}(t^2+\frac{1}{4})\pi^{-1/4}\lvert\Gamma(\frac{1}{4}+\mathrm{i}\frac{t}{2})\rvert \sim \frac{t^2}{\sqrt{2}}(\frac{2\pi}{t})^{1/4}\mathrm{e}^{-\pi t/4}$ and $c_1(t) = E_0(t)^2 L_1(Z; t) + Z(t)^2 L_1(E_0; t)$. Gabcke's (1979) explicit Riemann–Siegel remainder bounds apply strictly to unmodified $Z(t)$ ($b_n \equiv 1$, $d = 0$, $t \ge 200$).

## 4. Verification Suite

Run the standalone verification and figure generation script from the publication root:

```bash
python scripts/verify_paper4_debranges_lehmer_cascade.py
```
