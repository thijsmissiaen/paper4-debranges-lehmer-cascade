# Shifted Hermite-Biehler Kernel Identities, Local Multiplicity Jets, and Truncated Prime-Power Multipliers for Riemann's $\Xi$-Function

**DOI (Zenodo / CERN):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23641702.svg)](https://doi.org/10.5281/zenodo.23641702) [`10.5281/zenodo.23641702`](https://doi.org/10.5281/zenodo.23641702)  
**Author:** Thijs Missiaen (Independent Researcher, Belgium; ORCID: [0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810))  
**Series:** Analytic and Geometric Properties of the Transverse Wronskian of Riemann's Xi-Function (Paper IV)  
**MSC 2020:** Primary 11M26; Secondary 46E22, 30D15, 11N36, 41A60  
**License:** MIT

> [!IMPORTANT]
> **Mathematical Scope & Prior-Art Context:** For the shifted entire family $E _ \alpha(z) = \Xi(z) - \mathrm{i}\alpha\Xi'(z)$, the diagonal Nevanlinna–Pick kernel $K _ \alpha(t-\mathrm{i}d, t-\mathrm{i}d) = \frac{\alpha}{\pi}\widetilde{W}(d,t)$ is an exact scalar multiple of $\widetilde{W}(d,t)$ unconditionally, whereas positive semidefiniteness of its Cauchy–Pick matrices is equivalent to the Riemann Hypothesis by Levinson (1974), de Branges (1968), and Lagarias (2005, 2006). Likewise, the multiplicity jet $c _ {2m-1}(f; t_0) = m L_m(f; t_0) = \frac{m}{(m!)^2}(f^{(m)}(t_0))^2 > 0$ builds on Jensen (1913) and Csordas and Varga (1990), and the bounded residual series $\zeta _ X(s) = E _ X(s)\zeta(s)$ on $\mathrm{Re}(s) > 1$ is subject to explicit Schwarz-reflection, functional-equation, and strip-residue obstructions on $0 < d \le \frac{1}{2}$.

## Overview and Main Results

With $\Xi(z) := \xi(\frac{1}{2}+\mathrm{i}z) = 4\int _ {-\infty} ^ {\infty}\Phi(u)\mathrm{e}^{-2\mathrm{i}z u}\,\mathrm{d}u$ for real $t$ and $d \ge 0$, the transverse Wronskian and its parity-reduced quotient are normalized canonically across the series by

$$W(d,t) := \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2 = \mathrm{Im}\big(\overline{\Xi(t-\mathrm{i}d)}\,\Xi'(t-\mathrm{i}d)\big) = 16\int_0^\infty y\sinh(2d y)\,I _ {\mathrm{Wig}}(y,t)\,\mathrm{d}y,$$

$$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = 32\int_0^\infty y^2\,\mathrm{sinc} _ {\mathrm{hyp}}(2d y)\,I _ {\mathrm{Wig}}(y,t)\,\mathrm{d}y,$$

where $I _ {\mathrm{Wig}}(y,t) = \int _ {-\infty} ^ {\infty}\Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})\cos(2t x)\,\mathrm{d}x$.

1. **Shifted Hermite–Biehler Kernels and Equivalence to RH (Section 2):** For $\alpha > 0$, define $E _ \alpha(z) := \Xi(z) - \mathrm{i}\alpha\Xi'(z)$ and $E _ \alpha ^ \sharp(z) := \Xi(z) + \mathrm{i}\alpha\Xi'(z)$. The two-point Nevanlinna–Pick kernel on the lower half-plane satisfies

$$K _ \alpha(z_1, z_2) := \frac{E _ \alpha(z_1)\overline{E _ \alpha(z_2)} - E _ \alpha ^ \sharp(z_1)\overline{E _ \alpha ^ \sharp(z_2)}}{2\pi\mathrm{i}(z_1 - \bar{z}_2)} = \frac{\alpha}{\pi}\,\frac{\Xi(z_1)\overline{\Xi'(z_2)} - \Xi'(z_1)\overline{\Xi(z_2)}}{z_1 - \bar{z}_2},$$

with diagonal specialization $K _ \alpha(t-\mathrm{i}d, t-\mathrm{i}d) = \frac{\alpha}{\pi}\widetilde{W}(d,t)$ and critical-line limit $\frac{\alpha}{\pi}L_1(\Xi; t)$. By Proposition 2.4, Cauchy–Pick positive semidefiniteness $[K _ \alpha(z_j, z_k)] _ {j,k=1} ^ M \succeq 0$ holds if and only if $\lvert E _ \alpha ^ \sharp(w)\rvert > \lvert E _ \alpha(w)\rvert$ on the upper half-plane, which holds if and only if the Riemann Hypothesis is true.

2. **Jensen–Laguerre Multiplicity Jets and Local Quadratic Bowls (Section 3):** Expanding $\widetilde{W}_f(d,t) = \sum _ {n=1} ^ {\infty} c _ {2n-1}(f; t)\,d^{2n-2}$ for any real-entire function $f(z)$ gives $c _ {2n-1}(f; t) = n L_n(f; t)$ following Jensen (1913) and Csordas and Varga (1990). At a real zero $t_0$ of multiplicity $m$, $c_1(f; t_0) = \dots = c _ {2m-3}(f; t_0) = 0$ and

$$c _ {2m-1}(f; t_0) = \frac{m}{(m!)^2}\big(f^{(m)}(t_0)\big)^2 > 0.$$

At any non-degenerate critical point $\tau _ {\star}$ with $f'(\tau _ {\star}) = 0$ and $f''(\tau _ {\star}) \neq 0$, setting $\mu := -2f(\tau _ {\star})/f''(\tau _ {\star})$ yields the unified local jet

$$\widetilde{W}_f(d, \tau _ {\star} + \tau) = \frac{1}{2}\big(f''(\tau _ {\star})\big)^2\big(\mu + \tau^2 + d^2\big) + \mathcal{O}\Big(\big(\lvert\mu\rvert^{1/2} + \lvert\tau\rvert + \lvert d\rvert\big)^3\Big),$$

which is strictly positive when $\mu = +\delta^2 > 0$ (for a real Lehmer pair such as $\gamma _ {6709}, \gamma _ {6710}$ near $\tau _ {\star} \approx 7005.0817$), vanishes only at the origin when $\mu = 0$ ($m = 2$), and is negative on the disk $\tau^2 + d^2 < \delta_0^2$ when $\mu = -\delta_0^2 < 0$ (off-critical conjugate pair). The first two positive roots of $\Xi'(t)=0$ lie at $\tau_1 \approx 15.5857085898$ with $\frac{1}{2}(\Xi''(\tau_1))^2 \approx 1.00779672 \times 10^{-7}$ and $\tau_2 \approx 22.0979772804$ with $\frac{1}{2}(\Xi''(\tau_2))^2 \approx 3.37096135 \times 10^{-11}$.

3. **Leibniz–Wronskian Multipliers, Prime-Power Recursion, and Explicit Obstructions (Sections 4–6):** For $\Xi _ {\mathcal{M}}(z) = \mathcal{M}(z)\Xi(z)$, $W _ {\mathcal{M}}(d,t) = \lvert\mathcal{M}(t-\mathrm{i}d)\rvert^2 W(d,t) + \lvert\Xi(t-\mathrm{i}d)\rvert^2 W _ {1,\mathcal{M}}(d,t)$. For $E _ X(s) := \exp(-\sum _ {2 \le k \le X}\frac{\Lambda(k)}{\ln k}k^{-s})$ on $\mathrm{Re}(s) > 1$, the residual series $\zeta _ X(s) = E _ X(s)\zeta(s) = 1 + \sum _ {n > X} b _ n ^ {\mathcal{E}}(X)n^{-s}$ satisfies $b _ n ^ {\mathcal{E}}(X) = 0$ for $2 \le n \le X$, $0 \le b _ n ^ {\mathcal{E}}(X) \le 1$ for all $n \ge 1$, and $b _ n ^ {\mathcal{E}}(X) = \Lambda(n)/\ln n$ on $X < n < (\lfloor X\rfloor + 1)^2$. Three structural obstructions prevent unconditional off-line positivity:
   - Unsymmetrized $E _ X(\frac{1}{2}+\mathrm{i}z)$ fails Schwarz reflection on $d = 0$, creating a $1/d$ pole in $d^{-1}W _ {1,E _ X}(d,t)$, whereas conjugate-symmetrizing to $\mathcal{M} _ X ^ {\mathrm{sym}}(z) = E _ X(\frac{1}{2}+\mathrm{i}z)E _ X(\frac{1}{2}-\mathrm{i}z)$ breaks the functional equation of $\zeta _ X(s)$ and yields a zero-mean sign-indefinite intrinsic term $\widetilde{W} _ {1,\mathcal{M} _ X ^ {\mathrm{sym}}}(d,t)$.
   - Convolving $\frac{W(d,t+u)}{\lvert\Xi(t+u-\mathrm{i}d)\rvert^2}$ with the Fejér kernel $K _ U(u) = \frac{U}{2\pi}\mathrm{sinc}^2(U u/2)$ ($U = \ln X$) truncates prime-power harmonics above $X$ unconditionally only on $d > \frac{1}{2}$ ($\sigma > 1$, outside the critical strip); inside $0 < d < \frac{1}{2}$, contour shifting crosses both $s = 1$ with residue $+2\pi\,\mathrm{Re}\,K _ U(t+\mathrm{i}(\frac{1}{2}-d))$ and any zeros with $d _ \rho > d$ with residue $-2\pi m _ \rho\,\mathrm{Re}\,K _ U((t-\gamma _ \rho)+\mathrm{i}(d _ \rho-d))$.
   - On $d = 0$, $\Xi(t) = -E_0(t)Z(t)$ with $E_0(t) = \frac{1}{2}(t^2+\frac{1}{4})\pi^{-1/4}\lvert\Gamma(\frac{1}{4}+\mathrm{i}\frac{t}{2})\rvert$ and $c_1(t) = E_0(t)^2 L_1(Z; t) + Z(t)^2 L_1(E_0; t)$, while Gabcke's (1979) remainder bounds hold strictly for unmodified $Z(t)$ ($b_n \equiv 1$, $d = 0$, $t \ge 200$).

## Summary of Quantitative Diagnostics

| Verification Tier | Domain / Scope | Key Quantity / Diagnostic | Verified Result |
| :--- | :--- | :--- | :--- |
| **Tier 1A: Canonical Wigner** | $(d,t) \in \lbrace(0.15,6), (0.35,\gamma_1), (0.25,18.5)\rbrace$ | Max relative error of $W = 16\int y\sinh(2d y)I _ {\mathrm{Wig}}\,\mathrm{d}y$ | $\le 1.05 \times 10^{-57}$ (`mp.dps = 65`) |
| **Tier 1B–1D: Kernel & Counterexample** | $z_1,\dots,z_5 \in \mathbb{C} _ -$ and $F _ {\mathrm{off}}(z)$ | `mpmath.iv` width for $K_1(\Xi)$ vs $\lambda _ {\min}(\mathbf{G}_1(F _ {\mathrm{off}}))$ | Width $\le 2.01 \times 10^{-39}$; $\lambda _ {\min} < 0$ on $F _ {\mathrm{off}}$ |
| **Tier 2: Multiplicity & Bowls** | $m = 1,\dots,5$; $\tau_1 \approx 15.58570859$ | $550$-digit coalescence floor $c _ {3,\mathrm{floor}} = \frac{1}{2}(\Xi''(\tau_1))^2$ | $1.0077967238 \times 10^{-7}$ down to $\Delta\gamma = 10^{-250}$ |
| **Tier 3: Multipliers & Residues** | $n \le 5000$, $X \le 30$; $d \in \lbrace 0.15, 0.85\rbrace$ | $b _ n ^ {\mathcal{E}}(X) \in [0,1]$ and strip zero-residue jump | Range $[0,1]$ exact; jump error $2.78 \times 10^{-10}$ |

## Repository Structure

| Path | Description |
| :--- | :--- |
| `paper/paper4_debranges_lehmer_cascade.tex` | AMS-LaTeX source manuscript |
| `paper/paper4_debranges_lehmer_cascade.pdf` | Compiled PDF manuscript (14 pages) |
| `paper/figures/` | 300-DPI vector (`.pdf`) and raster (`.png`) figures |
| `scripts/verify_paper4_debranges_lehmer_cascade.py` | Standalone 50-digit and 550-digit verification suite |
| `.agents/skills/debranges-kernels-lehmer-cascade/SKILL.md` | AI agent skill specification and guardrails |
| `AGENT_SKILL_MANIFEST.json` | Machine-readable artifact and skill manifest |

## Reproducibility

```bash
python scripts/verify_paper4_debranges_lehmer_cascade.py
```
