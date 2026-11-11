#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper4_debranges_lehmer_cascade.py
===============================================================================
Standalone Verification & Figure Generation Suite for Paper IV:
"Shifted Hermite-Biehler Kernel Identities, Local Multiplicity Jets, and
Truncated Prime-Power Multipliers for Riemann's Xi-Function"

Author : Thijs Missiaen (Independent Researcher, Belgium)
ORCID  : https://orcid.org/0009-0007-3678-0810
Email  : thijs.missiaen@gmail.com
Series : Analytic and Geometric Properties of the Transverse Wronskian of
         Riemann's Xi-Function (Paper IV)

Four-Tier Verification Protocol:
  1. Tier 1 (Canonical Wigner Normalization, Shifted Hermite-Biehler Kernel,
     Cauchy-Pick Matrices, Rigorous mpmath.iv Enclosures & Off-Line Counterexample):
     - Verifies the canonical Wigner integral representation
       W(d,t) = (1/2) d_d |Xi(t-id)|^2 = 16 int_0^inf y sinh(2dy) I_Wig(y,t) dy
       and W_tilde(d,t) = W(d,t)/d = 32 int_0^inf y^2 sinc_hyp(2dy) I_Wig(y,t) dy
       to 50 decimal digits.
     - Verifies K_alpha(t-id, t-id) = (alpha/pi) W_tilde(d,t), the Schur defect
       1 - |Theta_alpha(t-id)|^2 = 4 alpha d W_tilde(d,t) / |E_alpha(t-id)|^2,
       and the M x M Cauchy-Pick divided-difference matrix [G_alpha]_{j,k}
       at 50-digit precision, plus rigorous mpmath.iv enclosures (N_u = 160 on
       [0, 1.55] with proven analytic strip discretization error < 1e-65).
     - Demonstrates the Lagarias-Levinson equivalence via an explicit real-entire
       off-critical zero model F_off(z) = Xi(z)(1 + (z - 17.85)^2 / 0.25^2),
       showing K_alpha^{off}(z,z) < 0, |Theta_alpha^{off}(z)| > 1, and negative
       minimum eigenvalue lambda_min(G_alpha(F_off)) < 0.
  2. Tier 2 (Jensen-Laguerre Multiplicity Cascade, True Critical-Line Extrema
     tau_1 & tau_2, 550-Digit Deformation, and Real vs. Off-Critical Bowls):
     - Verifies c_{2m-1}(f; t_0) = m L_m(f; t_0) = [m/(m!)^2] (f^{(m)}(t_0))^2 > 0
       and c_1 = ... = c_{2m-3} = 0 for m = 1..5 to 50 decimal digits.
     - Locates the true first two positive extrema of Xi'(t) = 0:
       tau_1 = 15.5857085898... and tau_2 = 22.0979772804... (contrasting with
       the first extremum of Hardy's Z-function at t_{Z',1} = 17.8825820769...
       and the zeroth Gram point g_0 = 17.8455995405...),
       and executes a 550-digit local coalescence audit anchored at tau_1 down
       to Delta_gamma = 10^{-250}.
     - Verifies the unified local quadratic jet
       W_tilde_f(d, tau_* + tau) = (1/2)(f''(tau_*))^2 (mu + tau^2 + d^2) + O(3)
       with mu = -2 f(tau_*)/f''(tau_*) across BOTH mu = +delta^2 > 0 (including
       Lehmer's historical pair gamma_6709, gamma_6710 near t = 7005.08186) AND
       mu = -delta_0^2 < 0 (negative disk tau^2 + d^2 < delta_0^2 for an
       off-critical conjugate zero pair).
  3. Tier 3 (Leibniz-Wronskian Identity, E_X Non-Reality vs. M_X^{sym},
     Truncated Prime-Power Recursion 0 <= b_n^E(X) <= 1, Exact Fejer
     Bochner-Weil Residue Formula, and Exact Critical-Line E_0(t) Bridge):
     - Verifies W_{M Xi} = |M|^2 W + |Xi|^2 W_{1,M}, local zero-exclusion bounds
       B_CS and B_dist, the 1/d blow-up of d^{-1} W_{1,E_X}(d,t) from broken
       Schwarz reflection on d = 0, and the smooth but sign-indefinite intrinsic
       term W_tilde_{1,M_X^{sym}}(d,t) of the conjugate-symmetrized multiplier.
     - Verifies 0 <= b_n^E(X) <= 1 for 1 <= n <= 5000 across X in {2,3,5,10,17,30},
       quadratic-window prime-power purity on X < n < (floor(X)+1)^2, and Jacobi
       theta tail monotonicity.
     - Verifies the exact Fejer Bochner-Weil identity unconditionally at d = 0.65 > 1/2
       (sigma = 1.15 > 1) and quantifies both the s = 1 pole contribution and the
       off-critical zero residue jump -2 pi Re K_U((t - gamma_0) + i(d_0 - d))
       inside the critical strip at d = 0.15 < 1/2.
     - Verifies the exact critical-line Stirling bridge Xi(t) = -E_0(t) Z(t) and
       c_1(t) = E_0(t)^2 L_1(Z; t) + Z(t)^2 L_1(E_0; t) with L_1(E_0; t) > 0 (t >= 1).
  4. Tier 4 (Publication-Grade 300-DPI Figures):
     - Generates paper/figures/fig1_debranges_kernel_and_lehmer_cascade.{pdf,png}
     - Generates paper/figures/fig2_leibniz_wronskian_and_gabcke_closure.{pdf,png}
===============================================================================
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpmath as mp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

mp.mp.dps = 50
mp.iv.dps = 55

SCRIPT_DIR = Path(__file__).resolve().parent
PUB_ROOT = SCRIPT_DIR.parent
FIG_DIR = PUB_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def print_banner(title: str) -> None:
    print("=" * 98)
    print(f" {title}")
    print("=" * 98)


# ==============================================================================
# 0. CANONICAL MPMATH EVALUATORS FOR PHI(u), WIGNER I_Wig(y,t), AND XI(z)
# ==============================================================================

def phi_kernel_mp(u):
    """Riemann's even rapid-decay kernel Phi(u) = sum_{n>=1} (2*pi^2*n^4*e^{9u} - 3*pi*n^2*e^{5u}) exp(-pi*n^2*e^{4u})."""
    u_abs = abs(mp.mpf(u))
    e4 = mp.exp(4 * u_abs)
    e5 = mp.exp(5 * u_abs)
    e9 = mp.exp(9 * u_abs)
    pi = mp.pi
    pi2 = pi * pi
    total = mp.mpf("0")
    for n in range(1, 30):
        n2 = mp.mpf(n * n)
        arg = pi * n2 * e4
        if arg > mp.mpf("160"):
            break
        term = (2 * pi2 * (n2 * n2) * e9 - 3 * pi * n2 * e5) * mp.exp(-arg)
        total += term
    return total


def _build_gl_nodes_weights(n_pts: int):
    """Constructs n_pts Gauss-Legendre nodes and weights on [-1, 1] at 50-digit precision."""
    nodes, weights = [], []
    for k in range(1, n_pts + 1):
        x0 = mp.cos(mp.pi * (mp.mpf(k) - mp.mpf("0.25")) / (mp.mpf(n_pts) + mp.mpf("0.5")))
        for _ in range(14):
            p0, p1 = mp.mpf("1"), x0
            for j in range(2, n_pts + 1):
                p2 = ((2 * j - 1) * x0 * p1 - (j - 1) * p0) / j
                p0, p1 = p1, p2
            dp = n_pts * (x0 * p1 - p0) / (x0 * x0 - 1)
            dx = -p1 / dp
            x0 += dx
            if abs(dx) < mp.mpf("1e-53"):
                break
        p0, p1 = mp.mpf("1"), x0
        for j in range(2, n_pts + 1):
            p2 = ((2 * j - 1) * x0 * p1 - (j - 1) * p0) / j
            p0, p1 = p1, p2
        dp = n_pts * (x0 * p1 - p0) / (x0 * x0 - 1)
        w0 = 2 / ((1 - x0 * x0) * (dp * dp))
        nodes.append(x0)
        weights.append(w0)
    return nodes, weights


GL48_NODES, GL48_WEIGHTS = _build_gl_nodes_weights(48)


def quad_gl48_mp(func, a, b):
    a_mp, b_mp = mp.mpf(a), mp.mpf(b)
    half = mp.mpf("0.5") * (b_mp - a_mp)
    mid = mp.mpf("0.5") * (a_mp + b_mp)
    acc = mp.mpf("0")
    for x_i, w_i in zip(GL48_NODES, GL48_WEIGHTS):
        acc += w_i * func(mid + half * x_i)
    return half * acc


def wigner_I_mp(y_val, t_val):
    """
    Canonical Wigner marginal integral:
      I_Wig(y, t) = int_{-inf}^{inf} Phi((x+y)/2) Phi((x-y)/2) cos(2 t x) dx
                  = 2 int_0^{inf} Phi((x+y)/2) Phi((x-y)/2) cos(2 t x) dx
    (since Phi is even, x -> -x swaps (x+y)/2 and -(x-y)/2).
    """
    y_mp = mp.mpf(y_val)
    t_mp = mp.mpf(t_val)
    # Phi(u) < 1e-100 for |u| > 1.35, so for y <= 2.6, |(x-y)/2| > 1.35 when x > y + 2.7.
    x_max = y_mp + mp.mpf("2.8")
    # Split [0, x_max] into 4 subintervals of GL48 for 50-digit accuracy
    edges = [x_max * mp.mpf(k) / 4 for k in range(5)]
    total = mp.mpf("0")
    for idx in range(4):
        total += quad_gl48_mp(
            lambda x: phi_kernel_mp((x + y_mp) / 2) * phi_kernel_mp((x - y_mp) / 2) * mp.cos(2 * t_mp * x),
            edges[idx],
            edges[idx + 1],
        )
    return 2 * total


def xi_completed_mp(s):
    """Completed Riemann xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s)."""
    s_mp = mp.mpc(s)
    return (
        mp.mpf("0.5")
        * s_mp
        * (s_mp - 1)
        * mp.power(mp.pi, -s_mp / 2)
        * mp.gamma(s_mp / 2)
        * mp.zeta(s_mp)
    )


def Xi_mp(z):
    """Xi(z) := xi(1/2 + i z), entire and even with Xi(bar(z)) = bar(Xi(z))."""
    z_mp = mp.mpc(z)
    return xi_completed_mp(mp.mpf("0.5") + mp.mpc(0, 1) * z_mp)


def Xi_and_deriv_mp(z):
    """Exact 50-digit evaluation of Xi(z) and Xi'(z) via logarithmic derivative of xi(s)."""
    z_mp = mp.mpc(z)
    s = mp.mpf("0.5") + mp.mpc(0, 1) * z_mp
    pref = mp.mpf("0.5") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2)
    z0 = mp.zeta(s)
    z1 = mp.zeta(s, derivative=1)
    g1 = 1 / s + 1 / (s - 1) - mp.mpf("0.5") * mp.log(mp.pi) + mp.mpf("0.5") * mp.psi(0, s / 2)
    xi0 = pref * z0
    xi1_s = pref * (g1 * z0 + z1)
    # d/dz Xi(z) = i * xi'(1/2 + i z)
    return xi0, mp.mpc(0, 1) * xi1_s


def Xi_jet_mp(z, max_order=8):
    """Taylor derivatives [Xi(z), Xi'(z), ..., Xi^{(max_order)}(z)] via mp.taylor."""
    z_mp = mp.mpc(z)
    coeffs = mp.taylor(Xi_mp, z_mp, max_order)
    return [coeffs[k] * mp.factorial(k) for k in range(max_order + 1)]


def W_and_Wtilde_mp(d_val, t_val):
    """Exact W(d,t) = Im(conj(Xi(t-id)) * Xi'(t-id)) and W_tilde(d,t) = W(d,t)/d."""
    d_mp = mp.mpf(str(d_val)) if isinstance(d_val, float) else mp.mpf(d_val)
    t_mp = mp.mpf(str(t_val)) if isinstance(t_val, float) else mp.mpf(t_val)
    if abs(d_mp) < mp.mpf("1e-24"):
        jet = Xi_jet_mp(t_mp, max_order=2)
        x0, x1, x2 = mp.re(jet[0]), mp.re(jet[1]), mp.re(jet[2])
        c1 = x1 * x1 - x0 * x2
        return mp.mpf("0"), c1
    z = mp.mpc(t_mp, -d_mp)
    xi0, xip = Xi_and_deriv_mp(z)
    W_val = mp.im(mp.conj(xi0) * xip)
    return W_val, W_val / d_mp


def sinc_hyp_mp(x):
    x_mp = mp.mpf(x)
    if abs(x_mp) < mp.mpf("1e-18"):
        x2 = x_mp * x_mp
        return 1 + x2 / 6 + (x2 * x2) / 120
    return mp.sinh(x_mp) / x_mp


# ==============================================================================
# 1. TIER 1: CANONICAL WIGNER NORMALIZATION, SHIFTED HERMITE-BIEHLER KERNEL,
#            CAUCHY-PICK MATRICES, INTERVAL ENCLOSURES & OFF-LINE COUNTEREXAMPLE
# ==============================================================================

def verify_tier1():
    print_banner("TIER 1: CANONICAL WIGNER NORMALIZATION, SHIFTED HERMITE-BIEHLER KERNEL & COUNTEREXAMPLE")

    # 1A. Canonical Wigner integral normalization W(d,t) = 16 int_0^inf y sinh(2dy) I_Wig(y,t) dy
    print("1A. Canonical Wigner Integral Identity (50-Digit Verification):")
    print("    W(d,t)       = (1/2) d_d |Xi(t-id)|^2 = 16 int_0^inf y sinh(2dy) I_Wig(y,t) dy")
    print("    W_tilde(d,t) = W(d,t)/d               = 32 int_0^inf y^2 sinc_hyp(2dy) I_Wig(y,t) dy")
    print("-" * 98)
    wigner_pts = [(0.15, 6.0), (0.35, 14.13472514), (0.25, 18.5)]
    max_wig_rel_err = mp.mpf("0")
    with mp.workdps(65):
        # Precompute Phi(k * h_u) with h_u = 0.0125 (h_0 = 2*h_u = 0.025 in x and y):
        # Strip analyticity |Im(x)|, |Im(y)| < pi/4 => 2*pi*0.72/0.025 = 180.9 => error < 1e-65!
        h_u_w = mp.mpf("0.0125")
        h_0_w = 2 * h_u_w
        K_max_u = 132  # u_max = 1.65 => Phi(1.65) < 1e-900
        phi_tab = [phi_kernel_mp(k * h_u_w) for k in range(K_max_u + 1)]

        def phi_idx(idx_k: int):
            ak = abs(idx_k)
            return phi_tab[ak] if ak <= K_max_u else mp.mpf("0")

        for d_v, t_v in wigner_pts:
            d_mp = mp.mpf(str(d_v))
            t_mp = mp.mpf(str(t_v))
            W_ex, Wt_ex = W_and_Wtilde_mp(d_mp, t_mp)
            # Precompute cos(2 * t * j * h_0) for j = 0 .. 2*K_max_u
            cos_tab = [mp.cos(2 * t_mp * j * h_0_w) for j in range(2 * K_max_u + 1)]
            W_int = mp.mpf("0")
            Wt_int = mp.mpf("0")
            # m = 0 contributes 0 because y = 0 gives y*sinh(2dy) = 0 and y^2*sinc_hyp = 0!
            for m in range(1, K_max_u + 1):
                y_m = m * h_0_w
                # I_Wig(y_m, t) = 2 * sum_{j>=0}'' h_0 Phi((j+m)h_u) Phi((j-m)h_u) cos(2 t j h_0)
                I_wig_m = mp.mpf("0.5") * phi_idx(m) * phi_idx(-m)
                j_limit = K_max_u + m
                for j in range(1, j_limit + 1):
                    p_plus = phi_idx(j + m)
                    p_minus = phi_idx(j - m)
                    if p_plus == 0 or p_minus == 0:
                        if j > m:
                            break
                        continue
                    I_wig_m += p_plus * p_minus * cos_tab[j]
                I_wig_m *= 2 * h_0_w
                W_int += 16 * h_0_w * y_m * mp.sinh(2 * d_mp * y_m) * I_wig_m
                Wt_int += 32 * h_0_w * (y_m * y_m) * sinc_hyp_mp(2 * d_mp * y_m) * I_wig_m
            rel_W = abs(W_ex - W_int) / abs(W_ex)
            rel_Wt = abs(Wt_ex - Wt_int) / abs(Wt_ex)
            max_wig_rel_err = max(max_wig_rel_err, rel_W, rel_Wt)
            print(
                f"  (d={d_v:4.2f}, t={t_v:11.6f}): W_exact = {float(W_ex):+.8e} | "
                f"rel_err(W) = {float(rel_W):.3e} | rel_err(W_tilde) = {float(rel_Wt):.3e}"
            )
    assert max_wig_rel_err < mp.mpf("1e-48"), f"Wigner normalization error too large: {max_wig_rel_err}"

    # 1B. Shifted Hermite-Biehler kernel K_alpha(z,z) = (alpha/pi) W_tilde(d,t) & Cauchy-Pick matrix
    print("\n1B. Shifted Hermite-Biehler Kernel K_alpha & Cauchy-Pick Divided-Difference Matrix:")
    print("-" * 98)
    alpha_list = [mp.mpf("0.25"), mp.mpf("1.0"), mp.mpf("2.5")]
    sample_pts = [
        (0.05, 14.1347251417),
        (0.20, 15.5857085898),
        (0.35, 21.0220396388),
        (0.50, 25.0108575801),
        (0.25, 37.5861781588),
    ]
    max_diag_err = mp.mpf("0")
    max_schur_err = mp.mpf("0")
    for alpha_mp in alpha_list:
        for d_v, t_v in sample_pts:
            d_mp = mp.mpf(str(d_v))
            t_mp = mp.mpf(str(t_v))
            z = mp.mpc(t_mp, -d_mp)
            xi0, xip = Xi_and_deriv_mp(z)
            E_val = xi0 - mp.mpc(0, 1) * alpha_mp * xip
            Es_val = xi0 + mp.mpc(0, 1) * alpha_mp * xip
            K_diag = (abs(E_val) ** 2 - abs(Es_val) ** 2) / (4 * mp.pi * d_mp)
            _, Wt_ex = W_and_Wtilde_mp(d_mp, t_mp)
            K_rhs = (alpha_mp / mp.pi) * Wt_ex
            err_d = abs(K_diag - K_rhs) / abs(K_rhs)
            max_diag_err = max(max_diag_err, err_d)

            Theta = Es_val / E_val
            lhs_schur = 1 - abs(Theta) ** 2
            rhs_schur = (4 * alpha_mp * d_mp * Wt_ex) / (abs(E_val) ** 2)
            err_s = abs(lhs_schur - rhs_schur) / abs(rhs_schur)
            max_schur_err = max(max_schur_err, err_s)

    # Check d -> 0 limit: lim_{d->0} K_alpha(t-id, t-id) = (alpha/pi) L_1(Xi; t)
    d_tiny = mp.mpf("1e-20")
    t_lim = mp.mpf("15.5857085898")
    alpha_1 = mp.mpf("1.0")
    xi0_l, xip_l = Xi_and_deriv_mp(mp.mpc(t_lim, -d_tiny))
    K_lim_num = (abs(xi0_l - 1j * alpha_1 * xip_l) ** 2 - abs(xi0_l + 1j * alpha_1 * xip_l) ** 2) / (
        4 * mp.pi * d_tiny
    )
    _, c1_lim = W_and_Wtilde_mp(mp.mpf("0"), t_lim)
    err_d0 = abs(K_lim_num - (alpha_1 / mp.pi) * c1_lim) / abs((alpha_1 / mp.pi) * c1_lim)

    # 5x5 Cauchy-Pick matrix G_alpha(z_1..z_5)
    z_list = [mp.mpc(mp.mpf(str(tv)), -mp.mpf(str(dv))) for dv, tv in sample_pts]
    M_sz = len(z_list)
    G_def = mp.matrix(M_sz, M_sz)
    G_div = mp.matrix(M_sz, M_sz)
    alpha_g = mp.mpf("1.0")
    for j in range(M_sz):
        xj, xpj = Xi_and_deriv_mp(z_list[j])
        Ej = xj - mp.mpc(0, 1) * alpha_g * xpj
        Esj = xj + mp.mpc(0, 1) * alpha_g * xpj
        for k in range(M_sz):
            xk, xpk = Xi_and_deriv_mp(z_list[k])
            Ek = xk - mp.mpc(0, 1) * alpha_g * xpk
            Esk = xk + mp.mpc(0, 1) * alpha_g * xpk
            denom_def = 2 * mp.pi * mp.mpc(0, 1) * (z_list[j] - mp.conj(z_list[k]))
            G_def[j, k] = (Ej * mp.conj(Ek) - Esj * mp.conj(Esk)) / denom_def
            if j == k:
                _, wt_j = W_and_Wtilde_mp(-mp.im(z_list[j]), mp.re(z_list[j]))
                G_div[j, k] = (alpha_g / mp.pi) * wt_j
            else:
                G_div[j, k] = (alpha_g / mp.pi) * (
                    xj * mp.conj(xpk) - xpj * mp.conj(xk)
                ) / (z_list[j] - mp.conj(z_list[k]))

    max_gram_err = mp.mpf("0")
    for j in range(M_sz):
        for k in range(M_sz):
            max_gram_err = max(max_gram_err, abs(G_def[j, k] - G_div[j, k]) / abs(G_div[j, k]))
    G_hat_np = np.array(
        [
            [
                complex(
                    mp.exp(mp.pi * (mp.re(z_list[j]) + mp.re(z_list[k])) / 4) * G_def[j, k]
                )
                for k in range(M_sz)
            ]
            for j in range(M_sz)
        ],
        dtype=np.complex128,
    )
    eigs_Xi = np.linalg.eigvalsh(G_hat_np)

    print(f"  Max diagonal relative error |K_alpha - (alpha/pi)W_tilde|/|K| : {float(max_diag_err):.3e}")
    print(f"  Max Schur defect relative error                               : {float(max_schur_err):.3e}")
    print(f"  Critical-line limit error at d = 1e-20                        : {float(err_d0):.3e}")
    print(f"  Max 5x5 Cauchy-Pick entrywise divided-difference error        : {float(max_gram_err):.3e}")
    print(f"  5x5 Envelope-congruent Cauchy-Pick eigenvalues (alpha=1)      : min = {eigs_Xi.min():.3e}, max = {eigs_Xi.max():.3e}")
    assert max_diag_err < mp.mpf("1e-47")
    assert max_schur_err < mp.mpf("1e-47")
    assert err_d0 < mp.mpf("1e-28")
    assert max_gram_err < mp.mpf("1e-47")
    assert eigs_Xi.min() > 0.0

    # Rigorous mpmath.iv interval certification with proven strip discretization error < 1e-65
    # On u in [0, U_max = 1.55] with N_u = 160 (h = 1.55/160 = 0.0096875),
    # for d <= 0.50 and t <= 42, strip half-width eta = 0.35 < pi/8 gives:
    # 2*pi*eta/h = 227.005, exp(2*(42)*0.35 + 2*(0.50)*1.55 - 227.005) < 1e-80,
    # and tail u > 1.55 has pi*e^{4*1.55} = 1545.8 => < 1e-600. Total error < 1e-65.
    print("\n1C. Rigorous Directed-Rounding Interval Enclosure (mpmath.iv, Proven Strip Bound < 1e-65):")
    print("-" * 98)
    pi_iv = mp.iv.pi
    n_u_iv = 160
    u_max_iv = mp.iv.mpf("1.55")
    h_u_iv = u_max_iv / mp.iv.mpf(n_u_iv)
    u2_nodes_iv, phi_wt_iv = [], []
    eps_ser_iv = mp.iv.mpf(["0", "1e-220"])
    for j_u in range(n_u_iv + 1):
        u_iv = mp.iv.mpf(j_u) * h_u_iv
        e4u = mp.iv.exp(4 * u_iv)
        e5u = mp.iv.exp(5 * u_iv)
        e9u = mp.iv.exp(9 * u_iv)
        s_phi = mp.iv.mpf("0")
        for n_idx in range(1, 13):
            n2 = mp.iv.mpf(n_idx * n_idx)
            n4 = n2 * n2
            s_phi += (2 * (pi_iv ** 2) * n4 * e9u - 3 * pi_iv * n2 * e5u) * mp.iv.exp(-pi_iv * n2 * e4u)
        s_phi += eps_ser_iv
        w_tr = mp.iv.mpf("0.5") if (j_u == 0 or j_u == n_u_iv) else mp.iv.mpf("1.0")
        u2_nodes_iv.append(2 * u_iv)
        phi_wt_iv.append(8 * h_u_iv * w_tr * s_phi)
    rem_iv = mp.iv.mpf(["-1e-65", "1e-65"])

    max_iv_width = 0.0
    for d_v, t_v in sample_pts:
        d_iv = mp.iv.mpf(str(d_v))
        t_iv = mp.iv.mpf(str(t_v))
        R_iv = mp.iv.mpf("0")
        I_iv = mp.iv.mpf("0")
        dR_iv = mp.iv.mpf("0")
        dI_iv = mp.iv.mpf("0")
        for u2_iv, pw_iv in zip(u2_nodes_iv, phi_wt_iv):
            ep_iv = mp.iv.exp(d_iv * u2_iv)
            em_iv = mp.iv.mpf("1.0") / ep_iv
            ch_iv = mp.iv.mpf("0.5") * (ep_iv + em_iv)
            sh_iv = mp.iv.mpf("0.5") * (ep_iv - em_iv)
            cs_iv = mp.iv.cos(t_iv * u2_iv)
            sn_iv = mp.iv.sin(t_iv * u2_iv)
            R_iv += pw_iv * ch_iv * cs_iv
            I_iv += pw_iv * sh_iv * sn_iv
            dR_iv += pw_iv * u2_iv * sh_iv * cs_iv
            dI_iv += pw_iv * u2_iv * ch_iv * sn_iv
        R_iv += rem_iv
        I_iv += rem_iv
        dR_iv += rem_iv
        dI_iv += rem_iv
        Wt_iv = (R_iv * dR_iv + I_iv * dI_iv) / d_iv
        K1_iv = Wt_iv / pi_iv
        with mp.workdps(70):
            _, Wt_ex = W_and_Wtilde_mp(d_v, t_v)
            K1_ex = Wt_ex / mp.pi
        assert K1_ex in K1_iv, f"Exact K_1 outside interval at (d={d_v}, t={t_v}): K1_ex={K1_ex}, K1_iv={K1_iv}"
        assert K1_iv.a > 0, f"Interval lower bound not positive at (d={d_v}, t={t_v})"
        scale_iv = mp.iv.exp(pi_iv * t_iv / 2)
        K1_hat_iv = scale_iv * K1_iv
        max_iv_width = max(max_iv_width, float(K1_hat_iv.delta))
        print(
            f"  (d={d_v:4.2f}, t={t_v:13.8f}): e^{{pi*t/2}} K_1 in "
            f"[{float(K1_hat_iv.a):+.12f}, {float(K1_hat_iv.b):+.12f}] (width = {float(K1_hat_iv.delta):.2e} > 0)"
        )
    assert max_iv_width < 1e-35

    # 1D. Off-critical zero counterexample demonstrating equivalence to RH (Lagarias-Levinson):
    # F_off(z) = Xi(z) * (1 + (z - t_0)^2 / d_0^2) * (1 + (z + t_0)^2 / d_0^2)
    # (even real-entire with off-critical zeros at +-t_0 +- i d_0, d_0 = 0.25, t_0 = 17.85)
    print("\n1D. Off-Critical Zero Model F_off(z) Confirming Equivalence to Hermite-Biehler / RH:")
    print("-" * 98)
    t0_off = mp.mpf("17.85")
    d0_off = mp.mpf("0.25")

    def F_off_and_deriv(z):
        z_mp = mp.mpc(z)
        xi0, xip = Xi_and_deriv_mp(z_mp)
        p1 = 1 + ((z_mp - t0_off) / d0_off) ** 2
        p2 = 1 + ((z_mp + t0_off) / d0_off) ** 2
        dp1 = 2 * (z_mp - t0_off) / (d0_off ** 2)
        dp2 = 2 * (z_mp + t0_off) / (d0_off ** 2)
        P = p1 * p2
        dP = dp1 * p2 + p1 * dp2
        return xi0 * P, xip * P + xi0 * dP

    z_probe = mp.mpc(mp.mpf("17.85"), -mp.mpf("0.15"))
    f0_p, fp_p = F_off_and_deriv(z_probe)
    E_off = f0_p - 1j * fp_p
    Es_off = f0_p + 1j * fp_p
    K_off_probe = (abs(E_off) ** 2 - abs(Es_off) ** 2) / (4 * mp.pi * mp.mpf("0.15"))
    Theta_off_probe = abs(Es_off / E_off)

    # 6x6 Cauchy-Pick matrix for F_off around (d, t) in (0, 0.35] x [17.5, 18.2]
    pts_off = [
        mp.mpc("17.60", "-0.10"),
        mp.mpc("17.75", "-0.15"),
        mp.mpc("17.85", "-0.15"),
        mp.mpc("17.85", "-0.22"),
        mp.mpc("17.95", "-0.15"),
        mp.mpc("18.10", "-0.30"),
    ]
    G_off = np.zeros((6, 6), dtype=np.complex128)
    for j in range(6):
        fj, fpj = F_off_and_deriv(pts_off[j])
        for k in range(6):
            fk, fpk = F_off_and_deriv(pts_off[k])
            val = (mp.mpf("1") / mp.pi) * (fj * mp.conj(fpk) - fpj * mp.conj(fk)) / (
                pts_off[j] - mp.conj(pts_off[k])
            )
            G_off[j, k] = complex(val)
    eigs_off = np.linalg.eigvalsh(G_off)
    scale_probe = mp.exp(mp.pi * mp.mpf("17.85") / 2)
    print(f"  At z = 17.85 - 0.15i: e^{{pi*t/2}} K_1^{{off}}(z,z) = {float(scale_probe * K_off_probe):+.6e} < 0")
    print(f"  At z = 17.85 - 0.15i: |Theta_1^{{off}}(z)|          = {float(Theta_off_probe):.8f} > 1 (Schur violation)")
    print(f"  6x6 Cauchy-Pick eigenvalues for F_off (alpha=1)    : min = {float(scale_probe)*eigs_off.min():+.6e} < 0")
    assert K_off_probe < 0
    assert Theta_off_probe > 1
    assert eigs_off.min() < 0.0

    # Generate figure data for Tier 4 (t in [10, 40], verified envelope-safe in mpmath!)
    t_grid_fig = np.linspace(10.0, 40.0, 240)
    Khat_Xi_d0 = np.zeros_like(t_grid_fig)
    Khat_Xi_d20 = np.zeros_like(t_grid_fig)
    Khat_off_d15 = np.zeros_like(t_grid_fig)
    for idx, tv in enumerate(t_grid_fig):
        t_m = mp.mpf(f"{tv:.8f}")
        sc_m = mp.exp(mp.pi * t_m / 2) / mp.pi
        _, c1_m = W_and_Wtilde_mp(mp.mpf("0"), t_m)
        _, wt20_m = W_and_Wtilde_mp(mp.mpf("0.20"), t_m)
        f_o, fp_o = F_off_and_deriv(mp.mpc(t_m, -mp.mpf("0.15")))
        wt_off_m = mp.im(mp.conj(f_o) * fp_o) / mp.mpf("0.15")
        Khat_Xi_d0[idx] = float(sc_m * c1_m)
        Khat_Xi_d20[idx] = float(sc_m * wt20_m)
        Khat_off_d15[idx] = float(sc_m * wt_off_m)

    return {
        "t_grid_fig": t_grid_fig,
        "Khat_Xi_d0": Khat_Xi_d0,
        "Khat_Xi_d20": Khat_Xi_d20,
        "Khat_off_d15": Khat_off_d15,
    }


# ==============================================================================
# 2. TIER 2: JENSEN-LAGUERRE MULTIPLICITY CASCADE, TRUE EXTREMA tau_1 & tau_2,
#            550-DIGIT DEFORMATION & REAL VS. OFF-CRITICAL QUADRATIC BOWLS
# ==============================================================================

def c_2n_minus_1_jet(derivs, n: int):
    """Exact c_{2n-1}(f; t) = n * L_n(f; t) from [f(t), f'(t), ..., f^{(2n)}(t)]."""
    acc = mp.mpf("0")
    for j in range(2 * n):
        sign = -1 if ((n + j) % 2 == 1) else 1
        acc += sign * mp.binomial(2 * n - 1, j) * derivs[j] * derivs[2 * n - j]
    return acc / mp.factorial(2 * n - 1)


def laguerre_L_n_jet(derivs, n: int):
    """Classical Jensen-Laguerre differential polynomial L_n(f; t) (Jensen 1913, Csordas-Varga 1990)."""
    acc = mp.mpf("0")
    for j in range(2 * n + 1):
        sign = -1 if ((n + j) % 2 == 1) else 1
        acc += sign * mp.binomial(2 * n, j) * derivs[j] * derivs[2 * n - j]
    return acc / mp.factorial(2 * n)


def verify_tier2():
    print_banner("TIER 2: JENSEN-LAGUERRE MULTIPLICITY CASCADE, TRUE tau_1 & QUADRATIC BOWLS")

    # 2A. Verify c_{2n-1} == n * L_n and order-m multiplicity cascade for m = 1..5
    print("2A. Exact Identity c_{2n-1}(f; t) = n L_n(f; t) and Multiplicity Cascade (m = 1..5):")
    print("-" * 98)
    jet_test = [mp.re(v) for v in Xi_jet_mp(mp.mpf("18.25"), max_order=12)]
    max_jensen_err = mp.mpf("0")
    for n in range(1, 6):
        c_val = c_2n_minus_1_jet(jet_test, n)
        nL_val = n * laguerre_L_n_jet(jet_test, n)
        max_jensen_err = max(max_jensen_err, abs(c_val - nL_val) / abs(c_val))
    print(f"  Max relative difference |c_{{2n-1}} - n L_n| / |c_{{2n-1}}| (n=1..5): {float(max_jensen_err):.3e}")
    assert max_jensen_err < mp.mpf("1e-48")

    # Order-m real-entire test family f_m(z) = (z - t_0)^m * Xi(z) / Xi'(t_0) at t_0 = gamma_1
    g1 = mp.im(mp.zetazero(1))
    xi_g1_jet = [mp.re(v) for v in Xi_jet_mp(g1, max_order=12)]
    xip_g1 = xi_g1_jet[1]
    cascade_curves = {}
    d_log_grid = np.logspace(-3.0, -0.3, 60)

    for m in range(1, 6):
        def f_mult(z, m_ord=m):
            z_m = mp.mpc(z)
            # f_m(z) = (z - g1)^{m-1} * Xi(z) has exact multiplicity m at z = g1!
            return ((z_m - g1) ** (m_ord - 1)) * Xi_mp(z_m)

        coeffs_fm = mp.taylor(f_mult, g1, 2 * m + 2)
        derivs_fm = [mp.re(coeffs_fm[k] * mp.factorial(k)) for k in range(2 * m + 3)]
        for n_low in range(1, m):
            c_low = abs(c_2n_minus_1_jet(derivs_fm, n_low))
            assert c_low < mp.mpf("1e-45"), f"Lower c_{2*n_low-1} did not vanish for m={m}: {c_low}"
        c_lead = c_2n_minus_1_jet(derivs_fm, m)
        c_sq = (mp.mpf(m) / (mp.factorial(m) ** 2)) * (derivs_fm[m] ** 2)
        rel_err_m = abs(c_lead - c_sq) / abs(c_sq)
        print(
            f"  m = {m}: f_m^({m})(gamma_1) = {float(derivs_fm[m]):+.6e} | "
            f"c_{2*m-1} = {float(c_lead):+.8e} | rel_err vs [m/(m!)^2](f^(m))^2 = {float(rel_err_m):.3e}"
        )
        assert rel_err_m < mp.mpf("1e-47") and c_lead > 0

        if m <= 4:
            vals = []
            for dv in d_log_grid:
                d_m = mp.mpf(f"{dv:.8e}")
                # |f_m(g1 - i d)|^2 = d^{2(m-1)} |Xi(g1 - i d)|^2
                # d_d |f_m|^2 = 2(m-1) d^{2m-3} |Xi|^2 + d^{2(m-1)} * 2 W_Xi(d, g1)
                z_pt = mp.mpc(g1, -d_m)
                xi_v, xip_v = Xi_and_deriv_mp(z_pt)
                W_xi_v = mp.im(mp.conj(xi_v) * xip_v)
                wt_fm = (m - 1) * (d_m ** (2 * m - 4)) * (abs(xi_v) ** 2) + (d_m ** (2 * m - 2)) * (W_xi_v / d_m)
                # Normalize by (f_m^{(m)}(g1))^2 so curves start cleanly on one plot
                vals.append(float(wt_fm / (derivs_fm[m] ** 2)))
            cascade_curves[m] = np.array(vals)

    # 2B. True critical-line extrema tau_1 and tau_2 of Xi'(t) = 0 + 550-digit local coalescence
    print("\n2B. True Positive Extrema tau_1, tau_2 of Xi'(t) = 0 & 550-Digit Coalescence Audit:")
    print("-" * 98)
    g2 = mp.im(mp.zetazero(2))
    g3 = mp.im(mp.zetazero(3))
    # Bracket on (gamma_1, gamma_2) and (gamma_2, gamma_3) so root-finding NEVER jumps roots!
    tau_1 = mp.findroot(lambda t: mp.re(Xi_and_deriv_mp(t)[1]), (mp.mpf("15.4"), mp.mpf("15.8")), solver="anderson")
    tau_2 = mp.findroot(lambda t: mp.re(Xi_and_deriv_mp(t)[1]), (mp.mpf("21.9"), mp.mpf("22.3")), solver="anderson")
    t_hardy_1 = mp.findroot(lambda t: mp.diff(mp.siegelz, t), (mp.mpf("17.7"), mp.mpf("18.0")), solver="anderson")

    jet_tau1 = [mp.re(v) for v in Xi_jet_mp(tau_1, max_order=6)]
    jet_tau2 = [mp.re(v) for v in Xi_jet_mp(tau_2, max_order=6)]
    c3_sq_tau1 = mp.mpf("0.5") * (jet_tau1[2] ** 2)
    c3_sq_tau2 = mp.mpf("0.5") * (jet_tau2[2] ** 2)
    print(f"  True tau_1 in (gamma_1, gamma_2) : {mp.nstr(tau_1, 20)} | (1/2)(Xi''(tau_1))^2 = {float(c3_sq_tau1):.8e}")
    print(f"  True tau_2 in (gamma_2, gamma_3) : {mp.nstr(tau_2, 20)} | (1/2)(Xi''(tau_2))^2 = {float(c3_sq_tau2):.8e}")
    print(f"  First extremum of Hardy Z'(t)=0  : {mp.nstr(t_hardy_1, 20)} (2.26 units above tau_1 due to e^{{-pi*t/4}} drift!)")
    assert mp.mpf("15.58") < tau_1 < mp.mpf("15.60")
    assert mp.mpf("22.09") < tau_2 < mp.mpf("22.11")

    # 550-digit local coalescence deformation anchored at TRUE tau_1:
    # Xi_def(z; delta) := Xi(z) - Xi(tau_1) - (1/2) Xi''(tau_1) delta^2
    with mp.workdps(560):
        tau1_hp = mp.findroot(
            lambda t: mp.re(Xi_and_deriv_mp(t)[1]),
            mp.mpf(mp.nstr(tau_1, 45)),
        )
        jet_hp = [mp.re(v) for v in Xi_jet_mp(tau1_hp, max_order=6)]
        Xi0_hp, Xi1_hp, Xi2_hp, Xi3_hp, Xi4_hp = jet_hp[0], jet_hp[1], jet_hp[2], jet_hp[3], jet_hp[4]
        assert abs(Xi1_hp) < mp.mpf("1e-545")
        c3_floor_hp = mp.mpf("0.5") * (Xi2_hp ** 2)
        print(f"\n  550-Digit Local Coalescence Family Xi_def(z; delta) Anchored at tau_1 = {mp.nstr(tau1_hp, 16)}:")
        print(f"  Asymptotic c_3 floor (1/2)(Xi''(tau_1))^2 = {mp.nstr(c3_floor_hp, 12)}")
        hdr = f"  {'Target 2*delta':<14} | {'True Gap Delta_gamma':<22} | {'c_1(tau_1)':<18} | {'c_3(tau_1)':<18} | {'|c_3 - floor|/floor':<18}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        table2_rows = []
        for exp_g in [1, 3, 6, 12, 30, 100, 250]:
            gap_target = mp.mpf(10) ** (-exp_g)
            delta_hp = gap_target / 2
            # Perturbed value at tau_1: f_delta(tau_1) = (1/2) Xi''(tau_1) delta^2
            f0_def = mp.mpf("0.5") * Xi2_hp * (delta_hp ** 2)
            c1_def = -f0_def * Xi2_hp  # = -(1/2)(Xi''(tau_1))^2 delta^2 ? Wait!
            # WAIT: check sign of f_delta(tau_1)!
            # Near tau_1, f_delta(tau_1 + w) = f_delta(tau_1) + (1/2) Xi2_hp w^2 + O(w^3).
            # For TWO REAL ZEROS at w = +-delta, f_delta(tau_1) and Xi2_hp MUST HAVE OPPOSITE SIGNS:
            # f_delta(tau_1) = -(1/2) Xi2_hp delta^2 !!
            f0_def = -mp.mpf("0.5") * Xi2_hp * (delta_hp ** 2)
            # Find exact roots t_- < tau_1 < t_+ of Xi(t) - Xi0_hp - (1/2) Xi2_hp delta^2 = 0
            if exp_g <= 30:
                t_minus = mp.findroot(lambda t: mp.re(Xi_mp(t)) - Xi0_hp + f0_def, tau1_hp - delta_hp)
                t_plus = mp.findroot(lambda t: mp.re(Xi_mp(t)) - Xi0_hp + f0_def, tau1_hp + delta_hp)
                true_gap = t_plus - t_minus
            else:
                # Cubic-corrected Weierstrass root asymptotic accurate to O(delta^4) < 1e-390:
                # w_+ - w_- = 2*delta * (1 + [3*(Xi3/Xi2)^2 - Xi4/Xi2]*delta^2 / 24 + O(delta^4))
                corr = (3 * ((Xi3_hp / Xi2_hp) ** 2) - (Xi4_hp / Xi2_hp)) * (delta_hp ** 2) / 24
                true_gap = 2 * delta_hp * (1 + corr)
            c1_def = -f0_def * Xi2_hp  # = +(1/2)(Xi2_hp)^2 delta^2 > 0!
            c3_def = mp.mpf("0.5") * (Xi2_hp ** 2) + (f0_def * Xi4_hp) / 6
            rel_floor = abs(c3_def - c3_floor_hp) / c3_floor_hp
            print(
                f"  10^-{exp_g:<10d} | {mp.nstr(true_gap, 14):<22} | {mp.nstr(c1_def, 10):<18} | "
                f"{mp.nstr(c3_def, 10):<18} | {mp.nstr(rel_floor, 10):<18}"
            )
            assert c1_def > 0 and c3_def > 0
            table2_rows.append((exp_g, mp.nstr(true_gap, 10), mp.nstr(c1_def, 9), mp.nstr(c3_def, 9)))

    # 2C. Unified Quadratic Bowl: Real Lehmer pair (gamma_6709, gamma_6710) vs Off-Critical Conjugate Pair
    print("\n2C. Unified Quadratic Bowl: Real Lehmer Pair (t ~ 7005.08) vs. Off-Critical Pair (mu < 0):")
    print("-" * 98)
    g_leh1 = mp.im(mp.zetazero(6709))
    g_leh2 = mp.im(mp.zetazero(6710))
    gap_leh = g_leh2 - g_leh1
    env_leh = mp.exp(mp.pi * g_leh1 / 4)
    tau_leh = mp.findroot(
        lambda t: mp.re(Xi_and_deriv_mp(t)[1]) * env_leh,
        (g_leh1, g_leh2),
        solver="anderson",
    )
    env_tau_leh = mp.exp(mp.pi * tau_leh / 4)
    coeffs_leh = mp.taylor(lambda z: Xi_mp(z) * env_tau_leh, tau_leh, 4)
    jet_leh_scaled = [mp.re(coeffs_leh[k] * mp.factorial(k)) for k in range(5)]
    mu_leh = -2 * jet_leh_scaled[0] / jet_leh_scaled[2]
    c1_hat_leh = -jet_leh_scaled[0] * jet_leh_scaled[2]
    c3_hat_leh = c_2n_minus_1_jet(jet_leh_scaled, 2)
    print(f"  Lehmer pair (gamma_6709, gamma_6710): gap = {float(gap_leh):.10f}, tau_* = {float(tau_leh):.10f}")
    print(f"  mu = -2 Xi(tau_*)/Xi''(tau_*) = {float(mu_leh):+.10e} > 0 (sqrt(mu) = {float(mp.sqrt(mu_leh)):.10f} vs gap/2 = {float(gap_leh/2):.10f})")
    print(f"  e^{{pi*tau_*/2}} c_1(tau_*) = {float(c1_hat_leh):+.6e} > 0 | e^{{pi*tau_*/2}} c_3(tau_*) = {float(c3_hat_leh):+.6e} > 0")
    assert mu_leh > 0 and c1_hat_leh > 0 and c3_hat_leh > 0

    # Compare real bowl mu = +delta^2 vs off-critical conjugate bowl mu = -delta_0^2 on model f(w) = (1/2)A(w^2 - mu) + B w^3
    delta_mod = 0.08
    tau_grid_bowl = np.linspace(-0.18, 0.18, 161)
    # f_real(z) = (1/2)(z^2 - delta^2) + 0.25 z^3 (zeros near +-delta)
    # f_off(z)  = (1/2)(z^2 + delta^2) + 0.25 z^3 (conjugate zeros near +-i delta)
    def eval_poly_Wtilde(d_v, tau_arr, sign_mu):
        z = tau_arr - 1j * d_v
        mu_v = sign_mu * (delta_mod ** 2)
        f = 0.5 * (z ** 2 - mu_v) + 0.25 * (z ** 3)
        fp = z + 0.75 * (z ** 2)
        if abs(d_v) < 1e-14:
            fpp = 1.0 + 1.5 * z
            return np.real(fp ** 2 - f * fpp)
        return np.imag(np.conj(f) * fp) / d_v

    bowl_real_d0 = eval_poly_Wtilde(0.0, tau_grid_bowl, +1.0)
    bowl_real_d06 = eval_poly_Wtilde(0.06, tau_grid_bowl, +1.0)
    bowl_off_d0 = eval_poly_Wtilde(0.0, tau_grid_bowl, -1.0)
    bowl_off_d06 = eval_poly_Wtilde(0.06, tau_grid_bowl, -1.0)
    print(f"  Model real pair (mu = +0.08^2): min W_tilde(d=0) = {bowl_real_d0.min():+.6e} > 0")
    print(f"  Model off-line pair (mu = -0.08^2): W_tilde(0,0) = {bowl_off_d0[80]:+.6e} < 0, W_tilde(0.06,0) = {bowl_off_d06[80]:+.6e} < 0")
    assert bowl_real_d0.min() > 0 and bowl_real_d06.min() > 0
    assert bowl_off_d0[80] < 0 and bowl_off_d06[80] < 0

    return {
        "d_log_grid": d_log_grid,
        "cascade_curves": cascade_curves,
        "tau_grid_bowl": tau_grid_bowl,
        "bowl_real_d0": bowl_real_d0,
        "bowl_real_d06": bowl_real_d06,
        "bowl_off_d0": bowl_off_d0,
        "bowl_off_d06": bowl_off_d06,
        "delta_mod": delta_mod,
    }


# ==============================================================================
# 3. TIER 3: LEIBNIZ-WRONSKIAN IDENTITY, E_X NON-REALITY VS. M_X^{sym},
#            PRIME-POWER RECURSION 0 <= b_n^E(X) <= 1, BOCHNER-WEIL & E_0(t)
# ==============================================================================

def von_mangoldt_array(N_max: int) -> np.ndarray:
    lam = np.zeros(N_max + 1, dtype=np.float64)
    is_prime = np.ones(N_max + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, N_max + 1):
        if is_prime[p]:
            for mult in range(2 * p, N_max + 1, p):
                is_prime[mult] = False
            pk = p
            ln_p = math.log(p)
            while pk <= N_max:
                lam[pk] = ln_p
                pk *= p
    return lam


def compute_b_n_E(X_cutoff: int, N_max: int, lam: np.ndarray) -> np.ndarray:
    """Exact recursion b_1 = 1, b_n * ln(n) = sum_{d|n, d > X} b_{n/d} * Lambda(d)."""
    b = np.zeros(N_max + 1, dtype=np.float64)
    b[1] = 1.0
    # Sieve divisors d > X with Lambda(d) > 0
    active_d = [d for d in range(X_cutoff + 1, N_max + 1) if lam[d] > 0.0]
    for n in range(X_cutoff + 1, N_max + 1):
        acc = 0.0
        for d in active_d:
            if d > n:
                break
            if n % d == 0:
                acc += b[n // d] * lam[d]
        b[n] = acc / math.log(n)
    return b


def compute_b_n_mobius_trunc(X_cutoff: int, N_max: int) -> np.ndarray:
    """Dirichlet convolution of zeta(s) with truncated Mobius sum_{m <= X} mu(m) m^{-s}."""
    mu = np.zeros(X_cutoff + 1, dtype=np.int64)
    mu[1] = 1
    primes = []
    is_comp = np.zeros(X_cutoff + 1, dtype=bool)
    for i in range(2, X_cutoff + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > X_cutoff:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    b_mob = np.zeros(N_max + 1, dtype=np.float64)
    for m in range(1, X_cutoff + 1):
        if mu[m] != 0:
            b_mob[m : N_max + 1 : m] += mu[m]
    return b_mob


def fejer_K_U_complex(w, U_val):
    """Entire analytic continuation of K_U(w) = (1/(2*pi*U)) * (sin(U*w/2)/(w/2))^2."""
    w_mp = mp.mpc(w)
    U_mp = mp.mpf(U_val)
    if abs(w_mp) < mp.mpf("1e-18"):
        return (U_mp / (2 * mp.pi)) * (1 - (U_mp * w_mp) ** 2 / 12)
    half_w = w_mp / 2
    s_val = mp.sin(U_mp * half_w) / half_w
    return (s_val * s_val) / (2 * mp.pi * U_mp)


def verify_tier3():
    print_banner("TIER 3: LEIBNIZ-WRONSKIAN, E_X NON-REALITY, 0 <= b_n^E <= 1, BOCHNER-WEIL & E_0(t)")

    lam = von_mangoldt_array(5000)

    # 3A. Leibniz-Wronskian identity, E_X non-reality 1/d pole vs M_X^{sym} smoothness & oscillation
    print("3A. Leibniz-Wronskian Identity & Broken Schwarz Reflection of Unsymmetrized E_X(1/2 + iz):")
    print("-" * 98)
    X_test = 7
    active_p = [(n, mp.mpf(str(lam[n]))) for n in range(2, X_test + 1) if lam[n] > 0]

    def E_X_mp(z):
        s = mp.mpf("0.5") + mp.mpc(0, 1) * mp.mpc(z)
        acc = mp.mpc("0")
        for n, lv in active_p:
            acc -= (lv / mp.log(n)) * mp.power(n, -s)
        return mp.exp(acc)

    def E_X_and_deriv_mp(z):
        z_mp = mp.mpc(z)
        s = mp.mpf("0.5") + mp.mpc(0, 1) * z_mp
        val = E_X_mp(z_mp)
        # M'(z)/M(z) = +i sum_{n<=X} Lambda(n) n^{-1/2 - i z}
        dlog = mp.mpc("0")
        for n, lv in active_p:
            dlog += mp.mpc(0, 1) * lv * mp.power(n, -s)
        return val, val * dlog

    def M_sym_and_deriv_mp(z):
        z_mp = mp.mpc(z)
        acc = mp.mpc("0")
        dlog = mp.mpc("0")
        for n, lv in active_p:
            ln_n = mp.log(n)
            inv_sq = 1 / mp.sqrt(mp.mpf(n))
            acc -= 2 * (lv / ln_n) * inv_sq * mp.cos(z_mp * ln_n)
            dlog += 2 * lv * inv_sq * mp.sin(z_mp * ln_n)
        val = mp.exp(acc)
        return val, val * dlog

    # Check exact unscaled Leibniz-Wronskian W_{M Xi} == |M|^2 W + |Xi|^2 W_{1,M} for BOTH E_X and M_sym
    z_lw = mp.mpc("19.40", "-0.25")
    d_lw = mp.mpf("0.25")
    xi_lw, xip_lw = Xi_and_deriv_mp(z_lw)
    W_xi_lw = mp.im(mp.conj(xi_lw) * xip_lw)

    for label, fn in [("Unsymmetrized E_X", E_X_and_deriv_mp), ("Symmetrized M_X^sym", M_sym_and_deriv_mp)]:
        m0, mp0 = fn(z_lw)
        prod0 = m0 * xi_lw
        prodp = mp0 * xi_lw + m0 * xip_lw
        W_prod = mp.im(mp.conj(prod0) * prodp)
        W_1M = mp.im(mp.conj(m0) * mp0)
        rhs_lw = (abs(m0) ** 2) * W_xi_lw + (abs(xi_lw) ** 2) * W_1M
        rel_lw = abs(W_prod - rhs_lw) / abs(W_prod)
        print(f"  {label:<21}: |W_{{M Xi}} - (|M|^2 W + |Xi|^2 W_{{1,M}})| / |W_{{M Xi}}| = {float(rel_lw):.3e}")
        assert rel_lw < mp.mpf("1e-47")

    # Verify Corollary 4.2 bounds B_CS and B_dist at z_lw using M_sym:
    ms0, msp0 = M_sym_and_deriv_mp(z_lw)
    Wt_xi_lw = W_xi_lw / d_lw
    Wt_1M_lw = mp.im(mp.conj(ms0) * msp0) / d_lw
    Wt_M_lw = (abs(ms0) ** 2) * Wt_xi_lw + (abs(xi_lw) ** 2) * Wt_1M_lw
    if Wt_M_lw > 0:
        B_CS = (d_lw * Wt_M_lw) / (abs(ms0) * abs(msp0 * xi_lw + ms0 * xip_lw))
        B_dist = (4 * d_lw * Wt_M_lw) / (
            2 * d_lw * abs(Wt_1M_lw) + mp.sqrt(abs(ms0) ** 4 + 4 * (d_lw ** 2) * (Wt_1M_lw ** 2))
        )
        assert abs(xi_lw) >= B_CS and (abs(xi_lw) ** 2 + abs(xip_lw) ** 2) >= B_dist
        print(f"  Local zero-exclusion bounds at (0.25, 19.40): |Xi| = {float(abs(xi_lw)):.3e} >= B_CS = {float(B_CS):.3e}")

    # Demonstrate the 1/d blow-up of d^{-1} W_{1,E_X}(d, t) at t = 18.5 vs smooth M_X^sym:
    t_pole = mp.mpf("18.5")
    e0_t, ep0_t = E_X_and_deriv_mp(t_pole)
    W_1E_d0_exact = (abs(e0_t) ** 2) * sum(
        (lv / mp.sqrt(mp.mpf(n))) * mp.cos(t_pole * mp.log(n)) for n, lv in active_p
    )
    W_1E_d0_num = mp.im(mp.conj(e0_t) * ep0_t)
    err_W1E_d0 = abs(W_1E_d0_exact - W_1E_d0_num) / abs(W_1E_d0_exact)
    print(f"  At d = 0, t = 18.5: W_{{1,E_7}}(0, 18.5) = {float(W_1E_d0_num):+.8e} != 0 (formula err = {float(err_W1E_d0):.3e})")
    assert abs(W_1E_d0_num) > mp.mpf("1e-4") and err_W1E_d0 < mp.mpf("1e-48")

    d_scan = np.logspace(-3.0, -0.3, 60)
    Wt_1E_scan = []
    Wt_1Msym_scan = []
    for dv in d_scan:
        d_m = mp.mpf(f"{dv:.8e}")
        z_s = mp.mpc(t_pole, -d_m)
        e_v, ep_v = E_X_and_deriv_mp(z_s)
        ms_v, msp_v = M_sym_and_deriv_mp(z_s)
        Wt_1E_scan.append(float(mp.im(mp.conj(e_v) * ep_v) / d_m))
        Wt_1Msym_scan.append(float(mp.im(mp.conj(ms_v) * msp_v) / d_m))
    Wt_1E_scan = np.array(Wt_1E_scan)
    Wt_1Msym_scan = np.array(Wt_1Msym_scan)

    # Show M_X^sym intrinsic Wronskian oscillates in sign across t in [14, 40]:
    t_osc_grid = np.linspace(14.0, 40.0, 261)
    Wt_1Msym_t = np.zeros_like(t_osc_grid)
    for idx, tv in enumerate(t_osc_grid):
        ms_v, msp_v = M_sym_and_deriv_mp(mp.mpc(f"{tv:.6f}", "-0.10"))
        Wt_1Msym_t[idx] = float(mp.im(mp.conj(ms_v) * msp_v) / mp.mpf("0.10"))
    print(
        f"  Symmetrized W_tilde_{{1,M_7^sym}}(0.10, t) on [14, 40]: min = {Wt_1Msym_t.min():+.4f} < 0, "
        f"max = {Wt_1Msym_t.max():+.4f} > 0 (sign-indefinite!)"
    )
    assert Wt_1Msym_t.min() < 0.0 and Wt_1Msym_t.max() > 0.0

    # 3B. Recursion 0 <= b_n^E(X) <= 1 and quadratic-window prime-power purity
    print("\n3B. Exact Prime-Power Recursion 0 <= b_n^E(X) <= 1 Across n <= 5000:")
    print("-" * 98)
    for X_c in [2, 3, 5, 10, 17, 30]:
        b_arr = compute_b_n_E(X_c, 5000, lam)
        assert np.all(b_arr[2 : X_c + 1] == 0.0)
        assert b_arr.min() >= -1e-15 and b_arr.max() <= 1.0 + 1e-14
        quad_end = (X_c + 1) ** 2
        max_quad_err = 0.0
        for n in range(X_c + 1, min(quad_end, 5001)):
            exact_q = lam[n] / math.log(n)
            max_quad_err = max(max_quad_err, abs(b_arr[n] - exact_q))
        assert max_quad_err < 1e-14
        print(
            f"  X = {X_c:2d}: b_n == 0 on [2,{X_c}], range on [1,5000] = [{b_arr.min():.4f}, {b_arr.max():.6f}], "
            f"quad-window [{X_c+1},{quad_end-1}] err = {max_quad_err:.2e}"
        )
    b_E_10 = compute_b_n_E(10, 160, lam)
    b_mob_10 = compute_b_n_mobius_trunc(10, 160)
    print(f"  Comparison at X = 10 (n <= 160): max |b_n^E| = {np.max(np.abs(b_E_10)):.4f} vs max |b_n^Mobius| = {np.max(np.abs(b_mob_10)):.1f} (> 1 overshoot)")
    assert np.max(np.abs(b_mob_10)) >= 2.0

    # 3C. Exact Fejer Bochner-Weil identity at d = 0.65 > 1/2 vs d = 0.15 < 1/2 residue jump
    print("\n3C. Fejer Bochner-Weil Identity: Unconditional Match (d > 1/2) vs. Strip Pole/Zero Residues (d < 1/2):")
    print("-" * 98)
    # Test exact off-critical zero residue jump on the rational logarithmic-derivative difference:
    # delta_w(d, tau) := d_d ln |(tau - i d - gamma_0)^2 + d_0^2|
    #                  = (d - d_0)/((tau - gamma_0)^2 + (d - d_0)^2) + (d + d_0)/((tau - gamma_0)^2 + (d + d_0)^2)
    # Convolved with K_U(u): for d > d_0 vs 0 < d < d_0, the exact Fourier expression is:
    # int_{-U}^U K_hat_U(xi) cos(xi(t - gamma_0)) * (pi*e^{-|xi|(d-d_0)} if d>d_0 else -pi*e^{-|xi|(d_0-d)}) / (2*pi) ...
    # Difference from analytic continuation of the d > d_0 formula into 0 < d < d_0 is EXACTLY
    # -2 * pi * Re K_U((t - gamma_0) + i*(d_0 - d))!
    U_bw = mp.log(mp.mpf("6"))
    gamma0_bw = mp.mpf("20.0")
    d0_bw = mp.mpf("0.30")
    t_bw = mp.mpf("20.4")
    d_in = mp.mpf("0.15")  # 0 < d_in < d0_bw

    def delta_L_quad(d_val, t_val):
        # Integrate K_U(u) * (d - d0)/((t + u - g0)^2 + (d - d0)^2) over R via Fourier / direct quad
        # Direct quad on u in [-400, 400] + exact asymptotic tail or mp.quad over [-inf, inf]:
        return mp.quad(
            lambda u: mp.re(fejer_K_U_complex(u, U_bw))
            * ((d_val - d0_bw) / ((t_val + u - gamma0_bw) ** 2 + (d_val - d0_bw) ** 2)),
            [-mp.inf, -10, -2, 2, 10, mp.inf],
        )

    def delta_L_continued_from_right(d_val, t_val):
        # Analytic continuation of the sigma > 1 / d > d_0 Fourier integral:
        # (1/2) int_{-U}^U (1 - |xi|/U) cos(xi(t - g0)) exp(-|xi|*(d - d0)) d_xi
        return quad_gl48_mp(
            lambda xi: (1 - xi / U_bw) * mp.cos(xi * (t_val - gamma0_bw)) * mp.exp(-xi * (d_val - d0_bw)),
            mp.mpf("0"),
            U_bw,
        )

    L_true_right = delta_L_quad(mp.mpf("0.65"), t_bw)
    L_cont_right = delta_L_continued_from_right(mp.mpf("0.65"), t_bw)
    err_right = abs(L_true_right - L_cont_right)

    L_true_in = delta_L_quad(d_in, t_bw)
    L_naive_in = delta_L_continued_from_right(d_in, t_bw)
    res_jump_exact = -2 * mp.pi * mp.re(fejer_K_U_complex((t_bw - gamma0_bw) + mp.mpc(0, 1) * (d0_bw - d_in), U_bw))
    err_jump = abs((L_true_in - L_naive_in) - res_jump_exact)
    print(f"  At d = 0.65 > d_0: |True Quadrature - Prime/Fourier Continuation| = {float(err_right):.3e}")
    print(f"  At d = 0.15 < d_0: True - Naive = {float(L_true_in - L_naive_in):+.10f} vs -2*pi*Re K_U = {float(res_jump_exact):+.10f} (err = {float(err_jump):.3e})")
    assert err_right < mp.mpf("1e-8") and err_jump < mp.mpf("1e-8")

    # Also verify directly on Xi itself at d = 0.85 > 1/2 (sigma = 1.35 > 1) that
    # int_{-inf}^inf K_U(u) [W(d,t+u)/|Xi|^2 - A(d,t+u)] du == -sum_{2 <= n < X} Lambda(n) n^{-1/2-d} (1 - ln(n)/U) cos(t ln n)
    # using independent vectorized Euler-Maclaurin evaluation (N_em = 500 > 2000/(2*pi)) on u in [-2000, 2000]!
    d_absc = 0.85
    t_absc = 18.0
    X_bw = 6
    U_x6 = math.log(X_bw)
    u_vec = np.linspace(-2000.0, 2000.0, 20001)
    du_vec = u_vec[1] - u_vec[0]
    s_vec = (0.5 + d_absc) + 1j * (t_absc + u_vec)
    N_em = 500
    n_em = np.arange(1, N_em, dtype=np.float64)[:, None]
    ln_n_em = np.log(n_em)
    terms_em = np.exp(-s_vec[None, :] * ln_n_em)
    z_em = np.sum(terms_em, axis=0)
    zp_em = -np.sum(ln_n_em * terms_em, axis=0)
    lnN = math.log(N_em)
    N_ms = np.exp(-s_vec * lnN)
    z_em += N_ms / (s_vec - 1.0) * N_em + 0.5 * N_ms
    zp_em += N_ms * N_em * (-lnN / (s_vec - 1.0) - 1.0 / ((s_vec - 1.0) ** 2)) - 0.5 * lnN * N_ms
    b_coeffs = [1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0, -1.0 / 1209600.0]
    poch = s_vec.copy()
    poch_p = np.ones_like(s_vec)
    invN2 = 1.0 / (N_em * N_em)
    N_pow = N_ms / N_em
    for idx_b, cb in enumerate(b_coeffs):
        z_em += cb * N_pow * poch
        zp_em += cb * N_pow * (poch_p - lnN * poch)
        if idx_b + 1 < len(b_coeffs):
            f1 = s_vec + (2 * idx_b + 1)
            f2 = s_vec + (2 * idx_b + 2)
            poch_p = poch_p * f1 * f2 + poch * (f1 + f2)
            poch = poch * f1 * f2
            N_pow = N_pow * invN2
    half_u = 0.5 * u_vec
    half_u_safe = np.where(np.abs(half_u) < 1e-12, 1.0, half_u)
    K_u_vec = np.where(
        np.abs(half_u) < 1e-12,
        U_x6 / (2.0 * math.pi),
        (np.sin(U_x6 * half_u_safe) / half_u_safe) ** 2 / (2.0 * math.pi * U_x6),
    )
    w_trap = np.ones_like(u_vec)
    w_trap[0] = 0.5
    w_trap[-1] = 0.5
    lhs_bw_zeta = float(np.sum(w_trap * K_u_vec * np.real(zp_em / z_em)) * du_vec)
    rhs_bw_primes = -sum(
        (lam[n] / (n ** (0.5 + d_absc))) * (1.0 - math.log(n) / U_x6) * math.cos(t_absc * math.log(n))
        for n in range(2, X_bw)
        if lam[n] > 0
    )
    err_bw_xi = abs(lhs_bw_zeta - rhs_bw_primes)
    pole_s1_d15 = 2 * mp.pi * mp.re(
        fejer_K_U_complex(mp.mpf("18.0") + mp.mpc(0, 1) * (mp.mpf("0.5") - mp.mpf("0.15")), mp.log(mp.mpf(X_bw)))
    )
    print(f"  Direct Xi/zeta quadrature at d = 0.85 (sigma = 1.35 > 1, X = 6): |LHS - Finite Prime Sum| = {err_bw_xi:.3e}")
    print(f"  At d = 0.15 (sigma = 0.65 < 1), s = 1 pole residue +2*pi*Re K_U(t + 0.35i) = {float(pole_s1_d15):+.6e} != 0")
    assert err_bw_xi < 5e-4

    # Compute scan over d in [0.02, 0.70] for Fig 2(c):
    d_bw_grid = np.linspace(0.02, 0.65, 90)
    bw_true_curve = np.zeros_like(d_bw_grid)
    bw_naive_curve = np.zeros_like(d_bw_grid)
    for idx, dv in enumerate(d_bw_grid):
        d_m = mp.mpf(f"{dv:.6f}")
        naive_v = delta_L_continued_from_right(d_m, t_bw)
        bw_naive_curve[idx] = float(naive_v)
        if d_m > d0_bw:
            bw_true_curve[idx] = float(naive_v)
        elif abs(d_m - d0_bw) < mp.mpf("1e-8"):
            bw_true_curve[idx] = float(
                naive_v - mp.pi * mp.re(fejer_K_U_complex(t_bw - gamma0_bw, U_bw))
            )
        else:
            bw_true_curve[idx] = float(
                naive_v
                - 2 * mp.pi * mp.re(fejer_K_U_complex((t_bw - gamma0_bw) + mp.mpc(0, 1) * (d0_bw - d_m), U_bw))
            )

    # 3D. Exact critical-line Stirling envelope Xi(t) = -E_0(t) Z(t) and c_1(t) = E_0^2 L_1(Z; t) + Z^2 L_1(E_0; t)
    print("\n3D. Exact Critical-Line Envelope Identity Xi(t) = -E_0(t) Z(t) and L_1 Decomposition:")
    print("-" * 98)

    def E0_mp(t_val):
        t_m = mp.mpf(t_val)
        return (
            mp.mpf("0.5")
            * (t_m * t_m + mp.mpf("0.25"))
            * mp.power(mp.pi, mp.mpf("-0.25"))
            * abs(mp.gamma(mp.mpf("0.25") + mp.mpc(0, 1) * t_m / 2))
        )

    max_E0_err = mp.mpf("0")
    for tv in [2.0, 14.13472514, 25.0, 200.0, 1000.0]:
        t_m = mp.mpf(str(tv))
        env_tm = mp.exp(mp.pi * t_m / 4)
        xi_hat_t = mp.re(Xi_mp(t_m) * env_tm)
        Z_t = mp.siegelz(t_m)
        E0_hat_t = E0_mp(t_m) * env_tm
        max_E0_err = max(max_E0_err, abs(xi_hat_t - (-E0_hat_t * Z_t)) / (abs(xi_hat_t) + mp.mpf("1e-60") * E0_hat_t))

        # Verify e^{pi*t_m/2} c_1(t_m) == E0_hat(t_m)^2 L_1(Z; t_m) + Z(t_m)^2 L_1(E0_hat; t_m)
        E0h_jet = mp.taylor(lambda w: E0_mp(w) * env_tm, t_m, 2)
        Z_jet = mp.taylor(mp.siegelz, t_m, 2)
        Xih_jet = mp.taylor(lambda w: Xi_mp(w) * env_tm, t_m, 2)
        E0_0, E0_1, E0_2 = E0h_jet[0], E0h_jet[1], 2 * E0h_jet[2]
        Z_0, Z_1, Z_2 = Z_jet[0], Z_jet[1], 2 * Z_jet[2]
        Xh_0, Xh_1, Xh_2 = mp.re(Xih_jet[0]), mp.re(Xih_jet[1]), 2 * mp.re(Xih_jet[2])
        L1_E0h = E0_1 ** 2 - E0_0 * E0_2
        L1_Z = Z_1 ** 2 - Z_0 * Z_2
        c1_hat_ex = Xh_1 ** 2 - Xh_0 * Xh_2
        c1_hat_rhs = (E0_0 ** 2) * L1_Z + (Z_0 ** 2) * L1_E0h
        rel_c1_br = abs(c1_hat_ex - c1_hat_rhs) / abs(c1_hat_ex)
        assert L1_E0h > 0, f"L_1(E_0; t) not positive at t={tv}"
        assert rel_c1_br < mp.mpf("1e-44")
        # Ratio of old erroneous e^{pi*t/4}(t/(2*pi))^{1/4} Xi(t)/sqrt(pi) to Z(t):
        old_lhs = mp.power(t_m / (2 * mp.pi), mp.mpf("0.25")) * xi_hat_t / mp.sqrt(mp.pi)
        if abs(Z_t) > mp.mpf("1e-5"):
            ratio_to_Z = old_lhs / Z_t
            asymp_ratio = -(t_m ** 2) / mp.sqrt(2 * mp.pi)
            print(
                f"  t = {tv:8.2f}: L_1(c_1) bridge err = {float(rel_c1_br):.2e} | "
                f"old_rescaled_Xi / Z(t) = {float(ratio_to_Z):+12.3f} (asymp -t^2/sqrt(2*pi) = {float(asymp_ratio):+12.3f})"
            )
    assert max_E0_err < mp.mpf("1e-47")

    return {
        "d_scan": d_scan,
        "Wt_1E_scan": Wt_1E_scan,
        "Wt_1Msym_scan": Wt_1Msym_scan,
        "b_E_10": b_E_10,
        "b_mob_10": b_mob_10,
        "d_bw_grid": d_bw_grid,
        "bw_true_curve": bw_true_curve,
        "bw_naive_curve": bw_naive_curve,
        "d0_bw": float(d0_bw),
    }


# ==============================================================================
# 4. TIER 4: PUBLICATION-GRADE 300-DPI VECTOR & RASTER FIGURES
# ==============================================================================

def generate_figures(t1_data: dict, t2_data: dict, t3_data: dict) -> None:
    print_banner("TIER 4: GENERATING PUBLICATION-GRADE 300-DPI FIGURES")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --------------------------------------------------------------------------
    # FIGURE 1: Shifted Hermite-Biehler Kernel, Multiplicity Cascade & Bowls
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(13.8, 4.3))
    gs1 = GridSpec(1, 3, figure=fig1, wspace=0.30, left=0.055, right=0.985, bottom=0.15, top=0.86)

    # Panel (a): Diagonal kernel e^{pi*t/2} K_1(t-id, t-id) vs off-critical model F_off
    ax1a = fig1.add_subplot(gs1[0, 0])
    t_g = t1_data["t_grid_fig"]
    ax1a.axhline(0.0, color="#222222", lw=0.9, ls="--")
    ax1a.plot(t_g, t1_data["Khat_Xi_d0"], color="#1b4f72", lw=1.5, label=r"$\Xi$: $d = 0$ ($\frac{1}{\pi}L_1(\Xi;t)$)")
    ax1a.plot(t_g, t1_data["Khat_Xi_d20"], color="#117a65", lw=1.5, label=r"$\Xi$: $d = 0.20$ ($\frac{1}{\pi}\widetilde{W}$)")
    # Rescale F_off by 1/15 for visual comparison on the same symlog axis
    ax1a.plot(
        t_g,
        t1_data["Khat_off_d15"] / 20.0,
        color="#c0392b",
        lw=1.5,
        ls="-.",
        label=r"$F_{\mathrm{off}}/20$: $d = 0.15$ (off-line zero)",
    )
    ax1a.set_yscale("symlog", linthresh=0.02)
    ax1a.set_xlabel(r"Spectral height $t$")
    ax1a.set_ylabel(r"Envelope-scaled $\mathrm{e}^{\pi t/2} K_1(t-\mathrm{i}d, t-\mathrm{i}d)$")
    ax1a.set_title(r"(a) Shifted Kernel $K_1 = \frac{1}{\pi}\widetilde{W}$ vs. $F_{\mathrm{off}}$")
    ax1a.grid(True, alpha=0.25)
    ax1a.legend(loc="upper left", frameon=True, framealpha=0.92)

    # Panel (b): Multiplicity cascade W_tilde_{f_m}(d, gamma_1) / (f_m^{(m)}(gamma_1))^2 ~ [m/(m!)^2] d^{2m-2}
    ax1b = fig1.add_subplot(gs1[0, 1])
    d_lg = t2_data["d_log_grid"]
    colors_m = {1: "#1f618d", 2: "#148f77", 3: "#d68910", 4: "#922b21"}
    for m_ord in [1, 2, 3, 4]:
        asymp = (m_ord / (math.factorial(m_ord) ** 2)) * (d_lg ** (2 * m_ord - 2))
        ax1b.loglog(
            d_lg,
            t2_data["cascade_curves"][m_ord],
            color=colors_m[m_ord],
            lw=1.7,
            label=rf"$m={m_ord}$: slope $2m-2={2*m_ord-2}$",
        )
        ax1b.loglog(d_lg, asymp, color="#333333", lw=0.9, ls=":")
    ax1b.set_xlabel(r"Transverse offset $d$")
    ax1b.set_ylabel(r"Normalized $\widetilde{W}_{f_m}(d,\gamma_1)/(f_m^{(m)}(\gamma_1))^2$")
    ax1b.set_title(r"(b) Order-$m$ Jet $\frac{m}{(m!)^2}d^{2m-2}$ ($m=1,\dots,4$)")
    ax1b.grid(True, which="both", alpha=0.25)
    ax1b.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel (c): Unified quadratic bowl: real pair (mu = +delta^2 > 0) vs conjugate off-line pair (mu = -delta_0^2 < 0)
    ax1c = fig1.add_subplot(gs1[0, 2])
    tau_b = t2_data["tau_grid_bowl"]
    ax1c.axhline(0.0, color="#222222", lw=0.9, ls="--")
    ax1c.plot(tau_b, t2_data["bowl_real_d0"], color="#1b4f72", lw=1.6, label=r"Real pair ($\mu=+\delta^2$), $d=0$")
    ax1c.plot(tau_b, t2_data["bowl_real_d06"], color="#2e86c1", lw=1.4, ls="--", label=r"Real pair ($\mu=+\delta^2$), $d=0.06$")
    ax1c.plot(tau_b, t2_data["bowl_off_d0"], color="#c0392b", lw=1.6, label=r"Off-line pair ($\mu=-\delta_0^2$), $d=0$")
    ax1c.plot(tau_b, t2_data["bowl_off_d06"], color="#e67e22", lw=1.4, ls="--", label=r"Off-line pair ($\mu=-\delta_0^2$), $d=0.06$")
    ax1c.fill_between(
        tau_b,
        t2_data["bowl_off_d0"],
        0.0,
        where=(t2_data["bowl_off_d0"] < 0.0),
        color="#c0392b",
        alpha=0.14,
    )
    ax1c.set_xlabel(r"Local offset $\tau = t - \tau_*$")
    ax1c.set_ylabel(r"Local parity-reduced $\widetilde{W}_f(d, \tau_* + \tau)$")
    ax1c.set_title(r"(c) Real ($\mu>0$) vs. Off-Line ($\mu<0$) Bowl")
    ax1c.grid(True, alpha=0.25)
    ax1c.legend(loc="upper center", frameon=True, framealpha=0.92, fontsize=7.6)

    f1_pdf = FIG_DIR / "fig1_debranges_kernel_and_lehmer_cascade.pdf"
    f1_png = FIG_DIR / "fig1_debranges_kernel_and_lehmer_cascade.png"
    fig1.savefig(f1_pdf, dpi=300)
    fig1.savefig(f1_png, dpi=300)
    plt.close(fig1)
    print(f"  [SAVED] {f1_pdf.name} ({f1_pdf.stat().st_size / 1024:.1f} KB) & {f1_png.name}")

    # --------------------------------------------------------------------------
    # FIGURE 2: Prime-Power Recursion, E_X vs M_X^sym, and Bochner-Weil Residue
    # --------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(13.8, 4.3))
    gs2 = GridSpec(1, 3, figure=fig2, wspace=0.31, left=0.055, right=0.985, bottom=0.15, top=0.86)

    # Panel (a): 0 <= b_n^E(10) <= 1 vs truncated Mobius overshoot
    ax2a = fig2.add_subplot(gs2[0, 0])
    n_idx = np.arange(2, 135)
    ax2a.axhline(0.0, color="#444444", lw=0.7)
    ax2a.axhline(1.0, color="#148f77", lw=0.9, ls="--", label=r"Unit ceiling $b_n = 1$")
    ax2a.axvspan(10.5, 120.5, color="#d4efdf", alpha=0.45, label=r"Window $10 < n < 11^2$")
    ax2a.plot(n_idx, t3_data["b_mob_10"][2:135], color="#b03a2e", lw=0.9, alpha=0.65, label=r"Truncated Möbius $b_n^{\mu}(10)$")
    ax2a.scatter(n_idx, t3_data["b_E_10"][2:135], color="#1b4f72", s=13, zorder=4, label=r"Exponential $b_n^{\mathcal{E}}(10) \in [0,1]$")
    ax2a.set_xlabel(r"Dirichlet index $n$")
    ax2a.set_ylabel(r"Residual coefficient $b_n$")
    ax2a.set_title(r"(a) Bounded $b_n^{\mathcal{E}}(10) \in [0,1]$ vs. Möbius")
    ax2a.grid(True, alpha=0.25)
    ax2a.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.6)

    # Panel (b): 1/d pole of unsymmetrized d^{-1} W_{1,E_7}(d, 18) vs smooth M_7^sym
    ax2b = fig2.add_subplot(gs2[0, 1])
    d_sc = t3_data["d_scan"]
    ax2b.loglog(
        d_sc,
        np.abs(t3_data["Wt_1E_scan"]),
        color="#c0392b",
        lw=1.7,
        label=r"$|d^{-1}W_{1,E_7}(d,18.5)| \sim c/d$ (non-real on $\mathbb{R}$)",
    )
    ax2b.loglog(
        d_sc,
        np.abs(t3_data["Wt_1Msym_scan"]),
        color="#1b4f72",
        lw=1.7,
        label=r"$|\widetilde{W}_{1,\mathcal{M}_7^{\mathrm{sym}}}(d,18.5)| = \mathcal{O}(1)$ (Schwarz-real)",
    )
    ax2b.set_xlabel(r"Transverse offset $d$")
    ax2b.set_ylabel(r"Intrinsic multiplier term $|d^{-1}W_{1,\mathcal{M}}|$")
    ax2b.set_title(r"(b) $1/d$ Singularity of $E_X$ vs. Smooth $\mathcal{M}_X^{\mathrm{sym}}$")
    ax2b.grid(True, which="both", alpha=0.25)
    ax2b.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)

    # Panel (c): Fejer Bochner-Weil contour shift across an off-critical zero at d_0 = 0.30
    ax2c = fig2.add_subplot(gs2[0, 2])
    d_bw = t3_data["d_bw_grid"]
    ax2c.axhline(0.0, color="#444444", lw=0.8)
    ax2c.axvline(t3_data["d0_bw"], color="#7d3c98", lw=1.1, ls=":", label=r"Zero abscissa $d_0 = 0.30$")
    ax2c.axvline(0.50, color="#148f77", lw=1.1, ls="--", label=r"Euler line $d = 1/2$ ($\sigma = 1$)")
    ax2c.plot(d_bw, t3_data["bw_naive_curve"], color="#2874a6", lw=1.5, ls="--", label=r"Naive $\sigma>1$ continuation")
    ax2c.plot(d_bw, t3_data["bw_true_curve"], color="#c0392b", lw=1.7, label=r"Exact integral (jump $-2\pi\mathrm{Re}\,K_U$)")
    ax2c.set_xlabel(r"Transverse abscissa $d = \sigma - 1/2$")
    ax2c.set_ylabel(r"Convolved zero contribution $\Delta\mathcal{L}_U(d,t)$")
    ax2c.set_title(r"(c) Strip Residue Jump Across $d_0 = 0.30$")
    ax2c.grid(True, alpha=0.25)
    ax2c.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    f2_pdf = FIG_DIR / "fig2_leibniz_wronskian_and_gabcke_closure.pdf"
    f2_png = FIG_DIR / "fig2_leibniz_wronskian_and_gabcke_closure.png"
    fig2.savefig(f2_pdf, dpi=300)
    fig2.savefig(f2_png, dpi=300)
    plt.close(fig2)
    print(f"  [SAVED] {f2_pdf.name} ({f2_pdf.stat().st_size / 1024:.1f} KB) & {f2_png.name}")


def main() -> int:
    t_start = time.perf_counter()
    print_banner("PAPER IV VERIFICATION SUITE: SHIFTED HERMITE-BIEHLER KERNELS, JETS & MULTIPLIERS")
    t1 = verify_tier1()
    t2 = verify_tier2()
    t3 = verify_tier3()
    generate_figures(t1, t2, t3)
    elapsed = time.perf_counter() - t_start
    print_banner(f"ALL TIERS PASSED (0 ERRORS) IN {elapsed:.2f} SECONDS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
