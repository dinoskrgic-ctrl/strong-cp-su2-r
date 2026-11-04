"""The bidoublet vacuum and the phase of the CP-breaking doublets (Secs. 4.3 and 5.6).

Three checks, each printed with the paper's equation it tests:

1. c in theta_tree = c Im z (Eq. kappaphase): the trace formula against a
   finite-difference derivative of arg det(M_u M_d), on the random textures
   of Sec. 4.2, after removing the theta = pi branch of sign-flipped
   determinants.
2. Im z from minimizing the full bidoublet potential with 60-digit
   arithmetic (Eq. imz), for random order-one couplings.
3. The relative phase phi of the two chi_R VEVs from the phase potential
   with the cubic chi Delta chi term (Eq. phiquant), under no symmetry, the
   plain exchange chi_1 <-> chi_2, and the twisted Z_4 exchange E.

Run: python bidoublet.py     (about a minute)
"""
import numpy as np
import mpmath as mp
from scipy.optimize import minimize

import common as c

mp.mp.dps = 60


# ----------------------------------------------------------------------------
# 1. the coefficient c
# ----------------------------------------------------------------------------
def fold_pi(theta):
    """theta_bar is defined modulo pi once det m can be negative: fold to [0, pi/2]."""
    a = np.abs(theta) % np.pi
    return np.minimum(a, np.pi - a)


def coefficient_c(n=400, seed=1, y=1e-8):
    rng = np.random.default_rng(seed)
    c_trace, c_fd = [], []
    for _ in range(n):
        h = c.fn_texture(rng, (4, 2, 0), 173.0)      # M_u = kappa1 h       at z = 0
        ht = c.fn_texture(rng, (2, 1, 0), 2.9)       # M_d = kappa1 h~      at z = 0
        c_trace.append(np.trace(np.linalg.inv(ht) @ h) - np.trace(np.linalg.inv(h) @ ht))
        z = 1j * y                                    # x = Re z = 0
        Mu, Md = h + ht * np.conj(z), h * z + ht
        th = np.angle(np.linalg.det(Mu) * np.linalg.det(Md))
        # remove the theta = pi branch (det h det h~ < 0), keep the sign
        th = th - np.pi * np.round(th / np.pi)
        c_fd.append(th / y)
    c_trace, c_fd = np.array(c_trace), np.array(c_fd)
    rel = np.max(np.abs(c_fd - c_trace) / np.abs(c_trace))
    a = np.abs(c_trace)
    print("[Eq. kappaphase] c = Tr(Md^-1 Mu) - Tr(Mu^-1 Md) at x = 0")
    print(f"  trace formula vs finite difference: max relative difference {rel:.1e}")
    print(f"  median |c| = {np.median(a):.0f}, 10-90% = {np.percentile(a, 10):.0f} to {np.percentile(a, 90):.0f}"
          f"   (paper: 45, 9 to 300)")
    print("  note: without the mod-pi fold about half the draws read theta = pi and |c| ~ pi/y")
    return a


# ----------------------------------------------------------------------------
# 2. the bidoublet potential
# ----------------------------------------------------------------------------
S2 = mp.matrix([[0, -1j], [1j, 0]])


def tilde(P):
    """Phi~ = sigma_2 Phi^* sigma_2."""
    Pc = mp.matrix([[mp.conj(P[i, j]) for j in range(2)] for i in range(2)])
    return S2 * Pc * S2


def dag(A):
    return A.H


def tr(A):
    return A[0, 0] + A[1, 1]


def potential(k1, X, Y, p):
    """Renormalizable potential of Phi with Delta_R and two chi_R at their VEVs (real couplings).

    Phi = diag(kappa1, X + iY),  Delta_R = [[0,0],[v_R,0]],  chi_a = (w_a e^{i theta_a}, 0).
    Terms that do not involve Phi are constants here and are dropped.
    """
    P = mp.matrix([[k1, 0], [0, mp.mpc(X, Y)]])
    Pt = tilde(P)
    D = mp.matrix([[0, 0], [p["vR"], 0]])
    chi = [mp.matrix([[p["w"][a] * mp.exp(1j * p["th"][a])], [0]]) for a in range(2)]
    T1 = tr(dag(P) * P)
    T2 = tr(Pt * dag(P))
    V = (-p["mu1"] * T1 - p["mu2"] * (T2 + mp.conj(T2))
         + p["l1"] * T1 ** 2 + p["l2"] * (T2 ** 2 + mp.conj(T2) ** 2) + p["l3"] * T2 * mp.conj(T2)
         + p["l4"] * T1 * (T2 + mp.conj(T2))
         + p["a1"] * T1 * tr(D * dag(D)) + p["a2"] * (T2 + mp.conj(T2)) * tr(D * dag(D))
         + p["a3"] * tr(dag(P) * P * D * dag(D)))
    for a in range(2):
        for b in range(2):
            cab = (dag(chi[a]) * chi[b])[0, 0]
            # chi^dag chi Tr(Phi^dag Phi), chi^dag Phi^dag Phi chi, chi^dag Phi~^dag Phi~ chi (all real)
            V += p["f1"][a][b] * cab * T1
            V += p["f2"][a][b] * (dag(chi[a]) * dag(P) * P * chi[b])[0, 0]
            V += p["f3"][a][b] * (dag(chi[a]) * dag(Pt) * Pt * chi[b])[0, 0]
            # the two portals that carry the bidoublet phase, Eq. (portal) and its trace partner
            t = p["g"][a][b] * (dag(chi[a]) * dag(P) * Pt * chi[b])[0, 0] + p["k"][a][b] * cab * T2
            V += t + mp.conj(t)
    return mp.re(V)


def random_couplings(rng, vR, w, phi):
    sym = lambda: (lambda m: (m + m.T) / 2)(rng.normal(size=(2, 2)))
    return dict(
        vR=mp.mpf(vR), w=[mp.mpf(w), mp.mpf(w)], th=[mp.mpf(0), mp.mpf(phi)],
        mu1=mp.mpf(0),                                  # fixed below by kappa1 = 174 GeV
        mu2=mp.mpf(rng.uniform(0.2, 1.0)) * vR ** 2,
        l1=mp.mpf(rng.uniform(0.1, 1)), l2=mp.mpf(rng.normal() * 0.3), l3=mp.mpf(rng.uniform(0.1, 1)),
        l4=mp.mpf(rng.normal() * 0.3),
        a1=mp.mpf(rng.uniform(0.1, 1)), a2=mp.mpf(rng.normal() * 0.3), a3=mp.mpf(rng.uniform(0.5, 2)),
        f1=sym().tolist(), f2=sym().tolist(), f3=sym().tolist(),
        g=rng.normal(size=(2, 2)).tolist(), k=rng.normal(size=(2, 2)).tolist())


H_STEP = mp.mpf("1e-35")   # finite-difference step in units of v_R


def minimize_bidoublet(p, k1):
    """Stationary point in (X, Y) with mu1^2 tuned so that kappa1 = 174 GeV; Hessian checked."""
    def grad(mu1, X, Y):
        q = dict(p, mu1=mu1)
        f = lambda a, b, cc: potential(a, b, cc, q)
        return [mp.diff(lambda t: f(t, X, Y), k1, h=H_STEP), mp.diff(lambda t: f(k1, t, Y), X, h=H_STEP),
                mp.diff(lambda t: f(k1, X, t), Y, h=H_STEP)]
    # linearized starting point
    sol = mp.findroot(lambda m, X, Y: grad(m, X, Y), (p["a1"] * p["vR"] ** 2, k1 / 10, mp.mpf(0)),
                      tol=mp.mpf(10) ** -50, maxsteps=100)
    mu1, X, Y = sol
    q = dict(p, mu1=mu1)
    H = mp.matrix(3, 3)
    v = [k1, X, Y]
    f = lambda *a: potential(*a, q)
    for i in range(3):
        for j in range(3):
            o = [0, 0, 0]
            o[i] += 1
            o[j] += 1
            H[i, j] = mp.diff(f, v, tuple(o), h=H_STEP)
    eig = mp.eigsy(H)[0]
    Dy = mp.diff(lambda t: f(k1, X, t), Y, 2, h=H_STEP) / 2              # exact Y^2 curvature: D of the paper
    return X, Y, Dy, eig


def check_imz(n=5, seed=11, vR=1e15, w=1e9):
    """All dimensionful quantities in units of v_R."""
    rng = np.random.default_rng(seed)
    k1 = mp.mpf(174) / vR
    w, vR = mp.mpf(w) / vR, 1
    print("[Eq. imz] Im z from the full potential at 60 digits, v_R = 1e15 GeV, <chi> = 1e9 GeV, phi = pi/2")
    i = rejected = 0
    while i < n:
        p = random_couplings(rng, vR, w, mp.pi / 2)
        X, Y, Dy, eig = minimize_bidoublet(p, k1)
        if not all(e > 0 for e in eig):       # a saddle, not a vacuum: draw again
            rejected += 1
            continue
        i += 1
        imz = Y / k1
        g, k = p["g"], p["k"]
        # (Phi^dag Phi~)_11 = kappa1 kappa2^* and Tr(Phi~ Phi^dag) = 2 kappa1 kappa2^*, so k counts twice
        gA = (g[0][1] - g[1][0]) / 2 + (k[0][1] - k[1][0])
        pred = -2 * gA * p["w"][0] * p["w"][1] * mp.sin(p["th"][1] - p["th"][0]) / Dy
        pred_a3 = -2 * gA * p["w"][0] * p["w"][1] / (p["a3"] * p["vR"] ** 2)
        print(f"  vacuum {i}: Im z = {mp.nstr(imz, 8)}, Eq. imz with exact D: rel. diff "
              f"{mp.nstr(abs(imz / pred - 1), 2)}; D/(alpha3 vR^2) = {mp.nstr(Dy / (p['a3'] * p['vR'] ** 2), 4)}; "
              f"Re z = {mp.nstr(X / k1, 3)}")
    print(f"  ({rejected} draws rejected because the stationary point was a saddle)")
    print("  note: g^A here is the antisymmetric part of g_ab plus twice that of k_ab, the off-diagonal")
    print("  chi_a^dag chi_b Tr(Phi~ Phi^dag) term, which is allowed and has the same vacuum structure.")
    # the phase vanishes when the antisymmetric parts vanish
    p = random_couplings(rng, vR, w, mp.pi / 2)
    p["g"] = ((np.array(p["g"]) + np.array(p["g"]).T) / 2).tolist()
    p["k"] = ((np.array(p["k"]) + np.array(p["k"]).T) / 2).tolist()
    X, Y, Dy, eig = minimize_bidoublet(p, k1)
    print(f"  symmetric g, k (exchange-symmetric potential): Im z = {mp.nstr(Y / k1, 3)}")


# ----------------------------------------------------------------------------
# 3. the relative phase of the chi_R VEVs
# ----------------------------------------------------------------------------
def phase_potential(t, h11, h22, h12, e1, e2):
    """chi-sector phase potential divided by v_R w^2, for w_1 = w_2 = w:

    cubic   h_ab chi_a^T i sigma_2 Delta_R chi_b + h.c.  ->  h11 cos 2t1 + h22 cos 2t2 + 2 h12 cos(t1+t2)
    e1 cos(t2 - t1) collects m12^2 (chi_1^dag chi_2 + h.c.) and lambda_6 (|chi_1|^2+|chi_2|^2)(chi_1^dag chi_2 + h.c.),
    e2 cos 2(t2 - t1) is lambda_5 ((chi_1^dag chi_2)^2 + h.c.).
    """
    t1, t2 = t
    return (h11 * np.cos(2 * t1) + h22 * np.cos(2 * t2) + 2 * h12 * np.cos(t1 + t2)
            + e1 * np.cos(t2 - t1) + e2 * np.cos(2 * (t2 - t1)))


def best_phi(args, rng, starts=40):
    best = None
    for _ in range(starts):
        r = minimize(phase_potential, rng.uniform(-np.pi, np.pi, 2), args=args, method="BFGS",
                     options=dict(gtol=1e-14))
        if best is None or r.fun < best.fun - 1e-15:
            best = r
    phi = np.degrees(np.abs((best.x[1] - best.x[0] + np.pi) % (2 * np.pi) - np.pi))
    return phi


def check_phiquant(n=300, seed=4, eps=1e-6):
    """eps sets the subleading quartics relative to the cubic: e ~ m12^2 / v_R."""
    rng = np.random.default_rng(seed)
    print("[Eq. phiquant] |phi| in degrees at the global minimum, subleading terms of relative size", eps)
    for case in ("E", "exchange", "none"):
        phis, shifts, signs = [], [], []
        for _ in range(n):
            h, e1, e2 = rng.normal(), eps * rng.normal(), eps * rng.normal()
            if case == "E":          # chi1 -> -i chi2, chi2 -> -i chi1: h22 = -h11, h12 = 0
                args = (h, -h, 0.0, e1, e2)
            elif case == "exchange":  # chi1 <-> chi2: h22 = h11
                args = (h, h, rng.normal(), e1, e2)
            else:
                args = (h, rng.normal(), rng.normal(), e1, e2)
            phi = best_phi(args, rng)
            phis.append(phi)
            signs.append(np.sign(args[0] * args[1]))
            if case == "E":
                shifts.append(abs(abs(phi - 90) - np.degrees(abs(e1) / (2 * abs(h)))))
        phis, signs = np.array(phis), np.array(signs)
        if case == "E":
            print(f"  Z4 exchange E: max ||phi| - 90| = {np.max(np.abs(phis - 90)):.2e} deg; "
                  f"agrees with e1/(2|h|) to {np.max(shifts):.1e} deg")
        elif case == "exchange":
            print(f"  plain exchange: |phi| in {{0, 180}} in {np.mean((phis < 1e-3) | (phis > 180 - 1e-3)):.0%} of draws")
        else:
            neg, pos = phis[signs < 0], phis[signs > 0]
            print(f"  no symmetry, h11 h22 > 0: |phi| in {{0, 180}} in "
                  f"{np.mean((pos < 1e-3) | (pos > 180 - 1e-3)):.0%} of draws")
            print(f"  no symmetry, h11 h22 < 0: |phi| - 90: median {np.median(np.abs(neg - 90)):.0f} deg, "
                  f"within 5 deg in {np.mean(np.abs(neg - 90) < 5):.0%}, within 20 deg in "
                  f"{np.mean(np.abs(neg - 90) < 20):.0%} of draws (h12 of order h11)")


if __name__ == "__main__":
    coefficient_c()
    print()
    check_imz()
    print()
    check_phiquant()
