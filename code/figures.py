"""Make all figures of the paper (scienceplots style, no titles).

Run jarlskog_test.py and scan.py first; they write data/*.npz and data/scan.json.
Usage: python figures.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import common as c
import scan

plt = c.setup_style()
OUT = Path("../paper/figures")
OUT.mkdir(exist_ok=True)
C1, C2, C3, CG = "#0C5DA5", "#FF2C00", "#00B945", "0.55"


def fold_pi(theta):
    a = abs(theta) % np.pi
    return min(a, np.pi - a)


def fig_jarlskog():
    d = np.load("data/jarlskog.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    a.hist(np.log10(d["J2"]), bins=30, color=C1, alpha=0.8)
    a.axvline(np.log10(c.J_MEAS), color="k", ls="--", lw=1, label=r"measured $J$")
    a.set_xlabel(r"$\log_{10}|J|$")
    a.set_ylabel("textures")
    a.legend(loc="upper left", frameon=False)
    # panel b: the texture zero versus a restored forbidden link
    rng = np.random.default_rng(7)
    th0, th1 = [], []
    for _ in range(400):
        m = c.fn_texture(rng, (2, 1, 0), 2.9)
        B = rng.normal(size=3) + 1j * rng.normal(size=3)
        B /= np.linalg.norm(B)
        M = np.zeros((4, 4), complex)
        M[:3, :3], M[3, :3], M[3, 3] = m, B, 1.0
        # theta_bar is defined mod pi: a negative real det m means theta = 0, not pi
        th0.append(fold_pi(np.angle(np.linalg.det(M))))
        M[:3, 3] = 1e-6 * rng.normal(size=3)        # restore q-D^c with epsilon = 1e-6
        th1.append(fold_pi(np.angle(np.linalg.det(M))))
    bins = np.linspace(-18, -2, 41)
    b.hist(np.log10(np.maximum(th0, 1e-18)), bins=bins, color=C1, alpha=0.8,
           label=r"$\bar q\Phi D^c$ absent")
    b.hist(np.log10(th1), bins=bins, color=C2, alpha=0.8,
           label=r"restored, $\epsilon=10^{-6}$")
    b.set_xlabel(r"$\log_{10}|\bar\theta_{\rm tree}|$")
    b.legend(loc="upper left", frameon=False)
    for ax, t in ((a, "(a)"), (b, "(b)")):
        ax.text(0.97, 0.93, t, transform=ax.transAxes, ha="right")
    fig.savefig(OUT / "fig_jarlskog.pdf")
    plt.close(fig)


def fig_beta_delta():
    delta = np.radians(np.linspace(0, 180, 721))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhspan(22.1, 23.1, color=CG, alpha=0.25, lw=0)
    for vub, col in ((0.00382, C1), (0.00370, C2), (0.00360, C3)):
        beta = [c.ut_angles(c.ckm_from_magnitudes(c.VUS, c.VCB, vub, d))[1] for d in delta]
        ax.plot(np.degrees(delta), beta, color=col,
                label=rf"$|V_{{ub}}|={vub * 1e3:.2f}\times10^{{-3}}$")
        # mark the point with alpha = 90 deg on each curve
        al = np.array([c.ut_angles(c.ckm_from_magnitudes(c.VUS, c.VCB, vub, d))[0] for d in delta])
        i = np.argmin(abs(al - 90))
        ax.plot(np.degrees(delta[i]), beta[i], "o", color=col, ms=3.5)
    ax.set_xlabel(r"$\delta$ [deg]")
    ax.set_ylabel(r"$\beta$ [deg]")
    ax.set_xlim(20, 140)
    ax.set_ylim(14, 25)
    ax.legend(loc="lower center", frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_beta_delta.pdf")
    plt.close(fig)


def minimal_graph():
    Yu = np.eye(3, dtype=bool)
    m = np.zeros((3, 3), bool)
    for i, j in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]:
        m[i, j] = True
    return Yu, m, np.array([0, 1, 1], bool)


def fig_alpha_phi():
    g = minimal_graph()
    start = scan.fit(g, n_starts=120, seed=3)["branches"]
    phis = np.radians(np.linspace(90, 22.5, 28)), np.radians(np.linspace(90, 112.5, 10))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for br, col in zip(sorted(start, key=lambda b: -b["angles"][0]), (C1, C2)):
        pts = []
        for branch_phis in phis:
            x = np.array(br["x"])
            for p in branch_phis:
                ls = least_squares(scan.residuals, x, args=(g, p), xtol=1e-15, ftol=1e-15, gtol=1e-15)
                if np.sum(ls.fun ** 2) > 1e-14:
                    break
                x = ls.x
                _, V = scan.observables(g, x, p)
                pts.append((np.degrees(p), c.ut_angles(V)[0]))
        pts = np.array(sorted(pts))
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=col, ms=2.5, lw=1,
                label=rf"solution at $\varphi=\pi/2$: $\alpha={br['angles'][0]:.1f}^\circ$")
    ax.plot([20, 115], [20, 115], color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\varphi$ [deg]")
    ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_alpha_phi.pdf")
    plt.close(fig)
    return pts


def fig_classes():
    d = json.load(open("data/scan.json"))
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.axhspan(22.1, 23.1, color=CG, alpha=0.25, lw=0)
    first = dict(excluded=True, cpc=True, cpv=True, near=True)
    for r in d["classes"]:
        k = r["index"]
        if r["outcome"] == "excluded":
            ax.plot(k, -2, "x", color=CG, ms=4, label="no exact fit" if first.pop("excluded", 0) else None)
            continue
        for b in r["branches"]:
            if abs(b["J"]) < 1e-8:
                ax.plot(k, 0, "s", color="#FF9500", ms=3,
                        label=r"exact, $J=0$" if first.pop("cpc", 0) else None)
            else:
                a, be, ga = b["angles"]
                near = abs(a - 91.0) < 3 * 2.8 and abs(ga - 66.4) < 3 * 2.8
                ax.plot(k, be, "o", ms=3.5, color=C1 if near else C2, mfc=C1 if near else "none",
                        label=(r"exact, $\alpha,\gamma$ within $3\sigma$" if near and first.pop("near", 0)
                               else ("exact, other" if (not near) and first.pop("cpv", 0) else None)))
    ax.set_xlabel("class index")
    ax.set_ylabel(r"$\beta$ [deg]")
    ax.set_xlim(-1, len(d["classes"]))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_classes.pdf")
    plt.close(fig)


def fig_leakage():
    MPL = 1.22e19
    rng = np.random.default_rng(11)
    res = {"aligned": [], "anarchic": []}
    for case in res:
        for _ in range(300):
            m = c.fn_texture(rng, (2, 1, 0), 2.9)
            mu = 10 ** rng.uniform(4, 13)
            F = 10 ** rng.uniform(-1, 0)
            w = mu / F                                    # |<chi>|, so that |B| ~ mu
            vev = w * np.array([1.0, np.exp(1j * rng.uniform(0, 2 * np.pi))])
            Fab = rng.normal(size=(2, 3)) * F / np.sqrt(2)
            k = rng.normal(size=(2, 3))
            M = np.zeros((4, 4), complex)
            M[:3, :3] = m
            M[3, :3] = vev @ Fab
            M[3, 3] = mu
            if case == "aligned":      # coefficient matrix aligned with the down Yukawas
                M[:3, 3] = m @ (np.conj(vev) @ k) / MPL
            else:                      # anarchic, with an overall y_b-sized coefficient
                M[:3, 3] = 2.9 * (np.conj(vev) @ k) / MPL
            # det M = det(m) (mu - B m^-1 Delta): use the Schur complement, since a direct
            # 4x4 determinant loses all precision for mu/m_d ~ 1e15
            schur = M[3, 3] - M[3, :3] @ np.linalg.solve(m, M[:3, 3])
            res[case].append((w ** 2 / mu, abs(np.angle(schur))))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for case, col in (("aligned", C1), ("anarchic", C2)):
        x, y = np.array(res[case]).T
        ax.loglog(x, y, ".", color=col, ms=2.5, alpha=0.6, label=f"{case} flavor")
    xs = np.logspace(5, 15, 10)
    ax.loglog(xs, xs / MPL, "k-", lw=0.8, label=r"$\langle\chi\rangle^2/(M_{\rm Pl}\mu)$")
    ax.axhline(1e-10, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\langle\chi_R\rangle^2/\mu$ [GeV]")
    ax.set_ylabel(r"$|\delta\bar\theta|$")
    ax.set_xlim(1e5, 1e15)
    ax.set_ylim(1e-18, 1e-2)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_leakage.pdf")
    plt.close(fig)
    out = {}
    for case in res:
        x, y = np.array(res[case]).T
        out[case] = float(np.median(y / (x / MPL)))
    return out


if __name__ == "__main__":
    fig_jarlskog()
    fig_beta_delta()
    print("alpha(phi) last points:", fig_alpha_phi()[-3:])
    print("leakage median ratio to <chi>^2/(M_Pl mu):", fig_leakage())
    if Path("data/scan.json").exists():
        fig_classes()
