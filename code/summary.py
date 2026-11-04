"""Numbers quoted in the text that are not produced by jarlskog_test.py or scan.py."""
import numpy as np
from scipy.optimize import least_squares

import common as c
import figures as f
import scan
import bidoublet


def minimal_graph_numbers():
    g = f.minimal_graph()
    res = scan.fit(g, n_starts=200, seed=5)
    for b in res["branches"]:
        print("minimal graph branch: angles", np.round(b["angles"], 2),
              " |B|/mu =", round(b["b_over_mu"] * np.sqrt(2), 2))
    # alpha - phi along the branch through alpha ~ 90 deg
    br = max(res["branches"], key=lambda b: b["angles"][0])
    devs = []
    for direction in (np.linspace(90, 22.5, 28), np.linspace(90, 112.5, 10)):
        x = np.array(br["x"])
        for p in np.radians(direction):
            x = least_squares(scan.residuals, x, args=(g, p), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
            _, V = scan.observables(g, x, p)
            devs.append(abs(c.ut_angles(V)[0] - np.degrees(p)))
    print(f"max |alpha - phi| for phi in [pi/8, 5pi/8]: {max(devs):.2f} deg")


def nine_link_example():
    """Example I of Arkani-Hamed et al.: Y_u on (12),(21),(22),(33), m on (12),(21),(22),(23),(33)."""
    iu, im = [(0, 1), (1, 0), (1, 1), (2, 2)], [(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)]

    def obs(x):
        Mu = np.zeros((3, 3), complex)
        m = np.zeros((3, 3))
        for k, ij in enumerate(iu):
            Mu[ij] = np.exp(x[k])
        Mu[0, 1] *= 1j                                  # phase pi/2 on Y^u_12
        for k, ij in enumerate(im):
            m[ij] = np.exp(x[4 + k])
        su, sd, V = c.ckm(Mu, (m @ m.T).astype(complex))
        return np.r_[su, sd, abs(V[0, 1]), abs(V[1, 2]), abs(V[0, 2])], V

    rng = np.random.default_rng(1)
    found = {}
    for _ in range(300):
        try:
            ls = least_squares(lambda x: np.log(obs(x)[0] / scan.TARGET), rng.uniform(-8, 5, 9),
                               bounds=(-12, 6), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.sum(ls.fun ** 2) < 1e-16:
            found.setdefault(tuple(np.round(c.ut_angles(obs(ls.x)[1]), 2)), 0)
    print("nine-link Example I exact solutions (alpha, beta, gamma):", list(found))


def dense_textures(n=600):
    rng = np.random.default_rng(3)
    al = []
    for _ in range(n):
        m = c.fn_texture(rng, (2, 1, 0), 2.9)
        mu_ = c.fn_texture(rng, (4, 2, 0), 173.0)
        B = np.abs(rng.normal(size=3)).astype(complex)
        B[2] *= 1j                                       # relative bridge phase pi/2
        B /= np.linalg.norm(B)
        _, _, V = c.ckm(mu_, c.heff(m, B))
        al.append(c.ut_angles(V)[0])
    print("dense textures: central 68% of alpha:", np.round(np.percentile(al, [16, 84]), 0))


if __name__ == "__main__":
    minimal_graph_numbers()
    nine_link_example()
    dense_textures()
    bidoublet.coefficient_c()   # the bidoublet coefficient c (Eq. kappaphase)
    print("leakage: median |dtheta| / (<chi>^2/(M_Pl mu)):", f.fig_leakage())
