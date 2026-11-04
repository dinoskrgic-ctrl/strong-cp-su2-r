"""Random Froggatt-Nielsen textures with the Nelson-Barr block structure.

Checks arg det M = 0 and the size of J for two CP-breaking doublets (and the
one-doublet control). Writes data/jarlskog.npz.
"""
import numpy as np
from pathlib import Path
import common as c

N_DRAWS, SEED = 400, 2026
MB, MT = 2.9, 173.0


def run(seed=SEED, n=N_DRAWS):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("J2", "J1", "imdet", "sv_d", "sv_u")}
    for _ in range(n):
        m = c.fn_texture(rng, (2, 1, 0), MB)
        mu_ = c.fn_texture(rng, (4, 2, 0), MT)
        # bridge with |B| = mu (mu = 1 units): two doublets -> independent phases
        B2 = rng.normal(size=3) + 1j * rng.normal(size=3)
        B2 /= np.linalg.norm(B2)
        # one doublet: a single common phase, removable by rephasing D_L
        B1 = rng.normal(size=3) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        B1 /= np.linalg.norm(B1)
        M = np.zeros((4, 4), complex)
        M[:3, :3], M[3, :3], M[3, 3] = m, B2, 1.0
        d = np.linalg.det(M)
        out["imdet"].append(abs(d.imag) / abs(d))
        for key, B in (("J2", B2), ("J1", B1)):
            su, sd, V = c.ckm(mu_, c.heff(m, B))
            out[key].append(abs(c.jarlskog(V)))
            if key == "J2":
                out["sv_d"].append(sd)
                out["sv_u"].append(su)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    r = run()
    Path("data").mkdir(exist_ok=True)
    np.savez("data/jarlskog.npz", **r)
    J = r["J2"]
    print(f"max |Im det|/|det| = {r['imdet'].max():.1e}")
    print(f"n_chi=2: median |J| = {np.median(J):.2e}, 10-90% = {np.percentile(J, 10):.1e}-{np.percentile(J, 90):.1e}")
    print(f"measured J at percentile {100 * np.mean(J < c.J_MEAS):.0f}")
    print(f"n_chi=1: max |J| = {r['J1'].max():.1e}")
    print("median singular values down", np.median(r["sv_d"], 0), "up", np.median(r["sv_u"], 0))
