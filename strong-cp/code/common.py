"""Shared inputs and helpers for the figures of the strong-CP / Nelson-Barr paper."""
import numpy as np

# --- inputs (MS-bar at M_Z, GeV) and CKM magnitudes ---------------------------
M_DOWN = np.array([0.0028, 0.055, 2.89])
M_UP = np.array([0.0013, 0.62, 172.0])
VUS, VCB, VUB = 0.2243, 0.0410, 0.00382
J_MEAS = 3.08e-5
ANGLES_MEAS = dict(alpha=(91.0, 2.8), beta=(22.6, 0.5), gamma=(66.4, 2.8))
LAM = 0.225  # Froggatt-Nielsen expansion parameter


def left_rotation(MMdag):
    """Eigenvalues (ascending) and eigenvectors of a Hermitian mass-squared matrix."""
    w, U = np.linalg.eigh(MMdag)
    return w, U


def heff(m, B):
    """Light left-handed mass-squared matrix after integrating out D (mu = 1 units)."""
    B = np.atleast_2d(B)
    return m @ m.conj().T - (m @ B.conj().T @ B @ m.conj().T) / (np.vdot(B, B).real + 1.0)


def ckm(Mu, H_d):
    """CKM matrix from the up mass matrix and the down left-handed mass-squared matrix."""
    wu, Uu = left_rotation(Mu @ Mu.conj().T)
    wd, Ud = left_rotation(H_d)
    return np.sqrt(np.abs(wu)), np.sqrt(np.abs(wd)), Uu.conj().T @ Ud


def jarlskog(V):
    return np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1]))


def ut_angles(V):
    """(alpha, beta, gamma) in degrees, on the branch J > 0."""
    if jarlskog(V) < 0:
        V = V.conj()
    a = np.angle(-V[2, 0] * np.conj(V[2, 2]) / (V[0, 0] * np.conj(V[0, 2])))
    b = np.angle(-V[1, 0] * np.conj(V[1, 2]) / (V[2, 0] * np.conj(V[2, 2])))
    g = np.angle(-V[0, 0] * np.conj(V[0, 2]) / (V[1, 0] * np.conj(V[1, 2])))
    return np.degrees([a, b, g]) % 360


def ckm_standard(s12, s23, s13, delta):
    c12, c23, c13 = np.sqrt(1 - s12**2), np.sqrt(1 - s23**2), np.sqrt(1 - s13**2)
    e = np.exp(1j * delta)
    return np.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])


def ckm_from_magnitudes(vus, vcb, vub, delta):
    s13 = vub
    c13 = np.sqrt(1 - s13**2)
    return ckm_standard(vus / c13, vcb / c13, s13, delta)


def fn_texture(rng, charges, scale):
    """Real Froggatt-Nielsen texture with random signs, |c| log-uniform in [1/2, 2]."""
    a = np.asarray(charges)
    c = rng.choice([-1.0, 1.0], (3, 3)) * np.exp(rng.uniform(np.log(0.5), np.log(2.0), (3, 3)))
    m = c * LAM ** np.add.outer(a, a)
    return m * scale / np.linalg.svd(m, compute_uv=False)[0]


def setup_style():
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the styles)
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight"})
    return plt
