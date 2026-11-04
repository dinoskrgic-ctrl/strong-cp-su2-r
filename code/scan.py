"""Enumerate eleven-link Yukawa graphs with the SU(2)_R-enforced zero and fit them.

Nodes: q_1..3, D_L (left); u^c_1..3, d^c_1..3, D^c (right).
Links: up Yukawas q-u^c, down Yukawas q-d^c, bridges D_L-d^c, and the mandatory
Dirac mass D_L-D^c. The links q-D^c are absent (SU(2)_R doublet, no invariant).
Writes data/scan.json.
"""
import itertools as it
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import common as c

PHI = np.pi / 2


# ----------------------------------------------------------------- enumeration
def has_perfect_matching(mask):
    return any(all(mask[i, p[i]] for i in range(3)) for p in it.permutations(range(3)))


def adjacency(Yu, Ym, Bv):
    # nodes: 0-2 q, 3 D_L, 4-6 u^c, 7-9 d^c, 10 D^c
    adj = {k: set() for k in range(11)}
    def link(a, b):
        adj[a].add(b); adj[b].add(a)
    for i, j in zip(*np.nonzero(Yu)):
        link(i, 4 + j)
    for i, j in zip(*np.nonzero(Ym)):
        link(i, 7 + j)
    for j in np.nonzero(Bv)[0]:
        link(3, 7 + j)
    link(3, 10)
    return adj


def connected(Yu, Ym, Bv):
    adj = adjacency(Yu, Ym, Bv)
    seen, stack = {0}, [0]
    while stack:
        for nb in adj[stack.pop()] - seen:
            seen.add(nb); stack.append(nb)
    return len(seen) == 11


def enumerate_graphs(min_bridges=2):
    masks = [np.array(b, bool).reshape(3, 3) for b in it.product([0, 1], repeat=9)]
    pm = [m for m in masks if has_perfect_matching(m)]
    out = []
    for nB in range(1, 4):
        for Bv in it.product([0, 1], repeat=3):
            if sum(Bv) != nB:
                continue
            Bv = np.array(Bv, bool)
            for Yu in pm:
                for Ym in pm:
                    if Yu.sum() + Ym.sum() + nB != 10:
                        continue
                    if connected(Yu, Ym, Bv):
                        out.append((Yu, Ym, Bv))
    allg = len(out)
    return [g for g in out if g[2].sum() >= min_bridges], allg


def canonical(g):
    Yu, Ym, Bv = g
    best = None
    for pq in it.permutations(range(3)):
        for pu in it.permutations(range(3)):
            for pd in it.permutations(range(3)):
                key = (Yu[np.ix_(pq, pu)].tobytes() + Ym[np.ix_(pq, pd)].tobytes()
                       + Bv[list(pd)].tobytes())
                if best is None or key < best:
                    best = key
    return best


def cycle_nodes(g):
    """Nodes on the unique cycle (prune leaves until none are left)."""
    adj = {k: set(v) for k, v in adjacency(*g).items()}
    leaves = [k for k, v in adj.items() if len(v) == 1]
    while leaves:
        k = leaves.pop()
        for nb in adj.pop(k):
            adj[nb].discard(k)
            if len(adj[nb]) == 1:
                leaves.append(nb)
    return set(adj)


def phase_bridge(g):
    """A bridge on the cycle carries the phase; if none is on it, the phase is unphysical."""
    cyc = cycle_nodes(g)
    on = [j for j in np.nonzero(g[2])[0] if 3 in cyc and 7 + j in cyc]
    return on[-1] if on else np.nonzero(g[2])[0][-1]


# ----------------------------------------------------------------------- model
def build(g, x, phi=PHI):
    """x = log-magnitudes of up links, down links, then log(b/mu)."""
    Yu, Ym, Bv = g
    nu, nm = Yu.sum(), Ym.sum()
    Mu = np.zeros((3, 3)); Mu[Yu] = np.exp(x[:nu])
    m = np.zeros((3, 3)); m[Ym] = np.exp(x[nu:nu + nm])
    b = np.exp(x[-1])
    B = np.zeros(3, complex); B[Bv] = b
    B[phase_bridge(g)] *= np.exp(1j * phi)   # the physical phase
    return Mu, m, B


def observables(g, x, phi=PHI):
    Mu, m, B = build(g, x, phi)
    su, sd, V = c.ckm(Mu.astype(complex), c.heff(m.astype(complex), B))
    return np.r_[su, sd, abs(V[0, 1]), abs(V[1, 2]), abs(V[0, 2])], V


TARGET = np.r_[c.M_UP, c.M_DOWN, c.VUS, c.VCB, c.VUB]


def residuals(x, g, phi=PHI):
    obs, _ = observables(g, x, phi)
    return np.log(np.maximum(obs, 1e-30) / TARGET)


def fit(g, n_starts=500, seed=0, phi=PHI, tol=1e-16):
    """Multi-start bounded least squares; returns every distinct exact solution branch.

    A class can have several discrete exact solutions of the nine magnitude
    equations; each is returned with its angles, J and b/mu.
    """
    n = g[0].sum() + g[1].sum() + 1
    lb, ub = [-12.0] * (n - 1) + [-6.0], [6.0] * (n - 1) + [4.0]
    rng = np.random.default_rng(seed)
    best_cost, branches = np.inf, {}
    for _ in range(n_starts):
        x0 = np.r_[rng.uniform(-8, 5, n - 1), rng.uniform(-4, 3)]
        try:
            ls = least_squares(residuals, x0, args=(g, phi), bounds=(lb, ub),
                               xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        except (np.linalg.LinAlgError, ValueError):
            continue
        cost = float(np.sum(ls.fun ** 2))
        best_cost = min(best_cost, cost)
        if cost > tol:
            continue
        _, V = observables(g, ls.x, phi)
        J = float(c.jarlskog(V))
        ang = c.ut_angles(V)
        key = "cpc" if abs(J) < 1e-8 else tuple(np.round(ang, 1))
        br = branches.setdefault(key, dict(count=0, J=J, angles=ang.tolist(),
                                           b_over_mu=float(np.exp(ls.x[-1])), x=ls.x.tolist()))
        br["count"] += 1
    return dict(best_cost=best_cost, branches=list(branches.values()))


def classify(res):
    if not res["branches"]:
        return "excluded"
    if all(abs(b["J"]) < 1e-8 for b in res["branches"]):
        return "cp_conserving"
    return "cp_violating"


def _work(args):
    k, g = args
    r = fit(g, seed=k)
    r.update(index=k, nu=int(g[0].sum()), nm=int(g[1].sum()), nB=int(g[2].sum()),
             Yu=g[0].astype(int).tolist(), m=g[1].astype(int).tolist(),
             B=g[2].astype(int).tolist(), outcome=classify(r))
    return r


if __name__ == "__main__":
    graphs, n_all = enumerate_graphs()
    classes = {}
    for g in graphs:
        classes.setdefault(canonical(g), g)
    print(f"graphs without n_B>=2 cut: {n_all}; with: {len(graphs)}; classes: {len(classes)}")
    from multiprocessing import Pool
    with Pool() as pool:
        rows = pool.map(_work, list(enumerate(classes.values())))
    for r in rows:
        print(r["index"], (r["nu"], r["nm"], r["nB"]), r["outcome"],
              [(np.round(b["angles"], 2).tolist(), b["count"]) for b in r["branches"]], flush=True)
    Path("data").mkdir(exist_ok=True)
    json.dump(dict(n_all=n_all, n_graphs=len(graphs), classes=rows),
              open("data/scan.json", "w"), indent=1)
