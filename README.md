# Strong CP without axions from the center of SU(2)_R

**Dino Škrgić and Dejan Stojkovic**, Department of Physics, University at Buffalo, SUNY

Claude Opus 5.5 assisted with code writing and sentence-level editing in the manuscript. 
Prepared for submission in SciPost Physics.

## The paper in brief

A Nelson–Barr solution of the strong CP problem needs one block of the
quark mass matrix to vanish, and a symmetry robust enough to keep it zero
once gravity is included. In a left–right completion of the Standard Model,
that symmetry is the center of SU(2)_R. It is a discrete remnant of a gauge
symmetry, so gravity does not break it. The CP-violating phase
arg det M then vanishes at tree level for every choice of the Yukawa
couplings. The phase of the bidoublet vacuum is fixed when the potential
is minimized, and it is suppressed by (⟨χ_R⟩/v_R)².

If the Yukawa sector is drawn as a link diagram, the one physical phase
becomes the relative phase φ of two vacuum expectation values. A scan of
all 6696 eleven-link graphs shows that every texture compatible with the
measured mixing has α ≃ φ. A Z₄ exchange symmetry of the potential sets
φ = π/2, which gives (α, β, γ) ≃ (90°, 23.9°, 66°), with β fixed by
|V_ub/V_cb| through sin β = R_b sin α.

## Contents

```
paper/
  strongcp_scipost.tex      the draft (SciPost Physics class)
  strongcp_scipost.pdf      compiled draft
  refs.bib                  bibliography
  SciPost.cls, SciPost_bibstyle.bst
  figures/                  all five figures (vector PDF), made by code/figures.py
code/
  common.py                 inputs (quark masses at M_Z, CKM magnitudes), CKM and angle helpers, plot style
  jarlskog_test.py          random textures: arg det M = 0, size of J, one-doublet control
  bidoublet.py              the bidoublet vacuum and the phase of the chi_R VEVs (see below)
  scan.py                   enumeration and exact fits of all eleven-link graphs
  figures.py                all figures
  summary.py                the remaining numbers quoted in the text
  requirements.txt
  data/                     outputs of jarlskog_test.py and scan.py, so the figures can be redrawn without rerunning the scan
```

## How to run

You need Python 3.10 or later.

```
cd code
pip install -r requirements.txt
python jarlskog_test.py     # seconds; writes data/jarlskog.npz
python bidoublet.py         # a few minutes (60-digit arithmetic)
python scan.py              # about 30 min on 4 cores; writes data/scan.json
python figures.py           # about 2 min; writes ../paper/figures/*.pdf
python summary.py           # about 2 min
```

Every random draw uses a fixed seed, so a rerun gives the same output.
`data/` already holds the scan results, so `figures.py` and `summary.py`
run without `scan.py`.

## Where each result comes from

| result in the paper | script | what it checks |
|---|---|---|
| arg det M = 0 for every texture; median \|J\| and the one-doublet control (Sec. 4.2, Fig. 1) | `jarlskog_test.py`, `figures.py` | 400 random real Froggatt–Nielsen textures with a complex bridge |
| θ̄_tree = c Im z and the size of c (Eq. kappaphase) | `bidoublet.py` | trace formula against a finite-difference derivative of arg det(M_u M_d) |
| Im z from the potential (Eq. imz) | `bidoublet.py` | full renormalizable potential of Φ with Δ_R and both χ_R, minimized at 60 digits with κ₁ = 174 GeV, v_R = 10¹⁵ GeV, ⟨χ_R⟩ = 10⁹ GeV |
| φ = ±π/2 from the Z₄ exchange (Eq. phiquant) | `bidoublet.py` | phase potential with the cubic χΔχ term, under no symmetry, the plain exchange, and the twisted exchange E |
| leakage bound ⟨χ_R⟩ ≲ 10⁹ GeV (Fig. 2) | `figures.py` | dimension-five leakage, from the Schur complement μ − B m⁻¹Δ |
| minimal graph, α(φ) and the second solution (Sec. 5.2, Fig. 3) | `figures.py`, `summary.py` | multistart exact fits |
| β(δ) and sin β = R_b sin α (Fig. 4) | `figures.py` | |
| 26 352 / 6696 graphs, 34 classes, the four branches (Sec. 5.3, Fig. 5) | `scan.py`, `figures.py` | every exact solution of each class at φ = π/2 |
| nine-link Example I, dense textures | `summary.py` | |

## What `bidoublet.py` prints

1. **The coefficient c.** The trace formula and the finite-difference
   derivative agree to better than 10⁻⁶. Over 400 textures, the median |c|
   is about 40 and the 10–90% range is roughly 8 to 250. The exact values
   depend on the seed. θ̄ is defined modulo π, because det m can be
   negative, so the code folds the phase before taking ratios. Without the
   fold, about half the draws would read θ̄ = π.
2. **Im z.** For each random set of couplings (saddles are rejected), the
   numerical minimum agrees with −2 g^A w₁w₂ sin φ / D to about 16 digits,
   where D is the exact curvature of the potential in Im κ₂. D is of order
   α₃ v_R², with an order-one factor from the other couplings. With
   antisymmetric portal couplings set to zero, Im z = 0. The off-diagonal
   term χ₁†χ₂ Tr(Φ̃Φ†) has the same vacuum structure as the portal, so
   g^A counts both.
3. **φ.** With the twisted exchange E, |φ| = 90° up to the predicted shift
   from the subleading quartics. With the plain exchange, φ is 0 or π.
   Without a symmetry, φ is 0 or π when h₁₁h₂₂ > 0. When h₁₁h₂₂ < 0, φ
   lands near 90° only if h₁₂ is small compared with h₁₁ and h₂₂.

## Building the PDF

Any full TeX Live installation, or Overleaf, compiles `paper/strongcp_scipost.tex`
with pdflatex and bibtex. The PDF included here was built in a minimal TeX
setup with a stand-in for the mathdesign Charter fonts, so its typeface
differs from the SciPost style. The content is the same.

## Contact

Dino Škrgić, Dejan Stojkovic: Department of Physics, University at Buffalo, SUNY.
