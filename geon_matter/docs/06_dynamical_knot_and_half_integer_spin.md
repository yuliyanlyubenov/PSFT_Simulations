# The knot in motion (full Maxwell dynamics) and half-integer spin
from topological charge

> **Numerical companions:** `examples/sim7_time_dependent_hopfion.py` (dynamical
> Maxwell, all PASS) and `examples/sim8_charge_monopole_half_integer_spin.py`
> (half-integer field angular momentum, all PASS).

Doc 05 left two precise loose ends. This document closes both:

1. **sim5 verified only the `t = 0` snapshot** — the Maxwell *constraints*
   (`∇·E = ∇·B = 0`) and nullness. Does the knotted light-geon satisfy the
   full *dynamics* (Faraday + Ampère) as it evolves? → **Part 1.**
2. **sim5's honest gap:** a bosonic Maxwell field carries *integer* angular
   momentum, so where does the electron's *half-integer* spin come from? →
   **Part 2**, which turns out to be answered by PSFT's own Postulate 4.

---

## Part 1 — The moving Hopfion is an exact *dynamical* Maxwell solution

### 1.1 Idea & math

Use the full time-dependent Bateman scalars (Kedia et al. 2013), natural units
`c = 1`, scale `a = 1`:

```
P     = r² − (t − i)² = (r² − t² + 1) + 2i t
α     = (r² − t² − 1 + 2iz) / P ,    β = 2(x − iy) / P
F     = E + iB = ∇α × ∇β .
```

In Riemann–Silberstein form the two *dynamical* Maxwell equations
(`∂_tE = ∇×B`, `∂_tB = −∇×E`) combine into the single complex equation

```
∂F/∂t = −i ∇×F .
```

`sim7` evaluates `F` from **analytic** spatial gradients (exact), then forms
the residual `R = ∂_tF + i∇×F` with one centred finite difference in time and
one in space — so any nonzero `R` is pure discretisation error.

### 1.2 Results

```
   t  | null RMS(F·F)/scl | Maxwell resid (rel) |    U       L_z
 0.00 |       1.9e-15     |      1.51e-02       | 19.734   -9.863
 0.50 |       1.6e-15     |      1.51e-02       | 19.734   -9.863
 1.00 |       1.2e-15     |      1.51e-02       | 19.733   -9.862
 1.50 |       9.6e-16     |      1.51e-02       | 19.731   -9.860
```

- **Stays null for all time** (`F·F → 0` at `10⁻¹⁵`).
- **Dynamical Maxwell holds:** the residual is `1.5×10⁻²` at `N=91` and drops
  to `8.5×10⁻³` at `N=121` — ratio `0.56 ≈ (dx₂/dx₁)² = 0.5625`, i.e. clean
  **2nd-order convergence to zero**. Faraday + Ampère are satisfied exactly;
  the residual is only the finite-difference stencil.
- **Conserved charges** `U` and `L_z` drift `< 0.03%` over `t ∈ [0, 1.5]` (the
  small drift is energy physically propagating out of the finite box as the
  knot moves).

### 1.3 Verdict

The knotted light-geon is a **bona-fide, self-consistent, propagating solution
of the full Maxwell dynamics** — not just a static snapshot. It lives natively
in the inviscid U(1) sector PSFT derives via Theorem 9.1. This is the
legitimate *propagating* precursor of a trapped geon: to make it stationary one
adds the high-`K` SU(3) viscosity (paper §7.4) as the binding mechanism.

---

## Part 2 — Half-integer spin from a topological charge (closing sim5's gap)

### 2.1 Idea

sim5 showed a pure bosonic Maxwell field carries integer angular momentum, and
flagged that the electron's spin-½ needs "extra structure." **That extra
structure is exactly what PSFT already postulates:** Postulate 4 says electric
charge *is* a topological winding number (and colour/isospin are topological
classes). A winding/monopole-type topological charge, combined with an electric
charge, makes the *electromagnetic field itself* carry **half-integer** angular
momentum — the classic "spin from a charge–monopole system" (Thomson 1904;
Saha 1936; "spin from isospin," Jackiw–Rebbi 1976; Goldhaber).

### 2.2 Math framework

For an electric charge `q_e` and a magnetic monopole `q_m` (Heaviside–Lorentz
units, `c = 1`),

```
E = (q_e/4π)(r−r_e)/|r−r_e|³ ,   B = (q_m/4π)(r−r_m)/|r−r_m|³ ,
L = ∫ r×(E×B) d³x  =  (q_e q_m / 4π) ẑ ,
```

directed along the axis joining the charges and — remarkably —
**independent of their separation**. Dirac's quantization condition
`q_e q_m = 2π n` (HL units, `ħ = 1`) then gives

```
L_z = (q_e q_m)/(4π) = (2π n)/(4π) = n/2   [units of ħ]   →   HALF-INTEGER.
```

The minimal monopole (`n = 1`) gives `L_z = ħ/2` — the electron's spin.

### 2.3 Results (`sim8`)

The field-angular-momentum integral has a slow `1/r` tail, so a finite box
undercounts it. `sim8` handles this rigorously:

```
(1) box-size convergence at fixed separation, Richardson L(R)=L∞−C/R:
     R= 6  |L_z|/(qq/4π)=0.909
     R= 9                =0.938
     R=12                =0.954
     R=18                =0.969
     Richardson L∞ = −0.079458   vs   analytic −0.079577   →  ratio 0.9985  ✓
(2) axiality:  L = [2e-8, −8e-6, −0.0723]  →  L_z fraction 0.99989           ✓
(3) separation-independence: deficit (L∞ − L_z(2a)) is linear in 2a with
     R² = 0.9999  →  the TRUE L_z is separation-independent (Saha/Thomson)   ✓
```

So the field-theoretic angular momentum equals `q_e q_m/4π` to 0.15%, is purely
axial, and is genuinely separation-independent — and Dirac quantization makes
it `nħ/2`.

### 2.4 Does it generalise / fit? — Yes, and it is the standard result

- **Generalises:** this is textbook classical electromagnetism + the Dirac
  quantization condition; it reproduces the well-known half-integer
  charge–monopole angular momentum.
- **Fits:** `L = ħ/2` for the minimal monopole is exactly the electron spin;
  the mechanism ("spin from isospin/topology") is established physics used in
  monopole and soliton physics (e.g. dyons, Skyrmions-as-fermions).
- **PSFT fit:** Postulate 4 (charge = topological winding) is precisely the
  monopole-like structure required. So **the electron geon's spin-½ is not an
  ad-hoc add-on — it is the field angular momentum sourced by the geon's own
  topological charge**, in a purely bosonic primitive photonic field. This is
  fully consistent with Postulate 1 ("matter = solitonic/topological pattern of
  the primitive photonic field").

### 2.5 How this reframes the doc-03 / doc-05 story

| stage | spin picture | status |
|---|---|---|
| doc 03 | point charge on a ring, `L = mcR = ħ/2` (heuristic) | suggestive |
| doc 05 (`sim5`) | bosonic light knot — *integer* angular momentum | rigorous, but spin-1 |
| **doc 06 (`sim8`)** | charge + **topological** winding ⇒ field `L = nħ/2` | **rigorous, half-integer** |

The half-integer spin is now derived (given a topological charge) rather than
imposed.

## Predictions

- **P-F1 — Spin-½ ⇔ unit topological charge.** In the geon picture, a
  particle's half-integer spin is locked to it carrying an *odd* unit of
  topological (winding/monopole) charge: `2S = n` (Dirac integer). Charged
  leptons/quarks (`n` odd) are fermions; pure-field excitations without net
  winding (photons, the Hopfion) are bosons. This ties the spin–statistics
  assignment to Postulate 4's topology — a falsifiable structural claim.
- **P-F2 — No isolated half-integer spin without topological charge.** A
  neutral, winding-free PSFT light configuration must be bosonic (integer
  spin). Observation of a fundamental, topologically-trivial, half-integer-spin
  light state would falsify the mechanism.
- **P-F3 — Dynamical knots are radiation, not matter.** The propagating Hopfion
  (sim7) carries integer spin and is *not* trapped; it is a radiation state.
  Only the SU(3)-viscosity-trapped, topologically-charged configuration is
  matter. Predicts a sharp ontological divide: free knotted light = boson;
  trapped winding geon = fermion.

## What remains open (honest)

- A **first-principles PSFT soliton** that literally carries this winding *and*
  reproduces the measured `e`, `m_e`, and the `λ̄_C` charge radius — i.e. the
  mass-spectrum problem (paper §17). sim8 imports the charge–monopole result
  into the PSFT setting; it does not yet construct the geon from the master
  equation.
- The **spin–statistics theorem** in PSFT (why odd winding ⇒ Fermi statistics,
  not just half-integer spin) — needs the quantised theory.
- Tying the trapped (stationary) geon to the propagating Hopfion explicitly via
  the high-`K` viscosity (the binding step) — numerically demonstrated only as
  a trapping geometry so far (examples 30/32).
