# Binding the geon, and deriving "quantum" facts deterministically
(no Born rule)

> **Numerical companions:** `examples/sim9_geon_binding_stability.py` and
> `examples/sim10_deterministic_quantization.py` (all PASS).

Two threads converge here, under one methodological commitment stated up front.

**Methodological commitment (per the project's stance).** PSFT is a
*deterministic* field theory. We do **not** quantise it by adjoining a
statistical Born rule. Instead we take experimental facts that orthodox quantum
mechanics explains *statistically* and show PSFT reproduces them
*deterministically* — as field intensities, resonant standing-wave eigenmodes,
and **topological integers** (winding numbers) — typically with more physical
detail about *why* the number is what it is. This is the same move PSFT already
makes for interference (paper §13: detection ∝ field energy density, no Born
rule) and that this module made for charge (Thm 9.1(iv)) and spin (sim8):
quantisation is topology, not chance.

- **Part 1** binds the geon into a stable, finite-size soliton — the
  mass-spectrum frontier — deterministically.
- **Part 2** reframes the canonical "statistical" QM result (atomic energy
  levels) as a deterministic standing-wave/topological fact, and states
  honestly where the deterministic programme is *not yet* complete.

---

## Part 1 — Binding the geon: Derrick instability cured by viscous confinement

### 1.1 Idea & math

Sims 5–8 built the *light* (Hopfion) and its angular momentum but left it
free — and free light disperses. Why isn't a ball of light already matter? **
Derrick's theorem.** Scale a localised massless-field configuration to size
`R`; the field (gradient) energy scales as

```
E_field(R) = A ħc / R          (monotonically falling → expands/radiates away)
```

so a pure Wheeler-geon — and the propagating Hopfion of sim5/sim7 — has no
stable size. It is *radiation*, not matter.

PSFT's stabiliser is Postulate 3: the SU(3) shear viscosity switches on
(Heaviside) only at `K > Kc^strong`, i.e. only at small size / high curvature,
and by **Theorem 10.1** produces **linear confinement** with the measured QCD
string tension `σ`:

```
E(R) = A ħc/R + σR ,   minimum at   R* = √(Aħc/σ),   Mc² = 2√(Aħc σ).
```

### 1.2 Results (`sim9`, with `σ = 0.18 GeV²` from the paper)

```
pure light  E = A/R          : monotonic → disperses (Derrick)   CONFIRMED
+ viscosity E = A/R + σR      : interior minimum → bound state    CONFIRMED

R* = √(A/σ) = 2.357 GeV⁻¹ = 0.465 fm
M  = 2√(Aσ) = 0.849 GeV   = 849 MeV          (A = 1)

light hadrons:  ρ 775,  ω 782,  proton 938,  η' 958 MeV
mode-constant A ∈ [0.5,4] → M ∈ [600, 1697] MeV   (brackets the light-hadron band)
```

A geon — light that cannot escape — has a **stable, finite size ≈ 0.5 fm and a
mass ≈ 0.85 GeV**, at the hadronic scale, with no tuning beyond an `O(1)` mode
constant.

### 1.3 Generalises / fits / honest scope

- **Generalises:** this is the Derrick + bag/flux-tube energy balance, here
  sourced specifically by PSFT's viscosity (Thm 10.1). It reduces to "free
  light disperses" when the viscosity is off (`K < Kc`).
- **Fits:** the predicted size and mass land squarely on the light-hadron
  scale; the mechanism is the standard confinement physics PSFT claims to
  reproduce.
- **Honest scope:** this is an energy-balance demonstration of the
  *stabilisation mechanism and its scale*, **not** a first-principles solution
  of the v2 master equation, and **not** the electron mass (a colourless lepton
  bound in the weak/EM regime — still open, paper §17). It shows the binding
  step exists and sets the right scale; it does not yet predict individual
  masses.

---

## Part 2 — "Quantum randomness" rederived deterministically

### 2.1 Idea

The textbook showcase of quantum statistics is the discrete atomic spectrum.
PSFT supplies a deterministic mechanism (docs 01–02, sim2): a bound geon carries
a **real internal light-clock** whose lab-frame appearance is the de Broglie
wave `λ_dB = h/p`. For a *bound* geon that wave must **close on itself** — the
internal phase must be single-valued around the orbit:

```
2πr = n λ_dB = n h/(mv)   ⇔   m v r = n ħ        (n ∈ ℤ, single-valuedness)
```

This is **not** a probability postulate. It is the *same topological
single-valuedness* that quantised electric charge (Thm 9.1(iv): winding number)
and spin (sim8: Dirac integer): `n` is a winding/node number. Combined with
deterministic Coulomb force balance it gives the entire gross spectrum.

### 2.2 Results (`sim10`)

```
derived a0 = 4πε₀ħ²/(me²)      = 5.291772e-11 m   (CODATA 5.291772e-11)  ✓
derived Ry = ½α²mc²            = 13.605693 eV     (CODATA 13.605693)     ✓

 n   r_n      v_n/c     E_n (eV)    2πr/λ_dB   (= n?)
 1    a0     0.00730   -13.60569    1.000000     1
 2   4a0     0.00365    -3.40142    2.000000     2
 3   9a0     0.00243    -1.51174    3.000000     3   ...
Balmer lines 3→2 … 6→2 match measured to 2.5e-4 (0.025%)
```

The hydrogen spectrum — `a0`, the Rydberg, the level ladder `E_n = −Ry/n²`, and
the visible Balmer lines — is reproduced **deterministically**, and the
standing-wave count is an **exact integer** `n` (the winding number).

### 2.3 What this buys, and what it does not (the honest boundary)

**What is shown deterministically (more detail than the statistical account):**

| QM "statistical" fact | PSFT deterministic origin | numeric |
|---|---|---|
| discrete energy levels | single-valued internal-clock standing wave; `n` = winding | `sim10` |
| angular-momentum quantisation `L=nħ` | same single-valuedness | `sim10` |
| spin `½`, `g=2` quantum | topological (monopole/winding) field angular momentum | `sim8`/`sim3` |
| charge quantisation | compact-Killing winding number | Thm 9.1(iv) |
| interference fringes | field energy density `∝|amplitude|²`, **no Born rule** | paper §13 |
| wave–particle duality | extended internal oscillation (wave) + whole-quantum deposit (particle) | docs 01–02 |

In every case the "quantum number" is a **deterministic integer set by a
boundary/topology condition**, and the "probability" is a **deterministic field
intensity** — exactly the reframing requested.

**What is *not* claimed (and must not be overclaimed):** fine structure, the
Lamb shift (the `α/π` self-energy of doc 05 — magnitude right, coefficient
open), multi-electron atoms, and — crucially — **Bell/entanglement
correlations**. Bell's theorem is firm: any deterministic theory reproducing
QM's correlations must be **nonlocal** (or superdeterministic). PSFT *does*
contain a candidate nonlocal medium — the spacetime fluid is a real field
spanning the manifold, much as Bohmian mechanics uses a nonlocal quantum
potential — so a deterministic account is *not excluded in principle*. But
constructing the explicit PSFT account of entanglement statistics is **open**,
and nothing here demonstrates it. The deterministic programme is established for
*single-particle quantisation*; the *many-body correlation* problem is the next
hard frontier.

### 2.4 Why this is the right division of labour

Evading the Born rule does not mean denying the data; it means **explaining the
data with a deterministic mechanism wherever one exists, and naming honestly the
places where it does not yet.** PSFT's gain over the statistical account is
*mechanistic detail*: it says *why* the spectrum is discrete (single-valued
trapped-light phase), *why* charge/spin are quantised (topology), and *why*
fringes appear (energy density) — not merely *that* the probabilities are what
the Born rule assigns.

## Predictions

- **P-G1 — Hadronic geon scaling `M ∝ √σ`.** A confined light-geon's mass and
  size scale as `M = 2√(Aσ)`, `R* = √(A/σ)` with the *same* `σ` that sets the
  QCD string tension and the Regge slope. Predicts a correlation between hadron
  size and mass fixed by `σ`, testable against the light-hadron spectrum and
  lattice `σ`.
- **P-G2 — Quantisation integers are physical winding numbers.** Every quantum
  integer (atomic `n`, charge, `2S`) is a topological winding/node count of a
  real field, so they obey integer *addition/selection rules* with no
  statistical smearing — and (P-F1) odd total winding ⇒ fermion. A measured
  "quantum number" that is not expressible as such an integer would falsify the
  picture.
- **P-G3 — Determinism is testable only via nonlocality.** Because Bell forces
  nonlocality, PSFT's deterministic claim is empirically meaningful only if the
  spacetime-fluid medium produces the *specific* correlated statistics of
  entanglement. This is the decisive open test: build the PSFT two-geon
  correlation and check it reproduces the Tsirelson bound. (Open.)

## Open problems (sharpened)

- First-principles soliton solution of the v2 master equation giving individual
  masses (`e`, `m_e`, hadron spectrum) — sim9 gives only the scale.
- Deterministic account of **entanglement / Bell correlations** via the
  nonlocal spacetime fluid — the central remaining QM challenge (P-G3).
- Fine structure / Lamb shift = the `α/π` self-energy series of doc 05
  (coefficient `½` needs the quantised photonic field — but *not* a Born rule).
