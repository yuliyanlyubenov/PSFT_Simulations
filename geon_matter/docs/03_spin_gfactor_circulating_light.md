# Spin-½ and g = 2 from circulating light (the zitterbewegung geon)

> **Numerical companion:** `examples/sim3_circulating_light_spin_gfactor.py`
> (`L = ħ/2`, `μ = μ_B`, `g = 2` at tree level — all PASS; QED anomaly flagged
> open).

## 1. Idea

If the electron geon is **charge-carrying light circulating at `c`** (U(1)
winding = electric charge, Postulate 4), then the circulation must carry both
intrinsic angular momentum and a magnetic moment. The claim:

**a light ring of energy `mc²` circulating at `c` reproduces, at tree level,
both halves of the Dirac electron — spin `ħ/2` and magnetic moment `μ_B` with
gyromagnetic ratio `g = 2`** — with the famous factor of 2 arising from the
mismatch between the mass-energy radius and the charge-current radius. This is
the PSFT-geon realisation of the long-standing zitterbewegung / circulating-
charge model (Huang 1952; Barut–Zanghì 1984; Hestenes 1990; Williamson–van der
Mark 1997).

## 2. Math framework

### 2.1 Spin from the mass-energy ring

Energy `E₀ = mc²` circulates at speed `c` on a ring of radius `R`. On the
ring the relativistic momentum density integrates to `p = E₀/c = mc`, so the
angular momentum is

```
L = p R = m c R .
```

Quantising at half a unit, `L = ħ/2`, fixes the **mass-energy ring radius**

```
R_mass = ħ / (2 m c) = λ̄_C / 2 ,
```

half the reduced Compton wavelength. (`sim3`: `R_mass = 1.931×10⁻¹³ m`,
`L = 5.27286×10⁻³⁵ J·s = ħ/2` to 12 digits.)

### 2.2 Magnetic moment from the charge-current ring

The U(1) charge `e` circulates with period `T = 2πR/c`, giving current and
moment

```
I = e / T = e c / (2π R) ,
μ = I · (π R²) = e c R / 2 .
```

Requiring the Dirac value `μ = μ_B = eħ/(2m)` fixes the **charge-current
radius**

```
R_charge = 2 μ_B / (e c) = ħ / (m c) = λ̄_C ,
```

the *full* reduced Compton wavelength. (`sim3`: `μ = 9.27401×10⁻²⁴ J/T = μ_B`
to 12 digits.)

### 2.3 The factor of 2 = the origin of g = 2

With spin `S = ħ/2` and moment `μ = μ_B = eħ/2m`, the gyromagnetic ratio
`μ = g (e/2m) S` gives

```
g = μ / [ (e/2m)(ħ/2) ] = 2 .
```

Physically: the **charge orbits at twice the radius of the mass-energy
centroid** (`R_charge = 2 R_mass`). Equivalently, the magnetic moment "sees"
the full `λ̄_C` while the spin "sees" `λ̄_C/2`. That ratio of 2 is exactly the
tree-level `g = 2` of the Dirac electron, here given a geometric picture.
(`sim3`: `g = 2.000000`.)

## 3. Does it generalise known physics? — Partially: tree-level Dirac, yes

- Reproduces the **Dirac magnetic moment** `μ_B` and **`g = 2`**, which the
  Dirac equation predicts and which Schrödinger theory cannot.
- Reproduces **spin-½** as genuine circulating angular momentum of the
  internal light — consistent with the zitterbewegung interpretation of the
  Dirac equation (the `c`-magnitude internal velocity operator and the
  `2mc²/ħ` trembling frequency match the geon's `ω₀` from doc 01/02).
- It is a *classical-field* (tree) result. It does **not** by itself contain
  the loop structure of QED.

So it generalises the Dirac-level electron, and stops honestly there.

## 4. Does it fit experiment? — Tree level yes; the 0.12% anomaly is open

| quantity | geon tree model | experiment (CODATA) |
|---|---|---|
| spin `S` | `ħ/2` (exact) | `ħ/2` |
| moment `μ` | `μ_B` (exact) | `1.00116 μ_B` |
| `g` | `2` (exact) | `2.00231930436` |
| anomaly `a = (g−2)/2` | **0** (not captured) | `1.15965×10⁻³` |

The anomaly is the most precisely tested number in physics. The tree-level
geon gives `g = 2` exactly and therefore **misses** `a`. In QED,
`a ≈ α/2π = 1.16141×10⁻³` (Schwinger) — a *radiative* correction from the
electron interacting with its own photon field.

**Honest status:** in PSFT the photonic field is primitive and
self-interacting, so the anomaly *should* arise from the geon's coupling to
its own `P_ab` fluctuations (the analogue of the QED photon loop). Computing
`a` from PSFT's photonic self-interaction is an **open problem** — the same
open problem as quantising PSFT (paper §17). I do not claim the geon model
reproduces `a`; I claim it reproduces the tree value `g = 2` and *localises*
where the anomaly must come from.

## 5. Predictions

- **P-C1 — Charge radius = 2 × spin radius.** A falsifiable structural
  statement: the geon's charge-current loop sits at `λ̄_C`, the mass-energy
  centroid at `λ̄_C/2`. Any internal-structure probe (form-factor /
  zitterbewegung-frequency measurement) should respect this 2:1 ratio.
- **P-C2 — Zitterbewegung frequency = 2ω₀.** The trembling frequency is
  `2mc²/ħ = 2ω₀` (electron: `2.47×10²⁰ Hz`), the orbital frequency of the
  charge around the ring. Connects to doc 01's internal clock.
- **P-C3 — Anomaly from photonic self-coupling.** PSFT predicts that when the
  geon's self-interaction with `P_ab` is included, `a` emerges and its leading
  term must match `α/2π`. This is a sharp target: a PSFT computation that
  produced `a ≠ α/2π` at leading order would falsify the framework.

## 6. What this does *not* settle

- The QED anomaly (above) — open.
- Why the half-quantum `L = ħ/2` (rather than `ħ`) is the stable
  configuration — tied to the unresolved soliton spectrum.
- A relativistically rigorous treatment of a spinning light ring (the naive
  ring has internal stresses); the result here is the standard
  circulating-`c` heuristic, which gets `L`, `μ`, `g` right but is not a full
  field solution.
