# From heuristic ring to a *genuine field-theoretic* geon, and the
g−2 anomaly

> **Numerical companions:** `examples/sim5_hopfion_field_geon.py` (exact null
> Maxwell solution; all checks PASS) and
> `examples/sim6_self_energy_anomaly_estimate.py` (anomaly = α/π expansion;
> reproduces CODATA to 3×10⁻⁹).

This document is the natural next step beyond doc 03. There the electron was a
*point charge on a wire* circulating at `c` — a heuristic that got `L = ħ/2`,
`μ = μ_B`, `g = 2` but was not an actual solution of any field equation. Here
I do two things the brief's pipeline demands:

1. **Promote "orbiting light" to a genuine, exact solution of Maxwell's
   equations** — the electromagnetic Hopfion — and verify its conserved
   charges by direct field integration.
2. **Attempt the QED anomaly `a = (g−2)/2` from photonic self-coupling**, and
   report honestly exactly how far that gets and where it stops.

---

## Part 1 — The electromagnetic Hopfion: orbiting light as a real field

### 1.1 Idea

"Light orbiting a centre" should not be a charge on a hoop; it should be a
*self-bound configuration of the electromagnetic field itself*. Such objects
exist and are exact: the **Rañada–Hopfion** electromagnetic knots (Rañada
1989; Irvine & Bouwmeester, *Nature Phys.* 2008; Kedia et al., *PRL* 2013).
Their field lines are linked circles (the Hopf fibration); the energy is pure
light that loops back on itself and carries intrinsic angular momentum. This
is the rigorous field-theoretic stand-in for a PSFT light-geon.

### 1.2 Math framework

Build the Riemann–Silberstein vector `F = E + icB` from two Bateman complex
scalars,

```
F = ∇α × ∇β ,
```

with the fundamental knot at `t = 0` (natural units `c = ε₀ = 1`, scale `a=1`):

```
α = (r² − 1 + 2iz)/(r² + 1) ,    β = 2(x − iy)/(r² + 1) .
```

The Bateman construction guarantees `F` solves the source-free Maxwell
equations. `sim5` evaluates `F` from analytic gradients on an 81³ grid and
checks the defining properties:

```
null field   RMS(E²−B²)/scale = 1.1e-15      E·B: RMS = 3.7e-16   ✓ (machine ε)
Maxwell      RMS(∇·E)/|∇F| = 1.1e-3           RMS(∇·B) = 1.1e-3    ✓ (2nd-order FD)
energy       U   = 19.72            (finite, localised)
momentum     P   = (0, 0, −9.87)    purely axial → the knot propagates along z
ang. mom.    L   = (0, 0, −9.85)    L_z fraction = 1.0000 → spin along the axis
|P|c/U = 0.50                       (< 1: a localised lump, not a plane wave)
```

So the configuration is **null light** (`|E| = c|B|`, `E·B = 0`), an **exact
divergence-free Maxwell field**, **finite-energy and localised**, and carries
**intrinsic angular momentum strictly along its symmetry axis**. The orbit is
the *linked field-line topology*, not a point on a wire.

> Unit caution: `L_z/U ≈ 0.4994` is a *length* (in knot-scale units `a`), not a
> dimensionless spin. The fundamental Hopfion is topologically **spin-1**. Do
> not read the 0.5 as "spin ½" — see the caveat below.

### 1.3 Does it generalise known physics? — Yes

- It is an exact solution of vacuum **Maxwell** (which PSFT derives via
  Theorem 9.1), so it lives natively in the PSFT U(1) sector.
- It carries the standard field-theoretic **energy, momentum, angular
  momentum, and helicity** of electromagnetism — computed here from the
  textbook integrals `U = ½∫(E²+B²)`, `P = ∫E×B`, `L = ∫r×(E×B)`.
- It demonstrates the general principle PSFT needs: **light can be a
  self-contained, localised, angular-momentum-carrying field object** —
  exactly the propagating-light precursor of a trapped geon.

### 1.4 Does it fit experiment? — Yes, and it is now lab-realisable

Electromagnetic Hopfions have been **constructed experimentally** in
structured-light and microwave systems and in plasmas (following Irvine &
Bouwmeester); their linked-field-line topology and angular-momentum content
match the theory. So this is not a thought experiment — knotted self-bound
light is real.

### 1.5 The honest gap: bosonic knot ≠ fermionic electron

A single fundamental Hopfion has **integer** angular momentum / helicity
(spin-1-like). The electron is **spin-½ with `g = 2`**. A pure *bosonic*
Maxwell field cannot, on its own, produce half-integer spin — that needs
fermionic / double-cover (spinor) structure. So:

- **What Part 1 proves:** orbiting/knotted light is a genuine, exact,
  experimentally realised field solution carrying intrinsic angular momentum
  — the rigorous upgrade of doc 03's heuristic.
- **What it does *not* prove:** that the electron *is* such a classical knot.
  It instead **localises the missing ingredient**: the geon needs the
  topological-charge / spinor structure of Postulate 4 (U(1) winding) layered
  on the bosonic field, plus the high-`K` SU(3) viscosity (paper §7.4) to trap
  it into a stationary bound state. Those remain open (paper §17).

This is the responsible outcome: a concrete, verified field object, and a
sharp statement of what extra structure the electron requires.

---

## Part 2 — The g−2 anomaly from photonic self-coupling

### 2.1 Idea

Doc 03's tree geon gives `g = 2` exactly and therefore misses the anomaly
`a = (g−2)/2 = 1.159652×10⁻³`. In PSFT the photonic field `P_ab` is primitive
and **self-interacting**, so the anomaly should come from the geon coupling to
its own photonic fluctuations — the PSFT analogue of the QED photon loop. How
far can that be carried honestly?

### 2.2 Math framework: the anomaly is an `α/π` expansion

QED organises the anomaly as a power series in `α/π`, one factor per exchanged
photon:

```
a = C₁(α/π) + C₂(α/π)² + C₃(α/π)³ + …
C₁ = 1/2 (Schwinger 1948), C₂ = −0.328479, C₃ = +1.181241, …
```

`sim6` sums this series with the known coefficients:

```
 n     C_n            term           partial sum
 1   0.500000000   1.161410e-03   1.16140973289e-03
 2  -0.328478966  -1.772305e-06   1.15963742782e-03
 3   1.181241457   1.480420e-08   1.15965223203e-03
 4  -1.912980000  -5.57e-11       1.15965217634e-03
 5   7.790000000   5.27e-13       1.15965217686e-03

 summed  = 1.15965217686e-03
 expt    = 1.15965218076e-03      relative agreement 3.4e-09
```

**Interpretation for PSFT:** the tree-level geon (`g = 2`) is the `n = 0`
term; **each factor of `α/π` is one photon the geon exchanges with its own
photonic field.** The dominant correction is the Schwinger term
`a₁ = α/2π = 1.161×10⁻³`, already within 0.1% of experiment.

### 2.3 Semiclassical magnitude (where it lands, and why it's ambiguous)

A Welton-style estimate of the geon jittering in its own zero-point photon
field gives a mean-square displacement

```
⟨Δr²⟩ ≈ (2α/π) λ̄_C² · ln Λ ,
```

`sim6` evaluates this at `ln Λ ∈ {1,2,5}`, giving `⟨Δr²⟩/λ̄_C² ≈ (0.5–2)×10⁻²`
— i.e. the moment correction sits at **order `α/π ≈ 2.3×10⁻³`, the right
size.** But the result depends on the logarithmic cutoff `Λ`, which a purely
classical treatment cannot fix. *That ambiguity is precisely why the clean
coefficient `C₁ = 1/2` cannot be derived classically.*

### 2.4 Does it generalise / fit? — Structurally yes; coefficient open

- **Generalises:** PSFT's self-interacting photonic field is exactly the
  object that must generate an `α/π` loop expansion; the geon supplies the
  `g = 2` zeroth term. The *structure* of the QED result is the structure PSFT
  predicts.
- **Fits (magnitude):** the semiclassical self-energy lands at the right order
  `α/π`; the one-loop Schwinger value matches experiment to 0.1%.
- **Open (coefficient):** deriving `C₁ = 1/2` from PSFT requires **quantising
  the photonic field** — the same open problem flagged in paper §17. I do
  **not** claim PSFT reproduces `a`; I claim it reproduces `g = 2` at tree
  level and predicts the anomaly is a photonic-self-coupling `α/π` series.

### 2.5 Prediction (sharpened P-C3)

> **P-E1 (sharp, falsifiable).** A PSFT photonic-loop computation of the geon
> magnetic moment **must** yield the leading coefficient `C₁ = 1/2`
> (`a ≈ α/2π`). Any other leading value falsifies the geon ontology. This
> converts the open quantisation problem into a *quantitative pass/fail test*
> for PSFT, not a vague aspiration.

---

## Summary

| Step | Part 1 (field geon) | Part 2 (anomaly) |
|---|---|---|
| Idea | orbiting light = exact field solution | anomaly from photonic self-coupling |
| Math | Rañada Hopfion `F = ∇α×∇β` | `a = ΣCₙ(α/π)ⁿ`, Welton jitter |
| Generalises? | yes — exact Maxwell, std. charges | yes — α/π loop structure |
| Fits experiment? | yes — knots realised in lab | magnitude yes; coeff. open |
| Numerics | `sim5`: null+div-free+axial L, PASS | `sim6`: series → CODATA 3e-9 |
| Honest gap | bosonic knot ≠ ½-spin electron | `C₁=1/2` needs quantisation |

**Net:** the "spinning light ring" is now a *real* field object (verified), and
the anomaly is *structurally* a photonic-self-energy `α/π` series with the tree
geon as its zeroth term — with two precisely-located open problems (half-integer
spin structure; the loop coefficient `1/2`) that become concrete falsifiable
targets rather than hand-waving.
