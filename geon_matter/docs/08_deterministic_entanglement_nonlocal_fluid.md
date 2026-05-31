# Entanglement deterministically: the spacetime fluid as the nonlocal
medium (P-G3)

> **Numerical companion:** `examples/sim11_two_geon_bell_correlations.py`
> (all checks PASS).

This is the decisive test of the deterministic programme, and the one demanding
the most discipline about what is proved versus hoped. Doc 07 established that
PSFT reproduces *single-particle* quantisation deterministically (energy
levels, spin, charge — all topology/standing-wave facts, no Born rule) but
flagged the hard frontier: **entanglement correlations**. Bell's theorem is
unforgiving — any deterministic theory matching the quantum correlations must be
**nonlocal**. So the question is sharp: does PSFT's distinguishing structure —
a real spacetime *fluid* spanning the manifold — provide the nonlocality needed,
without violating relativity?

I split the question into three numerically-separated parts and answer each
honestly.

## 1. The target is deterministic field geometry, not statistics (Q1)

The singlet correlation is `E(a,b) = −cos(a−b)`. **This cosine is the
deterministic field-overlap (Malus) law** — the inner product of analyzer modes
— which is *exactly* the energy-density rule PSFT already uses for interference
without a Born rule (paper §13). Its CHSH value at the optimal angles
(`0, 90, 45, 135°`) is

```
S = 2√2 = 2.8284   (Tsirelson bound)   — sim11 (Q1): S = 2.8284 ✓
```

So the "quantum" number `2√2` is the geometry of field-amplitude projection, not
a statistical axiom. This reframes the whole problem: entanglement's hallmark is
a *field-overlap* fact; the only genuinely statistical-looking thing is which
single outcome a single quantum gives.

## 2. Local determinism genuinely fails — Bell, demonstrated not dodged (Q2)

A geon pair that shares only a **local** hidden field orientation `λ` (fixed at
creation) with deterministic Malus-threshold detection:

```
A(a,λ) = sign cos(a−λ) ,   B(b,λ) = −sign cos(b−λ)
```

Monte-Carlo CHSH (`sim11`, Q2): **S = 2.001 ≤ 2.** It saturates the
local-realist bound and **cannot reach 2√2**. This is Bell's theorem made
concrete: a shared classical field orientation — local determinism — is
*insufficient*. The deterministic programme does **not** get to ignore this.

## 3. The spacetime fluid supplies the nonlocality (Q3)

PSFT's defining ingredient is a real fluid (`u^a`, Postulate 2) spanning the
manifold. Two geons created together were **one connected field configuration**
and remain coupled *through the fluid* — a physical, persistent nonlocal link,
not a spooky action. Modelling that shared medium with the Toner–Bacon
deterministic protocol (a shared hidden field state `(λ₁, λ₂)` plus a single
*relational* bit carried by the fluid):

```
A = sign(λ₁·â) ,   c = sign(λ₁·â) sign(λ₂·â) ,   B = −sign(λ₁·b̂ + c λ₂·b̂)
```

This is **deterministic** given `(λ₁, λ₂)` and **nonlocal** (the bit `c` is the
fluid-mediated relational connection). `sim11` (Q3) verifies it reproduces the
full cosine,

```
Δ:    0°    30°    45°    60°    90°   120°   135°   180°
E_nl −1.00 −0.867 −0.707 −0.499 +0.000 +0.500 +0.708 +1.00
−cos −1.00 −0.866 −0.707 −0.500  0.000 +0.500 +0.707 +1.00
```

and reaches **S = 2.828 ≈ 2√2** — the Tsirelson bound.

## 4. It is relativistically benign: no-signalling (all models)

Crucially, the fluid-mediated nonlocality carries **no usable signal**. `sim11`
checks Alice's marginal:

```
⟨A⟩_local = −0.0008 ,   ⟨A⟩_nonlocal = −0.0005     (both ≈ 0, setting-independent)
```

Alice's outcome statistics are independent of Bob's setting in every model, so
the correlation cannot transmit information. The fluid link operates purely at
the hidden-field level — exactly like the quantum-potential nonlocality of
Bohmian mechanics, and consistent with the relativistic causality PSFT's
covariant master equation enforces at the *signal* level.

## 5. The pipeline verdict

| step | result |
|---|---|
| Idea | the spacetime fluid is the nonlocal medium that lets deterministic geons entangle |
| Math | field-overlap `−cos`; local model; Toner–Bacon fluid-bit model |
| Generalises? | yes — contains the local bound (S≤2) and the quantum bound (2√2) as the local/nonlocal cases |
| Fits experiment? | reproduces the measured Tsirelson value `2√2` and the cosine correlation; preserves no-signalling | 
| Numerics | `sim11`: Q1 2.828, Q2 ≤2, Q3 2.828, no-signalling — all PASS |

## 6. What is proved, and what is not (the honest core)

**Proved (consistency / possibility).** PSFT's nonlocal spacetime fluid is the
*right kind of structure* to host quantum entanglement **deterministically and
without signalling**. The deterministic programme is therefore **not excluded by
Bell's theorem** — the standard objection ("no deterministic theory can give
2√2") does not apply to a theory with a real nonlocal medium, and the cosine
correlation is the deterministic field-overlap law PSFT already uses.

**Not proved (the deep open problem).** That PSFT's **master-equation dynamics**
actually *produce* this specific correlation — i.e. that the fluid's real
dynamics implement exactly the relational one-bit channel and yield `−cos(a−b)`
rather than some other nonlocal correlation. The Toner–Bacon channel is imported
as a stand-in for the fluid coupling; deriving it (and the fluid's information
capacity) from the field equations is genuinely open. `sim11` settles the
*consistency* half of P-G3; the *derivation* half remains.

## Predictions

- **P-H1 — Entanglement is mediated by a physical medium, not "nonlocal magic."**
  If the spacetime fluid carries the correlation, then conditions that perturb
  the fluid between the wings (extreme curvature, the high-`K` viscous phase,
  rapid expansion) should measurably *degrade* entanglement correlations beyond
  standard decoherence. Standard QM predicts no such medium-dependence. A
  curvature/medium dependence of Bell violation is a falsifiable PSFT signature.
- **P-H2 — Tsirelson, not super-quantum.** Because the correlation is the
  field-overlap geometry (inner products), PSFT predicts the bound is exactly
  `2√2` — never the algebraic maximum `4` of generic no-signalling (PR-box)
  theories. PSFT sits at the quantum bound for a structural reason (it is field
  projection), distinguishing it from post-quantum models.
- **P-H3 — No-signalling is exact and protected** by the covariant master
  equation (signal-level causality), even though hidden-field correlations are
  nonlocal. Any observed signalling via entanglement would falsify PSFT.

## Where this leaves the programme

The geon research now spans a full arc: light → special relativity
(derived, docs 01–03) → field-theoretic geon and spin (docs 05–06) → binding and
deterministic single-particle quantisation (doc 07) → and now deterministic,
no-signalling entanglement as a *consistent* possibility hosted by the spacetime
fluid (doc 08). The single deepest remaining task is the same one the paper flags
(§17): solve the v2 master equation for real bound states and show its fluid
dynamics *derive* — not merely *accommodate* — the quantum correlations and the
mass spectrum.
