# Bound-State Project — Deriving matter from the PSFT master equation

**Goal.** Move from "PSFT *accommodates* matter" to "PSFT *produces* matter":
solve (a faithful reduction of) the v2 master equation for a **stationary,
finite-energy, topologically-charged bound state** whose binding comes from
PSFT's *own* mechanism — the Kretschmann-gated SU(3) viscosity (Postulate 3,
Theorem 10.1) — and show it reproduces the hadronic mass/size scale and the
topological charge/spin. This is the deepest open problem flagged by the paper
(§17) and by this research module (RESULTS.md, P-G3).

This directory is **additive** and **reuses the `simulation/psft` package
read-only** (relaxer, ansätze, topology, viscosity). It does not modify any
existing file.

---

## What is already done vs. the gap

| done (in `simulation/`) | the gap this project fills |
|---|---|
| Skyrme baryon mass (ex. 11/12) via the **standard Skyrme model** with phenomenological `F_π, e_S` | bind matter with PSFT's **own** Kc-gated viscous phase, not Skyrme's 4th-order term |
| `GradientFlowRelaxer`, `EnergyFunctional` | a **PSFT-faithful energy functional** (viscous-phase bag) to relax |
| `VortexAnsatz`/`KnotAnsatz`, topology tools | relax these to a **self-consistent** profile and read off mass + charge |
| `HeavisideViscosity` (Kc gating) | use the Kc gate as the **bag/phase boundary** that stabilises the soliton |
| sim9 (0D energy balance `A/R+σR`) | a **real 3D/radial field soliton**, not a scaling argument |

## The reduced model (faithful, tractable)

The full v2 master equation is a gauge-valued relativistic Navier–Stokes coupled
to BSSN gravity and the photonic field. We reduce it to the **degrees of freedom
that carry the physics of binding**, keeping the PSFT mechanism exact:

1. **Trapped light** = a confined massless field mode (energy `∝ ħc/R`, the
   Derrick-dispersing term — sim5/sim9 showed free light cannot bind).
2. **Binding** = the Kretschmann-gated viscous phase. The paper's own language
   (Lorentz-covariance §) is a **phase transition** at `K = Kc`: a "GR phase"
   (inviscid, low energy) and a high-`K` "QCD phase" (viscous). Identify this
   with the **MIT/Friedberg–Lee bag**: the high-`K` interior costs a bag energy
   density `B`, and `B` is fixed by the PSFT scale `Kc^strong = 12/l_strong⁴`
   (`l_strong ≈ 1 fm`). The Heaviside `Θ(K−Kc)` **is** the bag boundary.
3. **Charge/spin** = U(1) topological winding (Postulate 4); spin-½ then follows
   from the charge-plus-winding field angular momentum (sim8).

So the geon is a **bag of trapped light with topological winding, the bag wall
being the Kc viscous-phase boundary** — every ingredient is a PSFT postulate,
no phenomenological Skyrme constants.

Energy functional (natural units, `ħc = 197.327 MeV·fm`):

```
E[φ] = ∫ d³x [ ½(∇φ)²  +  U_bag(φ; K, Kc) ]
       └ trapped-light gradient ┘   └ viscous-phase (bag) term, gated by Θ(K−Kc) ┘
```

minimised by gradient flow → stationary profile `φ*(r)`, mass `M = E[φ*]`,
size `R*`, winding `n`.

---

## Phases, milestones, success criteria

### Phase 1 — Viscous-phase bag geon (analytic/0D), `B` tied to PSFT `Kc`
*Deliverable:* `phase1_viscous_bag_geon.py`, `RESULTS_P1.md`.
- Build `E(R) = N x₁ ħc/R + (4π/3) B R³`, the MIT-bag energy, with the bag
  constant `B` derived from the PSFT critical-curvature scale (`B^{1/4} ∼
  ħc/l_strong`-class), **no Skyrme constants**.
- **Success:** stable minimum at `R* ∼ 0.5–1 fm`, `M ∼` light-hadron band
  (few hundred MeV–~1 GeV); reduces to "no binding" when the viscous phase is
  switched off (`B → 0`); upgrades sim9 (linear → bag, `σ`/`B` tied to `Kc`).

### Phase 2 — Relax a REAL field soliton bound by the viscous phase
*Deliverable:* `phase2_field_relaxation.py`, `RESULTS_P2.md`.
- Implement the PSFT-faithful `EnergyFunctional` (gradient + Kc-gated bag) and
  relax a radial profile with the existing `GradientFlowRelaxer`; cross-check
  with a 3D relaxation on a `CartesianGrid`.
- **Success:** (i) **Derrick demonstrated** — without the bag term the profile
  spreads/disperses (energy falls monotonically with size); with it the flow
  converges to a localised `φ*(r)`. (ii) The relaxed mass/size **match the
  Phase-1 bag estimate** within the model's accuracy. (iii) A clean,
  reproducible profile (core + wall + exterior vacuum).

### Phase 3 — Topological charge and spin from winding
*Deliverable:* `phase3_winding_charge_spin.py`, `RESULTS_P3.md`.
- Put U(1) winding `n` on the relaxed bag (reuse `VortexAnsatz` phase +
  `topology.winding_number_*`); confirm the conserved charge = `n` and that the
  bag mass is insensitive to it at leading order (charge ≠ mass driver).
- Tie to sim8: the field angular momentum of (winding + charge) → `L = nħ/2`
  (half-integer spin); confirm `2S = n` selection.
- **Success:** integer winding measured to <1%; spin assignment consistent with
  sim8; charged vs neutral configurations distinguished.

### Phase 4 — Dynamical stability (evolution) *(scoped, partial)*
*Deliverable:* `phase4_dynamical_stability.py` (may be a reduced 1D/spherical run).
- Seed the relaxed `φ*` into a time evolution (reuse `evolve/` hydro/photonic
  scaffolding where applicable) and check the configuration persists (energy &
  charge conserved, no dispersal) over many light-crossing times.
- **Success:** energy/charge conserved to a few %, no runaway growth/collapse
  over the simulated interval; honest report of any instability.

### Phase 5 — Two-geon correlation from field dynamics *(aspirational / open)*
- Attempt to *derive* (not import, as sim11 did) the `−cos` correlation and the
  one-bit channel from the shared-fluid coupling of two geons. This is the deep
  open half of P-G3 and may not close in this project; document partial progress
  and a concrete sub-problem (e.g. the fluid's information capacity between two
  cores).

---

## Honest scope & risks (stated up front)

- **This is a reduced model.** It keeps PSFT's *binding mechanism* (Kc-gated
  viscous phase) exact but replaces the full gauge-valued Navier–Stokes with a
  bag-type effective energy. It will get **scales and mechanisms** right, not
  individual masses to %.
- **The bag constant `B`** is tied to the PSFT scale `Kc^strong`, but the precise
  `O(1)` coefficient (cavity-mode number `x₁`, bag-wall structure) is
  model-dependent — reported as a band, like the MIT bag itself.
- **The electron (lepton sector)** is *not* strong-confined; its mass remains
  out of scope (weak/EM-bound, paper §17). Phases 1–3 target hadron-scale geons.
- **Phase 5 may not close.** Deriving the quantum correlation from dynamics is
  the genuinely hard frontier; partial progress is an acceptable outcome,
  clearly labelled.
- **No overclaiming.** Each phase prints explicit PASS/FAIL criteria and an
  honest "what this does / does not show" block, matching the module's standard.

## Execution order this session

Phases 1 → 2 → 3 are executed now (they build on each other and on existing
code). Phase 4 is started if time allows; Phase 5 is scoped for follow-up.
