# Future Work — Remaining Items

This document defines the items left open by the geon/matter research line
(the deterministic emergence of matter, mass, spin, and quantum phenomena from
the PSFT photonic field). Each item states **what it is**, **what it needs**
(prerequisites), **which existing tool** it builds on, its **difficulty**, and
**what it would buy**. Items are grouped and ordered by how foundational they
are.

Status legend: ⬛ deep (genuine research program) · 🟧 standard-but-laborious ·
🟦 incremental.

---

## A. Strong-sector mass spectrum

### A1 🟧 Absolute hyperfine strength (turn the fitted `κ` into a prediction)
**What.** Phase 7b reproduced the light-hadron *scale* from `l_strong` (no fit)
and the splitting *structure* from the spin operator, but the overall hyperfine
strength `κ` (one per sector) was fitted. Replace it with the first-principles
MIT-bag color-magnetic energy: `ΔE = α_c·Σ(λᵢ·λⱼ)(σᵢ·σⱼ)·I(R)`, with `I(R)` the
bag magnetic mode integral over the lowest cavity mode.
**Needs.** The massless spherical-cavity mode functions (spherical Bessel `j₀,
j₁`), the color SU(3) matrix elements `⟨λᵢ·λⱼ⟩`, and `α_c` at the bag scale
(`~0.5`, from the running coupling at `~ħc/l_strong`).
**Tool.** Extend `phase7b_hadron_mass_spectrum.py`; reuse `psft.gauge.algebra`.
**Buys.** The light spectrum (N, Δ, π, ρ, ω) with **zero** fitted hyperfine —
only `α_c`, which is itself running-coupling-fixed.

### A2 🟧 Strange and heavy sectors
**What.** Add quark masses `m_s, m_c, m_b` to the bag (massive-mode eigenvalues
`xᵢ(m R)`), predict K, D, B mesons and Λ, Σ, Ξ, Ω baryons.
**Needs.** Massive Dirac cavity-mode eigenvalues (transcendental boundary
condition); the quark masses as inputs (these remain free until A3/B1).
**Tool.** Generalize `phase1_viscous_bag_geon.py` mode constant `x₁` to `xᵢ(mR)`.
**Buys.** The full ground-state hadron spectrum from the bag + masses; tests the
universality of `B(l_strong)` across flavors.

### A3 ⬛ First-principles soliton from the FULL v2 master equation
**What.** Replace the reduced bag/Q-ball (Phases 1–2) with an actual stationary
solution of the *full* v2 master equation: the gauge-valued relativistic
Navier–Stokes (`eq:v2_master`) coupled to BSSN gravity and the photonic source,
with the SU(3) viscosity active at `K>Kc^strong`. Measure the soliton mass,
radius, and topological charge directly.
**Needs.** A coupled evolver: `psft.evolve.adm` (BSSN) + `psft.evolve.gauge_sectors`
(viscous SU(3) stress) + `psft.evolve.photonic_field`, with a relaxation/imaginary-time
driver to the stationary state. Constraint damping (Z4c) for stability.
**Tool.** All three `psft.evolve` modules already exist (examples 25–33); the
work is coupling them and adding the gauge-viscous stress to the matter source.
**Buys.** Particle masses as genuine **outputs** of the field equations — the
core of the mass-spectrum problem (paper §17). This is the single most
foundational remaining item: it would turn the strong-sector *scale* (already
derived) into individual *numbers*.

---

## B. Lepton sector

### B1 ⬛ Charged-lepton masses (electron, muon, tau)
**What.** Leptons are color-singlet; they are **not** strong-confined and so are
out of scope for the bag (Phases 1–2 are strong-sector). Their binding lives in
the electroweak viscous phase (`Kc^weak`, `l_weak ~ 10⁻¹⁸ m`). Build the
weak-sector geon and compute the lepton masses.
**Needs.** The SU(2) Hall-viscosity sector (parity-odd, paper Modification 4)
as the binding/chirality mechanism; the electroweak scale `l_weak`.
**Tool.** `psft.evolve.gauge_sectors` (weak sector), `psft.sectors.viscosity`
`HallViscosity`; example 04 (Hall parity) as a starting point.
**Buys.** The lepton masses and, crucially, an account of the **three
generations** and the mass hierarchy — entirely open.

---

## C. Absolute fluid noise spectrum (from dynamics, not estimate)

### C1 🟧 The metric→clock coupling (`O(1)` factor of Phase 7a)
**What.** Phase 7a fixed `ξ, τ_c, T` robustly but estimated the clock-rate
coupling `σ_g ≈ ½(δv/c)²` heuristically. Derive the exact coupling of a geon's
internal light-clock to a fluctuating fluid metric.
**Needs.** A linearized geon clock (the Phase-4 evolver) in a prescribed
fluctuating BSSN metric perturbation; read off `dω/ω` per unit metric fluctuation.
**Tool.** `phase4_dynamical_stability.py` + `psft.evolve.adm` perturbations.
**Buys.** The absolute decoherence rate `Γ` with no `O(1)` freedom.

### C2 ⬛ Fluid noise spectrum by evolving the viscous master equation
**What.** Phase 7a got `(σ, τ_c, ξ)` from the fluctuation-dissipation theorem +
the KSS bound. Instead, **evolve** the v2 viscous fluid (with thermal noise per
Landau–Lifshitz fluctuating hydrodynamics) and **measure** the stress
autocorrelation `⟨τ(x,t)τ(x',t')⟩` directly, recovering `σ, τ_c, ξ` from first
principles.
**Needs.** Stochastic viscous hydrodynamics on the grid; the noise amplitude
set by `η` (KSS) and `T`.
**Tool.** `psft.evolve.hydro_3d` + the v2 viscous term + a noise source.
**Buys.** The absolute noise spectrum from the dynamics, closing the last
estimate in the decoherence chain (Phases 6–7a).

---

## D. Deterministic quantum phenomena (no Born rule)

> Project stance (see the deterministic-physics note): derive QM facts
> deterministically — as field intensities, eigenmodes, and topological
> integers — and **evade** Born-rule/statistical quantization.

### D1 ⬛ Coherence `V` from two-soliton master-equation dynamics
**What.** Phase 6 derived the coherence law `V(t)=e^{−Γt}` and its dependences
from a reduced two-clock + stochastic-fluid model. Replace it with two **actual**
geon solitons (from A3) sharing the evolved fluid, and measure the relative-phase
coherence directly — deriving `V` from the genuine field dynamics.
**Needs.** A3 (real solitons) + C2 (real fluid noise).
**Tool.** Two coupled instances of the A3 evolver.
**Buys.** The two-geon Bell correlation (`S=2√2 V`) from first principles —
fully closing P-G3 (currently the consistency + reduced-dynamics halves are done).

### D2 🟧 Extend the deterministic program to more QM phenomena
**What.** Give deterministic field/topology accounts of phenomena QM treats
statistically: quantum tunneling (evanescent trapped-light fields), the
measurement/decoherence chain (via C2's fluid noise), identical-particle
statistics (spin–statistics from winding parity, P-F1), and delayed-choice /
quantum-eraser setups (flagged open in the paper §13).
**Needs.** The deterministic interference machinery (paper §13, example 10) +
the fluid-decoherence model (C2).
**Tool.** `examples/10_photon_double_slit.py` + the bound-state/coherence code.
**Buys.** A broader deterministic coverage of "quantum weirdness," each as a
field or topology fact rather than a probability postulate. (Honest limit: a
complete account of all correlations is nonlocal by Bell — the fluid is the
medium; each phenomenon must be derived, not assumed.)

---

## E. Precision / radiative structure

### E1 ⬛ The g−2 anomaly coefficient `C₁ = ½`
**What.** The tree geon gives `g=2` (sim3); the anomaly is the `α/π` series
(sim6) whose leading coefficient is `½` (Schwinger). Derive `½` from the geon
coupling to its own photonic field.
**Needs.** A treatment of the photonic-field self-interaction at one loop —
within the deterministic stance, the **field self-energy** structure, *not* a
Born-rule quantization. Open whether a deterministic field computation yields
exactly `½`.
**Tool.** The photonic-field evolver (`psft.evolve.photonic_field`) coupled to a
geon, computing the self-field correction to the magnetic moment.
**Buys.** The most precisely tested number in physics, as a sharp pass/fail
(`C₁` must equal `½`).

---

## F. Inherited from the paper (context, not this module)

Listed for completeness; these are the paper's own open items and predictions
(not part of the geon/matter line): zero GW damping (Einstein Telescope/LISA),
neutron-star post-merger `f₂` shift, QGP `η/s` phase transition at `Kc`,
gravitational Aharonov–Bohm phase, and vortex (filament) dark matter.

---

## Suggested ordering

1. **A3** (full-master-equation soliton) — the keystone; unlocks A2, B1, D1.
2. **C2** (fluid noise from dynamics) — unlocks D1, C1, D2.
3. **A1** (absolute hyperfine) — quick win once A3's bag is calibrated.
4. **B1** (lepton sector) — needs the weak-sector machinery.
5. **E1** (anomaly coefficient) — needs the photonic self-energy treatment.

A3 and C2 are the two genuine research programs; everything else is standard or
incremental once they are in place. Each would convert a *derived scaling or
mechanism* (already established) into an *absolute first-principles number*.
