# Constants and Free Parameters

This document (i) tabulates every constant that appeared in the geon/matter
research, classified by status, and (ii) analyses how many genuinely **free**
parameters PSFT carries, compared with the Standard Model + General Relativity.

---

## 1. Constants encountered, by status

**Status key:**
`U` = universal physical constant (not a PSFT choice) ·
`I` = PSFT structural/numerical *input* (free parameter) ·
`D` = *derived* within PSFT / universal bound (not free) ·
`F` = phenomenological *fit* (standard effective-model parameter, not first-principles here).

| Constant | Symbol | Value | Status | Where |
|---|---|---|---|---|
| Speed of light | `c` | 2.998e8 m/s | U | all |
| Reduced Planck | `ħ` | 1.055e−34 J·s | U | all |
| Newton constant | `G` | 6.674e−11 | U/I | gravity sector |
| Electron mass | `m_e` | 0.511 MeV | U(in)/open | sims 1–3 |
| Fine-structure | `α` | 1/137.036 | U | sim6, sim3 |
| **Confinement length** | `l_strong` | **1 fm** | **I (key)** | bag, fluid, Kc |
| Critical curvature (strong) | `Kc^strong = 12/l_strong⁴` | 1.2e61 m⁻⁴ | D(from `l_strong`) | viscosity gate |
| Electroweak length | `l_weak` | 1e−18 m | I | weak sector |
| Critical curvature (EM) | `Kc^EM` | ∞ | I (fixed) | inviscid U(1) |
| KSS viscosity ratio | `η/s` | 1/(4π) | D (universal bound) | Phase 7a |
| Planck viscosity | `η_P` | c³/(16πG) | D(from `G`) | viscosity normalization |
| QCD string tension | `σ` | 0.18 GeV² | F (↔ `l_strong`) | sim9, Phase 1 |
| Bag constant | `B^{1/4}` | ≈ κ·ħc/l_strong ≈ 145–235 MeV | D-scale/F-`O(1)` | Phases 1,7b |
| Cavity mode eigenvalue | `x₁` | 2.0428 | D (geometry) | Phases 1,7b |
| Bag zero-point | `Z₀` | 1.84 | F (MIT standard) | Phases 1,7b |
| Hyperfine strength | `κ_B, κ_M` | 49, 159 MeV | F (↔ `α_c`) | Phase 7b |
| Fluid correlation length | `ξ` | = `l_strong` = 1 fm | D(from `l_strong`) | Phases 6,7a |
| Fluid correlation time | `τ_c` | = `l_strong/c` = 3.34e−24 s | D(from `l_strong`) | Phases 6,7a |
| Viscous-phase temperature | `T` | = ħc/l_strong ≈ 197 MeV | D(from `l_strong`) | Phase 7a |
| Metric→clock coupling | `σ_g` | ≈ 0.13 | F (`O(1)`, FDT estimate) | Phase 7a |
| Compton frequency (e) | `mc²/ħ` | 7.76e20 rad/s | D | sims 1,2 |
| de Broglie relation | `λ=h/p` | — | D (derived) | sim2 |
| Tree gyromagnetic ratio | `g` | 2 | D (derived) | sim3 |
| Tsirelson bound | `S_max` | 2√2 | D (field-overlap) | sim11, Phase 5 |
| Bell visibility threshold | `V_crit` | 1/√2 | D | Phase 5 |
| Spin–statistics relation | `2S = n` | — | D (topology) | sim8, Phase 3 |

---

## 2. What a single input (`l_strong`) generated

A central finding of this work: in the strong sector, **one length scale
`l_strong ≈ 1 fm`** fixes a whole family of quantities that are independent
inputs in QCD/the Standard Model:

- the critical curvature `Kc^strong = 12/l_strong⁴` (viscosity activation);
- the bag-constant *scale* `B^{1/4} ~ ħc/l_strong` → the hadron mass *scale*
  (spin-averaged baryon to **0.1%**, Phase 7b);
- the string tension `σ` (confinement, sim9);
- the fluid coherence length `ξ = l_strong` and time `τ_c = l_strong/c`
  (Phases 6, 7a);
- the viscous-phase temperature `T = ħc/l_strong ≈ 197 MeV` (≈ QCD `T_c`);
- the entanglement coherence length / decoherence threshold (Phase 7a).

These collapse onto one parameter. Combined with the **universal** results
(`η/s=1/4π`, `g=2` tree, de Broglie, Tsirelson `2√2`, `2S=n`) which carry **no**
free parameter at all, this is substantial parameter economy *within the strong
sector*.

---

## 3. Free-parameter count: PSFT vs SM + GR

The Standard Model + GR carries **~26–27** free parameters:

| Group | Count |
|---|---|
| Gauge couplings `g₁,g₂,g₃` | 3 |
| Higgs (mass, self-coupling / vev) | 2 |
| Charged-fermion masses (6 quarks + 3 leptons) | 9 |
| CKM (3 angles + 1 phase) | 4 |
| Neutrino masses + PMNS | ~7 |
| QCD vacuum angle `θ` | 1 |
| Newton `G`, cosmological `Λ` | 2 |
| **Total** | **~27** |

### PSFT today (honest)

PSFT **does not currently reduce this count**, because the **fermion mass
spectrum is not yet derived** (the open soliton-mass problem, item A3/B1). What
it *does* do is **reorganize** the parameters and remove several:

| Parameter group | SM + GR | PSFT (current) |
|---|---|---|
| Gauge group `SU(3)×SU(2)×U(1)` | structural | structural (same) |
| Gauge couplings | 3 inputs | 3 inputs (viscosity-sector normalizations) |
| Higgs sector | 2 inputs | replaced by electroweak viscous phase (`l_weak`, Hall term) — count not yet settled |
| Scale-dependent `G` (v1 had `β_s,β_w,l_s,l_w`) | — | **removed**: `G_eff = G` universal (Modification 5) |
| Viscosity ratio `η/s` | — | **fixed** at `1/(4π)` (KSS), not free |
| Strong scale | `Λ_QCD` + bag/string params | **one** input `l_strong` (fixes `Kc^strong`, `B`-scale, `σ`, `ξ`, `T`) |
| Fermion masses (9) + mixings (~11) | inputs | **not yet derived** (open) |
| `G`, `Λ` | 2 inputs | 2 inputs (same) |

So the current honest count is **comparable to SM+GR** — PSFT has not yet
paid off on masses — but the strong-sector *scale* is a single parameter, and a
few SM/v1 parameters are eliminated (`η/s` fixed, `G` universal, the strong
scale unified).

### PSFT if the soliton spectrum is solved (the promise)

If items **A3** (mass spectrum from the master equation) and **B1** (lepton
sector) succeed, the ~9 fermion masses and ~11 mixings become **outputs** of the
soliton dynamics rather than inputs. The free-parameter list would then shrink
toward:

| Remaining free parameter | Role |
|---|---|
| Gauge group (structural) | which forces exist |
| `G` | gravitational coupling |
| `Λ` | cosmological constant |
| `l_strong` | strong critical curvature |
| `l_weak` | electroweak critical curvature |
| (gauge couplings) | sector viscosity normalizations |

i.e. of order **5–6 numerical parameters** instead of ~27 — with the entire
fermion mass/mixing spectrum derived. **This is the PSFT promise; it is not yet
achieved.** The present work has demonstrated the *mechanism* and *scales* that
make it plausible (strong scale from one input; masses as soliton energies), but
turning masses into first-principles numbers is the open keystone (A3).

---

## 4. Zero-parameter results established here

For the record, the following were obtained with **no** free parameter (pure
derivation or universal bound), and are the firmest outputs of the research:

- `η/s = 1/(4π)` (KSS bound; consistency with RHIC/LHC QGP).
- `E = γmc²`, time dilation, length contraction, `E²−(pc)²=(mc²)²` — special
  relativity, *derived* from confined light moving at `c`.
- de Broglie `λ = h/p`, with phase velocity `c²/v`, group velocity `v`.
- `g = 2` at tree level; the anomaly as an `α/π` series (`C₁=½` open).
- Spin–statistics `2S = n` (half-integer spin ⇔ odd topological winding).
- Bell/CHSH Tsirelson bound `S = 2√2` (field-overlap geometry); threshold
  `V_crit = 1/√2`; no-signalling.
- `ξ = l_strong`, `τ_c = l_strong/c` (fluid coherence scales).
- Light-hadron mass *ordering* (`N<Δ`, `π<ρ`) and the bag *scale* matching the
  spin-averaged baryon to 0.1%.

These require no tuning; they follow from the PSFT structure (inviscid U(1),
confined light, topological charge, the viscous phase) plus universal physics.
