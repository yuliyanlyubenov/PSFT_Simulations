# Geon Matter — Deterministic Emergence of Matter from Light

Numerical companion exploring how **matter, mass, spin, and quantum phenomena
emerge deterministically** from the PSFT photonic field, in which a particle is
a *geon* — light bound around a concentrated-energy core, stabilised by the
high-curvature SU(3) viscosity (paper §2, §7.4).

All scripts reuse the `psft` library (read-only) and self-test with explicit
PASS/FAIL criteria; dependencies are `numpy` only. Nothing here modifies the
core library or the paper.

## Method

Each result follows the same discipline: **idea → minimal math → does it
contain the established theory? → does it fit the measured numbers? →
prediction.** Two commitments run throughout:

- **Ontology (Postulate 1).** Light (the photonic field) is primitive; *matter*
  is the soliton. The photon is never treated as a soliton — the geon (the
  particle) is.
- **Determinism.** Quantum facts usually explained statistically are derived
  here *deterministically* — as field intensities, resonant/standing-wave
  eigenmodes, and **topological integers** (winding = charge, spin, level
  number) — rather than from a probability postulate.

## Results at a glance

| Topic | Result | Script |
|---|---|---|
| Special relativity from confined light | `E=γmc²`, time dilation, length contraction, mass shell — derived | `examples/sim1_confined_light_mass_and_lorentz.py` |
| de Broglie matter wave | `λ=h/p` to machine precision, as the moving-frame internal clock | `examples/sim2_de_broglie_internal_clock.py` |
| Spin-½ and `g=2` | from circulating charge-light (tree level) | `examples/sim3_circulating_light_spin_gfactor.py` |
| Field-theoretic geon | exact null Maxwell knot (Hopfion), conserved charges | `examples/sim5_hopfion_field_geon.py` |
| `g−2` anomaly structure | the measured anomaly as an `α/π` photonic-loop series | `examples/sim6_self_energy_anomaly_estimate.py` |
| Dynamical knot | full Maxwell evolution of the moving Hopfion | `examples/sim7_time_dependent_hopfion.py` |
| Half-integer spin | from charge + topological winding (`2S=n`) | `examples/sim8_charge_monopole_half_integer_spin.py` |
| Geon binding | viscous-phase confinement cures Derrick instability | `examples/sim9_geon_binding_stability.py` |
| Deterministic quantisation | hydrogen spectrum from standing-wave topology (no Born rule) | `examples/sim10_deterministic_quantization.py` |
| Deterministic entanglement | Tsirelson `2√2` from the nonlocal fluid, no-signalling | `examples/sim11_two_geon_bell_correlations.py` |

## The bound-state project (`bound_state/`)

A staged program *deriving* a bound particle from the master equation, binding
by PSFT's own Kc-gated viscous phase (see `bound_state/PROJECT_PLAN.md` and
`bound_state/RESULTS.md`):

| Phase | Result |
|---|---|
| 1 | viscous-phase bag geon: hadron-scale mass from the single input `l_strong`; Derrick cured |
| 2 | **real charged field soliton** (Q-ball) solved from its field equation; GeV/few-fm |
| 3 | winding → charge & spin: charge = winding `n`, `J_z = nQ`, `2S = n` |
| 4 | dynamical stability: the geon persists under time evolution (energy & charge conserved) |
| 5 | entanglement = fluid coherence: `S(V) = 2√2 V`, sharp Bell threshold `V = 1/√2` |
| 6 | coherence `V(t)=e^{−Γt}` **derived** from two-soliton + shared-fluid dynamics |
| 7a | absolute fluid noise spectrum `ξ=1 fm, τ_c=3.3e-24 s, T≈197 MeV` from FDT + KSS bound |
| 7b | light-hadron masses: bag scale matches spin-averaged baryon to **0.1%**; spectrum + orderings |

## Documentation

- `docs/` — the seven topic write-ups (math, generalisation, experimental fit,
  predictions) corresponding to the example scripts.
- `CONSTANTS_AND_PARAMETERS.md` — every constant used, classified, and a
  free-parameter analysis vs the Standard Model + GR.
- `FUTURE_WORK.md` — the remaining open items, each with prerequisites, the
  tool it builds on, and what it would buy.

## Running

```bash
cd geon_matter
python3 examples/sim1_confined_light_mass_and_lorentz.py     # ... any example
cd bound_state && python3 phase1_viscous_bag_geon.py          # ... any phase
```

## Honest scope

These are reduced, faithful models: they establish the **mechanisms, scales,
and structure** (special relativity from light; hadron scale and orderings from
one input; charge/spin from topology; deterministic quantisation and
entanglement). What they do **not** yet deliver — individual fermion masses, the
absolute hyperfine and decoherence normalisations, and the soliton spectrum from
the full master equation — is laid out in `FUTURE_WORK.md`.
