# The geon's internal light-clock *is* the de Broglie matter wave

> **Numerical companion:** `examples/sim2_de_broglie_internal_clock.py`
> (λ = h/p to machine precision; v_phase = c²/v, v_group = v — all PASS).

## 1. Idea

Doc 01 showed the geon carries an internal oscillation at the Compton
frequency `ω₀ = mc²/ħ` (its trapped-light clock). In the rest frame this
oscillation is **synchronous everywhere** — the whole geon "breathes in
phase." A moving observer, however, slices spacetime along *tilted*
simultaneity surfaces. The claim:

**the lab-frame appearance of the rest-frame-synchronous internal clock is a
travelling spatial phase — and that travelling phase is exactly the de Broglie
matter wave, `λ = h/p`.**

This is de Broglie's own 1924 "harmony of phases" argument, but PSFT supplies
the missing mechanism: the internal oscillation is not a postulate, it is the
geon's orbiting light. Wave–particle duality of *matter* then has the same
origin PSFT already gives the photon (paper §13, item 8).

## 2. Math framework

Rest-frame internal phase, constant over each rest-frame time slice:

```
Φ(t′) = ω₀ t′ ,        ω₀ = m c² / ħ   (Compton angular frequency).
```

Express `t′` in lab coordinates via the Lorentz transformation
`t′ = γ(t − v x / c²)`:

```
Φ(t, x) = ω₀ γ (t − v x / c²) = (γ ω₀) t − (γ ω₀ v / c²) x .
```

This is a plane wave `Φ = ω t − k x` with

```
ω = γ ω₀ = γ m c² / ħ = E / ħ          (frequency  ↔  total energy)
k = γ ω₀ v / c² = γ m v / ħ = p / ħ     (wavenumber ↔  momentum)

⇒  λ = 2π/k = h / p          ← de Broglie relation, exactly.
```

Two velocities drop out automatically:

```
phase velocity  v_φ = ω/k = c²/v   (> c, carries no signal/energy)
group velocity  v_g = dω/dk = v     (= the particle's velocity)
```

**Note the subtlety the model explains cleanly:** a lab clock sitting at a
fixed point sees the internal oscillation at `ω₀/γ` (time-dilated, slower),
yet the matter *wave* has frequency `γω₀` (faster). The two differ precisely
because the de Broglie wave is a pattern across *space* read on tilted
simultaneity slices, not a local clock rate. This is exactly de Broglie's
reconciliation of the slowed internal clock with the matter-wave frequency —
here it is a direct geometric consequence of the boost, not a separate
hypothesis.

### Numerical confirmation (`sim2`)

Reading `(ω, k)` off the lab-frame phase field by finite differences and
comparing to the relativistic `p = γmv` for a real electron:

```
 beta     v (m/s)  |  lambda_dB sim    h/p exact    rel.err |  v_phase/c  v_grp/c
 0.01  2.9979e+06  |  2.426189e-10   2.426189e-10   2.1e-16 |  100.0000   0.01000
 0.10  2.9979e+07  |  2.414148e-11   2.414148e-11   0.0e+00 |   10.0000   0.10000
 0.90  2.6981e+08  |  1.175116e-12   1.175116e-12   3.4e-16 |    1.1111   0.90000
```

`λ_sim = h/p` to machine precision; `v_φ = c²/v` exactly; `v_g = v` exactly.

## 3. Does it generalise known physics? — Yes: it yields the matter-wave
postulate of QM

- Recovers the de Broglie relation `λ = h/p` and `E = ħω` for matter —
  the empirical input of wave mechanics.
- The dispersion `ω(k) = √(ω₀² + c²k²)` (from `ω = γω₀`, `k = γβω₀/c`) is the
  relativistic matter-wave dispersion; its non-relativistic expansion
  `ħω ≈ mc² + ħ²k²/2m` is the free-particle Schrödinger dispersion (up to the
  rest-energy phase). So the Schrödinger wavelength–momentum relation is the
  low-`v` limit.
- Limit check: `m → 0` gives `ω₀ → 0`, `v → c`, `v_φ = v_g = c` — the photon.
  Matter waves and light waves are the same construction at different `m`.

It **generalises by unification**, not by deformation: the matter wave is the
moving-frame shadow of a real internal oscillation.

## 4. Does it fit experiment? — Yes, quantitatively

| experiment | geon-model number | observed |
|---|---|---|
| Davisson–Germer / LEED, 100 eV electron | `λ_dB = 122.6 pm` (`sim2` anchor) | diffraction off Ni (`d ≈ 215 pm`); textbook `≈123 pm` |
| Electron microscopy, neutron interferometry | `λ = h/p` at all `v` | standard, matches |
| Compton-clock atom interferometry (Lan et al., *Science* 2013) | internal clock at `ω₀ = mc²/ħ`, lab rate `ω₀/γ` | consistent with the measured Compton frequency tie-in |

The 100 eV electron de Broglie wavelength (`≈123 pm`) reproduced by `sim2` is
the very quantity whose diffraction founded matter-wave physics.

## 5. Predictions

- **P-B1 — Matter-wave phase = geon internal phase.** The interferometric
  phase of a matter wave is literally the accumulated internal-clock phase
  `∮(E dt − p·dx)/ħ` of the geon. PSFT's example-5/§13 picture predicts the
  *same* phase bookkeeping for photon and matter interference; any matter
  interferometer phase must equal the geon-clock phase with no extra term.
- **P-B2 — Gravitational/vorticity coupling of the clock.** Because the
  internal clock is the geon's trapped light, it should couple to the
  spacetime-fluid vorticity exactly as the paper's **Prediction 5
  (gravitational Aharonov–Bohm)** states:
  `Δφ = (m/ħ)∮ u_a dx^a`. This research re-derives that phase as the geon
  internal-clock phase shift, giving it a concrete carrier. Testable with
  atom interferometry near a rotating mass.
- **P-B3 — Exact tie between de Broglie wavelength and Compton clock.**
  `λ_dB · (γ p / m) = c² / (γ ω₀ v) · ...` ⇒ measuring the Compton frequency
  and the de Broglie wavelength of the *same* species must satisfy
  `f₀ λ_dB = c²/v_φ · …`, i.e. they are not independent. A precision joint
  measurement is a clean consistency test.

## 6. What this does *not* settle

- The **full Born rule** for arbitrary observables is not derived (consistent
  with the paper's stance: classical field energy density suffices for
  *interference*, but general quantisation of PSFT is open).
- The internal oscillation is treated as a single carrier frequency `ω₀`; a
  realistic geon has a structured internal field whose full spectrum is not
  modelled here.
