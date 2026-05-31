# A geon is confined light, and that *forces* exact special relativity

> **Numerical companion:** `examples/sim1_confined_light_mass_and_lorentz.py`
> (all four checks PASS). Numbers quoted below are reproduced by that script.

## 1. Idea

Take the PSFT geon literally: a matter particle is light (the primitive
photonic field) trapped/orbiting around a concentrated-energy core, stabilised
by the high-`K` SU(3) viscosity (paper §2, §7.4). The constituent light moves
at `c` in every frame because the U(1) sector is exactly inviscid
(`Kc^EM = ∞`).

**Claim:** every relativistic property of the *massive* geon — rest mass,
time dilation, length contraction, `E = γmc²`, `p = γmv`, the mass-shell
`E² − (pc)² = (mc²)²` — is a logical consequence of confining light that moves
at `c`, using nothing but the ordinary Lorentz transformation. PSFT does not
modify special relativity for matter; it explains *why* matter obeys it.

This is the conservative backbone. Ideas B and C build on it.

## 2. Math framework

### 2.1 Rest mass from trapped light (the "photon-in-a-box" theorem)

Consider light of total energy `E₀` confined to a bounded region with **zero
net momentum** (e.g. two equal counter-propagating beams, or a closed orbit).
The total 4-momentum in the rest frame is

```
P^μ_rest = (E₀/c, 0, 0, 0).
```

The invariant mass is `m² c² = −P_μ P^μ = (E₀/c)²`, i.e.

```
        m = E₀ / c² .
```

Massless constituents, confined, produce an object with **inertial rest
mass**. This is standard (Einstein's box; Rindler §§; it is the same
accounting as PSFT example 31, `m c² = ∫ (1/8π)(|E|²+|B|²) d³x = U_EM`).

### 2.2 Boosting the box: emergent `γ` energy–momentum

Boost along `x` at speed `v` (`β = v/c`, `γ = 1/√(1−β²)`). A null
4-momentum transforms by the same Lorentz matrix as a position 4-vector. For
the two-beam box (each beam energy `ε`, `E₀ = 2ε`):

```
forward beam:  ε → ε √[(1+β)/(1−β)]      (blueshift)
backward beam: ε → ε √[(1−β)/(1+β)]      (redshift)

E_lab  = ε(√[(1+β)/(1−β)] + √[(1−β)/(1+β)]) = 2εγ = γ E₀ = γ m c²
p_lab  = (ε/c)(√[(1+β)/(1−β)] − √[(1−β)/(1+β)]) = (2ε/c)γβ = γ m v
```

and therefore, identically,

```
E_lab² − (p_lab c)² = (m c²)² .
```

The relativistic dispersion of a massive particle is the Doppler-summed
energy of trapped light. `sim1` confirms this to 12 significant figures for
`β = 0.1 … 0.999`:

```
 beta    E/(m c^2)    gamma  |  p c/(m c^2)  gamma*beta | invariant/(mc^2)^2
 0.900    2.294157  2.294157 |    2.064742    2.064742  |   1.000000000000
 0.999   22.366272 22.366272 |   22.343906   22.343906  |   1.000000000000
```

### 2.3 The orbiting geon: the internal light-clock dilates by γ

Model the geon as a photon circulating in the `x–y` plane at radius `R`,
`x′ = R cos ω₀t′`, `y′ = R sin ω₀t′`, with tangential speed `ω₀R = c`
(`ω₀ = c/R`, rest period `T₀ = 2πR/c`). Boost to the lab frame:

```
t = γ(t′ + v x′/c²),   x = γ(x′ + v t′),   y = y′.
```

Over one internal revolution `Δt′ = T₀` the periodic `v x′/c²` term nets to
zero, so the lab-frame period is

```
        T_lab = γ T₀ .
```

The geon's *internal clock runs slow by exactly γ*. This is the ordinary
light-clock argument, but now the clock is the particle's own internal
structure — so the particle's proper time *is* the phase of its trapped light.
`sim1` (`T_lab/T₀` column) matches `γ` to 5 digits across all boosts.

### 2.4 The orbiting photon never exceeds `c`; the geon contracts by 1/γ

Two corollaries the brief asks for explicitly:

- **Constituent stays at `c`.** Numerically differentiating the boosted
  worldline gives `|dr/dt| = c` for every boost up to `β = 0.999` (deviations
  `< 10⁻³` are pure finite-difference error). Lorentz velocity-addition of a
  transverse `c` with a longitudinal `v` returns `c` with an aberrated
  direction — the orbit tips into a forward-beamed cycloid (relativistic
  aberration / headlight effect), but the speed is invariant. **The orbiting
  light is never superluminal, even when the geon centre moves at 0.999 c.**
- **Geon length-contracts.** Snapshotting the full ring of light at one lab
  instant (correctly accounting for relativity of simultaneity) gives an
  ellipse with `x`-extent `2R/γ` and unchanged `y`-extent: the soliton
  Lorentz-contracts along its motion. `sim1` (`x-extent/R` column) matches
  `1/γ` to 4 digits.

## 3. Does it generalise known physics? — Yes, it *is* SR

The construction reproduces, with no free parameters:

- mass–energy equivalence `E₀ = mc²`;
- relativistic energy and momentum `E = γmc²`, `p = γmv`;
- the mass shell `E² − (pc)² = (mc²)²`;
- time dilation and length contraction with the correct `γ` and `1/γ`;
- invariance of `c` for the constituent.

In the limit of a single free photon (`R → ∞`, no confinement) the rest mass
vanishes and we recover a massless particle on the light cone. The geon model
*contains* both massive matter and free light as the confined / unconfined
cases of one entity. This is the strongest possible "generalisation" outcome:
the idea does not extend SR, it **derives** the part of SR that applies to
matter from the part that applies to light.

## 4. Does it fit experiment? — Yes, by construction

| measurement | what the geon model says | status |
|---|---|---|
| Pair production / annihilation `2γ ↔ e⁺e⁻` | rest energy = released trapped-light energy `mc²` | matches `mc²` |
| Time dilation (muon lifetime, GPS, optical clocks at v) | internal light-clock slows by γ | matches γ to current precision |
| Relativistic dynamics in accelerators (`p = γmv`) | Doppler-summed momentum of trapped light | matches |
| Mass-shell relation in every collider event | `E² − (pc)² = (mc²)²` | matches identically |

Because the model output *is* the SR prediction, it inherits SR's entire
experimental record rather than competing with it.

## 5. Predictions (PSFT-specific, beyond bare SR)

- **P-A1 — Rest mass = integrated trapped photonic energy.**
  `m c² = ∫_core (1/8π)(|E|² + |B|²) d³x`. This is sharper than "E=mc²": it
  ties a particle's mass to a *spatial integral of its internal light field*,
  and predicts that the mass defect in binding equals the change in trapped
  field energy. Consistent with PSFT example 31's topological-cancellation
  accounting; falsifiable in principle by any first-principles soliton mass
  calculation that disagrees with the field-energy integral.
- **P-A2 — Internal Compton clock.** The geon carries a real internal
  oscillation at the Compton frequency `f₀ = mc²/h` (electron:
  `1.236 × 10²⁰ Hz`, reproduced by `sim1`). Its lab-frame rate is `f₀/γ`.
  This is the testable handle exploited in doc 02 (de Broglie) and is
  consistent with the atom-interferometry "Compton clock" program
  (Lan et al. 2013).
- **P-A3 — No internal degree of freedom beyond `c`-constituents.** Since the
  constituent is locked to `c`, the geon cannot store energy in a
  faster-than-light internal mode; any putative particle internal speed
  `> c` is excluded. (A consistency prediction, not a new signal.)

## 6. What this does *not* settle

- *Which* radii/energies form stable geons (the mass spectrum) — open
  (paper §17). The model fixes kinematics given a stable geon exists; it does
  not yet predict the electron mass from first principles.
- The *stabilisation* mechanism (high-`K` viscosity) is demonstrated only as
  a trapping geometry (examples 30/32), not yet as a dynamically stable bound
  state. That is the paper's own open item.
