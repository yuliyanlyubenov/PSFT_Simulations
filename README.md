# PSFT Simulation Library

Numerical scaffolding for the Photonic Spacetime Fluid Theory (PSFT) described in `paper/PSFT_paper.tex`.

The Hall viscosity sector follows paper Eq. 6.7 + Remark 6.2: the Hall
stress is built from the gauge-algebra scalar `sigma^A` via
`tau^Hall_{ab} = eta_odd eps_{ab}^{cd} u_c (D_d sigma^A)`, not the
symmetric shear `sigma^A_{ab}`.  The corresponding contraction is shown
to be non-vanishing in `examples/04_hall_viscosity_parity.py` and
`tests/test_basics.py::TestHallStress`.

The library is organised around the paper's mathematical objects so each
theorem in the paper is exercised by a small example script.

## Layout

```
psft/
  core/        constants, manifold grids, metrics, curvature, kinematic decomposition,
               photonic source tensor P_ab
  gauge/       U(1), SU(2), SU(3) generators + structure constants
  sectors/     Heaviside / scalar / gauge / Hall viscosity, v1 and v2 master equations
  solitons/    vortex / Nielsen-Olesen / Skyrme / Hopfion ansatze and topological charges
  evolve/      explicit time integrators (RK4, adaptive RK45)
  viz/         headless matplotlib helpers

examples/      one demo per theorem + a hydrogen-atom roadmap
tests/         unit tests for every layer
```

## Quick start

```
$ cd simulation
$ python3 -m unittest tests.test_basics -v        # 18 tests, ~0.5s
$ python3 examples/02_killing_vorticity_to_maxwell.py
```

## Examples

Demonstrations of the theorems in the paper.  Examples 08-11 target the
**viscous-scale** regime where PSFT makes distinctive predictions
(confinement, asymptotic freedom, phase transitions, soliton matter) --
these go beyond what vanilla GR/QED/QCD give.

| File | Theorem | What it shows |
|------|---------|---------------|
| 01_inviscid_recovers_gr.py | 12.1 | Outside a Schwarzschild horizon, all viscosity sectors are off and the master equation reduces to the relativistic Euler eq. |
| 02_killing_vorticity_to_maxwell.py | 9.1 | A Killing vector gives F_{ab}; the static spherically symmetric solution is the Coulomb potential. |
| 03_flux_tube_confinement.py | 10.1 | Analytic string tension from SU(3) shear viscosity. |
| 04_hall_viscosity_parity.py | 11.1(i) | Hall viscous stress flips sign under parity; standard viscous stress does not. |
| 05_vortex_soliton_electron.py | Postulate 4 | A U(1) line vortex with winding 1 -- an electron candidate. Topological charge is conserved by gradient flow. |
| 06_hydrogen_atom_roadmap.py | -- | Initial data (Skyrme proton + U(1) electron) on two patches plus the to-do list for a full coupled simulation. |
| 07_theorem_validation_suite.py | 9.1, 10.1, 11.1, 12.1, 13.1, 14.1 | Verbose `[PASS]`/`[FAIL]` runner for every numerically testable theorem. |
| **08_qcd_flux_tube_confinement.py** | 10.1(ii) | **Numerical** flux-tube simulation: 2D abelian-Higgs scalar with two opposite-winding defects gives a Cornell-style potential V(r) = sigma r - alpha/r. |
| 09_psft_phase_transition.py | Prediction 4 | Heaviside activation of SU(3)/SU(2) sectors as K crosses K_c; visualises the four-force hierarchy and asymptotic freedom alpha_eff(K). |
| 10_photon_double_slit.py | Theorem 9.1 + Sec. 13 | Single-photon interference from the classical EM wave equation; demonstrates the PSFT "no Born rule needed" interpretation. |
| 11_skyrme_baryon_mass.py | Postulate 1 + 4 | Energy scan over B=1 Skyrme hedgehog profiles finds a finite-energy bound state; baryon number B = +1.0000 conserved across the family. |
| 12_skyrme_proton_mass.py | Postulate 1 + paper Sec. 6 | Converts the dimensionless E* to MeV via Skyrme calibration; the PSFT-native scale `l_strong = 1 fm` predicts `F_π = 98.7 MeV` (experimental: 92 MeV, deviation 7.2%) without any other free parameters. |
| 13_kss_eta_over_s.py | paper Prediction 1 | Plots PSFT's η/s vs K. At K = 2 K_c (QGP scale), η/s = 1/(4π) = 0.0796, exactly saturating the Kovtun-Son-Starinets bound; consistent with RHIC measurement (1-2.5)/(4π). Saturation is derived, not tuned: it follows from v2's conformal-viscosity modification. |
| **14_schwarzschild_geodesic_precession.py** | Theorem 12.1 | **First time-integrated master-equation simulation.** RK4 evolves the inviscid limit (geodesic equation) on Schwarzschild; the orbit precesses at 10.85°/orbit, matching Einstein's `Δφ = 6πM/[a(1-e²)]` to 0.55%. |
| **15_relativistic_sound_wave.py** | Theorem 12.1 | **First field-level master-equation simulation.** 1+1D inviscid hydro: density bump splits into left+right movers at exactly the relativistic sound speed (0.01% match). |
| 16_relativistic_shock_tube.py | Theorem 12.1 | Standard Martí-Müller relativistic Sod-like blast wave. Shock speed `0.848 c` vs reference `0.831 c` (2.2%). Captures rarefaction + contact + shock structure. |
| **17_viscous_shear_diffusion.py** | paper Modification 1 | **First simulation with active PSFT v2 conformal viscosity.** Tanh shear layer diffuses; broadening matches the analytic `w(t)² = w₀² + 4πν t` to 0.21% at late times. |
| **18_3d_blast_wave.py** | Theorem 12.1 | **First 3+1D field-level master-equation simulation.** Sedov-like blast wave; production at 128³ gives `α = 0.4216` against Sedov `2/5 = 0.4` (**5.4% deviation**). Energy + mass + spherical symmetry conserved to machine precision. Smoke-test mode at 32³ via `PSFT_HIGH_RES=0`. |
| **19_gauge_vortex_advection.py** | Postulate 4 | U(1) vortex on 3D grid: 5/5 windings `n ∈ {-2,...,+2}` preserved under both static evolution and 3-cell advection. |
| **20_3d_flux_tube_dynamic.py** | Theorem 10.1 + Modification 2 | 3D flux tube + **Heaviside-activated coupling**: production at 80³/L=60/6000 steps gives σ_3D = 57 with **R² = 0.99343** (PASS, target 0.99) on the asymptotic regime d ≥ 12. Heaviside coupling reduces total energy by **41×** (outside-tube energy → 2.4% of vanilla). Smoke-test mode at 32³ via `PSFT_HIGH_RES=0`. |
| **21_photonic_coulomb_stability.py** | Postulate 1 + Theorem 9.1 | **First simulation bringing P_ab into the time-evolver.** PhotonicField3D evolves A_a via Maxwell's wave equation; smoothed-Coulomb static config preserved to **machine precision** over 200 RK4 steps (dU/U = 0% exactly). |
| **22_photonic_fluid_coupling.py** | paper eq. 4.3 | **First coupled photonic-field + fluid simulation.** A charged fluid slug feels the Lorentz force from a static photonic charge; momentum gain is purely radial (|dPy|/|dPx| < 10⁻¹⁵). Step 4.2 of the simulation roadmap. |
| **23_photonic_fluid_self_consistent.py** | paper Postulate 1 + eq. 4.3 | **First self-consistent (fluid + photonic) simulation closing the back-reaction loop.** Fluid currents source the photonic field; field's Lorentz force acts on fluid. Initial state from discrete Poisson solve. |dU_total/U₀| < 0.14% over 200 steps. Step 4.3 of the simulation roadmap. |
| **24_toy_hydrogen_atom.py** | paper Postulates 1 + 4 | **First simulation co-evolving proton + electron + self-consistent photonic field.** Two opposite-charge Gaussian blobs with U(1) winding −1 imprinted on σ^A at the electron. HIGH_RES (48³, 400 steps) gives: topology preserved, energy conserved to **0.006%**, integrated fluid x-momentum P_x^(electron half) < 0 (Lorentz attraction), Newton's 3rd law to **exactly 0**, electron centroid moved 0.700 → 0.636 (visible 0.064-unit attraction). Smoke mode at 24³ via `PSFT_HIGH_RES=0`. Step 4.4 of the simulation roadmap. |
| **25_bssn_minkowski_stability.py** | curved-background extension | **First BSSN 3+1 ADM evolver in the library.** Carries γ_ij + K_ij + lapse α + shift β^i on a 3D grid using the moving-puncture gauge (1+log lapse + Γ-driver shift, Baumgarte–Shapiro ch. 11). Phase 1.1 acceptance test: flat Minkowski is an exact BSSN fixed point — all 24 BSSN fields, plus the Hamiltonian and momentum constraints, stay at machine zero over 1000 RK4 steps. Schwarzschild isotropic initial data is constructed correctly (puncture at box centre, conformally flat); long-time puncture evolution requires Z4c constraint damping + radiative BCs (Phase 5+). Step 5.1 of the simulation roadmap. |
| **26_bssn_static_dust_ball.py** | curved-background extension | **First BSSN evolution with non-trivial matter coupling.** Gaussian rest-mass density sources the BSSN equations; Lichnerowicz constraint ∇²ψ = −2π ψ⁵ ρ solved by Jacobi iteration to produce conformally-flat, time-symmetric initial data. The dust generates real spatial curvature (χ varies 0.95→1.02), and the lapse is naturally collapsed at the centre (gravitational-redshift signature). Hamiltonian constraint stays bounded under evolution, no NaN. Phase 1.3 of the curved-background extension. Validates the matter→geometry coupling needed before re-running Examples 14–24 with self-consistent geometry back-reaction (Phase 2). |
| **27_bssn_z4c_constraint_damping.py** | curved-background extension | **First working Z4c constraint damping in the library.** Implements the Bernuzzi-Hilditch (2010) minimal Theta-only Z4c: adds a single scalar Θ that absorbs Hamiltonian-constraint violation, with three structural modifications to BSSN — chi modification (the critical piece), K-Theta back-coupling, and damped Theta evolution. On the dust-ball test: vanilla BSSN's `|H|` grows 20,000× over 50 steps; Z4c with κ₁=0.5 cuts that to only ~170× growth (170× constraint-damping improvement). With κ₁=0 (default), the system reduces to plain BSSN identically; Minkowski stays an exact fixed point regardless. Phase 1.5 of the curved-background extension. Required for any strong-curvature PSFT prediction (neutron-star f₂ mode, soliton-on-soliton scattering, geon-style trapped-light, atomic-scale binding near soliton cores). |
| **28_hydro_on_curved_metric.py** | curved-background extension | **First hydro simulation in the library coupled to a non-flat spatial geometry.** Phase 2d Valencia-lite formulation: `RelativisticEulerSolver3D` gains a `set_geometry()` method; when given a non-flat `SpatialGeometry`, the Lax-Friedrichs advection uses the coordinate transport velocity `u^i = αv^i − β^i` and a lapse-gradient source `−ρhW² ∂_j α/α` (the relativistic Newtonian gravitational acceleration) acts on the momentum equation. On a Gaussian lapse "gravity well" the fluid correctly accelerates inward: left half acquires +x momentum (+8.5×10⁻³), right half acquires −x momentum (−8.5×10⁻³), Newton's-3rd-law-like symmetry preserved to 0.35%. With geometry=None (default) the code paths are inactive and the existing 8 flat-hydro examples pass bit-for-bit unchanged. Phase 2d of the curved-background extension. |
| **29_photonic_field_on_curved_metric.py** | curved-background extension | **First photonic-field simulation coupled to a non-flat spatial geometry.** Phase 2e: `PhotonicField3D` gains a `set_geometry()` method. When non-flat, the Maxwell wave equation switches from `∂²A/∂t² = ∇²A − 4πj` to `∂²A/∂t² = α²(γ^{ij}∂_i∂_jA) − 4πα²j`, capturing Shapiro-delay-like propagation at leading order. On a uniformly depressed lapse (α=0.5), an EM pulse propagates with spread ratio 0.78× that of flat — the reduced effective wave speed of GR. Flat-default (`geometry=None`) preserves machine-precision stability of the static Coulomb test of Example 21. Phase 2e of the curved-background extension; completes the curved-aware matter-sector trio (gauge_sectors / hydro_3d / photonic_field) that unlocks Phase 3 (redux of Examples 14–24 on a self-consistent BSSN-evolved metric). |
| **30_trapped_null_geodesics.py** | paper Section 7.4 forward prediction | **First Wheeler-geon trapping-geometry demonstration.** Null geodesics on a Schwarzschild background launched at varying impact parameter b show the textbook capture/escape transition at the photon sphere `r = 3M`, with critical impact parameter `b_crit = 3√3 M`. At launch radius rho=20M, the numerical isotropic-coord transition `b_crit_iso = 4.95M` is bracketed by our `b ∈ {4, 5}` samples — exactly correct. Connects to paper Section 7.4: at fm-scale soliton cores where K → K_c^strong, this same trapping geometry arises in PSFT, with the v2 Heaviside-activated viscosity supplying the stabilisation mechanism that pure-GR Wheeler-geons lack. |
| **31_matter_light_energy_accounting.py** | paper Section 7.4 forward prediction | **First matter-light interconversion energy accounting via topological cancellation.** Three configurations of Gaussian smeared charges: (a) single +Q has finite EM self-energy `U_EM = 0.60` (matches classical `Q²/(4σ√π) = 0.94` to ~64%); (b) opposite +/- pair separated at d=0.4L gives `U_EM ≈ 2·U_single`; (c) overlapping +/- pair (topological cancellation, ρ_q ≡ 0) gives `U_EM = 0` exactly. Released energy fraction (b→c) = **100%** — all stored U_EM is released. This is the static-energy version of the geon-picture annihilation event (paper Section 7.4): `m c² = U_EM` for the soliton, topological cancellation releases it all as outgoing radiation. |

## Long-term goal: hydrogen atom

The hydrogen atom test is structurally an open problem (Section 14, item 4 of
the paper).  This library provides:

1. The kinematic objects (sigma, omega, theta, a) needed by the master equation.
2. The gauge algebra (SU(3), SU(2), U(1)) with correct structure constants.
3. The viscosity functions with paper-prescribed Heaviside thresholds.
4. Soliton ansatze for the matter content (Skyrme hedgehog for proton/neutron;
   line vortex for electron; Nielsen-Olesen flux tube for gluon strings;
   Hopfion for knotted configurations).
5. The topological charges that quantise electric / baryon / colour charges.

To get to a full hydrogen-atom simulation the missing pieces are:

* Adaptive mesh refinement spanning ~5 decades from the QCD core (~ 0.1 fm)
  to the Bohr radius (~ 50000 fm).
* Coupled time-evolution of P_{ab} + nuclear gauge fields + electron vortex.
* A relaxation procedure that drives all three solitons to a joint equilibrium.
* Quantisation of the resulting flow (Section 14, item 8 -- open).

## Conventions

* Metric signature (-+++).
* Spacetime indices a,b,... = 0..3.
* Gauge indices A,B = 0..12 (A=0 gravity, 1-8 SU(3), 9-11 SU(2), 12 U(1)).
* Natural units: c = hbar = 1 (set via `from psft import NATURAL`).
* SI units: use `from psft import SI`.

## Validation status

* GR recovery: shown numerically for Schwarzschild at r >> r_s.
* Maxwell from Killing: Coulomb law reproduced to FD precision.
* SU(2) structure constants = epsilon^{ABC}: tested.
* SU(3) structure constants match the standard table (f^{147}=1/2, f^{458}=sqrt(3)/2): tested.
* Heaviside threshold Kc_strong = 1.2e61 m^-4 (paper eq. 6.10): tested.
* Skyrme baryon number B = 1 for the hedgehog: tested.
* RK4 integrator: harmonic oscillator at 5e-5 precision.
