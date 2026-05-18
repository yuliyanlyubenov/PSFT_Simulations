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
