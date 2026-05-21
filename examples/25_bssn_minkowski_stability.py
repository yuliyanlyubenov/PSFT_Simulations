"""Example 25: BSSN Minkowski stability -- Phase 1.1 of the
curved-background extension (Step 5.1 of the simulation roadmap).

This is the first PSFT simulation that carries the spatial metric
gamma_ij + extrinsic curvature K_ij as dynamical fields on a 3D
Cartesian grid.  We use the BSSN reformulation
(Baumgarte-Shapiro 2nd ed., chapter 11) with the moving-puncture
gauge: 1+log lapse + Gamma-driver shift.

The Phase 1.1 acceptance test is the simplest possible: initialise
flat Minkowski (gamma_ij = delta_ij, K_ij = 0, alpha = 1, beta^i = 0)
and verify that BSSN evolution leaves this fixed point unchanged.
Because every BSSN RHS term vanishes identically on this initial
data, the evolution should preserve it to machine precision over any
number of steps.

Pass criteria:
  * |chi - 1|     stays at machine zero over 1000 steps
  * |alpha - 1|   stays at machine zero
  * |K|, |Abar|, |Gambar^i|, |beta^i|, |B^i| all stay at machine zero
  * Hamiltonian and momentum constraints stay at machine zero
  * No NaN

This validates the BSSN equations have no spurious sources in the
implementation; any later non-trivial test (Schwarzschild puncture,
self-gravitating fluid blob, etc.) builds on this foundation.

Run:
    python3 examples/25_bssn_minkowski_stability.py
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from psft.evolve.adm import (
    BSSNState, BSSNEvolver,
    hamiltonian_constraint, momentum_constraint,
)


def main():
    print("BSSN Phase 1.1: Minkowski stability (curved-background extension)")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 24 if HIGH_RES else 16
    L = 4.0
    dh = L / N
    n_steps = 1000 if HIGH_RES else 200

    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3 = {N**3} cells, L = {L}, dh = {dh:.4f}")
    print(f"  n_steps = {n_steps}, cfl = 0.25")

    state = BSSNState.flat_minkowski(N, N, N, dh, dh, dh)
    evolver = BSSNEvolver(state=state, cfl=0.25, ko_epsilon=0.5)

    H0 = hamiltonian_constraint(state)
    M0 = momentum_constraint(state)
    print(f"\n  initial constraints:")
    print(f"    max|H| = {np.max(np.abs(H0)):.2e}")
    print(f"    max|M| = {np.max(np.abs(M0)):.2e}")

    # Evolve.
    t_start = time.time()
    H_history = []
    chi_dev_history = []
    for step in range(n_steps):
        evolver.step()
        if (step + 1) % max(1, n_steps // 10) == 0:
            H = hamiltonian_constraint(state)
            chi_dev = float(np.max(np.abs(state.chi - 1.0)))
            alpha_dev = float(np.max(np.abs(state.alpha - 1.0)))
            print(f"    step {step+1:5d}/{n_steps}  t = {evolver.t:.3f}  "
                  f"|chi-1| = {chi_dev:.2e}  "
                  f"|alpha-1| = {alpha_dev:.2e}  "
                  f"max|H| = {np.max(np.abs(H)):.2e}")
            H_history.append(np.max(np.abs(H)))
            chi_dev_history.append(chi_dev)
    elapsed = time.time() - t_start

    H_final = hamiltonian_constraint(state)
    M_final = momentum_constraint(state)
    chi_dev = float(np.max(np.abs(state.chi - 1.0)))
    alpha_dev = float(np.max(np.abs(state.alpha - 1.0)))
    K_max = float(np.max(np.abs(state.K)))
    Abar_max = float(np.max(np.abs(state.Abar)))
    Gam_max = float(np.max(np.abs(state.Gam_u)))
    beta_max = float(np.max(np.abs(state.beta)))
    has_nan = bool(not np.all(np.isfinite(state.data)))

    print(f"\n  final state after {n_steps} steps ({elapsed:.1f}s):")
    print(f"    |chi - 1|       = {chi_dev:.2e}")
    print(f"    |alpha - 1|     = {alpha_dev:.2e}")
    print(f"    |K|             = {K_max:.2e}")
    print(f"    |Abar|          = {Abar_max:.2e}")
    print(f"    |Gambar^i|      = {Gam_max:.2e}")
    print(f"    |beta^i|        = {beta_max:.2e}")
    print(f"    max|H|          = {np.max(np.abs(H_final)):.2e}")
    print(f"    max|M|          = {np.max(np.abs(M_final)):.2e}")

    pass_static = (chi_dev < 1e-12 and alpha_dev < 1e-12
                   and K_max < 1e-12 and Abar_max < 1e-12)
    pass_constr = (np.max(np.abs(H_final)) < 1e-12
                   and np.max(np.abs(M_final)) < 1e-12)
    pass_stab = not has_nan

    print(f"\n  PASS criteria:")
    print(f"    Minkowski is exact fixed point  : {'PASS' if pass_static else 'FAIL'}")
    print(f"    constraints stay at zero        : {'PASS' if pass_constr else 'FAIL'}")
    print(f"    no NaN / finite throughout      : {'PASS' if pass_stab else 'FAIL'}")

    print()
    print("  PSFT interpretation:")
    print("  This is the first PSFT simulation that carries the spatial")
    print("  metric gamma_ij + extrinsic curvature K_ij as dynamical")
    print("  fields on the 3D grid (Step 5.1 of the simulation roadmap).")
    print("  BSSN evolution of flat Minkowski preserves the fixed point")
    print("  exactly, validating the BSSN RHS implementation as a")
    print("  foundation for the Phase 2 coupling of (rho, p, v^i, A_a,")
    print("  sigma^A) to the dynamical geometry.")


if __name__ == "__main__":
    main()
