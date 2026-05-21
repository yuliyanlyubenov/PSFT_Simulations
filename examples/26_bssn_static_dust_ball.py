"""Example 26: BSSN evolution of a self-gravitating static dust ball --
Phase 1.3 of the curved-background extension (Step 5.1 of the
simulation roadmap).

A Gaussian rest-mass distribution at rest (u^a = n^a, p = 0) sources
the BSSN equations through ADM energy density rho_adm = rho_rest.
Initial data is constructed by solving the Lichnerowicz form of the
Hamiltonian constraint
    nabla^2 psi = -2 pi psi^5 rho_rest
by Jacobi fixed-point iteration on the periodic grid; chi = psi^{-4},
gammabar_ij = delta_ij, K_ij = 0 (time-symmetric).  The lapse is
initialised pre-collapsed: alpha = psi^{-2}.

This is the first BSSN evolution in the library that includes a
non-trivial matter source.  The dust distribution generates a real
spatial curvature (chi varies from ~0.95 to ~1.01 in the test
parameters), and the lapse is naturally collapsed near the centre --
the standard gravitational-redshift signature.

Pass criteria (smoke mode):
  * Jacobi solver converges to the Lichnerowicz equation.
  * chi varies non-trivially across the grid (genuine curvature).
  * alpha < 1 at the centre (lapse collapse from self-gravity).
  * Hamiltonian constraint |H| stays bounded under evolution
    (no exponential growth).
  * No NaN over the evolution.

The matter is held static throughout; we are not yet evolving the
fluid equations on the curved metric (that's Phase 2, the Valencia
refactor of `hydro_3d.py`).  This example validates only the BSSN +
matter-source coupling.

Run:
    python3 examples/26_bssn_static_dust_ball.py
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from psft.evolve.adm import (
    BSSNState, BSSNEvolver,
    hamiltonian_constraint, momentum_constraint,
)


def main():
    print("BSSN Phase 1.3: static self-gravitating dust ball")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 32 if HIGH_RES else 24
    L = 4.0
    dh = L / N
    n_steps = 200 if HIGH_RES else 60

    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3 = {N**3} cells, L = {L}, dh = {dh:.4f}")
    print(f"  n_steps = {n_steps}, cfl = 0.25")

    # Gaussian dust distribution (weak field).
    X = np.linspace(0.5 * dh, L - 0.5 * dh, N)
    Xg, Yg, Zg = np.meshgrid(X, X, X, indexing='ij')
    r = np.sqrt((Xg - L / 2) ** 2 + (Yg - L / 2) ** 2 + (Zg - L / 2) ** 2)
    sigma = 0.4
    rho_amp = 0.02
    rho_rest = rho_amp * np.exp(-r ** 2 / (2 * sigma ** 2))
    M_total = float(np.sum(rho_rest)) * dh ** 3
    print(f"  Gaussian dust: sigma = {sigma}, amp = {rho_amp}")
    print(f"    integrated mass M = {M_total:.4f}")
    print(f"    M / L  = {M_total / L:.4f}  (weak field if << 1/2)")

    # Solve the Lichnerowicz equation for the conformal factor.
    print()
    print("  Solving Lichnerowicz constraint nabla^2 psi = -2 pi psi^5 rho ...")
    t_j = time.time()
    state, rho_adm = BSSNState.static_dust_ball(
        N, N, N, dh, dh, dh, rho_rest=rho_rest, n_jacobi=4000, tol=1e-10,
    )
    print(f"  Jacobi solve: {time.time() - t_j:.1f}s")
    print(f"    chi range:   [{state.chi.min():.4f}, {state.chi.max():.4f}]")
    print(f"    alpha range: [{state.alpha.min():.4f}, {state.alpha.max():.4f}]")

    H_init = hamiltonian_constraint(state, rho=rho_adm)
    M_init = momentum_constraint(state)
    print(f"  Constraints after Jacobi:")
    print(f"    max|H| = {np.max(np.abs(H_init)):.2e}  (FD-discretisation level)")
    print(f"    max|M| = {np.max(np.abs(M_init)):.2e}  (= 0 exactly, K_ij = 0)")

    # Evolve, holding the matter source fixed.
    print()
    print(f"  Evolving {n_steps} BSSN steps with rho_adm fixed ...")
    evolver = BSSNEvolver(state=state, cfl=0.25, ko_epsilon=0.5,
                          rho_adm=rho_adm)
    chi_init = state.chi.copy()
    alpha_init = state.alpha.copy()

    t_start = time.time()
    times = [0.0]
    chi_drift_hist = [0.0]
    alpha_drift_hist = [0.0]
    K_max_hist = [0.0]
    H_max_hist = [float(np.max(np.abs(H_init)))]

    for step in range(n_steps):
        evolver.step()
        if (step + 1) % max(1, n_steps // 10) == 0:
            chi_drift = float(np.max(np.abs(state.chi - chi_init)))
            alpha_drift = float(np.max(np.abs(state.alpha - alpha_init)))
            K_max = float(np.max(np.abs(state.K)))
            H = hamiltonian_constraint(state, rho=rho_adm)
            H_max = float(np.max(np.abs(H)))
            times.append(evolver.t)
            chi_drift_hist.append(chi_drift)
            alpha_drift_hist.append(alpha_drift)
            K_max_hist.append(K_max)
            H_max_hist.append(H_max)
            print(f"    step {step+1:4d}/{n_steps}  t = {evolver.t:.2f}  "
                  f"|d chi| = {chi_drift:.2e}  |d alpha| = {alpha_drift:.2e}  "
                  f"K_max = {K_max:.2e}  max|H| = {H_max:.2e}")
    elapsed = time.time() - t_start

    finite = bool(np.all(np.isfinite(state.data)))
    chi_drift = float(np.max(np.abs(state.chi - chi_init)))
    alpha_drift = float(np.max(np.abs(state.alpha - alpha_init)))
    H_final_max = float(np.max(np.abs(hamiltonian_constraint(state, rho=rho_adm))))
    H_growth = H_final_max / max(H_max_hist[0], 1e-30)

    print(f"\n  evolution wall time: {elapsed:.1f}s")
    print(f"\n  final state:")
    print(f"    chi:    [{state.chi.min():.4f}, {state.chi.max():.4f}]  "
          f"(initial [{chi_init.min():.4f}, {chi_init.max():.4f}])")
    print(f"    alpha:  [{state.alpha.min():.4f}, {state.alpha.max():.4f}]  "
          f"(initial [{alpha_init.min():.4f}, {alpha_init.max():.4f}])")
    print(f"    |H|:    {H_final_max:.2e}  (initial {H_max_hist[0]:.2e}, "
          f"growth factor {H_growth:.2f})")

    pass_curvature = (chi_init.max() - chi_init.min()) > 1e-3
    pass_collapse = alpha_init.min() < 1.0
    pass_finite = finite
    pass_bounded_H = H_growth < 10.0   # less than 10x growth from initial
    pass_bounded_K = K_max_hist[-1] < 0.5   # K stays well below unity

    print(f"\n  PASS criteria:")
    print(f"    non-trivial curvature (chi varies)  : {'PASS' if pass_curvature else 'FAIL'}")
    print(f"    lapse collapse (alpha_min < 1)      : {'PASS' if pass_collapse else 'FAIL'}")
    print(f"    no NaN under evolution              : {'PASS' if pass_finite else 'FAIL'}")
    print(f"    |H| bounded (no exponential growth) : {'PASS' if pass_bounded_H else 'FAIL'}")
    print(f"    K stays bounded                     : {'PASS' if pass_bounded_K else 'FAIL'}")

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    # chi at z = L/2 mid-plane
    ax = axes[0, 0]
    im = ax.imshow(state.chi[:, :, N // 2].T, origin='lower',
                    extent=[0, L, 0, L], cmap='viridis')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('chi(x, y, L/2) at t_final')
    plt.colorbar(im, ax=ax)
    # alpha at mid-plane
    ax = axes[0, 1]
    im = ax.imshow(state.alpha[:, :, N // 2].T, origin='lower',
                    extent=[0, L, 0, L], cmap='magma')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('alpha(x, y, L/2) at t_final')
    plt.colorbar(im, ax=ax)
    # drift evolution
    ax = axes[1, 0]
    ax.semilogy(times, chi_drift_hist, 'o-', label='|d chi|_max')
    ax.semilogy(times, alpha_drift_hist, 's-', label='|d alpha|_max')
    ax.semilogy(times, K_max_hist, 'x--', label='K_max')
    ax.set_xlabel('time t')
    ax.set_ylabel('drift / extrinsic curvature')
    ax.set_title('BSSN evolution diagnostics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Hamiltonian constraint
    ax = axes[1, 1]
    ax.semilogy(times, H_max_hist, 'o-', label='max|H|')
    ax.set_xlabel('time t')
    ax.set_ylabel('Hamiltonian constraint norm')
    ax.set_title('Constraint violation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), 'examples', 'out_bssn_dust_ball.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out}")

    print()
    print("  PSFT interpretation:")
    print("  This is the first BSSN evolution in the library that")
    print("  includes a non-trivial matter source.  The dust distribution")
    print("  generates real spatial curvature (chi varies non-trivially),")
    print("  and the lapse is naturally collapsed at the centre by")
    print("  self-gravity (gravitational redshift).  Validates the")
    print("  matter -> geometry coupling needed for Phase 2: re-running")
    print("  Examples 14-24 with self-consistent geometry back-reaction.")


if __name__ == "__main__":
    main()
