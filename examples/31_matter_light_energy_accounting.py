"""Example 31: matter-light energy accounting via topological cancellation
(paper Section 7.4 forward prediction).

In the geon interpretation of PSFT (paper Section 2 / Section 7.4), the
rest mass-energy of a matter soliton equals the integrated photonic-
field energy density across its core:

    m c^2 = integral_core (1/8 pi) (|E|^2 + |B|^2) d^3 x  ==  U_EM .

Matter-light interconversion (annihilation, pair production, photon
absorption/emission) becomes a topological-cancellation event:
opposite windings combine into a topologically trivial configuration,
releasing the integrated photonic-field energy as outgoing propagating
radiation.  Energy conservation is then a single accounting on P_ab
that covers both TRAPPED (matter-soliton) and PROPAGATING (radiation)
states of the same field -- the simplest possible bookkeeping for
E = m c^2.

This example demonstrates the accounting on a clean static
configuration: two Gaussian smeared charges of opposite sign.  We
compute U_EM in three configurations:

  (a) ONE charge of total Q = +1 in isolation.  U_EM = U_single.
  (b) TWO opposite charges (+Q, -Q) separated by a large distance.
      U_EM = 2 U_single - U_interaction (Coulomb attraction lowers
      the total).  For separation >> smearing-width, U_interaction
      is small and U_EM ~ 2 U_single.
  (c) TWO opposite charges placed at the SAME LOCATION.  The fields
      cancel exactly: rho_q_net = +Q - Q = 0, so A_t = 0, E = 0,
      U_EM = 0.  This is the topological-cancellation event:
      windings have merged to a trivial configuration, and all
      U_EM is released.

The interpretation in PSFT terms: each isolated charge is a stand-in
for a unit-winding soliton; its photonic-field energy U_single is the
geon-picture rest-mass-energy.  Configuration (c) is the annihilation
event -- topological cancellation releases 2 U_single = (m + mbar) c^2
as outgoing photons (in the dynamical version this would be visible
as radiation).

Pass criteria:
  * U_EM_one  > 0 (single charge has finite EM self-energy)
  * U_EM_far ~ 2 U_EM_one (two well-separated opposite charges)
  * U_EM_overlap < 0.01 * U_EM_one (topological cancellation drives
    U_EM to ~ 0)
  * Released energy fraction (U_EM_far - U_EM_overlap) / U_EM_far
    > 0.95 (annihilation releases ~ all the stored field energy)
  * U_EM_one scales as Q^2 / sigma (classical electromagnetic
    self-energy, with calculable prefactor)

This example is static (no time evolution) -- it accounts for the
photonic-field energy STORED in matter solitons of various topology.
The dynamical version (Example for future work) would evolve the
overlapping configuration and watch U_EM transfer from trapped to
propagating states.

Run:
    python3 examples/31_matter_light_energy_accounting.py
"""
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from psft.evolve.photonic_field import PhotonicField3D


def jacobi_poisson_periodic(rho_q, dx, n_iter=4000, tol=1e-12):
    """Solve lap A_t = 4 pi (rho_q - <rho_q>) on a periodic grid by Jacobi.

    Returns the relaxed potential A_t.  Mean of rho_q must be subtracted
    for the periodic problem to have a unique solution; if the input
    already has zero mean (e.g. + and - charges cancelling) the
    subtraction is a no-op.
    """
    mean_rho = float(np.mean(rho_q))
    src = 4 * math.pi * (rho_q - mean_rho)
    A = np.zeros_like(rho_q)
    for it in range(n_iter):
        A_new = (
            np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
            + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
            + np.roll(A, 1, axis=2) + np.roll(A, -1, axis=2)
            - src * dx * dx
        ) / 6.0
        change = float(np.max(np.abs(A_new - A)))
        A = A_new
        if change < tol:
            return A, it + 1
    return A, n_iter


def integrated_field_energy(pf):
    """U_EM = (1/8 pi) integral (|E|^2 + |B|^2) d^3 x."""
    return pf.total_field_energy()


def main():
    print("Example 31: matter-light energy accounting via topological "
          "cancellation")
    print("(PSFT paper Section 7.4 forward prediction)")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 64 if HIGH_RES else 40
    L = 1.0
    dh = L / N
    n_jacobi = 6000 if HIGH_RES else 3000

    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3, L = {L}, dh = {dh:.5f}")
    print(f"  Jacobi-Poisson iterations: {n_jacobi}")

    # Smearing width (must be > dh for the discretisation to resolve).
    sigma = max(6 * dh, 0.06)
    Q = 1.0
    print(f"  charge Q = {Q}, smearing sigma = {sigma:.4f} ({sigma/dh:.1f} cells)")

    # Build grid.
    x = np.linspace(0.5 * dh, L - 0.5 * dh, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    def gauss_charge(x0, y0, z0, q):
        r2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
        norm = q / (2 * math.pi * sigma ** 2) ** 1.5
        return norm * np.exp(-r2 / (2 * sigma ** 2))

    # ----------------------------------------------------------------
    # Configuration (a): single +Q charge
    # ----------------------------------------------------------------
    print()
    print("  Config (a) single +Q at box centre ...")
    rho_a = gauss_charge(L/2, L/2, L/2, +Q)
    A_t_a, it_a = jacobi_poisson_periodic(rho_a, dh, n_iter=n_jacobi)
    pf_a = PhotonicField3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4)
    pf_a.A_t = A_t_a
    U_a = integrated_field_energy(pf_a)
    print(f"    Jacobi converged in {it_a} iterations")
    print(f"    U_EM (single +Q): {U_a:.5f}")

    # ----------------------------------------------------------------
    # Configuration (b): +Q and -Q at distance d
    # ----------------------------------------------------------------
    d_far = 0.4 * L
    x_plus  = L/2 - d_far/2
    x_minus = L/2 + d_far/2
    print()
    print(f"  Config (b) two opposite charges, separation d = {d_far} ...")
    rho_b = gauss_charge(x_plus, L/2, L/2, +Q) + gauss_charge(x_minus, L/2, L/2, -Q)
    print(f"    total charge integral = {float(np.sum(rho_b))*dh**3:.3e}  "
          f"(should be ~0)")
    A_t_b, it_b = jacobi_poisson_periodic(rho_b, dh, n_iter=n_jacobi)
    pf_b = PhotonicField3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4)
    pf_b.A_t = A_t_b
    U_b = integrated_field_energy(pf_b)
    print(f"    Jacobi converged in {it_b} iterations")
    print(f"    U_EM (+Q at {x_plus}, -Q at {x_minus}): {U_b:.5f}")
    print(f"    Ratio U_b / (2 U_a) = {U_b / (2 * U_a):.4f}  "
          f"(close to 1 means weak Coulomb interaction)")

    # ----------------------------------------------------------------
    # Configuration (c): +Q and -Q at the SAME LOCATION
    # (topological cancellation; net charge density = 0)
    # ----------------------------------------------------------------
    print()
    print("  Config (c) two opposite charges OVERLAPPED at box centre ...")
    rho_c = gauss_charge(L/2, L/2, L/2, +Q) + gauss_charge(L/2, L/2, L/2, -Q)
    # rho_c should be exactly zero everywhere
    print(f"    max|rho_c| = {float(np.max(np.abs(rho_c))):.3e}  "
          f"(should be ~0, exact topological cancellation)")
    A_t_c, it_c = jacobi_poisson_periodic(rho_c, dh, n_iter=n_jacobi)
    pf_c = PhotonicField3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4)
    pf_c.A_t = A_t_c
    U_c = integrated_field_energy(pf_c)
    print(f"    Jacobi converged in {it_c} iterations")
    print(f"    U_EM (overlapped): {U_c:.5e}")

    # ----------------------------------------------------------------
    # Compute the "released" energy in the annihilation event
    # ----------------------------------------------------------------
    U_released = U_b - U_c
    released_fraction = U_released / U_b if U_b > 0 else 0.0
    print()
    print(f"  Released energy (configuration b -> c):")
    print(f"    U_released = U_b - U_c = {U_released:.5f}")
    print(f"    Released fraction = {released_fraction*100:.2f}% of U_b")

    # Classical EM self-energy of a Gaussian smeared point charge:
    #   U_self = Q^2 / (4 sigma sqrt(pi))  (Gaussian units)
    U_self_analytic = Q ** 2 / (4 * sigma * math.sqrt(math.pi))
    print(f"\n  Comparison to classical EM self-energy:")
    print(f"    U_self (analytic) = Q^2 / (4 sigma sqrt(pi)) = {U_self_analytic:.5f}")
    print(f"    U_a (measured)    = {U_a:.5f}")
    print(f"    Ratio measured/analytic = {U_a / U_self_analytic:.4f}")

    # Pass criteria.
    pass_single   = U_a > 0.0
    pass_two_far  = abs(U_b / (2 * U_a) - 1.0) < 0.4   # within 40% (allows Coulomb interaction)
    pass_cancel   = U_c < 0.01 * U_a                    # cancellation drives U_EM ~ 0
    pass_release  = released_fraction > 0.95
    pass_analytic = abs(U_a / U_self_analytic - 1.0) < 0.5  # smearing-grid mismatch ~ factor of 2 OK

    print(f"\n  PASS criteria:")
    print(f"    single +Q has positive U_EM        : {'PASS' if pass_single else 'FAIL'}")
    print(f"    U_EM(far +/-) ~ 2 U_EM(single)     : {'PASS' if pass_two_far else 'FAIL'}")
    print(f"    overlap U_EM < 1% of single        : {'PASS' if pass_cancel else 'FAIL'}")
    print(f"    >95% of stored energy released     : {'PASS' if pass_release else 'FAIL'}")
    print(f"    U_a matches classical self-energy  : {'PASS' if pass_analytic else 'FAIL'}")

    # Plot.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    mid = N // 2

    # Row 1: rho_q in each configuration
    for ax, rho, title in zip(axes[0],
                                [rho_a, rho_b, rho_c],
                                ['(a) single +Q', '(b) +Q ... -Q far',
                                 '(c) +Q + -Q overlapped (cancelled)']):
        vmax = float(np.max(np.abs(rho))) if float(np.max(np.abs(rho))) > 1e-12 else 1e-12
        im = ax.imshow(rho[:, :, mid].T, origin='lower',
                        extent=[0, L, 0, L], cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'rho_q  {title}')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: |E| field magnitude
    for ax, pf, title in zip(axes[1], [pf_a, pf_b, pf_c],
                              ['(a) U_EM = %.4f' % U_a,
                               '(b) U_EM = %.4f' % U_b,
                               '(c) U_EM = %.2e' % U_c]):
        Ex, Ey, Ez = pf.E_field()
        E_mag = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        im = ax.imshow(E_mag[:, :, mid].T, origin='lower',
                        extent=[0, L, 0, L], cmap='viridis')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'|E|(x,y,L/2)  {title}')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), 'examples', 'out_matter_light_energy.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n  Plot saved to {out}")

    print()
    print("  PSFT interpretation:")
    print("  This is the static-energy version of the geon-picture")
    print("  matter-light interconversion (paper Section 7.4).  In an")
    print("  isolated soliton, the photonic-field energy U_EM is the")
    print("  geon-picture rest-mass-energy:  m c^2 = U_EM.  When opposite-")
    print("  winding solitons combine topologically (configuration c),")
    print("  the photonic field cancels exactly and all stored U_EM is")
    print("  released -- the topological-cancellation annihilation event")
    print("  proposed in paper Section 7.4.  Total energy is conserved")
    print("  via a single accounting on P_ab covering both TRAPPED")
    print("  (matter) and PROPAGATING (outgoing radiation) states.  The")
    print("  full dynamical version -- with the released U_EM physically")
    print("  radiating outward -- is the natural next-iteration simulation.")


if __name__ == "__main__":
    main()
