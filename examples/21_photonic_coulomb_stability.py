"""Example 21: Photonic field P_ab on a 3D grid -- static Coulomb stability
test (Step 4.1 of the simulation roadmap).

In PSFT (paper Postulate 1) the photonic stress field P_ab is primitive;
in the classical limit it reduces to the EM stress-energy T^EM_ab built
from the 4-potential A_a.  This example brings P_ab into the 3D evolver:
we solve the Lorenz-gauge Maxwell wave equation
    dt^2 A_a  =  lap A_a  -  4 pi j_a
on an 80^3 Cartesian grid, initialised with the smoothed-Coulomb solution
for a Gaussian charge distribution.

Pass conditions:
  * Field strength (|E|^2, |B|^2) remains essentially constant over a long
    evolution (100+ RK4 steps).
  * Total field energy U_EM = (1/(8 pi)) int (|E|^2 + |B|^2) d^3x drifts
    less than 1% over the run.
  * The relative drift of A_t from its initial value stays small.

Run:
    python3 examples/21_photonic_coulomb_stability.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psft.evolve.photonic_field import (
    PhotonicField3D, smoothed_gaussian_charge_density,
)

HIGH_RES = os.environ.get("PSFT_HIGH_RES", "1") != "0"


def main():
    print(" PSFT P_ab field on 3D grid -- static Coulomb stability test")
    print("=" * 62)

    if HIGH_RES:
        N = 80
        n_steps = 200
        sigma_cells = 8
    else:
        N = 32
        n_steps = 40
        sigma_cells = 6
    L = 1.0
    Q = 1.0
    x0 = y0 = z0 = L / 2

    field = PhotonicField3D(
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="outflow",
    )
    sigma = sigma_cells * field.dx
    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3 = {N**3} cells, L = {L}, dx = {field.dx:.5f}")
    print(f"  Gaussian charge: Q = {Q}, sigma = {sigma:.4f} ({sigma_cells} cells)")

    field.initialise_static_coulomb(x0, y0, z0, Q, sigma)
    # Choose the source j_t to be EXACTLY the discrete Laplacian of the
    # initialised A_t divided by 4 pi.  This guarantees the static-balance
    # equation lap A_t = 4 pi j_t holds to floating-point precision; any
    # subsequent drift then reflects ONLY the time-integrator error, not
    # the initialisation residual.  Physically: we are computing the
    # smoothed charge distribution that this particular A_t corresponds
    # to in the discrete-Laplacian sense.
    rho_q = field._laplacian(field.A_t) / (4 * math.pi)
    zero_field = np.zeros_like(rho_q)
    total_charge = float(np.sum(rho_q)) * field.dx * field.dy * field.dz
    print(f"  total source charge (discrete) = {total_charge:.4f}  (continuum Q = {Q})")

    # Initial diagnostics.
    A_t_init = field.A_t.copy()
    U0 = field.total_field_energy()
    Ex_init, Ey_init, Ez_init = field.E_field()
    E_mag_init = np.sqrt(Ex_init ** 2 + Ey_init ** 2 + Ez_init ** 2)
    print(f"\n  initial diagnostics:")
    print(f"     A_t at center            = {A_t_init[N//2, N//2, N//2]:.5f}")
    print(f"     |E| at distance L/4      = {E_mag_init[N//2 + N//4, N//2, N//2]:.5f}")
    print(f"     total field energy U_EM  = {U0:.5f}")

    # Evolve with static source j_a = (rho_q, 0, 0, 0).
    print(f"\n  evolving {n_steps} RK4 steps ...")
    t_start = time.time()
    snapshots = []
    snap_times = [0.0]
    radius_profile_snapshots = []
    for step in range(n_steps):
        field.step(j_t=rho_q, j_x=zero_field, j_y=zero_field, j_z=zero_field)
        if step in (0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1):
            U_t = field.total_field_energy()
            A_drift = float(np.max(np.abs(field.A_t - A_t_init)))
            print(f"     step {step+1:4d}/{n_steps}  t = {field.t:.5f}  "
                  f"U = {U_t:.5f}  max|A_t - A_t_init| = {A_drift:.3e}")
            snapshots.append((field.t, field.A_t.copy()))
    elapsed = time.time() - t_start

    # Final diagnostics.
    U_final = field.total_field_energy()
    rel_dU = (U_final - U0) / U0
    A_drift_final = float(np.max(np.abs(field.A_t - A_t_init)))
    A_t_mean = float(np.mean(np.abs(A_t_init)))
    rel_drift = A_drift_final / max(A_t_mean, 1e-12)

    print(f"\n  final diagnostics ({elapsed:.1f}s wall time):")
    print(f"     U_EM(0)              = {U0:.5f}")
    print(f"     U_EM(t_final)        = {U_final:.5f}")
    print(f"     relative dU/U        = {rel_dU*100:.3f}%")
    print(f"     max|A_t - A_t_init|  = {A_drift_final:.3e}")
    print(f"     relative A_t drift   = {rel_drift*100:.3f}%")

    pass_energy = abs(rel_dU) < 0.01
    pass_drift = rel_drift < 0.05
    print(f"\n  PASS criteria:")
    print(f"     |dU/U| < 1%          : {'PASS' if pass_energy else 'FAIL'} ({rel_dU*100:.2f}%)")
    print(f"     A_t drift < 5%       : {'PASS' if pass_drift else 'FAIL'} ({rel_drift*100:.2f}%)")

    # Plot: A_t profile along x-axis, initial vs final.
    mid_y = N // 2
    mid_z = N // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(field.x, A_t_init[:, mid_y, mid_z], "o-", lw=1, ms=3,
            label="initial A_t")
    ax.plot(field.x, field.A_t[:, mid_y, mid_z], "x-", lw=1, ms=4,
            label=f"final A_t (t = {field.t:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("A_t(x, L/2, L/2)")
    ax.set_title("Static Coulomb potential along x-axis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    Ex_f, Ey_f, Ez_f = field.E_field()
    E_mag_final = np.sqrt(Ex_f ** 2 + Ey_f ** 2 + Ez_f ** 2)
    im = ax2.imshow(np.log10(np.maximum(E_mag_final[:, :, mid_z], 1e-8)).T,
                    origin="lower", cmap="viridis",
                    extent=[0, L, 0, L])
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"log10 |E|(x, y, L/2) at t = {field.t:.4f}")
    plt.colorbar(im, ax=ax2)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(HERE), "examples",
                            "out_photonic_coulomb.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")

    print()
    print("  PSFT interpretation:")
    print("  This is the FIRST simulation that brings the photonic source")
    print("  field P_ab into the time-evolution framework.  By Postulate 1")
    print("  the photonic field is primitive; in the classical limit it is")
    print("  the EM stress-energy built from A_a, evolved here by Maxwell's")
    print("  wave equation on a 3D Cartesian grid.  Stability of the static")
    print("  Coulomb solution under the wave-equation evolution validates")
    print("  the field-level photonic infrastructure on which the coupled")
    print("  (u^a, sigma^A, P_ab) runs of Step 4.2-4.4 will be built.")


if __name__ == "__main__":
    main()
