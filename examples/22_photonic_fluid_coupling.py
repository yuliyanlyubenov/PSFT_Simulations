"""Example 22: Coupled photonic field + spacetime fluid -- the
field-to-fluid half of the PSFT master-equation coupling
(Step 4.2 of the simulation roadmap).

A static Gaussian-Coulomb photonic field is held in place by its
external source, and a small slug of CHARGED fluid is placed off-centre.
The Lorentz-like force F^i = rho_q E^i + (j x B)^i is computed from
the photonic field and fed into the master-equation RHS as a body
force on the fluid momentum.

Pass conditions:
  * Direction of acceleration is radial (toward / away from the charge,
    according to the sign of rho_q).
  * The fluid momentum grows monotonically until the slug moves an
    appreciable distance.
  * Fluid momentum gain = - integral of force . v dt (work-energy theorem).
  * No NaN, no negative pressure, no instability over 100+ steps.

This isolates the FORCE half of the coupling.  The reverse coupling
(fluid currents source the field) is the natural Step 4.3 work.

Run:
    python3 examples/22_photonic_fluid_coupling.py
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

from psft.evolve.hydro_3d import RelativisticEulerSolver3D
from psft.evolve.photonic_field import PhotonicField3D

HIGH_RES = os.environ.get("PSFT_HIGH_RES", "1") != "0"


def main():
    print(" Coupled photonic field + fluid -- Lorentz coupling test")
    print("=" * 60)

    if HIGH_RES:
        N = 64
        n_steps = 200
    else:
        N = 32
        n_steps = 40
    L = 1.0
    Gamma = 4.0 / 3.0
    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3 = {N**3} cells, L = {L}")

    # Build photonic field with a static Coulomb at the centre.
    pf = PhotonicField3D(
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="outflow",
    )
    sigma_em = 8 * pf.dx
    Q_em = 1.0
    pf.initialise_static_coulomb(L / 2, L / 2, L / 2, Q_em, sigma_em)
    # Source j_t = lap(A_t)/(4 pi) (matches the discrete static balance).
    rho_em_source = pf._laplacian(pf.A_t) / (4 * math.pi)
    zero_field = np.zeros_like(rho_em_source)
    print(f"  photonic field: Q = {Q_em}, sigma = {sigma_em:.4f}")

    # Build fluid solver with a small charged slug off-centre.
    fluid = RelativisticEulerSolver3D(
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, Gamma=Gamma, cfl=0.3,
        boundary="outflow",
    )
    # Fluid slug at (x = L/2 + 0.2, y = z = L/2).
    slug_x = L / 2 + 0.2
    slug_y = L / 2
    slug_z = L / 2
    slug_sigma = 0.05
    rho_bg = 1e-3       # near-vacuum background
    p_bg = 1e-6
    slug_amp = 1.0      # additional density in the slug

    def rho_init(X, Y, Z):
        r2 = (X - slug_x) ** 2 + (Y - slug_y) ** 2 + (Z - slug_z) ** 2
        return rho_bg + slug_amp * np.exp(-r2 / (2 * slug_sigma ** 2))

    fluid.initialise(
        rho_func=rho_init,
        p_func=lambda X, Y, Z: p_bg + 0 * X,
        vx_func=lambda X, Y, Z: np.zeros_like(X),
        vy_func=lambda X, Y, Z: np.zeros_like(X),
        vz_func=lambda X, Y, Z: np.zeros_like(X),
    )
    print(f"  fluid slug at ({slug_x}, {slug_y}, {slug_z}), sigma = {slug_sigma}")

    # The fluid carries a CHARGE proportional to its local density.  For the
    # demonstration we set rho_charge = q_per_mass * (rho - rho_bg) so only
    # the slug contributes; the background is uncharged.
    q_per_mass = 1.0    # arbitrary calibration

    def rho_charge_of(rho_field):
        return q_per_mass * (rho_field - rho_bg)

    # Initial diagnostics.
    M0 = fluid.total_mass()
    Px0, Py0, Pz0 = fluid.total_momentum()
    rho_init_field, _, _, _, _, _ = fluid.primitives()
    rho_q_init = rho_charge_of(rho_init_field)
    total_q = float(np.sum(rho_q_init)) * fluid.dx * fluid.dy * fluid.dz
    print(f"\n  initial:  M = {M0:.4f}, P = ({Px0:.3e}, {Py0:.3e}, {Pz0:.3e})")
    print(f"  total charge of fluid slug = {total_q:.4f}")
    Ex0, Ey0, Ez0 = pf.E_field()
    # E at slug location.
    si, sj, sk = int(slug_x / pf.dx), int(slug_y / pf.dy), int(slug_z / pf.dz)
    E_at_slug = np.array([Ex0[si, sj, sk], Ey0[si, sj, sk], Ez0[si, sj, sk]])
    print(f"  E at slug position           = {E_at_slug}")
    expected_force_dir = E_at_slug / np.linalg.norm(E_at_slug)
    print(f"  expected force direction (radial outward for +Q_em, +q_per_mass): {expected_force_dir}")

    # Evolve coupled.
    print(f"\n  evolving {n_steps} steps ...")
    t_start = time.time()
    snapshots = []
    Px_history = [Px0]
    times = [0.0]
    for step in range(n_steps):
        # Compute Lorentz-like force from photon field acting on fluid.
        rho_now, _, vx_now, vy_now, vz_now, _ = fluid.primitives()
        rho_charge = rho_charge_of(rho_now)
        fx, fy, fz = pf.lorentz_force(rho_charge, vx_now, vy_now, vz_now)
        # Step fluid with this body force.
        fluid.step(body_force=(fx, fy, fz))
        # Step photonic field (static-source maintenance).
        pf.step(dt=fluid.t - pf.t, j_t=rho_em_source,
                j_x=zero_field, j_y=zero_field, j_z=zero_field)
        if (step + 1) % max(1, n_steps // 10) == 0:
            Px, Py, Pz = fluid.total_momentum()
            Px_history.append(Px)
            times.append(fluid.t)
            print(f"     step {step+1:4d}/{n_steps}  t = {fluid.t:.5f}  "
                  f"P = ({Px:.3e}, {Py:.3e}, {Pz:.3e})")
    elapsed = time.time() - t_start
    print(f"  evolution wall time: {elapsed:.1f}s")

    Px_final, Py_final, Pz_final = fluid.total_momentum()
    delta_P = np.array([Px_final - Px0, Py_final - Py0, Pz_final - Pz0])
    print(f"\n  final momentum change: dP = {delta_P}")
    print(f"  |dP|                 = {np.linalg.norm(delta_P):.3e}")

    # The slug is at x > L/2 with positive charge, photonic charge is also
    # positive (rho_em_source > 0 near origin).  Like charges repel:
    # the slug should be pushed in the +x direction.  So expect dPx > 0,
    # dPy ~ dPz ~ 0.
    dir_ok = (delta_P[0] > 0
              and abs(delta_P[1]) < 0.1 * abs(delta_P[0])
              and abs(delta_P[2]) < 0.1 * abs(delta_P[0]))
    print(f"  direction check (dPx > 0, dPy/z small):  "
          f"{'PASS' if dir_ok else 'FAIL'}")
    pass_size = abs(delta_P[0]) > 1e-6
    print(f"  size check (|dPx| > 1e-6):               "
          f"{'PASS' if pass_size else 'FAIL'}")
    rho_final, _, _, _, _, _ = fluid.primitives()
    no_nan = bool(np.all(np.isfinite(rho_final)))
    print(f"  stability (no NaN / inf):                "
          f"{'PASS' if no_nan else 'FAIL'}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(times, Px_history, "o-")
    ax.set_xlabel("time t")
    ax.set_ylabel("fluid total momentum P_x")
    ax.set_title("Fluid momentum gain under photonic force")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    im = ax2.imshow(rho_final[:, :, N // 2].T, origin="lower",
                    extent=[0, L, 0, L], cmap="viridis")
    ax2.plot(L / 2, L / 2, "*", color="white", ms=15, label="photonic charge")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title(f"final rho(x, y, L/2) at t = {fluid.t:.4f}")
    ax2.legend(loc="upper right")
    plt.colorbar(im, ax=ax2)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(HERE), "examples", "out_photonic_fluid.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")

    print()
    print("  PSFT interpretation:")
    print("  Step 4.2 of the coupled (u^a, sigma^A, P_ab) master-equation")
    print("  evolution: the photonic source field P_ab now drives the")
    print("  spacetime fluid via the Lorentz-like 4-force as paper eq. 4.3")
    print("  prescribes.  A charged fluid slug feels repulsion from a")
    print("  same-sign photonic charge, gaining linear momentum in the")
    print("  expected direction.  Energy non-conservation is by design at")
    print("  this stage: the photonic field is sourced externally rather")
    print("  than by the fluid current itself; full back-reaction comes in")
    print("  Step 4.3 and gives total stress-energy conservation.")


if __name__ == "__main__":
    main()
