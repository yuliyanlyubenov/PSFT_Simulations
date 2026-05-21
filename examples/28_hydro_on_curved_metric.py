"""Example 28: relativistic hydrodynamics on a non-flat lapse field --
Phase 2d of the curved-background extension (Step 5.1 of the simulation
roadmap).

The simulation library's `RelativisticEulerSolver3D` gained an optional
`geometry` argument in Phase 2d (Valencia formulation lite).  When a
non-flat `SpatialGeometry` is attached, two changes activate:

  1. Lax-Friedrichs advection uses the coordinate transport velocity
         u^i = alpha v^i - beta^i
     instead of v^i.  In a non-trivial lapse / shift, this captures the
     correct propagation of the fluid in the curved background.

  2. A lapse-gradient source term enters the momentum equation:
         d_t S_j += -(rho h W^2) d_j alpha / alpha
     which is the relativistic version of Newton's gravitational
     acceleration -rho d_j Phi.  In the weak-field limit the
     correspondence is exact: a uniform fluid in a "gravity well"
     (region of smaller alpha) accelerates toward it.

This example demonstrates the mechanism on a Gaussian lapse well at the
box centre.  A uniform fluid initially at rest develops momentum pointing
inward (toward the well) within a few light-crossing times.  Total
x-momentum stays close to zero by the spatial symmetry of the setup --
the curved-background coupling does not break momentum conservation in
this minimal Valencia form.

Pass criteria:
  * Left half of the box acquires +x momentum (fluid moves toward well)
  * Right half acquires -x momentum
  * Total x-momentum stays small (Newton's-3rd-law-like, by symmetry)
  * No NaN

Run:
    python3 examples/28_hydro_on_curved_metric.py
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

from psft.evolve.hydro_3d import RelativisticEulerSolver3D
from psft.core.geometry_3d import SpatialGeometry


def main():
    print("Phase 2d: hydrodynamics on a non-flat lapse field (gravity well)")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 48 if HIGH_RES else 32
    L = 1.0
    dh = L / N
    n_steps = 60 if HIGH_RES else 30

    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3, L = {L}, dh = {dh:.4f}, n_steps = {n_steps}")

    # Lapse "gravity well": alpha < 1 near the box centre.
    x = np.linspace(0.5*dh, L-0.5*dh, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    delta = 0.05   # 5 percent lapse depression at the centre
    sigma = 0.15
    alpha_field = 1.0 - delta * np.exp(
        -((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2) / (2 * sigma**2)
    )
    print(f"  Lapse well: depth={delta} (alpha_min={alpha_field.min():.4f}), sigma={sigma}")

    # Geometry: flat gamma_ij, varying alpha, zero shift.
    geom = SpatialGeometry.flat(N, N, N, dx=dh)
    geom.alpha = alpha_field

    # Hydro solver.
    solver = RelativisticEulerSolver3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L,
                                        Gamma=4/3)
    solver.initialise(
        rho_func=lambda X, Y, Z: 1.0 + 0*X,
        p_func=lambda X, Y, Z: 0.01 + 0*X,
        vx_func=lambda X, Y, Z: np.zeros_like(X),
        vy_func=lambda X, Y, Z: np.zeros_like(X),
        vz_func=lambda X, Y, Z: np.zeros_like(X),
    )
    solver.set_geometry(geom)
    print(f"  Initial: rho_mean={solver.D.mean():.4f}, "
          f"total mass={solver.total_mass():.4f}")

    # Evolve and track momentum on each spatial half.
    mid = N // 2
    times = [0.0]
    Px_left_hist = [0.0]
    Px_right_hist = [0.0]
    Ptot_hist = [0.0]
    t_start = time.time()
    for step in range(n_steps):
        solver.step()
        if (step + 1) % max(1, n_steps // 10) == 0:
            Px_left  = float(np.sum(solver.Sx[:mid, :, :])) * dh**3
            Px_right = float(np.sum(solver.Sx[mid:, :, :])) * dh**3
            Ptot = float(np.sum(solver.Sx)) * dh**3
            finite = bool(np.all(np.isfinite(solver.D)))
            times.append(solver.t)
            Px_left_hist.append(Px_left)
            Px_right_hist.append(Px_right)
            Ptot_hist.append(Ptot)
            print(f"    step {step+1:3d}: t={solver.t:.3f}  "
                  f"P_x^L={Px_left:+.3e}  P_x^R={Px_right:+.3e}  "
                  f"P_x^tot={Ptot:+.3e}  finite={finite}")
    elapsed = time.time() - t_start

    finite = bool(np.all(np.isfinite(solver.D)))
    Px_left_final = Px_left_hist[-1]
    Px_right_final = Px_right_hist[-1]
    sym_violation = abs(Px_left_final + Px_right_final) / max(abs(Px_left_final), 1e-30)

    print(f"\n  evolution wall time: {elapsed:.1f}s")
    print(f"\n  final momentum diagnostics:")
    print(f"    P_x^left  = {Px_left_final:+.3e}  (expect POSITIVE, toward well at x=L/2)")
    print(f"    P_x^right = {Px_right_final:+.3e}  (expect NEGATIVE, toward well at x=L/2)")
    print(f"    total |P_x^L + P_x^R| / |P_x^L| = {sym_violation:.3e}  "
          f"(expect << 1 by symmetry)")

    pass_left  = Px_left_final > 0
    pass_right = Px_right_final < 0
    pass_sym   = sym_violation < 0.1
    pass_finite = finite

    print(f"\n  PASS criteria:")
    print(f"    Left half acquires +x momentum    : {'PASS' if pass_left else 'FAIL'}")
    print(f"    Right half acquires -x momentum   : {'PASS' if pass_right else 'FAIL'}")
    print(f"    Spatial symmetry preserved        : {'PASS' if pass_sym else 'FAIL'}")
    print(f"    No NaN                            : {'PASS' if pass_finite else 'FAIL'}")

    # Plots.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(times, Px_left_hist, 'o-', label='P_x^L (left half)')
    ax.plot(times, Px_right_hist, 's-', label='P_x^R (right half)')
    ax.plot(times, Ptot_hist, 'x--', label='P_x^total')
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xlabel('time t')
    ax.set_ylabel('integrated x-momentum')
    ax.set_title('Lapse-gradient acceleration: fluid falls into gravity well')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    Sx_slice = solver.Sx[:, :, N//2]
    vmax = float(np.max(np.abs(Sx_slice)))
    im = ax.imshow(Sx_slice.T, origin='lower', extent=[0, L, 0, L],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'S_x at z=L/2, t={solver.t:.3f}  (red: +x, blue: -x)')
    plt.colorbar(im, ax=ax, label='S_x')
    # Overlay the lapse contour
    contour = ax.contour(X[:, :, N//2].T, Y[:, :, N//2].T, alpha_field[:, :, N//2].T,
                          levels=8, colors='black', alpha=0.4, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=7)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), 'examples', 'out_hydro_curved.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out}")

    print()
    print("  PSFT interpretation:")
    print("  This is the first hydrodynamics simulation in the PSFT library")
    print("  with a non-flat spatial geometry coupling.  The Valencia-lite")
    print("  formulation correctly captures the Newtonian gravitational")
    print("  acceleration as a lapse-gradient source on the momentum")
    print("  equation -- the leading-order curved-spacetime effect on")
    print("  fluid motion.  In PSFT's framework, the same mechanism")
    print("  produces the gravitational attraction between solitonic")
    print("  matter configurations and ambient fluid (paper Theorem 12.1:")
    print("  inviscid limit recovers GR), and is the foundation for the")
    print("  fully-coupled BSSN + hydro + photonic + sigma^A roadmap.")


if __name__ == "__main__":
    main()
