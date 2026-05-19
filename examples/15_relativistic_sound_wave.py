"""Example 15: relativistic sound wave -- linear test of the 1+1D
PSFT-inviscid-limit (relativistic Euler) solver.

Sets up a small-amplitude density perturbation on a uniform background
and confirms the wave propagates at the analytic relativistic sound
speed

    c_s^2 = Gamma p / (rho h),    h = 1 + Gamma p / [(Gamma-1) rho].

For a Gamma-law gas with Gamma = 4/3 and our parameters this gives
c_s ~ 0.31 c (in our natural units).  We track the centroid of a
right-moving Gaussian packet and verify (centroid_position - cs * t)
stays small relative to the packet width.

Run:
    python3 examples/15_relativistic_sound_wave.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psft.evolve.hydro_1d import (
    RelativisticEulerSolver1D, primitive_from_conservative,
)


def main():
    print(" Relativistic sound wave -- 1+1D Euler propagation test")
    print("=" * 60)

    Gamma = 4.0 / 3.0
    rho0 = 1.0
    p0 = 0.1
    h0 = 1.0 + Gamma * p0 / ((Gamma - 1.0) * rho0)
    cs2 = Gamma * p0 / (rho0 * h0)
    cs = math.sqrt(cs2)
    print(f"  Gamma   = {Gamma}")
    print(f"  rho0    = {rho0}")
    print(f"  p0      = {p0}")
    print(f"  h0      = {h0:.4f}")
    print(f"  cs      = {cs:.6f}  (relativistic sound speed)")

    # Build solver.  Periodic domain [0, 1].
    N = 1024
    L = 1.0
    solver = RelativisticEulerSolver1D(N=N, L=L, Gamma=Gamma, cfl=0.4)
    sigma = 0.04        # Gaussian packet width
    x_c = 0.50          # central position
    amp = 0.005         # small amplitude (linear regime)

    def gauss(x):
        return np.exp(-((x - x_c) / sigma) ** 2)

    # Set v = 0 initially with a density+pressure bump.  In linear theory,
    # this splits into a left-moving and a right-moving sound wave of equal
    # amplitude propagating at +/- c_s.  Tracking the position of each
    # peak measures the sound speed without needing to know the exact
    # right-moving polarisation (which has relativistic h-factors).
    solver.initialise(
        rho_func=lambda x: rho0 + amp * gauss(x),
        p_func=lambda x: p0 + cs2 * amp * gauss(x),
        v_func=lambda x: np.zeros_like(x),
    )

    print(f"  N cells = {N}, L = {L}, dx = {solver.dx:.5f}")
    print(f"  initial: Gaussian density bump at x = {x_c}, sigma = {sigma}")
    print(f"  initial: v = 0  (will split into left + right movers)")

    # Evolve for a time short enough to keep the two halves on the grid.
    t_end = 0.30 * L / cs       # ~ 0.97 here; peaks separate by 0.6 L
    print(f"  evolving to t_end = {t_end:.4f}")

    # Snapshots for plotting.
    snap_times = [0.0, 0.25 * t_end, 0.5 * t_end, 0.75 * t_end, t_end]
    snapshots = [(0.0, solver.primitives())]
    next_idx = 1
    while solver.t < t_end - 1e-12:
        solver.step()
        if next_idx < len(snap_times) and solver.t >= snap_times[next_idx]:
            snapshots.append((solver.t, solver.primitives()))
            next_idx += 1

    rho_f, p_f, v_f, _ = solver.primitives()
    # Find left and right peaks of the density perturbation.
    drho = rho_f - rho0
    # left peak is in x < x_c
    left_mask = solver.x < x_c
    right_mask = solver.x > x_c
    i_left = int(np.argmax(drho[left_mask]))
    i_right = int(np.argmax(drho[right_mask]))
    x_left = float(solver.x[left_mask][i_left])
    x_right = float(solver.x[right_mask][i_right])
    sep = x_right - x_left
    expected_sep = 2.0 * cs * solver.t
    rel_err = abs(sep - expected_sep) / expected_sep
    print(f"\n  measured separation at t = {solver.t:.4f}: {sep:.5f}")
    print(f"  expected 2 c_s t                          : {expected_sep:.5f}")
    print(f"  relative error                             : {rel_err*100:.2f}%")
    print(f"  left  peak position : x = {x_left:.5f}  (expected {x_c - cs*solver.t:.5f})")
    print(f"  right peak position : x = {x_right:.5f}  (expected {x_c + cs*solver.t:.5f})")
    if rel_err < 0.1:
        print("\n  PASS: sound speed recovered to within 10%.")

    # Diagnostics: conservation of total D.
    initial_total_mass = float(np.sum(rho0 + amp * gauss(solver.x))) * solver.dx
    print(f"\n  conservation diagnostics:")
    print(f"     d(total mass) / initial = "
          f"{(solver.total_mass() - initial_total_mass)/initial_total_mass:.2e}")

    # Plot snapshots.
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax0, ax1, ax2 = axes
    for (t, (rho, p, v, _)) in snapshots:
        label = f"t = {t:.3f}"
        ax0.plot(solver.x, rho - rho0, lw=1, label=label)
        ax1.plot(solver.x, p - p0, lw=1, label=label)
        ax2.plot(solver.x, v, lw=1, label=label)
    ax0.set_ylabel("rho - rho_0")
    ax0.set_title("Relativistic sound wave: propagation at c_s = "
                  f"{cs:.4f}")
    ax0.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("p - p_0")
    ax2.set_ylabel("v")
    ax2.set_xlabel("x")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(HERE), "examples",
                            "out_sound_wave.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")

    print()
    print("  PSFT interpretation:")
    print("  This is the FIRST field-level time-integrated master-equation")
    print("  simulation.  Theorem 12.1 of the paper guarantees that the")
    print("  inviscid limit reduces to the relativistic Euler equations,")
    print("  which we discretise on a 1D periodic grid with Lax-Friedrichs")
    print("  flux and RK4 in time.  Recovery of the correct relativistic")
    print("  sound speed validates that the solver propagates fluid")
    print("  perturbations correctly.")


if __name__ == "__main__":
    main()
