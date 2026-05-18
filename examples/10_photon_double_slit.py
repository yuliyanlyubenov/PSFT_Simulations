"""Example 10: Classical photon double-slit -- PSFT's Born-rule-free
interpretation of single-photon interference.

PSFT Postulate 1 takes the photon as primitive (a quantum of the photonic
stress field F_{ab}), so the electromagnetic wave equation -- derived as
Theorem 9.1 -- is the FUNDAMENTAL equation for light, not a classical
approximation to a quantum theory.  Theorem 9.1 + the inviscid U(1) sector
(K_c^EM = infinity) means light propagates freely as a wave.

The double-slit detection statistics emerge classically: the field
amplitude |A(x)| at the screen gives the intensity I(x) = |A(x)|^2, which
is the local energy density of the EM field.  Detection probability is
proportional to energy density.  No Born-rule postulate is required.

This simulation propagates a scalar wave through a double-slit barrier and
shows that the screen intensity matches the standard fringe pattern.

Run:
    python3 examples/10_photon_double_slit.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    print(" PSFT single-photon interference (classical wave, no Born rule)")
    print("=" * 65)

    # Physical setup -- Fraunhofer limit of a 2-slit experiment.
    # Use natural units; results scale by wavelength.
    wavelength = 1.0
    k = 2 * math.pi / wavelength     # wavenumber
    slit_separation = 4.0 * wavelength   # d
    slit_width = 0.5 * wavelength        # a
    screen_distance = 50.0 * wavelength  # L
    screen_width = 30.0 * wavelength
    n_pts = 400

    print(f"  wavelength lambda  = {wavelength}")
    print(f"  slit separation d  = {slit_separation}")
    print(f"  slit width a       = {slit_width}")
    print(f"  screen distance L  = {screen_distance}")
    print()

    # Fraunhofer two-slit pattern derived from classical wave superposition:
    # I(y) = I_0 * (sin(beta) / beta)^2 * cos(delta/2)^2
    # where  beta = (pi a / lambda) * (y / L)
    #        delta = (2 pi d / lambda) * (y / L)
    ys = np.linspace(-screen_width / 2, screen_width / 2, n_pts)
    theta = ys / screen_distance
    beta = (math.pi * slit_width / wavelength) * theta
    delta = (2 * math.pi * slit_separation / wavelength) * theta
    # sinc envelope: handle beta=0 singularity.
    envelope = np.where(np.abs(beta) < 1e-12, 1.0, np.sin(beta) / beta)
    intensity_classical = (envelope ** 2) * (np.cos(delta / 2) ** 2)

    # PSFT prediction: detection probability proportional to local energy
    # density u = (1/8 pi) (|E|^2 + |B|^2), which for a propagating field is
    # proportional to |A|^2 -- the same as the classical wave intensity.
    intensity_psft = intensity_classical  # By construction.

    # Now do a small explicit grid simulation: propagate a Gaussian wave
    # packet through a barrier with two slits and measure the screen
    # intensity by integrating |Phi(y, t)|^2 over time at the screen plane.
    Nx = 160
    Ny = 160
    Lx = 60.0 * wavelength
    Ly = 30.0 * wavelength
    dx = Lx / Nx
    dy = Ly / Ny
    xs = np.linspace(-Lx / 2, Lx / 2, Nx)
    yvals = np.linspace(-Ly / 2, Ly / 2, Ny)
    Xg, Yg = np.meshgrid(xs, yvals, indexing="ij")

    # Initial Gaussian wave packet centered at x = -20, moving in +x.
    sigma = 3.0 * wavelength
    Phi = np.exp(-((Xg + 20.0) ** 2 + Yg ** 2) / (2 * sigma ** 2))
    Phi = Phi.astype(complex) * np.exp(1j * k * Xg)
    Pi = np.zeros_like(Phi)   # time derivative

    # Barrier at x = 0 with two slits of width 0.5 lambda separated by 4 lambda.
    barrier_mask = (np.abs(Xg) < 0.5 * dx * 1.5).astype(float)  # narrow wall
    slit_open = (
        (np.abs(Yg - slit_separation / 2) < slit_width / 2)
        | (np.abs(Yg + slit_separation / 2) < slit_width / 2)
    )
    barrier_mask = barrier_mask * (~slit_open).astype(float)
    # Multiply Phi by (1 - mask) to zero out the barrier region every step.

    # Wave equation: d^2 Phi / dt^2 = c^2 * laplacian(Phi).  Use c = 1.
    dt = 0.25 * min(dx, dy)       # CFL-style
    n_steps = int(80 * wavelength / dt)

    # Screen position: x close to the right edge.
    screen_x_idx = int(Nx * 0.9)
    screen_intensity_accum = np.zeros(Ny)

    for step in range(n_steps):
        lap = (
            (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) - 2 * Phi) / dx ** 2
            + (np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1) - 2 * Phi) / dy ** 2
        )
        Pi = Pi + dt * lap
        Phi = Phi + dt * Pi
        # Apply barrier: zero the field on the wall (Dirichlet condition).
        Phi = Phi * (1.0 - barrier_mask)
        # Absorbing boundary: damp near the edges to suppress reflections.
        damp = 0.02
        Phi[:5, :]  *= (1 - damp)
        Phi[-5:, :] *= (1 - damp)
        Phi[:, :5]  *= (1 - damp)
        Phi[:, -5:] *= (1 - damp)
        # Accumulate energy density at the screen plane.
        if step > n_steps // 2:
            screen_intensity_accum += np.abs(Phi[screen_x_idx, :]) ** 2

    # Normalise.
    screen_intensity_accum /= screen_intensity_accum.max()
    intensity_classical_normed = intensity_classical / intensity_classical.max()

    # Plot.
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    ax = axes[0]
    ax.plot(ys / wavelength, intensity_classical_normed,
            label="Fraunhofer (analytic)", lw=2)
    # Resample the simulation to the analytic-y axis for visual comparison.
    sim_ys = yvals
    ax.plot(sim_ys / wavelength, screen_intensity_accum,
            label="Wave-eq simulation", lw=1, alpha=0.7)
    ax.set_xlabel("position on screen y / lambda")
    ax.set_ylabel("relative intensity (= local energy density)")
    ax.set_title("Single-photon two-slit interference (PSFT: classical wave)")
    ax.set_xlim(-screen_width / 2 / wavelength, screen_width / 2 / wavelength)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    ax2.imshow(np.abs(Phi).T,
               extent=[-Lx / 2 / wavelength, Lx / 2 / wavelength,
                       -Ly / 2 / wavelength, Ly / 2 / wavelength],
               origin="lower", cmap="viridis", aspect="auto")
    ax2.axhline(slit_separation / 2 / wavelength, color="red", lw=0.5, alpha=0.5)
    ax2.axhline(-slit_separation / 2 / wavelength, color="red", lw=0.5, alpha=0.5)
    ax2.set_xlabel("x / lambda")
    ax2.set_ylabel("y / lambda")
    ax2.set_title("|Phi(x, y)| at final time")

    out_path = os.path.join(os.path.dirname(HERE), "examples", "out_double_slit.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")
    print()
    print("  PSFT interpretation:")
    print("  The intensity I(y) = |Phi(y)|^2 IS the local EM energy density")
    print("  u(y) = (|E|^2 + |B|^2)/(8 pi).  By energy conservation, a single")
    print("  photon's detection probability at point y is proportional to u(y)")
    print("  -- no separate Born-rule postulate required (Section 13 of paper).")
    print("  Each photon arrives as one detection event; the fringe pattern")
    print("  emerges statistically over many events.")


if __name__ == "__main__":
    main()
