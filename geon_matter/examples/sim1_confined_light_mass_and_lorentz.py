"""sim1 -- Confined light -> rest mass + EXACT Lorentz behaviour.

Research question (Idea A):  In PSFT a matter soliton is a Wheeler-geon --
light trapped/orbiting around a concentrated-energy core (paper Sec 2,
Postulate 1; examples 30/31/32).  If the constituent light always moves at
c (the U(1) sector is exactly inviscid, Kc^EM = inf), what happens to the
orbiting photons and to the geon centre-of-mass when the whole geon moves at
speed v?

This script tests, numerically, the claim that PSFT does NOT modify the
Lorentz transformations -- it DERIVES them.  We build a geon out of light in
its rest frame, apply an ordinary Lorentz boost, and measure the emergent
particle kinematics:

  (1) the orbiting photon's speed in the lab frame is EXACTLY c for every
      boost (invariance of c -> orbiting light never goes superluminal even
      when the geon centre moves at 0.999 c);
  (2) the geon's internal "light clock" period dilates by exactly gamma
      (time dilation from first principles);
  (3) the geon's spatial extent contracts by 1/gamma along the boost
      (length contraction of the soliton);
  (4) two confined photons with zero net momentum behave as a particle of
      rest mass m = E0/c^2, with E = gamma m c^2, p = gamma m v, and the
      invariant  E^2 - (pc)^2 = (m c^2)^2  holding to machine precision.

All four are *consequences* of the constituent moving at c under the
standard boost -- no Lorentz transformation is modified.  This is the
conservative backbone of the research: PSFT geons REINFORCE special
relativity for matter rather than altering it.

Run:  python3 sim1_confined_light_mass_and_lorentz.py
"""
import os
import sys
import numpy as np

# Reuse the existing PSFT constants if the simulation package is importable;
# otherwise fall back to identical SI values.  (We never modify the package.)
_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    c = SI.c
    hbar = SI.hbar
    m_e = SI.m_electron
    print(f"[using psft.core.constants.SI]  c = {c:.6e} m/s")
except Exception:                                            # pragma: no cover
    c = 2.99792458e8
    hbar = 1.054571817e-34
    m_e = 9.1093837015e-31
    print(f"[fallback constants]  c = {c:.6e} m/s")


def boost_x(four_vec, beta):
    """Active Lorentz boost of a 4-vector (ct, x, y, z) to a frame in which
    the original rest frame moves at +beta along x."""
    g = 1.0 / np.sqrt(1.0 - beta * beta)
    ct, x, y, z = four_vec
    return np.array([g * (ct + beta * x),
                     g * (x + beta * ct),
                     y, z])


def orbit_worldline_rest(R, n=20000, n_turns=1.0):
    """Rest-frame null worldline of a single photon circulating in the x-y
    plane at radius R and speed c.  Returns 4-positions (ct', x', y', z')."""
    omega0 = c / R                       # angular speed s.t. omega0 * R = c
    T0 = 2.0 * np.pi / omega0            # rest-frame orbital period
    tprime = np.linspace(0.0, n_turns * T0, n)
    ct = c * tprime
    x = R * np.cos(omega0 * tprime)
    y = R * np.sin(omega0 * tprime)
    z = np.zeros_like(tprime)
    return np.vstack([ct, x, y, z]), T0


def speed_along_worldline(W4):
    """Coordinate 3-speed |dr/dt| sampled along a 4-worldline W4 (4 x N),
    using central differences (2nd-order accurate)."""
    ct, x, y, z = W4
    t = ct / c
    dt = t[2:] - t[:-2]
    dx = x[2:] - x[:-2]; dy = y[2:] - y[:-2]; dz = z[2:] - z[:-2]
    return np.sqrt(dx * dx + dy * dy + dz * dz) / dt


def ring_xextent_lab(R, beta, nphi=2000):
    """Lab-frame x-extent of a RING of light (radius R in its rest frame),
    measured as a snapshot at a single lab time.  Returns the ratio of the
    lab x-extent to the rest-frame diameter 2R; SR predicts 1/gamma."""
    g = 1.0 / np.sqrt(1.0 - beta * beta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    # To snapshot at lab time t = 0 we need rest-frame events with
    # ct' = -beta * x' (relativity of simultaneity).  x' = R cos(phi).
    xprime = R * np.cos(phi)
    yprime = R * np.sin(phi)
    ctprime = -beta * xprime
    x_lab = g * (xprime + beta * ctprime)         # = R cos(phi)/gamma
    return (x_lab.max() - x_lab.min()) / (2.0 * R)


def main():
    print("=" * 72)
    print("sim1: confined light -> rest mass + exact Lorentz behaviour")
    print("=" * 72)

    # --- Geon geometry (use the electron reduced Compton radius as the
    #     orbital radius so numbers are physical; the conclusions are
    #     radius-independent). ----------------------------------------------
    R = hbar / (m_e * c)                 # reduced Compton wavelength ~3.86e-13 m
    W_rest, T0 = orbit_worldline_rest(R, n=40000, n_turns=1.0)
    print(f"\norbital radius R = lambdabar_C = {R:.4e} m")
    print(f"rest-frame orbital period T0 = {T0:.4e} s")
    print(f"rest-frame internal frequency f0 = {1/T0:.4e} Hz  "
          f"(Compton freq m_e c^2/h = {m_e*c*c/(2*np.pi*hbar):.4e} Hz)")

    betas = [0.1, 0.5, 0.9, 0.99, 0.999]
    print("\n" + "-" * 72)
    print(f"{'beta':>7} {'gamma':>10} | {'photon |v|/c':>14} | "
          f"{'T_lab/T0':>10} {'(=gamma?)':>9} | {'x-extent/R':>11} {'(=1/g?)':>8}")
    print("-" * 72)

    results = []
    for beta in betas:
        g = 1.0 / np.sqrt(1.0 - beta * beta)
        # Boost the whole rest-frame worldline column by column.
        W_lab = np.empty_like(W_rest)
        for i in range(W_rest.shape[1]):
            W_lab[:, i] = boost_x(W_rest[:, i], beta)

        # (1) orbiting photon speed in lab frame -- must equal c everywhere.
        v_lab = speed_along_worldline(W_lab)
        vmax = v_lab.max() / c
        vmin = v_lab.min() / c

        # (2) lab-frame period: time for the orbital phase to complete one
        #     turn = lab time between first and last sample (one full turn).
        t_lab = W_lab[0] / c
        T_lab = t_lab[-1] - t_lab[0]
        period_ratio = T_lab / T0

        # (3) length contraction: snapshot the full light-ring at one lab
        #     instant (correctly accounting for relativity of simultaneity).
        x_extent_ratio = ring_xextent_lab(R, beta)        # SR predicts 1/gamma

        results.append((beta, g, vmax, vmin, period_ratio, x_extent_ratio))
        print(f"{beta:7.3f} {g:10.4f} | "
              f"[{vmin:.6f},{vmax:.6f}] | "
              f"{period_ratio:10.5f} {g:9.5f} | "
              f"{x_extent_ratio:11.5f} {1/g:8.5f}")

    # ---- assertions -------------------------------------------------------
    print("\n" + "-" * 72)
    ok_c = all(abs(r[2] - 1.0) < 1e-3 and abs(r[3] - 1.0) < 1e-3 for r in results)
    ok_time = all(abs(r[4] - r[1]) / r[1] < 1e-4 for r in results)
    ok_len = all(abs(r[5] - 1.0 / r[1]) < 1e-4 for r in results)
    print(f"  (1) orbiting photon |v| = c to <1e-3 (FD error) for all boosts : "
          f"{'PASS' if ok_c else 'FAIL'}")
    print(f"  (2) internal light-clock period = gamma * T0         : "
          f"{'PASS' if ok_time else 'FAIL'}")
    print(f"  (3) orbital x-extent = (1/gamma) * rest extent       : "
          f"{'PASS' if ok_len else 'FAIL'}")

    # ---- energy-momentum of a 2-photon "box" geon -------------------------
    print("\n" + "=" * 72)
    print("two-photon box geon: emergent relativistic energy-momentum")
    print("=" * 72)
    # Rest frame: photon A along +x, photon B along -x, each energy eps.
    # Choose eps so that E0 = 2 eps = m_e c^2.
    E0 = m_e * c * c
    eps = 0.5 * E0
    print(f"rest energy E0 = 2*eps = m_e c^2 = {E0:.6e} J  "
          f"(=> rest mass m = E0/c^2 = {E0/c/c:.6e} kg = m_e)")
    print("\n" + "-" * 72)
    print(f"{'beta':>7} {'E/(m c^2)':>12} {'gamma':>10} | "
          f"{'p c/(m c^2)':>13} {'gamma*beta':>11} | {'invariant/(mc^2)^2':>18}")
    print("-" * 72)
    ok_em = True
    for beta in betas:
        g = 1.0 / np.sqrt(1.0 - beta * beta)
        # Doppler-shifted photon energies (4-momentum boost of a null vector).
        # photon A (+x):  (eps/c)(1, +1, 0, 0); photon B (-x): (eps/c)(1,-1,0,0)
        pA = (eps / c) * np.array([1.0, 1.0, 0.0, 0.0])
        pB = (eps / c) * np.array([1.0, -1.0, 0.0, 0.0])
        pA_lab = boost_x(pA, beta)       # same transform (E/c, px, py, pz)
        pB_lab = boost_x(pB, beta)
        P = pA_lab + pB_lab
        E_lab = P[0] * c
        px_lab = P[1]
        invariant = (E_lab / c) ** 2 - (px_lab ** 2)   # (mc)^2
        E_ratio = E_lab / E0
        p_ratio = px_lab * c / E0
        inv_ratio = invariant / (m_e * c) ** 2
        if not (abs(E_ratio - g) < 1e-9 and abs(p_ratio - g * beta) < 1e-9
                and abs(inv_ratio - 1.0) < 1e-9):
            ok_em = False
        print(f"{beta:7.3f} {E_ratio:12.6f} {g:10.6f} | "
              f"{p_ratio:13.6f} {g*beta:11.6f} | {inv_ratio:18.12f}")
    print("-" * 72)
    print(f"  (4) E=gamma m c^2, p=gamma m v, E^2-(pc)^2=(mc^2)^2  : "
          f"{'PASS' if ok_em else 'FAIL'}")

    all_ok = ok_c and ok_time and ok_len and ok_em
    print("\n" + "=" * 72)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAIL'}")
    print("=" * 72)
    print("""
Interpretation:
  Every relativistic feature of a massive particle -- rest mass, time
  dilation, length contraction, gamma-scaled energy & momentum, and the
  mass-shell invariant -- falls out of confining light that always moves at
  c, using ONLY the ordinary Lorentz boost.  In PSFT the U(1) photonic
  sector is exactly inviscid (Kc^EM = inf), so the constituent light is
  forced to move at c in every frame; therefore the geon's external
  kinematics are EXACTLY special-relativistic.  PSFT does not modify the
  Lorentz transformations for matter -- it explains why they hold.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
