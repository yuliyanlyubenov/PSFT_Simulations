"""sim2 -- de Broglie matter waves from the geon's internal light-clock.

Research question (Idea B):  A PSFT geon has an internal oscillation -- the
orbiting/breathing light -- whose rest-frame frequency is the Compton
frequency  omega0 = m c^2 / hbar.  In the rest frame this oscillation is
synchronous everywhere ("the whole geon breathes in phase").  What does a lab
observer, for whom the geon moves at v, see?

We construct the rest-frame internal phase  Phi(t') = omega0 t'  as a field
that is constant over each rest-frame simultaneity slice, then express it in
lab coordinates via the Lorentz transformation t' = gamma (t - v x / c^2).
The claim to test: the tilt of simultaneity converts the in-phase internal
breathing into a TRAVELLING spatial phase -- the de Broglie matter wave --
with

    omega_lab = gamma omega0 = E / hbar           (total energy)
    k_lab     = gamma omega0 v / c^2 = p / hbar    (=> lambda_dB = h / p)
    v_phase   = omega/k = c^2 / v   (> c)
    v_group   = d omega / d k = v   (= the particle velocity)

This is de Broglie's 1924 "harmony of phases", here given a concrete PSFT
mechanism: the matter wave IS the moving-frame appearance of the geon's
internal light clock.  We verify k_lab = p/hbar numerically against the
relativistic momentum for a real electron across a range of speeds, and
confirm the group velocity equals v and the phase velocity equals c^2/v.

Run:  python3 sim2_de_broglie_internal_clock.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    c, hbar, m_e = SI.c, SI.hbar, SI.m_electron
    h = 2.0 * np.pi * hbar
    print(f"[using psft.core.constants.SI]  c = {c:.6e} m/s, m_e = {m_e:.6e} kg")
except Exception:                                            # pragma: no cover
    c, hbar = 2.99792458e8, 1.054571817e-34
    m_e = 9.1093837015e-31
    h = 2.0 * np.pi * hbar
    print("[fallback constants]")


def lab_phase_field(omega0, beta, t, x):
    """Internal phase Phi = omega0 * t' expressed in lab coords (t, x)."""
    g = 1.0 / np.sqrt(1.0 - beta * beta)
    return omega0 * g * (t - beta * x / c)        # t' = gamma(t - v x / c^2)


def extract_k_omega(omega0, beta):
    """Read off (omega_lab, k_lab) of the lab-frame phase field by finite
    differencing Phi(t, x) -- the genuinely measured wave parameters."""
    # Sample a small (t, x) patch and fit the linear phase.
    x = np.linspace(0.0, 1e-12, 5)
    t = np.linspace(0.0, 1e-21, 5)
    # d Phi / d t  at fixed x  -> omega ;  -d Phi / d x at fixed t -> k
    dPhi_dt = (lab_phase_field(omega0, beta, t[1], 0.0)
               - lab_phase_field(omega0, beta, t[0], 0.0)) / (t[1] - t[0])
    dPhi_dx = (lab_phase_field(omega0, beta, 0.0, x[1])
               - lab_phase_field(omega0, beta, 0.0, x[0])) / (x[1] - x[0])
    omega_lab = dPhi_dt
    k_lab = -dPhi_dx                              # plane wave ~ exp[i(k x - w t)]
    return omega_lab, k_lab


def main():
    print("=" * 74)
    print("sim2: de Broglie matter waves from the geon internal light-clock")
    print("=" * 74)

    omega0 = m_e * c * c / hbar                   # Compton angular frequency
    print(f"\nelectron Compton angular frequency omega0 = m c^2/hbar = "
          f"{omega0:.6e} rad/s")
    print(f"(rest-frame internal period = {2*np.pi/omega0:.4e} s)\n")

    betas = [0.01, 0.05, 0.1, 0.3, 0.6, 0.9]
    print("-" * 74)
    print(f"{'beta':>6} {'v (m/s)':>12} | {'lambda_dB sim':>15} "
          f"{'h/p exact':>14} {'rel.err':>9} | {'v_phase/c':>10} {'v_grp/c':>8}")
    print("-" * 74)

    ok = True
    rows = []
    for beta in betas:
        g = 1.0 / np.sqrt(1.0 - beta * beta)
        v = beta * c
        p = g * m_e * v                           # relativistic momentum
        lam_exact = h / p                         # de Broglie wavelength

        omega_lab, k_lab = extract_k_omega(omega0, beta)
        lam_sim = 2.0 * np.pi / k_lab
        rel_err = abs(lam_sim - lam_exact) / lam_exact

        v_phase = omega_lab / k_lab               # should be c^2 / v
        # group velocity: centred numerical d omega / d k about beta.
        db = 1e-5
        w1, k1 = extract_k_omega(omega0, beta - db)
        w2, k2 = extract_k_omega(omega0, beta + db)
        v_group = (w2 - w1) / (k2 - k1)

        if not (rel_err < 1e-9
                and abs(v_phase - c * c / v) / (c * c / v) < 1e-9
                and abs(v_group - v) / v < 1e-6):
            ok = False
        rows.append((beta, v, lam_sim, lam_exact, rel_err, v_phase / c, v_group / c))
        print(f"{beta:6.2f} {v:12.4e} | {lam_sim:15.6e} {lam_exact:14.6e} "
              f"{rel_err:9.1e} | {v_phase/c:10.4f} {v_group/c:8.5f}")

    print("-" * 74)
    # Sanity anchor: 100 eV electron (typical LEED / Davisson-Germer energy).
    E_kin = 100.0 * 1.602176634e-19               # 100 eV in joules
    # non-relativistic p = sqrt(2 m E):
    p_nr = np.sqrt(2.0 * m_e * E_kin)
    lam_100eV = h / p_nr
    print(f"\nanchor: 100 eV electron -> lambda_dB = {lam_100eV:.4e} m "
          f"(~{lam_100eV*1e12:.1f} pm; Davisson-Germer used Ni d~215 pm)")
    print(f"        (electron-diffraction regime; matches textbook ~123 pm "
          f"at 100 eV)")

    print("\n" + "=" * 74)
    print(f"  k_lab = p/hbar  (lambda = h/p)             : "
          f"{'PASS' if all(r[4] < 1e-9 for r in rows) else 'FAIL'}")
    print(f"  v_phase = c^2/v (>c, non-signalling)       : "
          f"{'PASS' if ok else 'CHECK'}")
    print(f"  v_group = v (matter wave tracks particle)  : "
          f"{'PASS' if ok else 'CHECK'}")
    print(f"\nOVERALL: {'ALL PASS' if ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation:
  The de Broglie wavelength lambda = h/p is not an independent quantum
  postulate here: it is the lab-frame shadow of the geon's rest-frame
  internal light-clock, produced purely by the Lorentz tilt of simultaneity.
  Wave-particle duality of MATTER thus has the same origin PSFT already
  gives to the photon (Sec 13, item 8): an extended internal oscillation
  whose phase pattern is the wave, and whose localised energy is the
  particle.  The construction uses the ordinary Lorentz transformation --
  again, SR is derived, not modified.""")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
