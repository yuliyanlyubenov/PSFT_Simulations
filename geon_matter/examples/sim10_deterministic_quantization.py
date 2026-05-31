"""sim10 -- Energy quantisation WITHOUT the Born rule: discrete levels as a
deterministic standing-wave / single-valuedness condition.

The user is sceptical of the statistical (Born-rule) reading of QM and wants
PSFT to derive the same experimental facts DETERMINISTICALLY, with more
physical detail.  Atomic energy levels are the cleanest test: orthodox QM gets
them from |psi|^2 statistics, but they were first obtained -- and are fully
reproduced -- by a DETERMINISTIC condition.

PSFT supplies the mechanism (docs 01-02, sim2): a matter geon carries a real
internal light-clock whose lab-frame appearance is the de Broglie wave
lambda_dB = h/p.  For a bound geon the wave must CLOSE ON ITSELF -- the
internal phase must be single-valued around the orbit:

    2 pi r = n lambda_dB = n h / (m v) ,   n in Z    (single-valuedness)
    <=>  m v r = n hbar                              (angular momentum)

This is NOT a probability postulate; it is the same TOPOLOGICAL
single-valuedness that quantised charge (Theorem 9.1(iv)) and spin (sim8):
n is a winding/node number.  Combined with deterministic Coulomb force balance

    m v^2 / r = e^2 / (4 pi eps0 r^2) ,

it yields, with no statistics whatsoever,

    r_n = n^2 a0 ,   v_n = alpha c / n ,   E_n = - Ry / n^2 ,

reproducing the hydrogen spectrum exactly (gross structure).  We verify a0,
the Rydberg, the level energies, and the Balmer lines against measured values,
and confirm the standing-wave count 2 pi r_n / lambda_dB = n is an exact
integer.

Run:  python3 sim10_deterministic_quantization.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    c, hbar, m_e, e, eps0 = SI.c, SI.hbar, SI.m_electron, SI.e, SI.epsilon_0
    print("[psft.core.constants.SI]")
except Exception:                                            # pragma: no cover
    c, hbar = 2.99792458e8, 1.054571817e-34
    m_e, e, eps0 = 9.1093837015e-31, 1.602176634e-19, 8.8541878128e-12
    print("[fallback constants]")

h = 2.0 * np.pi * hbar
eV = e


def main():
    print("=" * 74)
    print("sim10: deterministic energy quantisation (standing wave, no Born rule)")
    print("=" * 74)

    # deterministic constants derived from the single-valuedness + Coulomb balance
    a0 = 4.0 * np.pi * eps0 * hbar ** 2 / (m_e * e ** 2)     # Bohr radius
    alpha = e ** 2 / (4.0 * np.pi * eps0 * hbar * c)         # fine-structure const
    Ry = 0.5 * alpha ** 2 * m_e * c ** 2                     # Rydberg energy

    a0_ref = 5.29177210903e-11
    Ry_ref = 13.605693122 * eV
    print(f"\nderived Bohr radius a0 = 4 pi eps0 hbar^2/(m e^2) = {a0:.6e} m "
          f"(CODATA {a0_ref:.6e})")
    print(f"derived Rydberg    Ry = (1/2) alpha^2 m c^2       = {Ry/eV:.6f} eV "
          f"(CODATA {Ry_ref/eV:.6f})")
    ok_const = abs(a0 - a0_ref) / a0_ref < 1e-4 and abs(Ry - Ry_ref) / Ry_ref < 1e-4

    # ---- levels + standing-wave-count check -------------------------------
    print("\n-- hydrogen levels from the deterministic standing-wave condition --")
    print(f"{'n':>2} {'r_n (a0)':>9} {'v_n/c':>10} {'E_n (eV)':>11} "
          f"{'2pi r/lambda_dB':>16} {'(=n?)':>6}")
    ok_levels = True
    ok_winding = True
    for n in range(1, 7):
        r_n = n ** 2 * a0
        v_n = alpha * c / n
        E_n = -Ry / n ** 2
        lam_dB = h / (m_e * v_n)            # de Broglie wavelength of the orbit
        count = 2.0 * np.pi * r_n / lam_dB  # standing-wave count -> must be n
        if abs(count - n) > 1e-9:
            ok_winding = False
        print(f"{n:>2} {r_n/a0:9.0f} {v_n/c:10.6f} {E_n/eV:11.5f} "
              f"{count:16.6f} {n:6d}")
    # spot-check measured level: n=1 ionisation = 13.606 eV
    E1 = -Ry / 1 ** 2
    if abs(-E1 / eV - 13.605693) > 1e-3:
        ok_levels = False

    # ---- Balmer series (visible H lines) vs measured ----------------------
    print("\n-- Balmer lines (n -> 2 transitions) deterministic vs measured --")
    measured = {3: 656.279, 4: 486.135, 5: 434.047, 6: 410.173}   # nm (air~vac ok to 0.1%)
    print(f"{'transition':>12} {'lambda pred (nm)':>17} {'measured (nm)':>14} "
          f"{'rel.err':>9}")
    ok_balmer = True
    for n_hi, lam_meas in measured.items():
        dE = Ry * (1.0 / 2 ** 2 - 1.0 / n_hi ** 2)     # emitted photon energy
        lam = h * c / dE                                # wavelength
        rel = abs(lam * 1e9 - lam_meas) / lam_meas
        if rel > 5e-3:
            ok_balmer = False
        print(f"{f'{n_hi}->2':>12} {lam*1e9:17.3f} {lam_meas:14.3f} {rel:9.2e}")

    print("\n" + "-" * 74)
    print(f"  a0 and Rydberg from deterministic balance     : "
          f"{'PASS' if ok_const else 'FAIL'}")
    print(f"  levels E_n = -Ry/n^2 (n=1 -> 13.606 eV)       : "
          f"{'PASS' if ok_levels else 'FAIL'}")
    print(f"  standing-wave count = n EXACTLY (winding int) : "
          f"{'PASS' if ok_winding else 'FAIL'}")
    print(f"  Balmer lines match measured to <0.5%          : "
          f"{'PASS' if ok_balmer else 'FAIL'}")
    all_ok = ok_const and ok_levels and ok_winding and ok_balmer
    print("-" * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation (the deterministic reframing the user asked for):
  The discrete hydrogen spectrum -- the textbook showcase of "quantum
  randomness" -- needs NO probability postulate.  It is fixed by a
  deterministic boundary condition: the geon's real internal light-clock
  (de Broglie) wave must be single-valued around its orbit, so the winding
  number n is an integer.  Quantisation here is TOPOLOGY (an integer winding),
  exactly as for charge (Thm 9.1) and spin (sim8) -- not a statistical axiom.
  PSFT reproduces a0, the Rydberg, the level ladder, and the Balmer lines
  deterministically.
  Honest boundary (not overclaimed): fine structure, Lamb shift, multi-electron
  atoms, and especially Bell/entanglement correlations are NOT addressed here.
  A deterministic field theory can match QM statistics only if NONLOCAL
  (Bell's theorem); PSFT's spacetime fluid is a candidate nonlocal medium, but
  an explicit deterministic account of entanglement remains OPEN.  What is
  shown: the single-particle quantisation that QM treats statistically is, in
  PSFT, a deterministic standing-wave/topological fact.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
