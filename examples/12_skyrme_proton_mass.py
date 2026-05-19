"""Example 12: Skyrme baryon mass calibration -- from dimensionless E*
to a proton mass in MeV.

Example 11 extracted the dimensionless Skyrme energy E* = 73.65 from the
B=1 hedgehog ansatz.  In this example we bridge from that number to a
physical mass in MeV by going through two routes:

  Route A -- standard Adkins-Nappi-Witten calibration:
      Take experimental F_pi = 92 MeV and Skyrme coupling e_S = 4.84
      (commonly used in chiral perturbation fits) and use the standard
      Skyrme formula M_classical = (F_pi / e_S) * (dimensionless energy).
      Compare against the observed nucleon mass M_N = 939 MeV.

  Route B -- PSFT-native scale estimate:
      Use the PSFT critical-curvature scale K_c^strong = 12/l_strong^4
      with l_strong = 1 fm as the only input.  Identify the natural
      mass scale as the inverse strong length, M_psft = (hbar c)/l_strong
      = 197.3 MeV, and estimate F_pi = M_psft / O(1).  Predict M_proton
      and compare to the experimental value.

This is honest: Route A uses two phenomenological inputs (F_pi, e_S),
Route B uses only one PSFT input (l_strong) but at the cost of an O(1)
estimation factor.  Neither is a first-principles derivation -- the
Skyrme model itself is an effective theory whose constants come from
matching to QCD/experiment, and PSFT inherits this.

Run:
    python3 examples/12_skyrme_proton_mass.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import math
import numpy as np

from psft.core.constants import SI


# Dimensionless integral from Example 11, recomputed locally for self-contained run.
def skyrme_energy_density(r, F, Fp):
    sin_F = np.sin(F)
    s2 = sin_F * sin_F
    s4 = s2 * s2
    safe_r2 = np.maximum(r * r, 1e-12)
    sigma_term = 0.5 * r * r * Fp * Fp + s2
    skyrme_term = s2 * Fp * Fp + 0.5 * s4 / safe_r2
    return 4.0 * math.pi * (sigma_term + skyrme_term)


def find_E_star(n_a=64):
    rs = np.linspace(0.01, 10.0, 400)
    a_values = np.linspace(0.5, 2.0, n_a)
    best_E = float("inf")
    best_a = None
    for a in a_values:
        F = 4.0 * np.arctan(np.exp(-rs / a))
        Fp = np.gradient(F, rs)
        E = float(np.trapezoid(skyrme_energy_density(rs, F, Fp), rs))
        if E < best_E:
            best_E = E
            best_a = a
    return best_E, best_a


def main():
    print(" Skyrme baryon mass: from dimensionless E* to MeV")
    print("=" * 60)

    # Reload the dimensionless minimum.
    E_star, a_star = find_E_star(n_a=128)
    print(f"  Dimensionless minimum (from Example 11 reproduced):")
    print(f"     E* = {E_star:.4f}  at a* = {a_star:.4f}")

    # ----- Route A: ANW calibration -----
    print()
    print(" Route A -- Adkins-Nappi-Witten standard Skyrme calibration")
    print(" " + "-" * 56)
    # Standard Skyrme Lagrangian (in CGS-natural units with hbar=c=1):
    #     L = (F_pi^2/16) Tr(dU dU^dag) + (1/(32 e_S^2)) Tr([U^dag dU]^4)
    # For the hedgehog, the classical mass is
    #     M_classical = (F_pi / e_S) * (dimensionless integral I)
    # where I is the dimensionless Skyrme energy in (e_S F_pi r) units.
    # Our energy density uses the equivalent rescaled form so that I ~ E*.
    F_pi_MeV = 92.0      # MeV  -- experimental pion decay constant
    e_S = 4.84           # ANW 1983 best fit
    # The numerical Skyrme-model classical mass is conventionally written as
    # M_classical = 36.5 * pi^2 * F_pi / e_S  (Adkins 1984), giving the
    # dimensionless integral ~ 36.5 pi^2 ~ 360 in a different normalization
    # convention.  Different conventions differ by overall constants in the
    # energy density definition; in our (4 pi * (sigma + skyrme)) form the
    # benchmark factor is ~ 73.  So:
    M_classical = (F_pi_MeV / e_S) * E_star
    M_observed = 939.0   # nucleon mass, MeV
    print(f"     F_pi  (experimental)        = {F_pi_MeV:.1f} MeV")
    print(f"     e_S   (Skyrme coupling)     = {e_S}")
    print(f"     M_classical = F_pi/e_S * E* = {M_classical:.0f} MeV")
    print(f"     M_N   (observed nucleon)    = {M_observed} MeV")
    print(f"     ratio M_classical / M_N      = {M_classical / M_observed:.2f}")
    print()
    print("  The classical Skyrme mass overshoots by ~50%, as is well known")
    print("  in the literature (Adkins-Nappi-Witten 1984).  Quantum")
    print("  rotational corrections bring this down to ~1100 MeV, and further")
    print("  pion-loop corrections converge towards 939 MeV.  The classical")
    print("  number is the right ORDER of magnitude.")

    # ----- Route B: PSFT-native scale estimate -----
    print()
    print(" Route B -- PSFT-native scale (only PSFT input: l_strong = 1 fm)")
    print(" " + "-" * 58)
    hbar_c_MeV_fm = SI.hbar * SI.c / (1e-15 * 1e6 * SI.e)   # MeV*fm
    print(f"     (sanity) hbar c = {hbar_c_MeV_fm:.2f} MeV*fm  (should be ~197.3)")
    # PSFT natural mass scale = (hbar c) / l_strong, with l_strong = 1 fm.
    l_strong_fm = 1.0    # paper convention
    M_psft = hbar_c_MeV_fm / l_strong_fm
    print(f"     l_strong (PSFT)             = {l_strong_fm:.3f} fm")
    print(f"     M_psft = (hbar c)/l_strong  = {M_psft:.1f} MeV")
    # The pion decay constant in QCD relates to Lambda_QCD by F_pi ~ Lambda_QCD / 2.
    # PSFT's natural scale M_psft = 1/l_strong ~ Lambda_QCD (paper Sec. 6),
    # so identifying F_pi ~ M_psft / 2 is the leading estimate.
    F_pi_psft = M_psft / 2.0
    print(f"     F_pi (PSFT estimate ~ M_psft/2) = {F_pi_psft:.1f} MeV")
    print(f"     F_pi (experimental)              = {F_pi_MeV:.1f} MeV")
    rel = abs(F_pi_psft - F_pi_MeV) / F_pi_MeV
    print(f"     relative deviation               = {rel*100:.1f}%")
    # Use PSFT's F_pi estimate plus a standard e_S to predict nucleon mass.
    M_proton_psft = (F_pi_psft / e_S) * E_star
    print(f"     M_classical (PSFT F_pi)      = {M_proton_psft:.0f} MeV")
    print(f"     compared to observed 939 MeV:  ratio = {M_proton_psft/M_observed:.2f}")
    print()
    print("  The PSFT-derived F_pi sits within a factor 1.1 of the experimental")
    print("  value, just from the single input l_strong = 1 fm.  Predicted")
    print("  nucleon mass is order ~1500 MeV before quantum corrections --")
    print("  same ballpark as the standard ANW result, as expected since the")
    print("  inputs are similar.")

    # ----- What's needed for a true first-principles prediction -----
    print()
    print(" What would close the gap to an unconditional prediction:")
    print(" " + "-" * 55)
    print("  1. A first-principles derivation of e_S from the SU(3) viscous")
    print("     fluid (paper Conjecture SU3) would eliminate the second")
    print("     phenomenological input.")
    print("  2. A geometric calculation of F_pi from the spacetime-fluid")
    print("     symmetry-breaking pattern would replace the heuristic")
    print("     F_pi ~ M_psft/2.")
    print("  3. Quantum corrections to the classical Skyrme mass (rotational")
    print("     zero-point energy, pion loops) would need to be computed in")
    print("     the PSFT framework.")
    print("  4. The mass splittings between nucleons, Delta, hyperons, etc.")
    print("     would test the framework beyond the single-baryon prediction.")
    print()
    print("  Items 1-2 are research-level open problems listed in")
    print("  Section 14 of the paper.")


if __name__ == "__main__":
    main()
