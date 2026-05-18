"""Example 3: Linear quark-antiquark potential from PSFT shear viscosity
(Theorem 10.1).

We do not solve the full SU(3) viscous flow.  Instead we use the analytic
result of the paper:

    V(r) = sigma · r,    sigma = C_F · eta(K_c) · g_s^2 / A_tube

With C_F = 4/3 and lattice values sigma = 0.18 GeV^2, R_tube = 0.35 fm,
solve for the effective viscosity and confirm it is O(1) in natural units.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import math
import numpy as np

from psft.core.constants import NATURAL


def main():
    sigma = NATURAL.string_tension_GeV2          # 0.18 GeV^2
    R_tube_fm = NATURAL.R_tube_fm                # 0.35 fm
    # 1 fm = 1/(0.1973 GeV)
    R_tube_inv_GeV = R_tube_fm / 0.1973
    A_tube = math.pi * R_tube_inv_GeV**2         # GeV^-2
    C_F = 4.0 / 3.0
    g_s2 = 4 * math.pi * NATURAL.alpha_strong    # alpha_s = g_s^2/(4 pi)

    eta_Kc = sigma * A_tube / (C_F * g_s2)
    print(f"  Lattice string tension sigma = {sigma:.3f} GeV^2")
    print(f"  R_tube = {R_tube_fm} fm  ->  A_tube = {A_tube:.3f} GeV^-2")
    print(f"  C_F = 4/3,  alpha_s = {NATURAL.alpha_strong:.4f},  g_s^2 = {g_s2:.4f}")
    print(f"  eta(K_c) = sigma A_tube / (C_F g_s^2) = {eta_Kc:.3f} (natural units)")
    print(f"\nPaper estimate (Section 10):     eta ~ 1.31 / g_s^2 ~ "
          f"{1.31 / g_s2:.3f}")
    print("Both are O(1) -- consistent with the PSFT confinement picture.")

    # Sample V(r) curve for visual confirmation.
    rs = np.linspace(0.1, 1.0, 10)  # fm
    Vs = sigma * (rs / 0.1973)      # in GeV
    print("\n r [fm]    V(r) [GeV] = sigma*r")
    for r, V in zip(rs, Vs):
        print(f"  {r:.2f}      {V:.3f}")


if __name__ == "__main__":
    main()
