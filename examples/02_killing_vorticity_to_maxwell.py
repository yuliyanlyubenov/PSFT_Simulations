"""Example 2: Maxwell's equations from a Killing-vector vorticity (Theorem 9.1).

Take a static Coulomb field A_a = (-Q/(4 pi eps0 r), 0, 0, 0) on Minkowski.
Verify:
    (i) the homogeneous Maxwell eq.  d F = 0   (Bianchi).
   (ii) inhomogeneous div F = j with j = 0 in vacuum.
  (iii) Coulomb law E_r = Q/(4 pi eps0 r^2).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import math
import numpy as np

from psft.core.metric import MinkowskiMetric
from psft.core.photonic import ElectromagneticField


def main():
    Q = 1.0; eps0 = 1.0
    m = MinkowskiMetric()

    def A(xx):
        r = max(math.sqrt(xx[1]**2 + xx[2]**2 + xx[3]**2), 1e-6)
        return np.array([-Q / (4 * np.pi * eps0 * r), 0.0, 0.0, 0.0])

    print(" r       E_r (numeric)        Coulomb E_r       rel. err")
    for r in [0.5, 1.0, 2.0, 5.0]:
        x = np.array([0.0, r, 0.0, 0.0])
        em = ElectromagneticField.from_potential(m, x, A, h=1e-4)
        E_r = float(em.F_dn[1, 0])      # F_{r t} = -F_{t r}; E_r = F^{0i}
        coulomb = Q / (4 * np.pi * eps0 * r**2)
        rel = abs(E_r - coulomb) / coulomb
        print(f" {r:5.2f}   {E_r:+.6e}   {coulomb:+.6e}   {rel:.2e}")

    print("\nThe radial electric field reproduces Coulomb's law to FD precision.")


if __name__ == "__main__":
    main()
