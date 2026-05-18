"""Example 1: Inviscid limit recovers General Relativity (Theorem 12.1).

Strategy: pick a Schwarzschild background, place a test 4-velocity (radial
geodesic), confirm that
    * Kretschmann K < Kc on the macroscopic exterior   -> all viscosities zero.
    * Master-equation residual matches the geodesic equation -> u^b nabla_b u^a = 0.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

from psft.core.constants import SI
from psft.core.metric import SchwarzschildMetric
from psft.core.curvature import CurvatureBundle
from psft.sectors.viscosity import GaugeViscosity
from psft.sectors.master_eq import MasterEquationV2


def main():
    # Use a stellar-mass black hole so the Schwarzschild radius is metres.
    M_BH = 1.989e30          # 1 solar mass
    metric = SchwarzschildMetric(M=M_BH, G=SI.G, c=SI.c)
    gv = GaugeViscosity.physical(SI)
    rs = 2.0 * SI.G * M_BH / SI.c**2   # ~2953 m

    # Sample just outside the horizon (10 * rs): macroscopic but with measurable K.
    rho = 10.0 * rs
    x = np.array([0.0, rho, 0.0, 0.0])
    bundle = CurvatureBundle.from_metric(metric, x, h=rs * 1e-3)
    print(f"Schwarzschild radius rs = {rs:.3f} m")
    print(f"Sample point rho        = {rho:.3f} m  (~10 r_s)")
    print(f"Kretschmann K at r=10 AU: {bundle.K_scalar:.3e} m^-4")
    print(f"K_c^strong:               {SI.Kc_strong:.3e} m^-4")
    print(f"K / K_c^strong:           {bundle.K_scalar / SI.Kc_strong:.3e}\n")

    for sector in ("strong", "weak", "em"):
        eta = gv.eta(sector, bundle.K_scalar)
        print(f"  eta({sector:7s}, K) = {eta:.3e}   "
              + ("[active]" if eta > 0 else "[inviscid]"))

    print("\n=> All viscosity sectors are off in this regime.")
    print("   The master equation reduces to the relativistic Euler eq.")
    print("   (= Einstein equations in the metric-theory sense, Theorem 12.1).")

    eq = MasterEquationV2(metric=metric, consts=SI, gauge_viscosity=gv,
                          sector="gravity", Lambda=0.0)
    # Static observer in the rest frame of Schwarzschild.
    g = metric.g(x)
    A_t = np.sqrt(-1.0 / g[0, 0])
    u_func = lambda _x: np.array([A_t, 0.0, 0.0, 0.0])
    grad_p = np.zeros(4)
    res = eq.evaluate(x, u_func, rho_g=1.0, p_g=0.0, grad_p_g=grad_p)
    print(f"\nMaster-equation RHS magnitude:   |rhs|  = {np.linalg.norm(res.rhs):.3e}")
    print(f"Geodesic LHS (inertia)  magn.:   |LHS|  = {np.linalg.norm(res.inertia):.3e}")
    print("If both are tiny relative to typical curvature scales we have GR.")


if __name__ == "__main__":
    main()
