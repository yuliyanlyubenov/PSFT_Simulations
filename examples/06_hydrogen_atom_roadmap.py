"""Example 6: Hydrogen atom -- compositional roadmap.

PSFT identifies particles with topological solitons of the fluid:

    quark        SU(3) colour vortex (open string in viscous medium)
    gluon        flux tube connecting colour charges
    proton (p)   3-quark colour singlet, baryon number B = +1 Skyrme hedgehog
    neutron (n)  same -- different SU(2) isospin orientation
    electron     U(1) line vortex (winding n = -1)
    photon       primitive (not soliton); drives the entire stack via P_ab

For a single hydrogen atom one needs to set up:

    1. A spatial lattice that resolves both the QCD scale (~0.1 fm) AND the
       atomic Bohr radius (~50000 fm).  An adaptive multi-grid is required.
    2. A Skyrme-style B=1 hedgehog ansatz for the proton (centred at the
       nucleus), with optional internal SU(3) shear modes (Conjecture SU3).
    3. A U(1) electron vortex orbiting the proton, with quantised winding.
    4. The photonic source tensor P_ab seeded by the proton's bound EM
       field -- this drives the curvature that activates the SU(2) and SU(3)
       viscosity in the nucleus.

This script does NOT run the simulation -- it only assembles initial data on
a coarse grid and prints sanity checks.  Running the full coupled evolution
is a future-work item enumerated in Section 14 of the paper.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

from psft.core.constants import NATURAL, SI
from psft.core.manifold import CartesianGrid
from psft.solitons.ansatz import VortexAnsatz, SkyrmeHedgehogAnsatz
from psft.solitons.topology import (
    topological_charge_skyrme, winding_number_2d_loop,
)


def main():
    print("=" * 70)
    print(" Hydrogen atom roadmap -- initial data assembly")
    print("=" * 70)

    # ----- Step 1: choose scales -------------------------------------------
    fm = 1e-15
    a0 = 5.29177e-11        # Bohr radius, m
    R_proton = 0.8412 * fm  # proton charge radius
    print(f"  proton charge radius : {R_proton:.3e} m  ({R_proton/fm:.3f} fm)")
    print(f"  Bohr radius          : {a0:.3e} m")
    print(f"  scale ratio          : {a0 / R_proton:.3e}\n")

    # A single uniform grid cannot resolve both scales; use a logarithmically
    # zoomed three-region grid in a real simulation.  Here we set up two
    # separate Cartesian patches: a 'nucleus' patch and an 'atomic' patch.

    nucleus = CartesianGrid(shape=(48, 48, 48),
                             bounds=((-2*fm, 2*fm), (-2*fm, 2*fm), (-2*fm, 2*fm)))
    atomic = CartesianGrid(shape=(48, 48, 48),
                            bounds=((-3*a0, 3*a0), (-3*a0, 3*a0), (-3*a0, 3*a0)))

    # ----- Step 2: proton as Skyrme hedgehog -------------------------------
    proton_skyrme = SkyrmeHedgehogAnsatz(core_radius=R_proton).evaluate(nucleus)
    dx_n = nucleus.deltas()[0]
    B = topological_charge_skyrme(proton_skyrme["N4"], dx_n)
    print(f"  proton baryon number B (Skyrme):  {B:+.4f}  (expected +/- 1)")

    # ----- Step 3: electron vortex on the atomic patch ---------------------
    electron = VortexAnsatz(winding=-1, core_radius=0.05 * a0).evaluate(atomic)
    z_mid = atomic.shape[2] // 2
    w_e = winding_number_2d_loop(
        electron["phi_complex"][:, :, z_mid],
        center=(atomic.shape[0] // 2, atomic.shape[1] // 2),
        radius=atomic.shape[0] // 3,
    )
    print(f"  electron U(1) winding:            {w_e}    (expected -1)")

    # ----- Step 4: confirm the inviscid status of the EM sector ------------
    print(f"\n  K_c^strong = {SI.Kc_strong:.3e} m^-4")
    print(f"  K_c^weak   = {SI.Kc_weak:.3e} m^-4")
    print(f"  K_c^EM     = {SI.Kc_em}")
    print("  => Inside the nucleus, K vastly exceeds K_c^strong (Theorem 10.1):")
    print("     SU(3) viscosity locks the quark-gluon plasma into colour-singlet")
    print("     bound states.  Outside the nucleus, K << K_c^strong: the fluid")
    print("     is inviscid and only the Coulomb part of P_{ab} survives.")

    # ----- Step 5: outstanding tasks ---------------------------------------
    print("\nNot done in this scaffold (future work):")
    print("  * coupled evolution of nucleus + atomic patches with shared P_ab")
    print("  * quark sub-solitons inside the Skyrme hedgehog (SU(3) substructure)")
    print("  * neutron variant via isospin flip (SU(2) re-orientation)")
    print("  * binding-energy extraction from the relaxed equilibrium")
    print("  * adaptive mesh refinement across the 4-5 decade scale ratio")


if __name__ == "__main__":
    main()
