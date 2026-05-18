"""Example 9: PSFT Heaviside phase transition -- viscosity activation
at K_c^strong and K_c^weak.

A signature PSFT prediction (Prediction 4 in the paper) is a SHARP phase
transition in the gauge viscosity as the Kretschmann scalar K crosses
K_c^A from below.  This contrasts with lattice QCD's smooth crossover at
finite temperature and with conventional Standard-Model effective-field
treatments where the gauge running is logarithmic and continuous.

We plot eta^A(K) for the SU(3), SU(2) and U(1) sectors across many decades
of K and show that:

  * Below K_c^strong  : all sectors inviscid (pure GR regime).
  * K_c^strong < K < K_c^weak  : SU(3) viscosity on, SU(2)/U(1) still inviscid.
  * K > K_c^weak             : SU(3) and SU(2) on, U(1) still inviscid.
  * U(1) sector                : never freezes (K_c^EM = infinity).

This visualisation makes the four-force hierarchy of PSFT v2 concrete.
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

from psft.core.constants import SI
from psft.sectors.viscosity import (
    GaugeViscosity, HeavisideViscosity, profile_QCD_log, profile_constant,
)


def main():
    print(" PSFT phase transition demonstration")
    print("=" * 60)

    gv = GaugeViscosity.physical(SI,
                                  strong_profile=profile_QCD_log,
                                  weak_profile=profile_constant)
    # Override eta_0 for visualisation -- we just want the qualitative shape.
    gv.sectors["strong"].eta_0 = 1.0
    gv.sectors["weak"].eta_0 = 1.0

    # Sweep K across 20 decades and evaluate each sector.
    K_min = 1e40           # well below K_c^strong = 1.2e61
    K_max = 1e80           # well above K_c^weak = 1.2e73
    Ks = np.logspace(math.log10(K_min), math.log10(K_max), 400)
    eta_strong = np.array([gv.eta("strong", K) for K in Ks])
    eta_weak = np.array([gv.eta("weak",   K) for K in Ks])
    eta_em = np.array([gv.eta("em",     K) for K in Ks])  # always zero

    Kc_strong = SI.Kc_strong
    Kc_weak = SI.Kc_weak
    print(f"  K_c^strong = {Kc_strong:.3e} m^-4")
    print(f"  K_c^weak   = {Kc_weak:.3e} m^-4")
    print(f"  K_c^em     = +infinity (U(1) sector never freezes)")
    print()
    # Find the K values where each sector activates.
    idx_strong = int(np.argmax(eta_strong > 0))
    idx_weak = int(np.argmax(eta_weak > 0))
    print(f"  SU(3) activates first non-zero at K = {Ks[idx_strong]:.3e}")
    print(f"  SU(2) activates first non-zero at K = {Ks[idx_weak]:.3e}")

    # Compute the effective coupling alpha_eff(K) = 1/(8 b_0 ln(K/Kc^strong))
    # to visualise asymptotic freedom.
    b0 = 27.0 / (48 * math.pi)
    above_threshold = Ks > Kc_strong
    alpha_eff = np.zeros_like(Ks)
    alpha_eff[above_threshold] = 1.0 / (8 * b0 * np.log(Ks[above_threshold] / Kc_strong))

    # Plot.
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax = axes[0]
    ax.loglog(Ks, np.maximum(eta_strong, 1e-30), label="SU(3) (strong)", lw=2)
    ax.loglog(Ks, np.maximum(eta_weak, 1e-30),   label="SU(2) (weak)",   lw=2)
    ax.loglog(Ks, np.maximum(eta_em + 1e-30, 1e-30), label="U(1) (EM) -- always inviscid",
              lw=2, linestyle="--")
    ax.axvline(Kc_strong, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(Kc_weak,   color="gray", linestyle=":", alpha=0.7)
    ax.text(Kc_strong * 1.1, 1e-2, "K_c^strong", rotation=90, fontsize=8)
    ax.text(Kc_weak * 1.1, 1e-2, "K_c^weak",   rotation=90, fontsize=8)
    ax.set_ylabel("gauge-sector viscosity eta^A(K) [arb. units]")
    ax.set_ylim(1e-3, 10)
    ax.set_title("PSFT Heaviside activation of gauge sectors")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    ax2.loglog(Ks[above_threshold], alpha_eff[above_threshold], lw=2,
               color="C2", label=r"$\alpha_{\rm s}^{\rm eff}(K) = 1/(8 b_0 \ln(K/K_c^{\rm strong}))$")
    ax2.axvline(Kc_strong, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Kretschmann scalar K  [m^-4]")
    ax2.set_ylabel("alpha_eff(K)  [SU(3)]")
    ax2.set_title("PSFT prediction of QCD asymptotic freedom")
    ax2.legend(loc="upper right")
    ax2.set_ylim(1e-3, 10)

    out_path = os.path.join(os.path.dirname(HERE), "examples", "out_phase_transition.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")
    print()
    print("  PSFT predictions illustrated:")
    print("  1. Sharp Heaviside activation at K_c^A -- predicts a *first-order*")
    print("     phase transition in heavy-ion collisions at the QGP threshold,")
    print("     distinguishable from the smooth crossover of standard QCD.")
    print("  2. The U(1) sector (electromagnetism) never freezes, so the")
    print("     photon remains exactly massless at all energies.")
    print("  3. alpha_eff(K) decreases logarithmically above K_c^strong --")
    print("     asymptotic freedom of QCD emerges from the QCD-log profile.")


if __name__ == "__main__":
    main()
