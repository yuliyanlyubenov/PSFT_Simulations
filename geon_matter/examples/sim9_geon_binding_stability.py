"""sim9 -- Binding the geon: why free light disperses, and how the high-K
viscosity (flux-tube confinement) makes a STABLE, finite-size matter soliton.

This is the natural next step after sims 5-8: those built and characterised
the LIGHT (Hopfion) and its angular momentum; here we ask whether PSFT can
TRAP it into a stationary bound state -- the mass-spectrum frontier (paper
Sec 17), attacked deterministically (no quantisation, no Born rule).

The obstruction is Derrick's theorem: a localised configuration of pure
massless field in 3D has no stable size.  Scaling a configuration to size R,
the field (gradient) energy scales as

    E_field(R) = A * hbar c / R          (wants to EXPAND: lower energy at
                                          large R -> the Wheeler-geon radiates
                                          away; free light disperses)

PSFT supplies the missing stabiliser: the SU(3) shear viscosity that switches
on (Heaviside) only at K > Kc^strong, i.e. only at small size / high
curvature.  By Theorem 10.1 it produces LINEAR confinement (a flux tube) with
string tension sigma:

    E_conf(R) = sigma * R                 (wants to SHRINK: lower energy at
                                          small R)

The competition  E(R) = A hbar c / R + sigma R  has a genuine minimum:

    R* = sqrt(A hbar c / sigma) ,   M c^2 = E(R*) = 2 sqrt(A hbar c sigma) .

With the measured QCD string tension sigma ~ 0.18 GeV^2 this lands at the
HADRONIC scale -- size ~ 0.5 fm, mass ~ 0.85 GeV -- with NO free parameters
beyond an O(1) mode constant A.  We demonstrate (i) no minimum without the
viscosity term, (ii) a stable minimum with it, (iii) the resulting scale is
hadronic, matching light-hadron masses.

Honest scope: this is a Derrick/bag-model energy balance demonstrating the
STABILISATION MECHANISM and its scale -- not a first-principles solution of
the v2 master equation, and not the electron mass (a colourless lepton, bound
in the weak/EM regime, remains unexplained -- paper Sec 17).

Run:  python3 sim9_geon_binding_stability.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    sigma_GeV2 = SI.string_tension_GeV2          # 0.18 GeV^2 (paper value)
    print(f"[psft.core.constants.SI]  string tension sigma = {sigma_GeV2} GeV^2")
except Exception:                                            # pragma: no cover
    sigma_GeV2 = 0.18
    print(f"[fallback]  sigma = {sigma_GeV2} GeV^2")

HBARC_GeV_fm = 0.1973269804                       # hbar c in GeV.fm


def energy_GeV(R_invGeV, A, sigma, with_conf=True):
    """E(R) = A/R + sigma R  in natural units (hbar=c=1), R in GeV^-1."""
    E = A / R_invGeV
    if with_conf:
        E = E + sigma * R_invGeV
    return E


def main():
    print("=" * 74)
    print("sim9: geon binding -- Derrick instability of light + viscous "
          "confinement")
    print("=" * 74)

    A = 1.0                                       # O(1) lowest-mode constant
    sigma = sigma_GeV2

    # ---- (1) Derrick scan: with vs without the confinement term -----------
    R = np.linspace(0.3, 8.0, 400)               # GeV^-1
    E_free = energy_GeV(R, A, sigma, with_conf=False)
    E_bound = energy_GeV(R, A, sigma, with_conf=True)

    # free light: monotonic decreasing -> no minimum (disperses)
    free_monotonic = np.all(np.diff(E_free) < 0)
    # bound: interior minimum
    i_min = int(np.argmin(E_bound))
    has_min = 0 < i_min < len(R) - 1

    print("\n-- (1) energy vs size (natural units, GeV) --")
    print(f"  pure light  E = A/R          : monotonic decreasing "
          f"(disperses)  -> {'CONFIRMED' if free_monotonic else 'no'}")
    print(f"  + viscosity E = A/R + sigma R: interior minimum "
          f"(bound state) -> {'CONFIRMED' if has_min else 'no'}")

    # ---- (2) analytic minimum --------------------------------------------
    R_star = np.sqrt(A / sigma)                   # GeV^-1
    M_star = 2.0 * np.sqrt(A * sigma)             # GeV
    R_star_fm = R_star * HBARC_GeV_fm
    print("\n-- (2) stable geon (analytic minimum of A/R + sigma R) --")
    print(f"  R* = sqrt(A/sigma)      = {R_star:.4f} GeV^-1 = "
          f"{R_star_fm:.4f} fm")
    print(f"  M  = 2 sqrt(A sigma)    = {M_star:.4f} GeV = "
          f"{M_star*1000:.0f} MeV")
    # numeric check vs grid minimum
    ok_minmatch = abs(R[i_min] - R_star) / R_star < 0.05

    # ---- (3) compare to the light-hadron scale ----------------------------
    print("\n-- (3) is the scale hadronic? --")
    hadrons = [("rho(770)", 0.775), ("proton", 0.938),
               ("omega(782)", 0.782), ("eta'(958)", 0.958)]
    print(f"  predicted geon mass (A=1): {M_star*1000:.0f} MeV, "
          f"size {R_star_fm:.2f} fm")
    print("  light hadrons:")
    for name, m in hadrons:
        print(f"     {name:12s} {m*1000:6.0f} MeV  "
              f"(geon/measured = {M_star/m:.2f})")
    # scale match: within a factor ~1.5 of the rho/proton band, for A in [0.5,4]
    Ms = [2.0*np.sqrt(a*sigma) for a in (0.5, 1.0, 2.0, 4.0)]
    print(f"  mode-constant sensitivity: A in [0.5,4] -> M in "
          f"[{min(Ms)*1000:.0f}, {max(Ms)*1000:.0f}] MeV "
          f"(brackets the light-hadron band)")
    ok_scale = 0.4 < M_star < 1.5                 # GeV, light-hadron band

    print("\n" + "-" * 74)
    print(f"  pure light has NO stable size (Derrick)        : "
          f"{'PASS' if free_monotonic else 'FAIL'}")
    print(f"  viscous confinement gives a stable minimum     : "
          f"{'PASS' if (has_min and ok_minmatch) else 'FAIL'}")
    print(f"  stable scale is hadronic (~0.5 fm, ~0.85 GeV)  : "
          f"{'PASS' if ok_scale else 'FAIL'}")
    all_ok = free_monotonic and has_min and ok_minmatch and ok_scale
    print("-" * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation (deterministic, no Born rule):
  A geon is light that cannot escape.  Pure light always lowers its energy by
  spreading out (Derrick) -- which is exactly why a free Wheeler-geon and the
  propagating Hopfion (sim5/sim7) are NOT matter: they disperse/radiate.  The
  PSFT high-K SU(3) viscosity (Postulate 3) switches on only at small size
  (K>Kc^strong) and, via Theorem 10.1, confines linearly.  The balance pins a
  STABLE, finite-size, self-bound light-soliton -- a deterministic field
  configuration, not a probability cloud -- whose size (~0.5 fm) and mass
  (~0.85 GeV, bracketing rho/proton for an O(1) mode constant) come out at the
  hadronic scale with no tuning.  This is the binding step the geon programme
  needed; the precise mass spectrum (and the lepton sector) remain the open
  master-equation problem.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
