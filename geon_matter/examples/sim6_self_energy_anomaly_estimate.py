"""sim6 -- The magnetic-moment anomaly a = (g-2)/2 from photonic self-coupling:
what is honestly derivable, and what is not.

Doc 03's geon gives g = 2 at tree level and MISSES the anomaly
a = (g-2)/2 = 1.159652e-3.  In PSFT the photonic field P_ab is primitive and
self-interacting, so the anomaly should arise from the geon coupling to its
own photonic fluctuations -- the analogue of the QED photon loop.  This script
does three honest things:

  (1) BENCHMARK: show that the measured anomaly is an expansion in (alpha/pi),
      a_e = C1 (a/pi) + C2 (a/pi)^2 + C3 (a/pi)^3 + ...,  with the known QED
      coefficients (Schwinger C1 = 1/2, ...).  Summing the series reproduces
      the CODATA value to ~1e-10.  Each extra power of alpha/pi = one extra
      photon exchange.  This is the STRUCTURE PSFT must reproduce: tree geon =
      the "g=2" zeroth term; each photonic self-coupling = one factor alpha/pi.

  (2) SEMICLASSICAL MAGNITUDE: a Welton-style estimate of the electron's
      jitter in its own zero-point photon field, <dr^2> = (2 alpha/pi)
      lambdabar_C^2 ln(Lambda).  This lands the correction at order
      (alpha/pi) x O(1) -- the right size -- but the logarithmic cutoff
      ambiguity is exactly why the clean coefficient 1/2 cannot be fixed
      classically.

  (3) VERDICT: PSFT predicts the anomaly is a photonic-self-energy effect of
      order alpha/pi; deriving C1 = 1/2 from PSFT's (quantised) photonic field
      is open problem P-C3.  A PSFT computation giving any leading coefficient
      other than 1/2 would falsify the framework.

Run:  python3 sim6_self_energy_anomaly_estimate.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    alpha = SI.alpha_em
    print(f"[psft.core.constants.SI]  alpha = 1/{1/alpha:.6f}")
except Exception:                                            # pragma: no cover
    alpha = 1.0 / 137.035999084
    print(f"[fallback]  alpha = 1/{1/alpha:.6f}")


def main():
    print("=" * 74)
    print("sim6: g-2 anomaly as a photonic-self-coupling (alpha/pi) expansion")
    print("=" * 74)

    a_exp = 1.15965218076e-3          # CODATA electron anomaly (g-2)/2
    api = alpha / np.pi
    print(f"\nalpha/pi = {api:.8e}")
    print(f"experimental a_e = (g-2)/2 = {a_exp:.11e}\n")

    # ---- (1) the QED (alpha/pi) series ------------------------------------
    # Known analytic/numeric mass-independent coefficients for a_e:
    C = [0.5,                 # C1  Schwinger 1948
         -0.328478965579,     # C2  Petermann/Sommerfield 1957
         1.181241456587,      # C3  Laporta-Remiddi 1996
         -1.91298,            # C4  Laporta 2017 (numeric)
         7.79]                # C5  (approx)
    print("-- (1) photonic-loop expansion  a = sum C_n (alpha/pi)^n --")
    print(f"{'n':>2} {'C_n':>16} {'term':>16} {'partial sum':>18}")
    partial = 0.0
    for n, Cn in enumerate(C, start=1):
        term = Cn * api ** n
        partial += term
        print(f"{n:>2} {Cn:>16.9f} {term:>16.6e} {partial:>18.11e}")
    rel = abs(partial - a_exp) / a_exp
    print(f"\n  summed series      = {partial:.11e}")
    print(f"  experiment         = {a_exp:.11e}")
    print(f"  relative agreement = {rel:.2e}")
    ok_series = rel < 1e-6
    print(f"  => the anomaly IS an alpha/pi (photon-loop) expansion: "
          f"{'CONFIRMED' if ok_series else 'CHECK'}")
    print(f"     tree geon (g=2) is the n=0 term; n=1 (Schwinger 1/2) dominates.")

    # ---- (2) semiclassical Welton magnitude -------------------------------
    print("\n-- (2) semiclassical self-field magnitude (Welton jitter) --")
    # <dr^2> = (2 alpha/pi) lambdabar_C^2 * ln(Lambda); take ln in [1, 10].
    for lnL in (1.0, 2.0, 5.0):
        dr2_over_lc2 = (2.0 * alpha / np.pi) * lnL
        # crude moment correction ~ <dr^2>/lambdabar_C^2 in natural smearing
        print(f"   ln(Lambda)={lnL:>3}:  <dr^2>/lambdabar_C^2 = "
              f"{dr2_over_lc2:.3e}  (order alpha/pi x O(1))")
    print("   => correction sits at order alpha/pi ~ 2.3e-3, the right size;")
    print("      the log-cutoff ambiguity is why the coefficient 1/2 is NOT")
    print("      classically fixed -- it needs the quantised photon loop.")

    # ---- (3) what the tree geon gives vs needs ----------------------------
    print("\n-- (3) tree geon vs full anomaly --")
    g_tree = 2.0
    g_full = 2.0 * (1.0 + a_exp)
    print(f"   tree-level geon (doc 03):  g = {g_tree:.9f}   (a = 0)")
    print(f"   measured              :    g = {g_full:.9f}   (a = {a_exp:.3e})")
    print(f"   Schwinger one-loop    :    a1 = alpha/2pi = {alpha/(2*np.pi):.8e}")
    print(f"   one-loop vs experiment:    {abs(alpha/(2*np.pi)-a_exp)/a_exp:.2e} "
          f"(0.1% -- the rest is higher photon loops)")

    print("\n" + "=" * 74)
    print(f"VERDICT: anomaly = photonic-self-coupling expansion in alpha/pi : "
          f"{'PASS' if ok_series else 'CHECK'}")
    print("=" * 74)
    print("""
Honest conclusion (prediction P-C3):
  * The geon picture gets g = 2 exactly at tree level (doc 03, sim3).
  * The anomaly is, structurally, an expansion in alpha/pi -- one factor per
    photon the geon exchanges with its own photonic field P_ab.  PSFT's
    primitive, self-interacting photonic field is exactly the object that
    must generate these terms.
  * What is NOT yet derived from PSFT: the leading coefficient C1 = 1/2.
    Classically (Welton) the magnitude comes out at order alpha/pi but the
    coefficient is cutoff-ambiguous; pinning it to 1/2 requires QUANTISING
    the PSFT photonic field -- the same open problem the paper flags (Sec 17).
  * Sharp falsifiable target: a PSFT photonic-loop computation MUST yield
    C1 = 1/2.  Any other leading coefficient falsifies the geon ontology.""")
    return 0 if ok_series else 1


if __name__ == "__main__":
    raise SystemExit(main())
