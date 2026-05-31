"""Phase 5 -- Two-geon correlation from fluid coherence: entanglement as a
PHYSICAL, degradable medium property (the dynamical half of P-G3).

sim11 showed (consistency) that a deterministic nonlocal model can reach the
Tsirelson bound 2sqrt2 if the spacetime fluid supplies the nonlocal link.  It
left open the *dynamical* question: what sets the strength of that link, and is
it degradable like a real medium?

Here we make the link a physical quantity -- the FLUID COHERENCE V: the
fraction of geon pairs that remain a single, coherently-connected field object
through the spacetime fluid between the wings.  A pair that stays coherent is
the nonlocal single-object (Toner-Bacon / sim11) -> contributes -cos(a-b).  A
pair whose fluid link is broken (decohered by curvature, the viscous phase, or
distance) falls back to a SEPARABLE state -> contributes zero correlation.
The deterministic per-event model (shared field state + coherence coin = the
hidden variables) then gives

    E(a,b; V) = -V cos(a-b) ,     S(V) = 2 sqrt(2) * V .

Consequences, all checked numerically:
  * S(V) is LINEAR in the fluid coherence and reaches Tsirelson (2sqrt2) at V=1;
  * Bell violation (S>2) occurs ONLY above a sharp threshold V > 1/sqrt2 ~ 0.707
    (the Werner-state visibility threshold) -- below it the pair is effectively
    classical/separable;
  * NO-SIGNALLING holds at every V (marginals stay 1/2).

This operationalises prediction P-H1: because entanglement strength = fluid
coherence, PERTURBING the spacetime fluid between the wings (extreme curvature,
the high-K viscous phase) must reduce V and hence S, and DESTROY Bell violation
once V drops below 1/sqrt2 -- a medium-dependent, falsifiable effect ABSENT
from orthodox QM (where reduced visibility is mere apparatus imperfection, not
fundamental medium physics).

Honest status: V is still a parameter (set by the fluid state), not yet derived
from the master-equation two-soliton dynamics -- that final derivation is the
remaining open step.  What is new vs sim11: the nonlocal link is now a physical,
quantitative, degradable medium property with a sharp, testable threshold.

Run:  python3 phase5_two_geon_from_fluid.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI       # provenance only
    print("[psft.core.constants available]")
except Exception:                                            # pragma: no cover
    print("[fallback]")

RNG = np.random.default_rng(20260531)


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def tb_outcomes(a, b, N):
    """Per-event deterministic Toner-Bacon outcomes (the coherent / single-
    object pair connected through the fluid).  Returns (A, B) arrays of +/-1."""
    av, bv = _unit(a), _unit(b)
    l1 = RNG.normal(size=(N, 3)); l1 /= np.linalg.norm(l1, axis=1, keepdims=True)
    l2 = RNG.normal(size=(N, 3)); l2 /= np.linalg.norm(l2, axis=1, keepdims=True)
    d1a, d2a = l1 @ av, l2 @ av
    d1b, d2b = l1 @ bv, l2 @ bv
    A = np.sign(d1a)
    c = np.sign(d1a) * np.sign(d2a)
    B = -np.sign(d1b + c * d2b)
    return A, B


def E_fluid(a, b, V, N=300000):
    """Correlation with fluid coherence V: fraction V of pairs stay coherent
    (TB single-object), fraction (1-V) decohere to a separable/random state."""
    A_c, B_c = tb_outcomes(a, b, N)
    coherent = RNG.random(N) < V                  # the coherence coin (hidden)
    # decohered pairs: independent random outcomes (separable, zero correlation)
    A_d = RNG.choice([-1.0, 1.0], size=N)
    B_d = RNG.choice([-1.0, 1.0], size=N)
    A = np.where(coherent, A_c, A_d)
    B = np.where(coherent, B_c, B_d)
    return float(np.mean(A * B)), float(np.mean(A)), float(np.mean(B))


def chsh_S(V, N=300000):
    a, ap, b, bp = 0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4
    E = lambda x, y: E_fluid(x, y, V, N)[0]
    return abs(E(a, b) - E(a, bp) + E(ap, b) + E(ap, bp))


def main():
    print("=" * 76)
    print("Phase 5: entanglement as fluid coherence -- S(V) = 2sqrt2 V (P-G3/P-H1)")
    print("=" * 76)

    tsirelson = 2 * np.sqrt(2)
    Vthr = 1.0 / np.sqrt(2)
    print(f"\nTsirelson bound 2sqrt2 = {tsirelson:.4f};  "
          f"Bell-violation threshold V = 1/sqrt2 = {Vthr:.4f}\n")

    # ---- (1) S(V) curve ----------------------------------------------------
    print("-" * 76)
    print(f"{'fluid coherence V':>18} {'CHSH S(V)':>11} {'2sqrt2 V':>10} "
          f"{'Bell violated?':>15}")
    print("-" * 76)
    Vs = [0.0, 0.25, 0.5, Vthr, 0.85, 1.0]
    rows = []
    for V in Vs:
        S = chsh_S(V)
        rows.append((V, S))
        print(f"{V:18.4f} {S:11.4f} {tsirelson*V:10.4f} "
              f"{('YES' if S > 2.0 else 'no'):>15}")
    print("-" * 76)

    # linearity & endpoints
    Vfit = np.array([r[0] for r in rows]); Sfit = np.array([r[1] for r in rows])
    slope = np.polyfit(Vfit, Sfit, 1)[0]
    ok_linear = abs(slope - tsirelson) < 0.05
    S1 = chsh_S(1.0); S0 = chsh_S(0.0)
    ok_endpoints = abs(S1 - tsirelson) < 0.03 and S0 < 0.05
    ok_thresh = abs(chsh_S(Vthr) - 2.0) < 0.03

    print(f"\n  slope dS/dV = {slope:.4f}  (predicted 2sqrt2 = {tsirelson:.4f})")
    print(f"  S(V=1) = {S1:.4f} (Tsirelson),  S(V=0) = {S0:.4f} (no correlation)")
    print(f"  S(V=1/sqrt2) = {chsh_S(Vthr):.4f}  (exactly the classical bound 2)")

    # ---- (2) no-signalling at all V ---------------------------------------
    print("\n-- no-signalling: Alice marginal independent of V and of Bob --")
    for V in [0.0, 0.5, 1.0]:
        _, mA, mB = E_fluid(0.3, 1.1, V)
        print(f"   V={V}:  <A> = {mA:+.4f},  <B> = {mB:+.4f}  (both ~0)")
    _, mA0, _ = E_fluid(0.3, 0.0, 0.8)
    _, mA1, _ = E_fluid(0.3, 2.0, 0.8)
    ok_ns = abs(mA0 - mA1) < 0.01
    print(f"   <A> for Bob setting b=0 vs b=2 (V=0.8): {mA0:+.4f} vs {mA1:+.4f} "
          f"-> independent: {'PASS' if ok_ns else 'FAIL'}")

    # ---- (3) P-H1 quantitative: medium perturbation degrades S ------------
    print("\n-- (3) P-H1 made quantitative: fluid perturbation degrades Bell --")
    print("   model a fluid perturbation of strength p reducing coherence as")
    print("   V(p) = exp(-p)  (decoherence by curvature / viscous phase):")
    print(f"{'perturbation p':>15} {'V=exp(-p)':>10} {'S':>8} {'entangled?':>12}")
    for p in [0.0, 0.2, 0.35, 0.7, 1.2]:
        V = np.exp(-p)
        S = chsh_S(V)
        print(f"{p:15.2f} {V:10.4f} {S:8.4f} "
              f"{('Bell-violating' if S > 2 else 'classical'):>12}")
    p_crit = -np.log(Vthr)
    print(f"   => Bell violation lost once p > {p_crit:.3f} (V < 1/sqrt2): a")
    print(f"      sharp, medium-dependent threshold absent from orthodox QM.")

    print("\n" + "-" * 76)
    print(f"  S(V) linear with slope 2sqrt2                  : "
          f"{'PASS' if ok_linear else 'FAIL'}")
    print(f"  endpoints: S(1)=2sqrt2 (Tsirelson), S(0)=0     : "
          f"{'PASS' if ok_endpoints else 'FAIL'}")
    print(f"  Bell threshold at V=1/sqrt2 (S=2)              : "
          f"{'PASS' if ok_thresh else 'FAIL'}")
    print(f"  no-signalling at all V                         : "
          f"{'PASS' if ok_ns else 'FAIL'}")
    all_ok = ok_linear and ok_endpoints and ok_thresh and ok_ns
    print("-" * 76)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 76)
    print("""
Interpretation:
  Entanglement strength is identified with a PHYSICAL, degradable property --
  the coherence V of the spacetime-fluid link between the two geons.  The CHSH
  value is exactly S = 2sqrt2 V: full fluid coherence reaches the Tsirelson
  bound, and Bell violation survives only above the sharp threshold V > 1/sqrt2.
  No-signalling holds throughout (the link lives at the hidden-field level).
  This advances sim11 from an abstract one-bit channel to a physical medium with
  a quantitative, testable degradation law (P-H1): perturbing the fluid between
  the wings must reduce S and destroy entanglement past a sharp threshold --
  a medium-dependence orthodox QM does not predict.

Honest open step:
  V is set by the fluid state but is not YET computed from the master-equation
  two-soliton dynamics (the fluid's actual coherence/information capacity
  between two cores).  Deriving V(curvature, separation, viscous-phase) from the
  field equations is the final, still-open piece of P-G3.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
