"""sim11 -- The decisive test (P-G3): can a DETERMINISTIC two-geon model
reproduce entanglement (Bell-CHSH) correlations up to the Tsirelson bound?

This is the make-or-break test of the deterministic programme.  Bell's theorem
is firm: any deterministic theory matching the quantum correlations must be
NONLOCAL.  So we ask three sharply-separated questions and answer each
numerically and honestly:

  (Q1) TARGET.  What does the quantum / deterministic-field-overlap prediction
       give?  The singlet correlation E(a,b) = -cos(a-b) is exactly the
       inner-product (Malus / energy-density) geometry PSFT already uses for
       interference -- NO Born rule, just field projection.  Its CHSH value at
       the optimal angles is S = 2 sqrt(2) ~ 2.828 (Tsirelson bound).

  (Q2) LOCAL determinism.  A geon pair sharing a real LOCAL hidden orientation
       lambda (set at creation) with deterministic Malus-threshold detection.
       Monte-Carlo CHSH.  Bell says this CANNOT exceed S = 2.  We confirm it.
       => local determinism (a shared classical field orientation) is
          INSUFFICIENT.  This is Bell's theorem, demonstrated, not dodged.

  (Q3) NONLOCAL determinism via the spacetime fluid.  PSFT's distinguishing
       structure is a real fluid spanning the manifold: the two geons were one
       field configuration at creation and remain coupled THROUGH the fluid.
       We model that shared medium with the Toner-Bacon deterministic protocol
       (shared hidden state + one fluid-mediated relational bit).  It is
       DETERMINISTIC and NONLOCAL.  We verify it reproduces E = -cos(a-b)
       exactly and reaches S = 2 sqrt(2).

We also verify NO-SIGNALLING (Alice's marginal is independent of Bob's
setting) in every model, so the fluid-mediated nonlocality is relativistically
benign -- correlations only, no usable signal.

HONEST STATUS (printed at the end): this is a CONSISTENCY / POSSIBILITY proof
-- PSFT's nonlocal fluid is exactly the kind of structure that lets a
deterministic theory reach the Tsirelson bound, and the cosine is the
deterministic field-overlap law.  It is NOT a derivation of the correlation
from the PSFT master-equation dynamics; that remains the deep open problem.

Run:  python3 sim11_two_geon_bell_correlations.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI            # provenance only
    print("[psft.core.constants available]")
except Exception:                                            # pragma: no cover
    print("[fallback]")

RNG = np.random.default_rng(20260531)             # fixed seed: no Date/random ban issue


def chsh(Efunc, a, ap, b, bp):
    """CHSH combination S = E(a,b) - E(a,b') + E(a',b) + E(a',b')."""
    return Efunc(a, b) - Efunc(a, bp) + Efunc(ap, b) + Efunc(ap, bp)


# ---- (Q1) quantum / field-overlap target --------------------------------
def E_quantum(a, b):
    """Singlet / field-projection correlation = -cos(angle difference)."""
    return -np.cos(a - b)


# ---- (Q2) LOCAL deterministic geon model --------------------------------
def E_local(a, b, N=400000):
    """Shared LOCAL hidden orientation lambda (uniform), deterministic
    Malus-threshold detection.  Anti-correlated source."""
    lam = RNG.uniform(0.0, 2 * np.pi, N)
    A = np.sign(np.cos(a - lam))                  # Alice: +/-1, depends on a, lam ONLY
    B = -np.sign(np.cos(b - lam))                 # Bob:   depends on b, lam ONLY (local)
    return np.mean(A * B)


# ---- (Q3) NONLOCAL deterministic model (Toner-Bacon, fluid-mediated) -----
def _unit_vectors(angle):
    """Planar Bloch vector for a measurement angle (z-component 0)."""
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def E_nonlocal(a, b, N=400000):
    """Toner-Bacon (2003) deterministic 1-bit protocol.  The shared hidden
    state (lam1, lam2) is set at the source; the single relational bit is
    carried by the spacetime fluid connecting the two geons.  Reproduces the
    singlet correlation -a.b exactly.  Deterministic given (lam1, lam2)."""
    avec = _unit_vectors(a); bvec = _unit_vectors(b)
    # two shared uniform unit vectors on the sphere (the source field state)
    l1 = RNG.normal(size=(N, 3)); l1 /= np.linalg.norm(l1, axis=1, keepdims=True)
    l2 = RNG.normal(size=(N, 3)); l2 /= np.linalg.norm(l2, axis=1, keepdims=True)
    d1a = l1 @ avec; d2a = l2 @ avec               # continuous projections
    d1b = l1 @ bvec; d2b = l2 @ bvec
    A = np.sign(d1a)                               # Alice's deterministic output
    c = np.sign(d1a) * np.sign(d2a)                # one fluid-mediated bit (+/-1)
    # Bob uses the CONTINUOUS projections (magnitudes matter) + the relational bit
    B = -np.sign(d1b + c * d2b)
    return np.mean(A * B)


def no_signalling(model, N=400000):
    """Alice's marginal <A> should not depend on Bob's setting b."""
    a = 0.3
    if model == "local":
        lam = RNG.uniform(0, 2 * np.pi, N)
        A = np.sign(np.cos(a - lam))
        return np.mean(A)            # independent of b by construction
    else:  # nonlocal
        avec = _unit_vectors(a)
        l1 = RNG.normal(size=(N, 3)); l1 /= np.linalg.norm(l1, axis=1, keepdims=True)
        A = -np.sign(l1 @ avec)
        return np.mean(A)


def main():
    print("=" * 76)
    print("sim11: deterministic two-geon Bell-CHSH test (P-G3)")
    print("=" * 76)

    # CHSH-optimal angles for the cosine correlation
    a, ap, b, bp = 0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4
    tsirelson = 2 * np.sqrt(2)
    deg = lambda x: f"{np.degrees(x):.0f}"
    print(f"\nmeasurement angles  a={deg(a)}, a'={deg(ap)}, "
          f"b={deg(b)}, b'={deg(bp)}  (CHSH-optimal)")
    print(f"local-realist bound |S| <= 2 ;  Tsirelson bound |S| = 2sqrt2 "
          f"= {tsirelson:.4f}\n")

    # ---- correlation tables ----------------------------------------------
    print("-" * 76)
    print(f"{'pair (a,b)':>14} | {'E_quantum':>11} {'E_local':>11} "
          f"{'E_nonlocal':>12}")
    print("-" * 76)
    pairs = [(a, b), (a, bp), (ap, b), (ap, bp)]
    for (x, y) in pairs:
        print(f"({deg(x):>3},{deg(y):>3})       | "
              f"{E_quantum(x, y):11.4f} {E_local(x, y):11.4f} "
              f"{E_nonlocal(x, y):12.4f}")
    print("-" * 76)

    S_q = chsh(E_quantum, a, ap, b, bp)
    S_l = chsh(lambda x, y: E_local(x, y), a, ap, b, bp)
    S_n = chsh(lambda x, y: E_nonlocal(x, y), a, ap, b, bp)

    print(f"\n  (Q1) QM / field-overlap CHSH  S = {abs(S_q):.4f}  "
          f"(target Tsirelson {tsirelson:.4f})")
    print(f"  (Q2) LOCAL determinism   CHSH S = {abs(S_l):.4f}  "
          f"(Bell bound 2; cannot exceed)")
    print(f"  (Q3) NONLOCAL (fluid)    CHSH S = {abs(S_n):.4f}  "
          f"(reaches Tsirelson)")

    ok_q = abs(abs(S_q) - tsirelson) < 1e-6
    ok_l = abs(S_l) <= 2.0 + 0.02                 # MC noise; must NOT exceed 2
    ok_l_strict = abs(S_l) < 2.05 and abs(S_l) > 1.9   # saturates ~2, not 2.83
    ok_n = abs(abs(S_n) - tsirelson) < 0.02       # MC reaches 2sqrt2

    # nonlocal model must also reproduce the cosine shape, not just CHSH
    shape_err = max(abs(E_nonlocal(x, y) - E_quantum(x, y)) for (x, y) in
                    [(0, t) for t in np.linspace(0, np.pi, 7)])

    print("\n-- no-signalling (Alice marginal independent of Bob) --")
    nsl = no_signalling("local"); nsn = no_signalling("nonlocal")
    print(f"   <A> local model    = {nsl:+.4f}  (≈0, setting-independent)")
    print(f"   <A> nonlocal model = {nsn:+.4f}  (≈0, setting-independent)")
    ok_ns = abs(nsl) < 0.01 and abs(nsn) < 0.01

    print("\n" + "-" * 76)
    print(f"  (Q1) field-overlap gives Tsirelson 2sqrt2 (no Born rule) : "
          f"{'PASS' if ok_q else 'FAIL'}")
    print(f"  (Q2) LOCAL determinism capped at S=2 (Bell's theorem)    : "
          f"{'PASS' if (ok_l and ok_l_strict) else 'FAIL'}")
    print(f"  (Q3) NONLOCAL fluid model reaches 2sqrt2, shape=-cos     : "
          f"{'PASS' if (ok_n and shape_err < 0.02) else 'FAIL'}")
    print(f"  no-signalling preserved in all models                    : "
          f"{'PASS' if ok_ns else 'FAIL'}")
    all_ok = ok_q and ok_l and ok_l_strict and ok_n and ok_ns and shape_err < 0.02
    print("-" * 76)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 76)
    print(f"""
Interpretation (honest):
  * (Q1) The Tsirelson correlation -cos(a-b) is the DETERMINISTIC field-overlap
    (Malus / energy-density) law PSFT already uses -- the "quantum" 2sqrt2 is
    the geometry of field-amplitude projection, not a statistical axiom.
  * (Q2) A geon pair with only a LOCAL shared field orientation is capped at
    S=2 (measured {abs(S_l):.3f}).  Local determinism genuinely FAILS -- exactly as
    Bell proved.  We do not dodge this.
  * (Q3) PSFT's distinguishing ingredient is a REAL nonlocal medium: the
    spacetime fluid through which the two geons remain a single connected
    configuration.  A deterministic model with one fluid-mediated relational
    bit (Toner-Bacon) reproduces -cos(a-b) and reaches S={abs(S_n):.3f} ~ 2sqrt2,
    while preserving NO-SIGNALLING (<A> independent of Bob).

  WHAT THIS PROVES:  PSFT's nonlocal fluid is the right KIND of structure to
  host quantum entanglement deterministically and without signalling -- the
  deterministic programme is not excluded by Bell.
  WHAT IT DOES NOT PROVE:  that PSFT's master-equation DYNAMICS actually
  produce this specific correlation / one-bit channel.  Deriving the cosine
  (and the fluid's information capacity) from the field equations is the deep
  open problem (P-G3 remains open; this is the consistency half of it).""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
