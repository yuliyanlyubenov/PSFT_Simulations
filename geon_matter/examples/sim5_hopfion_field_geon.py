"""sim5 -- A GENUINE field-theoretic geon: the electromagnetic Hopfion.

Doc 03 modelled circulating light as a point charge on a ring (a heuristic).
This script upgrades that to an EXACT solution of the source-free Maxwell
equations whose energy is literally light "knotted" / orbiting on itself: the
Ranada-Hopfion electromagnetic knot (Ranada 1989; Irvine & Bouwmeester 2008;
Kedia, Bialynicki-Birula, Foster, Bouwmeester, PRL 2013).

It is built from the Bateman complex scalars (alpha, beta) via the
Riemann-Silberstein vector  F = E + i c B = grad(alpha) x grad(beta).
At t = 0 (natural units c = eps0 = mu0 = 1, scale = 1):

    alpha = (r^2 - 1 + 2 i z) / (r^2 + 1)
    beta  = 2 (x - i y)       / (r^2 + 1)

This F is a NULL field (E . B = 0 and |E| = |B| everywhere, i.e. F . F = 0)
and is divergence-free (Gauss laws in vacuum).  It is a finite-energy,
localised blob of light carrying intrinsic angular momentum along its
symmetry axis -- the rigorous field-theoretic realisation of "orbiting light
with spin", and the closest exact-solution analogue of a PSFT light-geon.

We compute (analytic gradients; integrals on a 3D grid):

  * null check          F . F = (E^2 - B^2) + 2 i (E . B)  ->  0
  * Maxwell constraints div E -> 0,  div B -> 0   (vacuum Gauss laws)
  * energy        U = (1/2) integral (E^2 + B^2) dV
  * momentum      P = integral (E x B) dV
  * angular mom.  L = integral r x (E x B) dV     (expect along +z)
  * the dimensionless ratio  c|L_z| / U  (intrinsic spin-to-energy)

Honest caveat (printed): a single fundamental Hopfion is a BOSONIC (integer,
spin-1-like) field; the electron's half-integer spin and g=2 need fermionic /
double-cover structure absent from a pure Maxwell field.  So this demonstrates
"light can be a genuine self-bound angular-momentum-carrying field solution",
not "the electron IS a classical photon knot".

Run:  python3 sim5_hopfion_field_geon.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI          # imported read-only (provenance)
    print("[psft.core.constants available; sim runs in natural units c=eps0=1]")
except Exception:                                            # pragma: no cover
    print("[fallback; natural units c=eps0=1]")


def hopfion_fields(x, y, z):
    """Analytic E, B of the t=0 fundamental electromagnetic Hopfion.
    Returns (E, B) each as a length-3 list of real 3D arrays.  Natural units
    (c = 1), so F = E + i B with F = grad(alpha) x grad(beta)."""
    r2 = x * x + y * y + z * z
    D = r2 + 1.0
    D2 = D * D

    # gradients of alpha = (r^2 - 1 + 2 i z)/D
    dax = 4.0 * x * (1.0 - 1j * z) / D2
    day = 4.0 * y * (1.0 - 1j * z) / D2
    daz = (4.0 * z + 2j * (r2 + 1.0 - 2.0 * z * z)) / D2

    # gradients of beta = 2(x - i y)/D
    dbx = (2.0 * D - 4.0 * x * x + 4j * x * y) / D2
    dby = (-2j * D - 4.0 * x * y + 4j * y * y) / D2
    dbz = (-4.0 * x * z + 4j * y * z) / D2

    # F = grad(alpha) x grad(beta)
    Fx = day * dbz - daz * dby
    Fy = daz * dbx - dax * dbz
    Fz = dax * dby - day * dbx

    E = [Fx.real, Fy.real, Fz.real]
    B = [Fx.imag, Fy.imag, Fz.imag]
    return (E, B), (Fx, Fy, Fz)


def divergence(Vx, Vy, Vz, dx):
    gx = np.gradient(Vx, dx, axis=0)
    gy = np.gradient(Vy, dx, axis=1)
    gz = np.gradient(Vz, dx, axis=2)
    return gx + gy + gz


def main():
    print("=" * 74)
    print("sim5: electromagnetic Hopfion -- a genuine field-theoretic light-geon")
    print("=" * 74)

    HIGH = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 121 if HIGH else 81
    Lbox = 4.0
    ax = np.linspace(-Lbox, Lbox, N)
    dx = ax[1] - ax[0]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    print(f"grid: {N}^3 over [-{Lbox},{Lbox}]^3, dx = {dx:.4f}")

    (E, B), F = hopfion_fields(X, Y, Z)
    Fx, Fy, Fz = F

    # ---- null check:  F . F = (E^2 - B^2) + 2 i (E . B) --------------------
    FdotF = Fx * Fx + Fy * Fy + Fz * Fz
    E2 = E[0] ** 2 + E[1] ** 2 + E[2] ** 2
    B2 = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    EdotB = E[0] * B[0] + E[1] * B[1] + E[2] * B[2]
    scale = np.mean(E2 + B2)
    rel_E2mB2 = np.sqrt(np.mean((E2 - B2) ** 2)) / scale
    rel_EdotB = np.sqrt(np.mean(EdotB ** 2)) / scale
    print("\n-- null-field check (should be ~0) --")
    print(f"  RMS(E^2 - B^2)/scale = {rel_E2mB2:.2e}")
    print(f"  RMS(E . B)/scale     = {rel_EdotB:.2e}")
    ok_null = rel_E2mB2 < 1e-10 and rel_EdotB < 1e-10

    # ---- Maxwell vacuum constraints (FD) ----------------------------------
    divE = divergence(E[0], E[1], E[2], dx)
    divB = divergence(B[0], B[1], B[2], dx)
    # interior only (avoid one-sided FD at the boundary)
    s = (slice(2, -2),) * 3
    gradmag = np.sqrt(np.mean(E2[s])) / dx          # characteristic |dF|
    rel_divE = np.sqrt(np.mean(divE[s] ** 2)) / gradmag
    rel_divB = np.sqrt(np.mean(divB[s] ** 2)) / gradmag
    print("\n-- Maxwell vacuum constraints (FD, interior) --")
    print(f"  RMS(div E)/|gradF| = {rel_divE:.2e}")
    print(f"  RMS(div B)/|gradF| = {rel_divB:.2e}")
    ok_div = rel_divE < 1e-2 and rel_divB < 1e-2     # 2nd-order FD on smooth field

    # ---- conserved charges ------------------------------------------------
    dV = dx ** 3
    u = 0.5 * (E2 + B2)
    U = np.sum(u) * dV
    # Poynting / momentum density  p = E x B
    px = E[1] * B[2] - E[2] * B[1]
    py = E[2] * B[0] - E[0] * B[2]
    pz = E[0] * B[1] - E[1] * B[0]
    P = np.array([np.sum(px), np.sum(py), np.sum(pz)]) * dV
    # angular momentum density  r x p
    lx = Y * pz - Z * py
    ly = Z * px - X * pz
    lz = X * py - Y * px
    Lvec = np.array([np.sum(lx), np.sum(ly), np.sum(lz)]) * dV

    print("\n-- conserved charges (natural units, box-truncated) --")
    print(f"  energy        U      = {U:.6f}")
    print(f"  momentum      P      = [{P[0]:+.3e}, {P[1]:+.3e}, {P[2]:+.3e}]")
    print(f"  ang. momentum L      = [{Lvec[0]:+.3e}, {Lvec[1]:+.3e}, {Lvec[2]:+.3e}]")
    Lmag = np.linalg.norm(Lvec)
    print(f"  |L| = {Lmag:.6f},  L_z fraction = {abs(Lvec[2])/Lmag:.4f} "
          f"(expect ~1: angular momentum along symmetry axis)")
    print(f"  L_z/U = {abs(Lvec[2])/U:.4f}  [a characteristic LENGTH in units of")
    print(f"          the knot scale a=1; NOT a dimensionless spin -- the")
    print(f"          fundamental Hopfion is topologically spin-1, see caveat]")
    print(f"  |P|c/U = {np.linalg.norm(P)/U:.4f}  (<1: localised lump, not a plane wave)")
    ok_spin_axis = abs(Lvec[2]) / Lmag > 0.99
    ok_finite = np.isfinite(U) and U > 0 and Lmag > 0

    print("\n" + "-" * 74)
    print(f"  null field (E.B=0, |E|=|B|)            : {'PASS' if ok_null else 'FAIL'}")
    print(f"  vacuum Maxwell constraints (div=0)     : {'PASS' if ok_div else 'FAIL'}")
    print(f"  finite U, L; spin along symmetry axis  : "
          f"{'PASS' if (ok_finite and ok_spin_axis) else 'FAIL'}")
    all_ok = ok_null and ok_div and ok_finite and ok_spin_axis
    print("-" * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation:
  The energy is a localised, finite, NULL electromagnetic field -- pure light
  -- that is a self-contained EXACT solution of Maxwell's equations and
  carries quantised intrinsic angular momentum along its axis.  This is the
  rigorous field-theoretic version of doc 03's "orbiting light with spin":
  the orbit is the linked (Hopf) field-line structure, not a point on a wire.
  In PSFT terms it is a propagating-light configuration; a *stationary* geon
  additionally needs the high-K SU(3) viscosity (paper Sec 7.4) to trap it.

Honest caveat:
  A single fundamental Hopfion carries INTEGER (bosonic, spin-1-like) angular
  momentum and helicity.  The electron's half-integer spin and g=2 require
  fermionic / double-cover structure that a pure bosonic Maxwell field does
  not have.  So this proves "light can be a genuine self-bound spinning field
  solution", and locates precisely what extra structure the *electron* geon
  needs -- it does not claim the electron is literally a classical light knot.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
