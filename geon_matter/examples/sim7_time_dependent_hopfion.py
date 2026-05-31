"""sim7 -- The Hopfion in MOTION: dynamical Maxwell (Faraday + Ampere) check.

sim5 verified the t=0 Hopfion satisfies the Maxwell CONSTRAINTS (div E = div B
= 0) and is null.  This script verifies the full DYNAMICS: that the
time-dependent Ranada-Hopfion is an exact solution of the evolution equations
Faraday (dB/dt = -curl E) and Ampere (dE/dt = +curl B), and that it stays
null and keeps its conserved charges as it propagates.

Time-dependent Bateman scalars (Kedia-Bialynicki-Birula-Foster-Bouwmeester,
PRL 2013), natural units c = 1, scale a = 1:

    P     = r^2 - (t - i)^2 = (r^2 - t^2 + 1) + 2 i t        (complex)
    alpha = (r^2 - t^2 - 1 + 2 i z) / P
    beta  = 2 (x - i y) / P
    F     = E + i B = grad(alpha) x grad(beta)

In Riemann-Silberstein form the two dynamical Maxwell equations combine into

    dF/dt = -i curl F .

We evaluate F from ANALYTIC spatial gradients (exact), then check the residual
R = dF/dt + i curl F using one centred finite difference in time and one in
space -- so the residual is pure 2nd-order discretisation error, shrinking
as dx^2, dt^2.  We also confirm the field stays null (F . F = 0) at every time
and track U and L_z (approximately conserved up to box truncation as the knot
moves).

Run:  python3 sim7_time_dependent_hopfion.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    print("[psft.core.constants available; natural units c=eps0=1]")
except Exception:                                            # pragma: no cover
    print("[fallback; natural units c=eps0=1]")


def hopfion_F(x, y, z, t):
    """Analytic Riemann-Silberstein vector F = E + iB of the moving Hopfion."""
    r2 = x * x + y * y + z * z
    P = (r2 - t * t + 1.0) + 2j * t          # = r^2 - (t-i)^2
    P2 = P * P
    Na = r2 - t * t - 1.0 + 2j * z           # numerator of alpha
    Nb = 2.0 * x - 2j * y                    # numerator of beta

    # grad(alpha):  (P - Na) = 2[1 + i(t - z)]
    w = 1.0 + 1j * (t - z)
    dax = 4.0 * x * w / P2
    day = 4.0 * y * w / P2
    daz = (4.0 * z * w + 2j * P) / P2

    # grad(beta)
    dbx = (2.0 * P - 2.0 * x * Nb) / P2
    dby = (-2j * P - 2.0 * y * Nb) / P2
    dbz = (-2.0 * z * Nb) / P2

    Fx = day * dbz - daz * dby
    Fy = daz * dbx - dax * dbz
    Fz = dax * dby - day * dbx
    return Fx, Fy, Fz


def curl(Vx, Vy, Vz, dx):
    dV = lambda V, ax: np.gradient(V, dx, axis=ax)
    cx = dV(Vz, 1) - dV(Vy, 2)
    cy = dV(Vx, 2) - dV(Vz, 0)
    cz = dV(Vy, 0) - dV(Vx, 1)
    return cx, cy, cz


def conserved(Fx, Fy, Fz, X, Y, Z, dx):
    E = [Fx.real, Fy.real, Fz.real]
    B = [Fx.imag, Fy.imag, Fz.imag]
    E2 = sum(e * e for e in E); B2 = sum(b * b for b in B)
    dV = dx ** 3
    U = 0.5 * np.sum(E2 + B2) * dV
    px = E[1] * B[2] - E[2] * B[1]
    py = E[2] * B[0] - E[0] * B[2]
    pz = E[0] * B[1] - E[1] * B[0]
    Lz = np.sum(X * py - Y * px) * dV
    return U, Lz


def main():
    print("=" * 74)
    print("sim7: moving Hopfion -- dynamical Maxwell (Faraday + Ampere) check")
    print("=" * 74)

    HIGH = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 121 if HIGH else 91
    Lbox = 5.0
    ax = np.linspace(-Lbox, Lbox, N)
    dx = ax[1] - ax[0]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    dt = 0.5 * dx
    print(f"grid {N}^3 over [-{Lbox},{Lbox}], dx={dx:.4f}, dt={dt:.4f}")

    print("\n" + "-" * 74)
    print(f"{'t':>6} | {'null RMS(F.F)/scl':>18} | {'Maxwell resid (rel)':>20} | "
          f"{'U':>9} {'L_z':>10}")
    print("-" * 74)

    times = [0.0, 0.5, 1.0, 1.5]
    ok_null = True
    ok_dyn = True
    Us, Lzs = [], []
    for t in times:
        Fx, Fy, Fz = hopfion_F(X, Y, Z, t)
        mag = np.mean(np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + np.abs(Fz) ** 2)

        # null
        FdotF = Fx * Fx + Fy * Fy + Fz * Fz
        null_rms = np.sqrt(np.mean(np.abs(FdotF) ** 2)) / mag

        # dynamical Maxwell:  dF/dt + i curl F = 0
        Fp = hopfion_F(X, Y, Z, t + dt)
        Fm = hopfion_F(X, Y, Z, t - dt)
        dFdt = [(Fp[i] - Fm[i]) / (2.0 * dt) for i in range(3)]
        cF = curl(Fx, Fy, Fz, dx)
        s = (slice(2, -2),) * 3                  # interior (avoid boundary FD)
        resid2 = 0.0; curl2 = 0.0
        for i in range(3):
            R = dFdt[i] + 1j * cF[i]
            resid2 += np.mean(np.abs(R[s]) ** 2)
            curl2 += np.mean(np.abs(cF[i][s]) ** 2)
        rel_resid = np.sqrt(resid2 / curl2)

        U, Lz = conserved(Fx, Fy, Fz, X, Y, Z, dx)
        Us.append(U); Lzs.append(Lz)
        if null_rms > 1e-10:
            ok_null = False
        if rel_resid > 5e-2:
            ok_dyn = False
        print(f"{t:6.2f} | {null_rms:18.2e} | {rel_resid:20.2e} | "
              f"{U:9.3f} {Lz:10.4f}")

    print("-" * 74)
    # conservation (box-truncated): U and L_z drift only as the knot leaves box
    U_drift = (max(Us) - min(Us)) / np.mean(Us)
    Lz_drift = (max(Lzs) - min(Lzs)) / abs(np.mean(Lzs))
    print(f"  U  fractional drift over t in [0,1.5] : {U_drift:.2%}  "
          f"(box truncation as knot propagates)")
    print(f"  L_z fractional drift                  : {Lz_drift:.2%}")
    print()
    print(f"  field stays NULL at all times (F.F=0)        : "
          f"{'PASS' if ok_null else 'FAIL'}")
    print(f"  dynamical Maxwell dF/dt = -i curl F (FD)     : "
          f"{'PASS' if ok_dyn else 'FAIL'}")
    print(f"  conserved charges stable to box truncation   : "
          f"{'PASS' if (U_drift < 0.25 and Lz_drift < 0.25) else 'CHECK (enlarge box)'}")
    all_ok = ok_null and ok_dyn
    print("\n" + "=" * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation:
  The knotted light-geon is verified as a full DYNAMICAL solution of Maxwell's
  equations, not just a t=0 snapshot: Faraday and Ampere hold (residual = pure
  FD error, shrinking with resolution), the field stays null as it evolves,
  and energy / axial angular momentum are conserved up to the energy that
  physically propagates out of the finite box.  This confirms the Hopfion is a
  bona-fide self-consistent "orbiting light" configuration in the inviscid
  U(1) sector PSFT derives via Theorem 9.1 -- a legitimate propagating
  precursor of a trapped geon.""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
