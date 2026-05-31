"""sim8 -- Half-integer spin from a topological charge: the charge+monopole
field angular momentum (the missing ingredient flagged by sim5).

sim5 showed a bosonic Maxwell field (Hopfion) carries INTEGER angular
momentum, and that the electron's half-integer spin needs extra structure.
This script supplies it -- using only classical electromagnetism plus a
TOPOLOGICAL charge, exactly the kind PSFT postulates (Postulate 4: electric
charge = winding number; charges are topological invariants of the flow).

Classic result (Thomson 1904; Saha 1936; "spin from isospin", Jackiw-Rebbi):
the electromagnetic field of an electric charge q_e together with a magnetic
monopole q_m carries angular momentum

    L = (q_e q_m / 4 pi) z_hat      (Heaviside-Lorentz units, c=1)

directed along the axis joining them and -- remarkably -- INDEPENDENT of their
separation.  Dirac's quantization condition q_e q_m = 2 pi n (HL units, hbar=1)
then forces

    L_z = n / 2   (in units hbar) ,    n in Z   ->   HALF-INTEGER spin.

So a purely bosonic field acquires spin-1/2 the moment a topological
(monopole / winding) charge is present.  We verify the field-integral formula
numerically (value, axiality, separation-independence) and then apply Dirac
quantization to land on hbar/2.

Fields (HL units, eps0 = mu0 = c = 1):
    E = (q_e/4pi) (r - r_e)/|r - r_e|^3      (electric charge at r_e)
    B = (q_m/4pi) (r - r_m)/|r - r_m|^3      (monopole at r_m)
    L = integral  r x (E x B)  d^3x

Run:  python3 sim8_charge_monopole_half_integer_spin.py
"""
import os
import sys
import numpy as np

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    sys.path.insert(0, _SIM)
    from psft.core.constants import SI
    print("[psft.core.constants available; Heaviside-Lorentz units c=eps0=1]")
except Exception:                                            # pragma: no cover
    print("[fallback; Heaviside-Lorentz units c=eps0=1]")


def coulomb_field(X, Y, Z, q, pos):
    """E (or B) of a point charge q at pos = (x0,y0,z0). HL units (q/4pi)."""
    dx = X - pos[0]; dy = Y - pos[1]; dz = Z - pos[2]
    r2 = dx * dx + dy * dy + dz * dz
    r3 = np.power(r2, 1.5)
    k = q / (4.0 * np.pi)
    return k * dx / r3, k * dy / r3, k * dz / r3


def field_angular_momentum(Lbox, a, dx=0.2, q_e=1.0, q_m=1.0):
    """Compute L = integral r x (E x B) on a grid of spacing dx over the box
    [-Lbox, Lbox]^3, charge at +a z, monopole at -a z.  Singular points placed
    at sub-cell offsets so no node coincides with them."""
    N = int(round(2 * Lbox / dx)) + 1
    ax = np.linspace(-Lbox, Lbox, N)
    dx = ax[1] - ax[0]
    # half-cell shift keeps the two singular points off every grid node
    X, Y, Z = np.meshgrid(ax + 0.5 * dx, ax + 0.37 * dx, ax + 0.21 * dx,
                          indexing="ij")
    Ex, Ey, Ez = coulomb_field(X, Y, Z, q_e, (0.0, 0.0, +a))
    Bx, By, Bz = coulomb_field(X, Y, Z, q_m, (0.0, 0.0, -a))
    # momentum density g = E x B
    gx = Ey * Bz - Ez * By
    gy = Ez * Bx - Ex * Bz
    gz = Ex * By - Ey * Bx
    dV = dx ** 3
    Lx = np.sum(Y * gz - Z * gy) * dV
    Ly = np.sum(Z * gx - X * gz) * dV
    Lz = np.sum(X * gy - Y * gx) * dV
    return np.array([Lx, Ly, Lz])


def richardson_Linf(a, boxes, dx=0.2):
    """The truncated integral obeys  L_z(R) = L_inf - C/R  (the field AM has a
    slow 1/r tail).  Fit that line in 1/R and return the extrapolated L_inf."""
    R = np.array(boxes, dtype=float)
    Lz = np.array([field_angular_momentum(Rb, a, dx=dx)[2] for Rb in boxes])
    # linear fit  Lz = L_inf - C*(1/R)
    A = np.vstack([np.ones_like(R), 1.0 / R]).T
    coef, *_ = np.linalg.lstsq(A, Lz, rcond=None)
    L_inf, negC = coef
    return L_inf, Lz


def main():
    print("=" * 74)
    print("sim8: half-integer spin from charge + topological (monopole) charge")
    print("=" * 74)

    dx = 0.15 if os.environ.get("PSFT_HIGH_RES", "0") != "0" else 0.2
    analytic = 1.0 / (4.0 * np.pi)          # q_e q_m / 4pi with q_e=q_m=1
    print(f"\ngrid spacing dx={dx}, q_e=q_m=1")
    print(f"analytic prediction  L_z = q_e q_m/(4 pi) = {analytic:.5f} "
          f"(independent of separation)")

    # ---- (1) magnitude via box-size convergence + Richardson --------------
    # The field AM has a slow 1/r tail, so a finite box undercounts it:
    #   L_z(R) = L_inf - C/R .  Fit the line in 1/R to recover L_inf.
    a_fix = 0.5
    boxes = [6.0, 9.0, 12.0, 18.0]
    print(f"\n-- (1) box-size convergence at fixed separation 2a={2*a_fix} --")
    print(f"{'box R':>8} {'L_z(R)':>12} {'|L_z|/(qq/4pi)':>16}")
    L_inf, Lz_R = richardson_Linf(a_fix, boxes, dx=dx)
    for R, lz in zip(boxes, Lz_R):
        print(f"{R:8.1f} {lz:12.6f} {abs(lz)/analytic:16.4f}")
    print(f"  Richardson L_inf (R->inf)  = {L_inf:12.6f}")
    print(f"  analytic   q_e q_m/(4 pi)  = {-analytic:12.6f}  "
          f"(sign: L points monopole->charge)")
    ratio = abs(L_inf) / analytic
    ok_value = abs(ratio - 1.0) < 0.03
    print(f"  |L_inf| / (q_e q_m/4pi)    = {ratio:.4f}  -> "
          f"{'PASS' if ok_value else 'CHECK'}")

    # ---- (2) axiality -----------------------------------------------------
    Lvec = field_angular_momentum(12.0, 1.0, dx=dx)
    axiality = abs(Lvec[2]) / (abs(Lvec[0]) + abs(Lvec[1]) + abs(Lvec[2]))
    ok_axial = axiality > 0.999
    print(f"\n-- (2) axiality: L = [{Lvec[0]:.2e}, {Lvec[1]:.2e}, {Lvec[2]:.5f}] "
          f"-> L_z fraction {axiality:.5f}  "
          f"{'PASS' if ok_axial else 'FAIL'}")

    # ---- (3) separation-INDEPENDENCE via tail model -----------------------
    # At fixed box, deficit = L_inf - L_z(2a) should grow linearly in 2a:
    #   deficit ~ k * (2a)/R .  A clean linear fit (high R^2) confirms the
    #   TRUE (untruncated) L_z is separation-independent.
    print("\n-- (3) separation-independence (fixed box, tail-deficit model) --")
    Rfix = 12.0
    seps = [0.5, 1.0, 1.5, 2.0, 3.0]
    Lz_sep = np.array([field_angular_momentum(Rfix, a, dx=dx)[2] for a in seps])
    twoa = np.array([2 * a for a in seps])
    # untruncated value at this box ~ L_inf from part 1; deficit:
    deficit = abs(L_inf) - np.abs(Lz_sep)
    # linear fit deficit = k*(2a)
    k, b = np.polyfit(twoa, deficit, 1)
    fit = k * twoa + b
    ss_res = np.sum((deficit - fit) ** 2)
    ss_tot = np.sum((deficit - deficit.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"{'2a':>6} {'L_z(2a)':>12} {'deficit':>10} {'fit k*(2a)+b':>14}")
    for ta, lz, df, ft in zip(twoa, Lz_sep, deficit, fit):
        print(f"{ta:6.1f} {lz:12.6f} {df:10.5f} {ft:14.5f}")
    print(f"  deficit ~ k*(2a):  k={k:.4f}, intercept b={b:.4f}, R^2={r2:.4f}")
    ok_indep = r2 > 0.99
    print(f"  => deficit is purely the truncated 1/R tail (linear in 2a, "
          f"R^2>0.99): {'PASS' if ok_indep else 'CHECK'}")
    print(f"     hence the TRUE L_z is separation-independent, as Saha/Thomson.")

    # ---- (2) Dirac quantization -> half-integer ---------------------------
    print("\n" + "-" * 74)
    print("  Dirac quantization (HL units, hbar=1):  q_e q_m = 2 pi n,  n in Z")
    print("  => L_z = (q_e q_m)/(4 pi) = (2 pi n)/(4 pi) = n/2   [units hbar]")
    print("  minimal monopole n=1:   L_z = 1/2  =  hbar/2   <-- HALF-INTEGER SPIN")
    print("-" * 74)

    print(f"\n  L_z = q_e q_m/(4 pi), Richardson-extrapolated      : "
          f"{'PASS' if ok_value else 'CHECK'}")
    print(f"  L purely axial                                     : "
          f"{'PASS' if ok_axial else 'FAIL'}")
    print(f"  L_z separation-independent (tail-model R^2>0.99)   : "
          f"{'PASS' if ok_indep else 'FAIL'}")
    all_ok = ok_axial and ok_indep and ok_value
    print("\n" + "=" * 74)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'CHECK'}")
    print("=" * 74)
    print("""
Interpretation -- this closes the gap sim5 left open:
  * A pure bosonic Maxwell field carries INTEGER angular momentum (sim5).
  * Add a TOPOLOGICAL (magnetic-monopole / winding) charge and the field
    angular momentum becomes q_e q_m/(4pi) -- separation-independent, axial,
    and HALF-INTEGER by Dirac quantization (L_z = n/2).
  * PSFT Postulate 4 says electric charge IS a topological winding number and
    colour/isospin are topological classes -- precisely the monopole-like
    structure needed.  So the half-integer spin of the electron geon is not an
    add-on: it is the field angular momentum sourced by the geon's own
    topological charge.  Spin-1/2 emerges from topology, in a bosonic
    photonic field -- fully consistent with "matter = solitonic (topological)
    pattern of the primitive photonic field" (Postulate 1).
  * Caveat: this is the standard charge-monopole (Saha/Thomson) result imported
    into the PSFT setting; a first-principles PSFT soliton that literally
    carries this winding and reproduces e, m_e remains the open mass-spectrum
    problem (paper Sec 17).""")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
