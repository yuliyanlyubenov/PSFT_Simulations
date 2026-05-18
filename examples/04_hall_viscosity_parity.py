"""Example 4: Hall viscosity is parity-odd (Theorem 11.1, proof of (i)).

This example exercises the v2 Hall stress (paper eq. 6.7 + Remark 6.2):

    tau^Hall_{ab} = eta_odd * eps_{ab}^{cd} u_c (D_d sigma^A)

where sigma^A is the gauge-algebra **scalar** order parameter (Higgs-like
VEV), distinct from the gauge-sector shear tensor sigma^A_{ab}.

Two demonstrations:

  Part A -- algebraic: show the contraction
            eps_{ab}^{cd} u_c (D_d sigma) -> antisymmetric rank-2 tensor
            is parity-odd, while the standard viscous stress is parity-even.

  Part B -- numerical: confirm the Hall stress is non-vanishing on a
            non-trivial sigma^A(x), and that the algebraic form
            eps^{bcde} u_c sigma^A_{de} (which appeared in earlier paper
            deposits) vanishes identically against any symmetric shear.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from itertools import permutations


def parity_vec(v):
    out = v.copy(); out[1:] = -out[1:]; return out


def parity_tensor2(T):
    out = T.copy()
    for a in range(4):
        for b in range(4):
            spatial = int(a > 0) + int(b > 0)
            if spatial % 2 == 1:
                out[a, b] = -T[a, b]
    return out


def main():
    eps = np.zeros((4, 4, 4, 4))
    for perm in permutations((0, 1, 2, 3)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        eps[perm] = 1.0 if inv % 2 == 0 else -1.0

    # Comoving observer.
    u = np.array([1.0, 0.0, 0.0, 0.0])
    # Use the consistent Hall stress form from paper eq. 6.7:
    #   tau^Hall_{ab} = eps_{ab}^{cd} u_c J_d        (J_d = D_d sigma^A here a 4-current)
    J = np.array([0.0, 0.3, 0.5, -0.2])

    tau_hall = np.einsum("abcd,c,d->ab", eps, u, J)

    # Plain (parity-even) viscous stress for comparison.
    sigma = np.zeros((4, 4))
    sigma[1, 2] = sigma[2, 1] = 0.5
    sigma[1, 1] = 1.0; sigma[2, 2] = -0.5; sigma[3, 3] = -0.5
    tau_visc = 2.0 * 0.1 * sigma

    # Apply parity to inputs and re-evaluate the integrand.
    u_P = parity_vec(u)
    J_P = parity_vec(J)
    sigma_P = parity_tensor2(sigma)
    tau_hall_from_P = np.einsum("abcd,c,d->ab", eps, u_P, J_P)
    tau_visc_from_P = 2.0 * 0.1 * sigma_P

    # Expected transformations:
    #   Hall stress is parity-odd:    tau_hall(Px) = -P tau_hall(x).
    #   Standard stress is parity-even: tau_visc(Px) =  P tau_visc(x).
    P_tau_hall = parity_tensor2(tau_hall)
    P_tau_visc = parity_tensor2(tau_visc)

    diff_hall = np.linalg.norm(tau_hall_from_P + P_tau_hall)
    diff_visc = np.linalg.norm(tau_visc_from_P - P_tau_visc)

    print("Part A -- algebraic check of parity transformation:")
    print("---------------------------------------------------")
    print("Hall stress sample (lower indices):")
    print(tau_hall)
    print(f"\n||tau_hall(P x) + P[tau_hall(x)]||  = {diff_hall:.3e}  (~ 0 -> parity-ODD)")
    print(f"||tau_visc(P x) - P[tau_visc(x)]||  = {diff_visc:.3e}  (~ 0 -> parity-EVEN)")
    print("\n=> Hall viscosity flips sign under parity (Theorem 11.1(i)).")

    # ------------------------------------------------------------------
    print("\nPart B -- Hall stress is non-vanishing on a non-trivial sigma^A:")
    print("----------------------------------------------------------------")
    from psft.core.metric import MinkowskiMetric
    from psft.core.curvature import CurvatureBundle
    from psft.sectors.master_eq import _levi_civita_at

    m = MinkowskiMetric()
    sigma_A = lambda xx: 0.3 * xx[1]**2 + 0.5 * xx[2] * xx[3] - 0.2 * xx[3]**2
    x0 = np.array([0.0, 0.1, 0.2, 0.3])
    u_up = np.array([1.0, 0.0, 0.0, 0.0])

    h_step = 1e-4
    curv = CurvatureBundle.from_metric(m, x0)
    eps_lower = _levi_civita_at(curv.g)
    d_sigma_dn = np.zeros(4)
    for d in range(4):
        xp = x0.copy(); xp[d] += h_step
        xm = x0.copy(); xm[d] -= h_step
        d_sigma_dn[d] = (sigma_A(xp) - sigma_A(xm)) / (2 * h_step)
    d_sigma_up = curv.g_inv @ d_sigma_dn
    eta_odd = 0.5
    tau_hall = eta_odd * np.einsum("abmn,m,n->ab", eps_lower, u_up, d_sigma_up)

    print("  tau^Hall_{ab} at x =", x0, ":")
    print(tau_hall)
    print(f"\n  ||tau^Hall||  = {np.linalg.norm(tau_hall):.3e}   (NON-zero)")
    print(f"  asymmetry check: ||tau + tau^T|| = {np.linalg.norm(tau_hall + tau_hall.T):.3e}   (~ 0 -> manifestly antisymmetric)")

    # Compare against the algebraic form eps^{bcde} u_c sigma^A_{de} that
    # appeared in earlier paper deposits -- it vanishes for any symmetric
    # shear, as required by the (d,e) anti-symmetry of eps and (d,e)
    # symmetry of sigma.
    sigma_symmetric = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 0.5, -0.5, 0.0],
        [0.0, 0.0, 0.0, -0.5],
    ])
    legacy_attempt = np.einsum("bcde,c,de->b", eps_lower, u_up, sigma_symmetric)
    print(f"\n  legacy contraction eps^{{bcde}} u_c sigma_{{de}} = {legacy_attempt}")
    print(f"  ||legacy attempt|| = {np.linalg.norm(legacy_attempt):.3e}   (identically 0)")

    print("""
  Notes:
    * The Hall *stress* is the meaningful tensor and it is manifestly
      non-zero and antisymmetric.
    * The Hall *4-force* h_a^b D_c tau^{Hall,c}_b in flat Minkowski with
      a constant comoving u^c collapses to eps^c_b^{de} u_d (D_c D_e sigma),
      which vanishes by Schwarz symmetry of mixed partials (D_c and D_e
      commute on a scalar).  Non-trivial Hall force generation requires
      curvature or a non-constant u^c -- precisely the regime
      K > K_w^c on the LHS of Theorem 11.1 of the paper.""")


if __name__ == "__main__":
    main()
