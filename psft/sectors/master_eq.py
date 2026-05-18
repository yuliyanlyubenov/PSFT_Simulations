"""PSFT master equations (v1 and v2).

We evaluate the RHS of the v2 master equation pointwise:

  (rho_g + p_g) u^b nabla_b u_a^A
      = - h_a^b nabla_b p_g            * delta^A_0
      + h_a^b D_c [ 2 eta^A_B(K) Theta(K-Kc^A) sigma^{cB}_b ]
      + h_a^b [ eta_odd^A(K) eps_b^{cde} u_c sigma_{de}^A ] Theta(K - K_w)
      + (8 pi G / c^4) h_a^b nabla_c P^c_b   * delta^A_0

The gauge-sector index A runs over 0..(dim-1).  Sector A=0 is gravity (no gauge
algebra, gets pressure + photonic force); A>=1 are gauge-covariant.

This module returns the symbolic / numerical RHS at a single point.  Field
evolution on a grid is handled by `psft.evolve.integrators`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np

from psft.core.constants import PSFTConstants
from psft.core.curvature import CurvatureBundle
from psft.core.kinematics import KinematicDecomposition
from psft.core.metric import Metric
from psft.core.photonic import PhotonicSource
from psft.gauge.algebra import GaugeAlgebra
from psft.sectors.viscosity import GaugeViscosity, HallViscosity


@dataclass
class MasterEquationResult:
    inertia: np.ndarray         # LHS: (rho_g+p_g) u^b nabla_b u_a    (4,)
    pressure_force: np.ndarray  # h_a^b nabla_b p_g                    (4,)
    viscous_force: np.ndarray   # h_a^b D_c tau^{cb}                   (4,)
    hall_force: np.ndarray      # parity-odd term                      (4,)
    photonic_force: np.ndarray  # photonic 4-force                     (4,)
    rhs: np.ndarray             # sum of RHS terms                     (4,)
    residual: np.ndarray        # inertia - rhs (should -> 0 on-shell) (4,)


@dataclass
class MasterEquationV2:
    """Builder/evaluator for the v2 PSFT master equation."""
    metric: Metric
    consts: PSFTConstants
    gauge_viscosity: GaugeViscosity
    hall_viscosity: Optional[HallViscosity] = None
    gauge_algebra: Optional[GaugeAlgebra] = None
    G_eff: float = None                          # default uses consts.G
    Lambda: float = 0.0
    sector: str = "gravity"                      # which gauge sector A is being evolved

    def __post_init__(self):
        if self.G_eff is None:
            self.G_eff = self.consts.G

    def evaluate(
        self,
        x: np.ndarray,
        u_func: Callable[[np.ndarray], np.ndarray],
        rho_g: float,
        p_g: float,
        grad_p_g: np.ndarray,
        photonic: Optional[PhotonicSource] = None,
        regenerate_photonic: Optional[Callable] = None,
        sigma_A_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> MasterEquationResult:
        """Evaluate the v2 master equation RHS at point x.

        `sigma_A_func(x) -> float` is the gauge-algebra scalar order parameter
        sigma^A (the Higgs-like vacuum-expectation field; see Remark 6.2 of
        the paper).  It is required only when the Hall term is active
        (sector="weak", K > K_w, hall_viscosity set).  D_d sigma^A is
        computed by central finite differences -- in flat gauge, this
        reduces to the partial derivative.
        """
        curv = CurvatureBundle.from_metric(self.metric, x)
        kin = KinematicDecomposition.from_velocity(
            self.metric, x, u_func, Gamma=curv.Gamma
        )

        # Inertia: (rho+p) u^b nabla_b u_a
        # nabla_b u_a = grad_u_dn (indexed [b,a])
        inertia = (rho_g + p_g) * np.einsum("b,ba->a", kin.u_up, kin.grad_u_dn)

        # Pressure force (gravity sector only).
        pressure_force = np.zeros(4)
        photonic_force = np.zeros(4)
        if self.sector == "gravity":
            pressure_force = -kin.h @ grad_p_g
            if photonic is not None:
                div = photonic.divergence(
                    self.metric, x, regenerate=regenerate_photonic,
                )
                photonic_force = (8 * np.pi * self.G_eff / self.consts.c**4) * (kin.h @ div)

        # Viscous force: D_c (2 eta sigma^c_b) projected by h_a^b.
        K = curv.K_scalar
        eta_val = self.gauge_viscosity.eta(self.sector, K)
        viscous_force = np.zeros(4)
        if eta_val != 0.0:
            # We need nabla_c (2 eta sigma^{cb}) ~ d_c on a single point with
            # gradient over surrounding samples; for the pointwise evaluator
            # we approximate via the trace of nabla_c sigma^{c}_b at x.  For a
            # static configuration we use the divergence of the spatial part
            # of sigma^{c}_b, which on Minkowski is just d_c sigma^{c}_b.
            div_sigma = self._divergence_sigma(x, u_func, h=1e-4)
            viscous_force = 2.0 * eta_val * (kin.h @ div_sigma)

        # Hall viscosity term (parity-odd) -- paper Sec. 6, eq. 6.7 +
        # Remark 6.2: tau^Hall_{ab} = eta_odd eps_{ab}^{cd} u_c (D_d sigma^A),
        # entering the master equation via h_a^b D_c tau^{Hall,c}_b.  Here
        # sigma^A is the gauge-algebra scalar order parameter, distinct
        # from the gauge-sector shear tensor sigma^A_{ab}.
        hall_force = np.zeros(4)
        if self.hall_viscosity is not None and self.sector == "weak":
            coeff = self.hall_viscosity.coeff(K)
            if coeff != 0.0:
                if sigma_A_func is None:
                    raise ValueError(
                        "Hall term active but no sigma_A_func supplied. "
                        "The Hall stress is built from the gauge-algebra "
                        "scalar sigma^A (not the shear tensor); please "
                        "pass sigma_A_func(x) -> float to evaluate."
                    )
                hall_force = self._hall_force(
                    x, u_func, sigma_A_func, coeff, h_step=1e-4,
                )

        rhs = pressure_force + viscous_force + hall_force + photonic_force
        residual = inertia - rhs
        return MasterEquationResult(
            inertia=inertia,
            pressure_force=pressure_force,
            viscous_force=viscous_force,
            hall_force=hall_force,
            photonic_force=photonic_force,
            rhs=rhs,
            residual=residual,
        )

    def _divergence_sigma(self, x, u_func, h=1e-4) -> np.ndarray:
        """nabla_c sigma^c_b at x via central differences of the kinematic decomp."""
        out = np.zeros(4)
        for c in range(4):
            xp = x.copy(); xp[c] += h
            xm = x.copy(); xm[c] -= h
            curv_p = CurvatureBundle.from_metric(self.metric, xp)
            curv_m = CurvatureBundle.from_metric(self.metric, xm)
            kp = KinematicDecomposition.from_velocity(
                self.metric, xp, u_func, Gamma=curv_p.Gamma
            )
            km = KinematicDecomposition.from_velocity(
                self.metric, xm, u_func, Gamma=curv_m.Gamma
            )
            sig_p = curv_p.g_inv @ kp.sigma     # sigma^c_b at xp
            sig_m = curv_m.g_inv @ km.sigma
            out += (sig_p[c] - sig_m[c]) / (2 * h)
        return out

    def _hall_force(self, x, u_func, sigma_A_func, coeff, h_step=1e-4) -> np.ndarray:
        """Hall 4-force: h_a^b D_c tau^{Hall,c}_b.

        With tau^{Hall}_{ab} = coeff * eps_{ab}^{cd} u_c (D_d sigma^A) and
        flat-gauge D_d -> partial_d, we form tau^{Hall,c}_b = g^{ca} tau^{Hall}_{ab}
        on a 3-point stencil and finite-difference its c-divergence.
        """
        def hall_stress_mixed(xx):
            curv = CurvatureBundle.from_metric(self.metric, xx)
            u_up = u_func(xx).astype(float)
            eps_lower = _levi_civita_at(curv.g)
            # D_d sigma^A via central differences (lower index by default).
            d_sigma_dn = np.zeros(4)
            for d in range(4):
                xp = xx.copy(); xp[d] += h_step
                xm = xx.copy(); xm[d] -= h_step
                d_sigma_dn[d] = (sigma_A_func(xp) - sigma_A_func(xm)) / (2 * h_step)
            d_sigma_up = curv.g_inv @ d_sigma_dn
            # tau_{ab} = coeff * eps_{ab}^{cd} u_c (D_d sigma^A)
            #          = coeff * eps_{abmn} u^m (D^n sigma^A)
            tau_dn = coeff * np.einsum("abmn,m,n->ab", eps_lower, u_up, d_sigma_up)
            # raise the first index: tau^c_b = g^{ca} tau_{ab}
            return curv.g_inv @ tau_dn

        # Spatial projector h_a^b at x (evaluate once).
        curv0 = CurvatureBundle.from_metric(self.metric, x)
        u0 = u_func(x).astype(float)
        u0_dn = curv0.g @ u0
        h_proj = curv0.g + np.einsum("a,b->ab", u0_dn, u0_dn)

        # D_c tau^{Hall,c}_b via central differences of the c-index.
        out = np.zeros(4)
        for c in range(4):
            xp = x.copy(); xp[c] += h_step
            xm = x.copy(); xm[c] -= h_step
            tau_p = hall_stress_mixed(xp)
            tau_m = hall_stress_mixed(xm)
            out += (tau_p[c, :] - tau_m[c, :]) / (2 * h_step)
        return h_proj @ out


def _levi_civita_at(g: np.ndarray) -> np.ndarray:
    """Levi-Civita tensor eps_{abcd} (fully lower) = sqrt(-det g) * eps_symbol.

    Used by the Hall force computation, which needs eps_{ab}^{cd}; callers
    raise the last two indices via einsum with g^{ce} g^{df} as needed.
    """
    eps = np.zeros((4, 4, 4, 4))
    from itertools import permutations
    for perm in permutations((0, 1, 2, 3)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        eps[perm] = 1.0 if inv % 2 == 0 else -1.0
    det = float(np.linalg.det(g))
    return eps * np.sqrt(max(-det, 1e-300))


@dataclass
class MasterEquationV1:
    """Legacy v1 master equation: Gaussian viscosity, scalar bulk term.

    Provided mainly for cross-checks against the v2 reductions.
    """
    metric: Metric
    consts: PSFTConstants
    eta_func: Callable[[float], float]
    zeta_func: Callable[[float], float]

    def evaluate(self, x, u_func, rho_g, p_g, grad_p_g, l_eff=None):
        curv = CurvatureBundle.from_metric(self.metric, x)
        kin = KinematicDecomposition.from_velocity(self.metric, x, u_func, Gamma=curv.Gamma)
        if l_eff is None:
            K = curv.K_scalar
            l_eff = K ** (-0.25) if K > 0 else float("inf")
        eta = float(self.eta_func(l_eff))
        zeta = float(self.zeta_func(l_eff))

        inertia = (rho_g + p_g) * np.einsum("b,ba->a", kin.u_up, kin.grad_u_dn)
        pressure_force = -kin.h @ grad_p_g
        # Viscous shear (eta sigma) -- evaluated as divergence in a follow-up step.
        # For demonstration return the algebraic ingredients.
        sigma_up = curv.g_inv @ kin.sigma
        viscous_alg = 2 * eta * sigma_up + zeta * kin.theta * kin.h
        return {
            "inertia": inertia,
            "pressure_force": pressure_force,
            "viscous_alg": viscous_alg,
            "shear": kin.sigma,
            "vorticity": kin.omega,
            "theta": kin.theta,
            "K": curv.K_scalar,
            "eta": eta,
            "zeta": zeta,
        }
