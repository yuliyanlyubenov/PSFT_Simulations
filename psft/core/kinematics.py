"""Kinematic decomposition of a 4-velocity field u^a (Definition 3.2).

For a timelike unit field u^a (u^a u_a = -1):

    nabla_a u_b = sigma_ab + omega_ab + (1/3) theta h_ab - u_a a_b

with
    h_{ab} = g_{ab} + u_a u_b       (spatial projector)
    sigma_{ab} = nabla_(a u_b) + u_(a a_b) - (1/3) theta h_{ab}
    omega_{ab} = nabla_[a u_b] + u_[a a_b]
    theta = nabla_a u^a
    a_b = u^c nabla_c u_b

Pointwise interface: takes a CurvatureBundle and a function u_func(x) returning
the 4-velocity at neighbouring points (for finite-difference nabla u).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from psft.core.metric import Metric


@dataclass
class KinematicDecomposition:
    u_up: np.ndarray            # u^a (4,)
    u_dn: np.ndarray            # u_a (4,)
    grad_u_dn: np.ndarray       # nabla_a u_b  (4,4) = d_a u_b - Gamma^c_{ab} u_c
    sigma: np.ndarray           # shear (4,4) symmetric trace-free spatial
    omega: np.ndarray           # vorticity (4,4) antisymmetric spatial
    theta: float                # expansion scalar
    a_dn: np.ndarray            # 4-acceleration a_b
    h: np.ndarray               # spatial projector h_{ab}

    @classmethod
    def from_velocity(
        cls,
        metric: Metric,
        x: np.ndarray,
        u_func: Callable[[np.ndarray], np.ndarray],
        Gamma: np.ndarray,
        h_step: float = 1e-5,
    ) -> "KinematicDecomposition":
        g = metric.g(x)
        u_up = u_func(x).astype(float)
        u_dn = g @ u_up
        # Spatial projector h_{ab} = g_{ab} + u_a u_b
        h_proj = g + np.einsum("a,b->ab", u_dn, u_dn)

        # nabla_a u_b = d_a u_b - Gamma^c_{ab} u_c
        # First we need d_a u_b on the lowered field.
        # Compute u_dn at neighbouring points via finite difference.
        du_dn = np.zeros((4, 4))  # du_dn[a, b] = d_a u_b
        for a in range(4):
            xp = x.copy(); xp[a] += h_step
            xm = x.copy(); xm[a] -= h_step
            gp = metric.g(xp); gm = metric.g(xm)
            up_dn = gp @ u_func(xp).astype(float)
            um_dn = gm @ u_func(xm).astype(float)
            du_dn[a] = (up_dn - um_dn) / (2 * h_step)

        grad_u_dn = du_dn - np.einsum("cab,c->ab", Gamma, u_dn)

        # 4-acceleration a_b = u^c nabla_c u_b
        a_dn = np.einsum("c,cb->b", u_up, grad_u_dn)

        # Expansion theta = nabla_a u^a = g^{ab} nabla_a u_b
        g_inv = np.linalg.inv(g)
        theta = float(np.einsum("ab,ab->", g_inv, grad_u_dn))

        # Symmetric / antisymmetric parts of nabla_(a u_b) and nabla_[a u_b]
        sym = 0.5 * (grad_u_dn + grad_u_dn.T)
        antisym = 0.5 * (grad_u_dn - grad_u_dn.T)

        # Add u_(a a_b) and u_[a a_b]
        u_a_sym = 0.5 * (np.einsum("a,b->ab", u_dn, a_dn) + np.einsum("a,b->ab", a_dn, u_dn))
        u_a_anti = 0.5 * (np.einsum("a,b->ab", u_dn, a_dn) - np.einsum("a,b->ab", a_dn, u_dn))

        sigma = sym + u_a_sym - (theta / 3.0) * h_proj
        omega = antisym + u_a_anti

        return cls(
            u_up=u_up, u_dn=u_dn,
            grad_u_dn=grad_u_dn,
            sigma=sigma, omega=omega,
            theta=theta, a_dn=a_dn, h=h_proj,
        )

    def is_unit_timelike(self, tol: float = 1e-8) -> bool:
        return abs(float(self.u_up @ self.u_dn) + 1.0) < tol

    def shear_norm_squared(self) -> float:
        """sigma^{ab} sigma_{ab} -- the viscous heating coefficient."""
        return float(np.sum(self.sigma * self.sigma))
