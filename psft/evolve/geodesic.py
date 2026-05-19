"""Geodesic time evolution -- the inviscid limit of the PSFT master equation.

When the master equation's RHS vanishes (no pressure gradient, no viscous
force, no Hall force, no photonic source -- i.e. a "test fluid" of vanishing
density on a fixed background), the master equation
    (rho_g + p_g) u^b nabla_b u_a^A = 0
reduces to the geodesic equation
    u^b nabla_b u^a = 0,   or equivalently   d^2 x^a/dtau^2 + Gamma^a_{bc} dx^b/dtau dx^c/dtau = 0.

This module integrates the geodesic equation on any Metric using RK4.
It is the simplest non-trivial time-evolution of a master-equation solution
and serves as a foundation for the field-level evolver that will follow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np

from psft.core.metric import Metric
from psft.core.curvature import _christoffel_at


@dataclass
class GeodesicState:
    """Position and 4-velocity of a test particle at a single proper-time slice."""
    tau: float                # proper time
    x: np.ndarray             # spacetime position (4,)
    u: np.ndarray             # 4-velocity (4,)

    def copy(self) -> "GeodesicState":
        return GeodesicState(tau=self.tau, x=self.x.copy(), u=self.u.copy())


def geodesic_rhs(metric: Metric, x: np.ndarray, u: np.ndarray,
                 fd_step: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Right-hand side of the geodesic equation:
        dx^a / dtau = u^a
        du^a / dtau = -Gamma^a_{bc}(x) u^b u^c
    Returns (dx/dtau, du/dtau).
    """
    Gamma = _christoffel_at(metric, x, fd_step)            # shape (4,4,4)
    dudtau = -np.einsum("abc,b,c->a", Gamma, u, u)
    return u.copy(), dudtau


@dataclass
class GeodesicEvolver:
    """RK4-integrated geodesic on a fixed metric background.

    `metric` is held fixed (we do NOT solve Einstein's equations here);
    `fd_step` controls the finite-difference accuracy of the Christoffel
    computation at each step.

    The evolver enforces the normalisation constraint u^a u_a = -1
    (timelike unit vector) by projection after every step, which keeps
    the trajectory on the true mass-shell despite numerical drift.
    """
    metric: Metric
    dt: float = 0.1                         # proper-time step
    fd_step: float = 1e-3                   # for Christoffel FD
    enforce_normalisation: bool = True

    def step(self, state: GeodesicState) -> GeodesicState:
        h = self.dt
        f = lambda x, u: geodesic_rhs(self.metric, x, u, self.fd_step)

        k1_x, k1_u = f(state.x, state.u)
        k2_x, k2_u = f(state.x + 0.5 * h * k1_x, state.u + 0.5 * h * k1_u)
        k3_x, k3_u = f(state.x + 0.5 * h * k2_x, state.u + 0.5 * h * k2_u)
        k4_x, k4_u = f(state.x + h * k3_x, state.u + h * k3_u)

        new_x = state.x + (h / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        new_u = state.u + (h / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

        if self.enforce_normalisation:
            new_u = self._renormalise(new_x, new_u)

        return GeodesicState(tau=state.tau + h, x=new_x, u=new_u)

    def _renormalise(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Project u onto the constraint surface u^a u_a = -1.

        We multiply the timelike component by the factor needed to bring
        the norm back to -1.  This is a minimal-disturbance correction.
        """
        g = self.metric.g(x)
        norm = float(np.einsum("ab,a,b->", g, u, u))
        if norm >= -1e-14:
            return u
        target = -1.0
        scale = float(np.sqrt(target / norm))
        return scale * u

    def run(self, state0: GeodesicState, n_steps: int) -> List[GeodesicState]:
        states = [state0.copy()]
        s = state0.copy()
        for _ in range(n_steps):
            s = self.step(s)
            states.append(s)
        return states

    def trajectory(self, state0: GeodesicState, n_steps: int) -> np.ndarray:
        """Return positions only, shape (n_steps+1, 4)."""
        return np.array([s.x for s in self.run(state0, n_steps)])
