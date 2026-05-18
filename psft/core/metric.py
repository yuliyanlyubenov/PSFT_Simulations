"""Spacetime metric classes.

Each Metric provides g_ab and g^{ab} at a coordinate x = (t,x,y,z), and
optionally analytic partial derivatives d_c g_ab.  When analytic derivatives
are absent the CurvatureBundle falls back to finite differences.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np

_ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


class Metric(ABC):
    name: str = "metric"

    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """g_{ab} at coordinate x; x has shape (4,) or (4, *grid)."""

    def g_inv(self, x: np.ndarray) -> np.ndarray:
        g = self.g(x)
        if g.ndim == 2:
            return np.linalg.inv(g)
        # Move (4,4) to the back, invert, move back.
        gm = np.moveaxis(g, [0, 1], [-2, -1])
        inv = np.linalg.inv(gm)
        return np.moveaxis(inv, [-2, -1], [0, 1])

    def dg(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """d_c g_{ab} via finite differences (override for analytic forms).

        Returns shape (4,4,4) at a single point or (4,4,4,*grid) on a grid.
        Axis order: (c, a, b) for the derivative slot c.
        """
        scalar_point = x.ndim == 1
        if scalar_point:
            out = np.zeros((4, 4, 4))
            for c in range(4):
                xp = x.copy(); xp[c] += h
                xm = x.copy(); xm[c] -= h
                out[c] = (self.g(xp) - self.g(xm)) / (2 * h)
            return out
        # Grid case: only valid for time derivative pointwise; spatial via
        # numpy.gradient on (Nx,Ny,Nz) axes assumed uniform.
        raise NotImplementedError(
            "Grid finite-difference dg should use manifold gradient + analytic "
            "time derivative; override dg() for stationary backgrounds."
        )


class MinkowskiMetric(Metric):
    name = "minkowski"

    def g(self, x):
        if x.ndim == 1:
            return _ETA.copy()
        out = np.broadcast_to(_ETA[..., None, None, None], (4, 4) + x.shape[1:])
        return np.array(out)

    def g_inv(self, x):
        return self.g(x)

    def dg(self, x, h=1e-5):
        if x.ndim == 1:
            return np.zeros((4, 4, 4))
        return np.zeros((4, 4, 4) + x.shape[1:])


class SchwarzschildMetric(Metric):
    """Schwarzschild in isotropic Cartesian coordinates (rho = sqrt(x^2+y^2+z^2)).

        ds^2 = -A(rho)^2 dt^2 + B(rho)^2 (dx^2 + dy^2 + dz^2)
        A(rho) = (1 - GM/2c^2 rho) / (1 + GM/2c^2 rho)
        B(rho) = (1 + GM/2c^2 rho)^2

    The Schwarzschild areal radius is r = rho B(rho).  Curvature invariants
    (e.g. Kretschmann K = 48 (GM)^2/r^6) are exact in this representation.
    """
    name = "schwarzschild"

    def __init__(self, M: float, G: float = 6.67430e-11, c: float = 2.99792458e8):
        self.M = M
        self.G = G
        self.c = c
        self.GM_c2 = G * M / c**2  # length scale; half-rs

    def _rho(self, x):
        if x.ndim == 1:
            return float(np.sqrt(x[1]**2 + x[2]**2 + x[3]**2))
        return np.sqrt(x[1]**2 + x[2]**2 + x[3]**2)

    def g(self, x):
        rho = self._rho(x)
        # Guard against rho = 0 (singular at origin).
        rho_safe = np.maximum(rho, self.GM_c2 / 2.0 + 1e-30)
        h = self.GM_c2 / (2.0 * rho_safe)
        A = (1.0 - h) / (1.0 + h)
        B2 = (1.0 + h)**4
        if x.ndim == 1:
            g = np.eye(4)
            g[0, 0] = -A * A
            g[1, 1] = B2
            g[2, 2] = B2
            g[3, 3] = B2
            return g
        grid_shape = x.shape[1:]
        g = np.zeros((4, 4) + grid_shape)
        g[0, 0] = -A * A
        g[1, 1] = B2
        g[2, 2] = B2
        g[3, 3] = B2
        return g

    def areal_radius(self, x):
        rho = self._rho(x)
        rho_safe = np.maximum(rho, self.GM_c2 / 2.0 + 1e-30)
        h = self.GM_c2 / (2.0 * rho_safe)
        return rho_safe * (1.0 + h)**2


class FLRWMetric(Metric):
    """Spatially flat FLRW: ds^2 = -dt^2 + a(t)^2 (dx^2+dy^2+dz^2)."""
    name = "flrw"

    def __init__(self, a_of_t: Callable[[float], float]):
        self.a_of_t = a_of_t

    def g(self, x):
        if x.ndim == 1:
            a = float(self.a_of_t(x[0]))
            g = np.diag([-1.0, a*a, a*a, a*a])
            return g
        t = x[0]
        a = self.a_of_t(t)
        grid_shape = x.shape[1:]
        g = np.zeros((4, 4) + grid_shape)
        g[0, 0] = -1.0
        g[1, 1] = a * a
        g[2, 2] = a * a
        g[3, 3] = a * a
        return g


class FunctionalMetric(Metric):
    """User-supplied callable g(x) -> (4,4) array.

    Optional `dg_func(x) -> (4,4,4)` can be supplied for analytic derivatives.
    """
    name = "custom"

    def __init__(self, g_func: Callable[[np.ndarray], np.ndarray],
                 dg_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.g_func = g_func
        self.dg_func = dg_func

    def g(self, x):
        return self.g_func(x)

    def dg(self, x, h=1e-5):
        if self.dg_func is not None:
            return self.dg_func(x)
        return super().dg(x, h)
