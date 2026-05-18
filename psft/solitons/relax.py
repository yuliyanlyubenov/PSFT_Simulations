"""Soliton relaxation: gradient flow toward stationary configurations.

For a scalar/gauge field with energy functional E[phi], we run

    d phi / d tau = - delta E / delta phi

until the energy stops decreasing.  This finds local minima of the static
PSFT energy in fixed (e.g. Minkowski) background -- the field-theoretic way
to locate stable solitonic patterns predicted by Postulates 1 and 4.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from psft.core.manifold import CartesianGrid


@dataclass
class EnergyFunctional:
    """Generic static-energy functional E[phi] on a grid.

    `density_func(phi, grid)` returns an energy density array (same spatial
    shape as phi); `grad_func` returns the variational derivative.  If
    grad_func is None we estimate it with a 4-point finite difference of E.
    """
    density_func: Callable[[np.ndarray, CartesianGrid], np.ndarray]
    grad_func: Optional[Callable[[np.ndarray, CartesianGrid], np.ndarray]] = None
    dx: float = 1.0

    def energy(self, phi: np.ndarray, grid: CartesianGrid) -> float:
        return float(self.density_func(phi, grid).sum()) * self.dx**3

    def gradient(self, phi: np.ndarray, grid: CartesianGrid) -> np.ndarray:
        if self.grad_func is not None:
            return self.grad_func(phi, grid)
        return self._fd_gradient(phi, grid)

    def _fd_gradient(self, phi: np.ndarray, grid: CartesianGrid,
                     eps: float = 1e-4) -> np.ndarray:
        flat = phi.ravel()
        out = np.zeros_like(flat)
        E0 = self.energy(phi, grid)
        for i in range(flat.size):
            backup = flat[i]
            flat[i] = backup + eps
            E_plus = self.energy(flat.reshape(phi.shape), grid)
            flat[i] = backup
            out[i] = (E_plus - E0) / eps
        return out.reshape(phi.shape)


@dataclass
class GradientFlowRelaxer:
    """Steepest-descent solver with adaptive step size."""
    energy: EnergyFunctional
    step: float = 1e-2
    max_iter: int = 2000
    tol: float = 1e-8
    record_history: bool = False

    def relax(self, phi0: np.ndarray, grid: CartesianGrid) -> dict:
        phi = phi0.copy()
        E_prev = self.energy.energy(phi, grid)
        history = []
        step = self.step
        for it in range(self.max_iter):
            grad = self.energy.gradient(phi, grid)
            phi_new = phi - step * grad
            E_new = self.energy.energy(phi_new, grid)
            if E_new > E_prev:
                step *= 0.5
                if step < 1e-12:
                    break
                continue
            phi = phi_new
            if self.record_history:
                history.append(E_new)
            if abs(E_prev - E_new) < self.tol * (1.0 + abs(E_prev)):
                E_prev = E_new
                break
            E_prev = E_new
            step = min(step * 1.05, self.step * 10)
        return {"phi": phi, "energy": E_prev, "iters": it, "history": history}
