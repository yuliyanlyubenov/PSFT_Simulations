"""Spacetime coordinate grids.

The simulator works in two regimes:

  * **Pointwise**: a single spacetime point x = (t,x,y,z), used for analytic
    tests where the metric is known and uniform.
  * **Grid**: a regular Cartesian grid in spatial coordinates, with time
    advanced by the integrator.  Fields on the grid have shape
    (tensor_indices..., Nx, Ny, Nz) -- spatial axes last.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CartesianGrid:
    """A regular 3D Cartesian spatial grid, with t carried explicitly.

    `shape` = (Nx, Ny, Nz).  Bounds are inclusive on the low side and
    exclusive on the high side (numpy linspace-like, with endpoint).
    """
    shape: Tuple[int, int, int]
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    t: float = 0.0

    def axes(self):
        return tuple(
            np.linspace(lo, hi, n) for (lo, hi), n in zip(self.bounds, self.shape)
        )

    def meshgrid(self):
        ax = self.axes()
        return np.meshgrid(*ax, indexing="ij")  # X, Y, Z

    def coords_4(self):
        """Return a (4, Nx, Ny, Nz) field x^a = (t, x, y, z)."""
        X, Y, Z = self.meshgrid()
        T = np.full_like(X, self.t)
        return np.stack([T, X, Y, Z], axis=0)

    def deltas(self):
        return tuple(
            (hi - lo) / (n - 1) if n > 1 else 1.0
            for (lo, hi), n in zip(self.bounds, self.shape)
        )

    def gradient(self, field: np.ndarray, spatial_index: int = 0) -> np.ndarray:
        """Return d_i F over the three spatial axes.

        `field` has shape (..., Nx, Ny, Nz).  Returns shape (3, *field.shape)
        with components (d_x, d_y, d_z) along its first axis.
        """
        dx, dy, dz = self.deltas()
        # Operate on the last 3 axes.
        axis0 = field.ndim - 3
        grads = np.stack([
            np.gradient(field, dx, axis=axis0),
            np.gradient(field, dy, axis=axis0 + 1),
            np.gradient(field, dz, axis=axis0 + 2),
        ], axis=0)
        return grads
