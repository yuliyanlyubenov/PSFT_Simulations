"""Soliton ansatze for PSFT matter (electron, proton/neutron, knot).

Per Postulate 1, matter is a stable soliton in the photonic stress field.
We provide three standard ansatze, parametrised by core radius and winding:

  * VortexAnsatz          U(1) line vortex   -- electron candidate
  * NielsenOlesenAnsatz   superconducting flux tube -- gluon/colour string
  * SkyrmeHedgehogAnsatz  3D hedgehog with baryon number 1 -- proton/neutron
  * KnotAnsatz            Hopfion configuration -- linked-flux variant

These produce the field profile (and its derivatives) on a CartesianGrid
so the master equation evaluator can use them as initial data for relaxation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from psft.core.manifold import CartesianGrid


class SolitonAnsatz(ABC):
    @abstractmethod
    def evaluate(self, grid: CartesianGrid) -> dict: ...


@dataclass
class VortexAnsatz(SolitonAnsatz):
    """U(1) line vortex along the z-axis with winding `n`.

    Returns complex-scalar order parameter Phi = f(r) e^{i n phi}, with
    f(r) the standard tanh core profile.
    """
    winding: int = 1
    core_radius: float = 1.0
    amplitude: float = 1.0
    axis: str = "z"

    def evaluate(self, grid: CartesianGrid) -> dict:
        X, Y, Z = grid.meshgrid()
        if self.axis == "z":
            r_perp = np.sqrt(X**2 + Y**2)
            phi = np.arctan2(Y, X)
        elif self.axis == "y":
            r_perp = np.sqrt(X**2 + Z**2)
            phi = np.arctan2(Z, X)
        else:
            r_perp = np.sqrt(Y**2 + Z**2)
            phi = np.arctan2(Z, Y)
        f = self.amplitude * np.tanh(r_perp / self.core_radius)
        Phi = f * np.exp(1j * self.winding * phi)
        return {
            "phi_complex": Phi,
            "amplitude": f,
            "phase": np.angle(Phi),
            "r_perp": r_perp,
        }


@dataclass
class NielsenOlesenAnsatz(SolitonAnsatz):
    """Nielsen-Olesen vortex with gauge field A_phi.

    Provides both scalar field |Phi| and the gauge connection A_phi(r) along
    the chosen axis -- a paper-style flux-tube template for PSFT colour
    confinement studies.
    """
    winding: int = 1
    core_radius: float = 1.0
    flux_radius: float = 1.5
    amplitude: float = 1.0

    def evaluate(self, grid: CartesianGrid) -> dict:
        X, Y, Z = grid.meshgrid()
        r = np.sqrt(X**2 + Y**2)
        phi_ang = np.arctan2(Y, X)
        f = self.amplitude * np.tanh(r / self.core_radius)
        a = self.winding * (1.0 - np.exp(-(r / self.flux_radius) ** 2))
        Phi = f * np.exp(1j * self.winding * phi_ang)
        # A_x = -sin(phi)/r * a, A_y = cos(phi)/r * a -- the canonical NO gauge.
        with np.errstate(divide="ignore", invalid="ignore"):
            A_x = np.where(r > 0, -np.sin(phi_ang) / r * a, 0.0)
            A_y = np.where(r > 0,  np.cos(phi_ang) / r * a, 0.0)
        return {
            "phi_complex": Phi,
            "A_x": A_x, "A_y": A_y, "A_z": np.zeros_like(A_x),
            "amplitude": f, "flux_profile": a,
        }


@dataclass
class SkyrmeHedgehogAnsatz(SolitonAnsatz):
    """B=1 Skyrme hedgehog: n^a = (sin F(r) hat r^a, cos F(r))."""
    core_radius: float = 1.0
    F_inf: float = 0.0       # F at r=infinity
    F_0: float = np.pi       # F at r=0

    def F(self, r: np.ndarray) -> np.ndarray:
        return self.F_inf + (self.F_0 - self.F_inf) * np.exp(-r / self.core_radius)

    def evaluate(self, grid: CartesianGrid) -> dict:
        X, Y, Z = grid.meshgrid()
        r = np.sqrt(X**2 + Y**2 + Z**2)
        Fr = self.F(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            hat_x = np.where(r > 0, X / r, 0.0)
            hat_y = np.where(r > 0, Y / r, 0.0)
            hat_z = np.where(r > 0, Z / r, 0.0)
        sigma = np.cos(Fr)
        n_vec = np.stack([
            np.sin(Fr) * hat_x,
            np.sin(Fr) * hat_y,
            np.sin(Fr) * hat_z,
        ], axis=0)
        # Full O(4) field N^a = (sigma, n_vec) of unit norm.
        N = np.stack([sigma, n_vec[0], n_vec[1], n_vec[2]], axis=0)
        return {"N4": N, "n_vec": n_vec, "sigma": sigma, "F": Fr}


@dataclass
class KnotAnsatz(SolitonAnsatz):
    """Hopfion-style ansatz for linked-flux solitons (knot configuration).

    Uses the standard rational map for an unknotted Hopf charge Q=1 vortex.
    For higher charges one composes with z -> z^p w^q maps; here we expose
    only the Q=1 baseline as a starting template.
    """
    radius: float = 1.0

    def evaluate(self, grid: CartesianGrid) -> dict:
        X, Y, Z = grid.meshgrid()
        # Map R^3 -> S^3 via stereographic projection from (0,0,0,-1):
        r2 = X**2 + Y**2 + Z**2
        denom = 1.0 + r2 / self.radius**2
        w = (2.0 * (X + 1j * Y) / self.radius) / denom
        z = (2.0 * Z / self.radius + 1j * (r2 / self.radius**2 - 1.0)) / denom
        # The unknotted Hopf map.  S^2 image:
        n_x = 2.0 * (w * np.conjugate(z)).real
        n_y = 2.0 * (w * np.conjugate(z)).imag
        n_z = (np.abs(w) ** 2 - np.abs(z) ** 2)
        n_vec = np.stack([n_x, n_y, n_z], axis=0)
        return {"n_vec": n_vec, "w": w, "z": z}
