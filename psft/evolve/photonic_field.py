"""Photonic field P_ab evolved on a 3D Cartesian grid (Item 4 of the
simulation roadmap).

In PSFT (paper Postulate 1) the photonic stress field is the primitive
dynamical entity that drives the spacetime fluid.  In the classical
limit (Q_ab -> 0, Lambda -> 0), the photonic source tensor reduces to
the electromagnetic stress-energy tensor

    P_ab  ->  T^EM_ab  =  (1/4 pi)[F_ac F^c_b - (1/4) g_ab F_cd F^cd]

with F_ab = d_a A_b - d_b A_a.  The 4-potential A_a evolves on a 3D
Cartesian spacetime slice via the Maxwell wave equation (Lorenz gauge,
flat Minkowski background, c = 1):

    dt^2 A_a  =  lap A_a  -  4 pi j_a.

We discretise as a first-order-in-time system with state
(A_t, A_x, A_y, A_z, pi_t, pi_x, pi_y, pi_z), where pi_a = dt A_a.

The Lorenz-gauge constraint d^a A_a = 0 is preserved by the evolution
when initial data satisfies it (this is enforced on initialisation).

Matter is solitonic in PSFT, so the source current j_a comes from
solitonic configurations of (u^a, sigma^A), NOT from a separately
postulated matter field.  At this layer we expose j_a as a callable
the caller supplies; an empty (zero) current yields vacuum
electrodynamics, useful for stability tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# Smoothed-Coulomb closed-form initialiser
# ---------------------------------------------------------------------------
def smoothed_gaussian_charge_density(X, Y, Z, x0, y0, z0, Q, sigma):
    """Gaussian-smoothed charge density:
        rho(r) = Q / (2 pi sigma^2)^{3/2} * exp(-r^2/(2 sigma^2))
    integrates to Q.
    """
    r2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
    norm = Q / (2 * math.pi * sigma * sigma) ** 1.5
    return norm * np.exp(-r2 / (2 * sigma * sigma))


def smoothed_coulomb_potential(X, Y, Z, x0, y0, z0, Q, sigma):
    """Electric potential for a Gaussian-smoothed point charge:

        phi(r) = Q / (4 pi r) * erf(r / (sigma sqrt(2)))

    Solves nabla^2 phi = -4 pi rho with rho the Gaussian above.
    Goes to Q/(4 pi r) at large r, finite at r=0.
    """
    from scipy.special import erf as _erf  # may not be available
    pass


def _erf_array(x):
    """Pure-numpy erf via series + asymptotic; we avoid scipy import."""
    # numpy >= 1.21 has np.special? No -- not directly.  Use math.erf via
    # np.vectorize since it's a one-shot initialisation.
    return np.vectorize(math.erf, otypes=[float])(x)


def smoothed_coulomb_potential(X, Y, Z, x0, y0, z0, Q, sigma):
    """Scalar potential phi(r) = Q/r * erf(r/(sigma sqrt(2))) for a
    Gaussian-smeared charge in Gaussian units (the same units used in
    the Maxwell wave equation
    dt^2 A_a = lap A_a - 4 pi j_a).

    Note: the metric convention (-,+,+,+) makes A_t = -phi, so the
    initialiser for the time-component of the 4-potential is the
    NEGATIVE of this function (handled in PhotonicField3D.initialise_*).
    Use this function as-is for diagnostic plotting of phi.
    """
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
    r_safe = np.maximum(r, 1e-12)
    return (Q / r_safe) * _erf_array(r_safe / (sigma * math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# PhotonicField3D
# ---------------------------------------------------------------------------
@dataclass
class PhotonicField3D:
    """EM 4-potential A_a on a 3D Cartesian grid, evolved by the Maxwell
    wave equation in Lorenz gauge on a flat Minkowski background.

    State arrays (each shape (Nx, Ny, Nz)):
        A_t, A_x, A_y, A_z       4-potential components
        pi_t, pi_x, pi_y, pi_z   conjugate momenta (= dt A_a)
    """
    Nx: int
    Ny: int
    Nz: int
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    cfl: float = 0.4
    boundary: str = "periodic"     # "periodic" or "outflow"

    A_t: np.ndarray = field(default=None, init=False)
    A_x: np.ndarray = field(default=None, init=False)
    A_y: np.ndarray = field(default=None, init=False)
    A_z: np.ndarray = field(default=None, init=False)
    pi_t: np.ndarray = field(default=None, init=False)
    pi_x: np.ndarray = field(default=None, init=False)
    pi_y: np.ndarray = field(default=None, init=False)
    pi_z: np.ndarray = field(default=None, init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.x = np.linspace(0.5 * self.dx, self.Lx - 0.5 * self.dx, self.Nx)
        self.y = np.linspace(0.5 * self.dy, self.Ly - 0.5 * self.dy, self.Ny)
        self.z = np.linspace(0.5 * self.dz, self.Lz - 0.5 * self.dz, self.Nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        zero = np.zeros((self.Nx, self.Ny, self.Nz))
        self.A_t = zero.copy()
        self.A_x = zero.copy()
        self.A_y = zero.copy()
        self.A_z = zero.copy()
        self.pi_t = zero.copy()
        self.pi_x = zero.copy()
        self.pi_y = zero.copy()
        self.pi_z = zero.copy()

    # ---- initial conditions -------------------------------------------------
    def initialise_static_coulomb(self, x0: float, y0: float, z0: float,
                                    Q: float, sigma: float):
        """Set up the static smoothed-Coulomb solution of Maxwell's eq
        for a point charge at (x0, y0, z0) of total charge Q smeared
        over Gaussian width sigma.

        With metric signature (-,+,+,+) we have A_t = -phi where phi is
        the standard scalar potential.  Under the wave equation
        dt^2 A_t = lap A_t - 4 pi j_t with j_t = rho:
            static balance => lap A_t = 4 pi rho
        which is satisfied by A_t = -phi (since lap phi = -4 pi rho).

        Sets A_t = -phi, A_i = 0, pi_a = 0.
        """
        phi = smoothed_coulomb_potential(
            self.X, self.Y, self.Z, x0, y0, z0, Q, sigma,
        )
        self.A_t = -phi
        self.A_x[:] = 0.0
        self.A_y[:] = 0.0
        self.A_z[:] = 0.0
        self.pi_t[:] = 0.0
        self.pi_x[:] = 0.0
        self.pi_y[:] = 0.0
        self.pi_z[:] = 0.0
        self.t = 0.0

    # ---- spatial derivatives ------------------------------------------------
    def _laplacian(self, F: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            lap = (
                (np.roll(F, 1, axis=0) + np.roll(F, -1, axis=0) - 2 * F) / self.dx ** 2
                + (np.roll(F, 1, axis=1) + np.roll(F, -1, axis=1) - 2 * F) / self.dy ** 2
                + (np.roll(F, 1, axis=2) + np.roll(F, -1, axis=2) - 2 * F) / self.dz ** 2
            )
        else:   # outflow: zero-gradient ghost cells
            lap = np.zeros_like(F)
            # interior: 6-point stencil
            lap[1:-1, :, :] += (F[2:, :, :] - 2 * F[1:-1, :, :] + F[:-2, :, :]) / self.dx ** 2
            lap[:, 1:-1, :] += (F[:, 2:, :] - 2 * F[:, 1:-1, :] + F[:, :-2, :]) / self.dy ** 2
            lap[:, :, 1:-1] += (F[:, :, 2:] - 2 * F[:, :, 1:-1] + F[:, :, :-2]) / self.dz ** 2
        return lap

    def _gradient(self, F: np.ndarray, axis: int) -> np.ndarray:
        """Central difference along given axis (0=x, 1=y, 2=z)."""
        h = (self.dx, self.dy, self.dz)[axis]
        if self.boundary == "periodic":
            return (np.roll(F, -1, axis=axis) - np.roll(F, 1, axis=axis)) / (2 * h)
        else:
            return np.gradient(F, h, axis=axis)

    # ---- field strength and stress-energy -----------------------------------
    def E_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electric field.  With A_a = (-phi, A_x, A_y, A_z) covariant
        4-potential and pi_a = d_t A_a:
            E^i = -d_i phi - d_t A^i  =  d_i A_t  -  pi_i
        (the standard relation E = -grad phi - dA/dt, written in terms
        of the covariant A_t and momenta).
        """
        Ex = self._gradient(self.A_t, axis=0) - self.pi_x
        Ey = self._gradient(self.A_t, axis=1) - self.pi_y
        Ez = self._gradient(self.A_t, axis=2) - self.pi_z
        return Ex, Ey, Ez

    def B_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Magnetic field B^i = (curl A)^i."""
        Bx = self._gradient(self.A_z, axis=1) - self._gradient(self.A_y, axis=2)
        By = self._gradient(self.A_x, axis=2) - self._gradient(self.A_z, axis=0)
        Bz = self._gradient(self.A_y, axis=0) - self._gradient(self.A_x, axis=1)
        return Bx, By, Bz

    def total_field_energy(self) -> float:
        """U = (1/(8 pi)) integral (|E|^2 + |B|^2) d^3x."""
        Ex, Ey, Ez = self.E_field()
        Bx, By, Bz = self.B_field()
        E2 = Ex * Ex + Ey * Ey + Ez * Ez
        B2 = Bx * Bx + By * By + Bz * Bz
        return float(np.sum(E2 + B2)) * self.dx * self.dy * self.dz / (8 * math.pi)

    def photonic_stress_T00(self) -> np.ndarray:
        """Field-energy density rho_EM = (1/(8 pi))(|E|^2 + |B|^2),
        i.e. the (0,0) component of T^EM_ab."""
        Ex, Ey, Ez = self.E_field()
        Bx, By, Bz = self.B_field()
        E2 = Ex * Ex + Ey * Ey + Ez * Ez
        B2 = Bx * Bx + By * By + Bz * Bz
        return (E2 + B2) / (8 * math.pi)

    # ---- evolution ----------------------------------------------------------
    def rhs(self, A_t, A_x, A_y, A_z, pi_t, pi_x, pi_y, pi_z,
             j_t, j_x, j_y, j_z):
        """Wave equation RHS: dt A = pi, dt pi = lap A - 4 pi j."""
        dA_t = pi_t
        dA_x = pi_x
        dA_y = pi_y
        dA_z = pi_z
        dpi_t = self._laplacian(A_t) - 4 * math.pi * j_t
        dpi_x = self._laplacian(A_x) - 4 * math.pi * j_x
        dpi_y = self._laplacian(A_y) - 4 * math.pi * j_y
        dpi_z = self._laplacian(A_z) - 4 * math.pi * j_z
        return dA_t, dA_x, dA_y, dA_z, dpi_t, dpi_x, dpi_y, dpi_z

    def step(self, dt: float = None,
             j_t: np.ndarray = None, j_x: np.ndarray = None,
             j_y: np.ndarray = None, j_z: np.ndarray = None):
        """Advance state by one RK4 step.

        Sources `j_a` default to zero (vacuum).  For a static Coulomb
        configuration, pass the charge density as j_t (and zero spatial
        currents) so the (lap A_t - 4 pi rho) cancellation holds.
        """
        if dt is None:
            dh_min = min(self.dx, self.dy, self.dz)
            dt = self.cfl * dh_min / math.sqrt(3.0)   # 3D Courant for c=1
        zero = np.zeros_like(self.A_t)
        j_t_arr = j_t if j_t is not None else zero
        j_x_arr = j_x if j_x is not None else zero
        j_y_arr = j_y if j_y is not None else zero
        j_z_arr = j_z if j_z is not None else zero

        U0 = (self.A_t, self.A_x, self.A_y, self.A_z,
              self.pi_t, self.pi_x, self.pi_y, self.pi_z)
        k1 = self.rhs(*U0, j_t_arr, j_x_arr, j_y_arr, j_z_arr)
        U2 = tuple(u + 0.5 * dt * k for u, k in zip(U0, k1))
        k2 = self.rhs(*U2, j_t_arr, j_x_arr, j_y_arr, j_z_arr)
        U3 = tuple(u + 0.5 * dt * k for u, k in zip(U0, k2))
        k3 = self.rhs(*U3, j_t_arr, j_x_arr, j_y_arr, j_z_arr)
        U4 = tuple(u + dt * k for u, k in zip(U0, k3))
        k4 = self.rhs(*U4, j_t_arr, j_x_arr, j_y_arr, j_z_arr)
        new = tuple(
            u + (dt / 6.0) * (a + 2 * b + 2 * c + d)
            for u, a, b, c, d in zip(U0, k1, k2, k3, k4)
        )
        (self.A_t, self.A_x, self.A_y, self.A_z,
         self.pi_t, self.pi_x, self.pi_y, self.pi_z) = new
        self.t += dt
        return dt

    def evolve(self, t_end: float, j_t=None, j_x=None, j_y=None, j_z=None,
                callback=None):
        while self.t < t_end - 1e-12:
            self.step(j_t=j_t, j_x=j_x, j_y=j_y, j_z=j_z)
            if callback is not None:
                callback(self)

    # ---- Lorentz force on a charge distribution (coupling to matter) -------
    def lorentz_force(self, rho_charge: np.ndarray,
                       vx: np.ndarray = None, vy: np.ndarray = None,
                       vz: np.ndarray = None):
        """Lorentz force density on a charge distribution.

            f^i  =  rho_charge * E^i  +  (j x B)^i
                 =  rho_charge * (E + v x B)^i

        For a stationary distribution (v = 0) reduces to f^i = rho * E^i.
        Returns (fx, fy, fz) each shape (Nx, Ny, Nz).
        """
        Ex, Ey, Ez = self.E_field()
        zero = np.zeros_like(rho_charge)
        if vx is None:
            vx = zero
        if vy is None:
            vy = zero
        if vz is None:
            vz = zero
        Bx, By, Bz = self.B_field()
        # E + v x B  -- the per-charge-density acceleration field.
        ax = Ex + (vy * Bz - vz * By)
        ay = Ey + (vz * Bx - vx * Bz)
        az = Ez + (vx * By - vy * Bx)
        return rho_charge * ax, rho_charge * ay, rho_charge * az
