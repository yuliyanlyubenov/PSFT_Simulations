"""Gauge-sector evolution on a 3D Cartesian grid.

In PSFT (paper Postulates 1+4) matter is solitonic in the fundamental fields
(u^a, sigma^A).  The scalar gauge-algebra fields sigma^A carry topological
content (electric charge, baryon number, etc.) and must be evolved alongside
the fluid:

    dt sigma^A + v^i D_i sigma^A = sources

In its simplest form, sigma^A is advected by the spacetime-fluid 3-velocity
v^i with periodic / outflow boundaries.  We use a Lax-Friedrichs (Rusanov)
flux per direction and method-of-lines RK4 in time.

Topological conservation: if we initialise sigma^A with a topological
configuration (vortex, Skyrme hedgehog, etc.) and evolve by pure advection,
the topological charge should be invariant under smooth evolution.  This is
a non-trivial numerical test: pure finite-difference schemes with strong
dissipation can wash out topology over time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Lax-Friedrichs advection of a scalar field by a velocity field on a 3D grid
# ---------------------------------------------------------------------------
@dataclass
class ScalarAdvector3D:
    """Advection of a scalar (or set of scalars) by a 3-velocity field.

    Solves
        dt phi + v^i d_i phi  =  0
    on a periodic / outflow 3D Cartesian grid using Lax-Friedrichs flux
    and RK4 in time.

    `phi` may be a single ndarray of shape (Nx, Ny, Nz) (one scalar) or
    (M, Nx, Ny, Nz) for M scalars evolved in parallel.
    """
    Nx: int
    Ny: int
    Nz: int
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    cfl: float = 0.4
    boundary: str = "periodic"

    phi: np.ndarray = field(default=None, init=False)        # current state
    vx: np.ndarray = field(default=None, init=False)         # velocity (frozen)
    vy: np.ndarray = field(default=None, init=False)
    vz: np.ndarray = field(default=None, init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.x = np.linspace(0.5 * self.dx, self.Lx - 0.5 * self.dx, self.Nx)
        self.y = np.linspace(0.5 * self.dy, self.Ly - 0.5 * self.dy, self.Ny)
        self.z = np.linspace(0.5 * self.dz, self.Lz - 0.5 * self.dz, self.Nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def set_velocity(self, vx_func, vy_func, vz_func):
        self.vx = vx_func(self.X, self.Y, self.Z)
        self.vy = vy_func(self.X, self.Y, self.Z)
        self.vz = vz_func(self.X, self.Y, self.Z)

    def set_scalar(self, phi: np.ndarray):
        self.phi = phi
        self.t = 0.0

    def _shift(self, U: np.ndarray, axis: int, shift: int) -> np.ndarray:
        """Shift along the SPATIAL axis (offset by phi.ndim - 3 if phi is
        multi-component)."""
        # Always shift along the spatial axes (the last three).
        spatial_axis = (U.ndim - 3) + axis
        if self.boundary == "periodic":
            return np.roll(U, -shift, axis=spatial_axis)
        elif self.boundary == "outflow":
            out = np.empty_like(U)
            slc_dst = [slice(None)] * U.ndim
            slc_src = [slice(None)] * U.ndim
            if shift > 0:
                slc_dst[spatial_axis] = slice(None, -shift)
                slc_src[spatial_axis] = slice(shift, None)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
                slc_dst[spatial_axis] = slice(-shift, None)
                slc_src[spatial_axis] = slice(-1, None)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
            elif shift < 0:
                slc_dst[spatial_axis] = slice(-shift, None)
                slc_src[spatial_axis] = slice(None, shift)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
                slc_dst[spatial_axis] = slice(None, -shift)
                slc_src[spatial_axis] = slice(None, 1)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
            else:
                out = U.copy()
            return out
        raise ValueError(f"unknown boundary: {self.boundary}")

    def _lf_flux_axis(self, axis: int, v_par: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Lax-Friedrichs flux at i+1/2 interface along given axis.

        flux_{i+1/2} = (v_i phi_i + v_{i+1} phi_{i+1}) / 2
                    - (max|v|/2) (phi_{i+1} - phi_i)
        """
        phi_r = self._shift(phi, axis, 1)
        v_r = self._shift(v_par, axis, 1)
        alpha = np.maximum(np.abs(v_par), np.abs(v_r))
        # When phi has leading component axis, broadcast alpha and velocity.
        if phi.ndim == 4 and v_par.ndim == 3:
            alpha = alpha[None, ...]
            v_par_b = v_par[None, ...]
            v_r_b = v_r[None, ...]
        else:
            v_par_b = v_par
            v_r_b = v_r
        return 0.5 * (v_par_b * phi + v_r_b * phi_r) - 0.5 * alpha * (phi_r - phi)

    def rhs(self, phi: np.ndarray) -> np.ndarray:
        flux_xp = self._lf_flux_axis(0, self.vx, phi)
        flux_yp = self._lf_flux_axis(1, self.vy, phi)
        flux_zp = self._lf_flux_axis(2, self.vz, phi)
        flux_xm = self._shift(flux_xp, 0, -1)
        flux_ym = self._shift(flux_yp, 1, -1)
        flux_zm = self._shift(flux_zp, 2, -1)
        return -((flux_xp - flux_xm) / self.dx
                 + (flux_yp - flux_ym) / self.dy
                 + (flux_zp - flux_zm) / self.dz)

    def step(self, dt: float = None):
        if dt is None:
            v_max = float(np.max(np.abs(np.stack([self.vx, self.vy, self.vz]))))
            dt = self.cfl * min(self.dx, self.dy, self.dz) / max(v_max, 1e-12)

        k1 = self.rhs(self.phi)
        k2 = self.rhs(self.phi + 0.5 * dt * k1)
        k3 = self.rhs(self.phi + 0.5 * dt * k2)
        k4 = self.rhs(self.phi + dt * k3)
        self.phi = self.phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt
        return dt

    def evolve(self, t_end: float, callback=None):
        while self.t < t_end - 1e-12:
            self.step()
            if callback is not None:
                callback(self)


# ---------------------------------------------------------------------------
# Topological diagnostics
# ---------------------------------------------------------------------------
def winding_number_in_plane(phi_complex: np.ndarray, axis: int = 2, slice_idx: int = None,
                            center=None, radius_frac: float = 0.4, n_samples: int = 256):
    """Winding of a complex U(1) field around (center) in the plane normal to `axis`.

    `phi_complex` shape (Nx, Ny, Nz).  axis 2 means we slice at z=slice_idx
    and measure winding in the (x, y) plane.  Default centre is the middle.
    """
    Nx, Ny, Nz = phi_complex.shape
    if axis == 2:
        if slice_idx is None:
            slice_idx = Nz // 2
        slc = phi_complex[:, :, slice_idx]
    elif axis == 1:
        if slice_idx is None:
            slice_idx = Ny // 2
        slc = phi_complex[:, slice_idx, :]
    else:
        if slice_idx is None:
            slice_idx = Nx // 2
        slc = phi_complex[slice_idx, :, :]
    N1, N2 = slc.shape
    if center is None:
        cx, cy = N1 // 2, N2 // 2
    else:
        cx, cy = center
    r = radius_frac * min(N1, N2) / 2
    angles = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    xi = np.clip(np.round(xs).astype(int), 0, N1 - 1)
    yi = np.clip(np.round(ys).astype(int), 0, N2 - 1)
    phases = np.angle(slc[xi, yi])
    closed = np.concatenate([phases, phases[:1]])
    unwrapped = np.unwrap(closed)
    return int(np.round((unwrapped[-1] - unwrapped[0]) / (2 * np.pi)))


def baryon_number_skyrme_3d(N4: np.ndarray, dx: float, dy: float = None,
                            dz: float = None) -> float:
    """3D Skyrme baryon number on a Cartesian grid.

    N4 shape: (4, Nx, Ny, Nz) with |N|^2 = sum_a (N^a)^2 = 1.

        B = (1/(2 pi^2)) int eps_{abcd} N^a d_x N^b d_y N^c d_z N^d  d^3x

    (See `psft.solitons.topology.topological_charge_skyrme` for the
    same expression in a slightly different normalisation; this is the
    integer-valued version.)
    """
    if N4.shape[0] != 4:
        raise ValueError("baryon_number_skyrme_3d expects shape (4, Nx, Ny, Nz)")
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    from itertools import permutations
    eps4 = np.zeros((4, 4, 4, 4))
    for perm in permutations((0, 1, 2, 3)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        eps4[perm] = 1.0 if inv % 2 == 0 else -1.0
    dN_x = np.gradient(N4, dx, axis=1)
    dN_y = np.gradient(N4, dy, axis=2)
    dN_z = np.gradient(N4, dz, axis=3)
    density = np.einsum("abcd,a...,b...,c...,d...->...", eps4, N4, dN_x, dN_y, dN_z)
    integral = density.sum() * (dx * dy * dz)
    return float(integral / (2 * np.pi ** 2))
