"""3+1D relativistic fluid solver -- direct lift of `hydro_1d.py` to a
3D Cartesian spatial grid.

State variables (conservative form, per cell):
    D            = gamma rho                lab-frame rest-mass density
    S^i = (S_x, S_y, S_z) = rho h gamma^2 v^i    momentum-density components
    tau          = rho h gamma^2 - p - D    energy density excess

Evolution equations (flat Minkowski background):
    dt D    + dx (D vx)        + dy (D vy)        + dz (D vz)        = 0
    dt S_j  + dx (S_j vx + p dxj) + dy (S_j vy + p dyj) + dz (S_j vz + p dzj) = 0
    dt tau  + dx (S_x - D vx)  + dy (S_y - D vy)  + dz (S_z - D vz)  = 0

Equation of state: Gamma-law ideal gas, p = (Gamma - 1) rho eps.

Numerics:
  * Conservative ↔ primitive recovery uses the same closed-form algebraic
    relation as `hydro_1d` -- with |S|^2 = S_x^2 + S_y^2 + S_z^2.
  * Per-direction Lax-Friedrichs flux.
  * Method-of-lines RK4 in time.
  * Joint advective + viscous CFL time-step (conformal viscosity flux
    added optionally as a small extension; the v2 modifications enter
    here as in 1D).

Memory: a 64^3 grid carries 262 144 cells, each with 5 doubles for state
and 4 RK4 stages -> ~75 MB total state during a step.  Comfortably
within laptop RAM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np

from psft.evolve.hydro_1d import sound_speed, signal_speed


# ---------------------------------------------------------------------------
# Primitive recovery (3D, |S|^2 = S_x^2 + S_y^2 + S_z^2)
# ---------------------------------------------------------------------------
def primitive_from_conservative_3d(
    D: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, Sz: np.ndarray, tau: np.ndarray,
    Gamma: float, max_iter: int = 80, tol: float = 1e-12,
):
    """Newton-Raphson primitive recovery on a 3D field.

    Returns (rho, p, vx, vy, vz, W), each with shape matching the inputs.
    """
    S2 = Sx * Sx + Sy * Sy + Sz * Sz
    Smag = np.sqrt(S2)
    A = tau + D
    safe_A = np.where(A > 0, A, 1e-12)
    x = A + 0.5 * S2 / safe_A
    x = np.maximum(x, 1.0001 * Smag)

    for _ in range(max_iter):
        x2_minus_S2 = np.maximum(x * x - S2, 1e-30)
        sqrt_term = np.sqrt(x2_minus_S2)
        F = (x * x - Gamma * A * x
             + (Gamma - 1.0) * S2
             + (Gamma - 1.0) * D * sqrt_term)
        Fp = (2.0 * x - Gamma * A
              + (Gamma - 1.0) * D * x / sqrt_term)
        Fp = np.where(np.abs(Fp) < 1e-15, 1e-15, Fp)
        dx = -F / Fp
        x_new = x + dx
        bad = x_new * x_new <= S2
        x_new = np.where(bad, 0.5 * (x + 1.0001 * Smag), x_new)
        x = x_new
        if np.max(np.abs(F)) < tol * max(float(np.max(np.abs(x * x))), 1.0):
            break

    p = np.maximum(x - A, 1e-20)
    inv_x = 1.0 / x
    vx = Sx * inv_x
    vy = Sy * inv_x
    vz = Sz * inv_x
    W = x / np.sqrt(np.maximum(x * x - S2, 1e-30))
    rho = np.maximum(D / W, 1e-20)
    return rho, p, vx, vy, vz, W


def conservative_from_primitive_3d(
    rho: np.ndarray, p: np.ndarray,
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    Gamma: float,
):
    """(rho, p, v) -> (D, Sx, Sy, Sz, tau)."""
    v2 = vx * vx + vy * vy + vz * vz
    W = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-30))
    eps = p / ((Gamma - 1.0) * np.maximum(rho, 1e-30))
    h = 1.0 + Gamma * eps
    D = rho * W
    rho_h_W2 = rho * h * W * W
    Sx = rho_h_W2 * vx
    Sy = rho_h_W2 * vy
    Sz = rho_h_W2 * vz
    tau = rho_h_W2 - p - D
    return D, Sx, Sy, Sz, tau


# ---------------------------------------------------------------------------
# Per-direction fluxes
# ---------------------------------------------------------------------------
def flux_x(D, Sx, Sy, Sz, tau, rho, p, vx):
    """Flux in the +x direction.  Returns (F_D, F_Sx, F_Sy, F_Sz, F_tau)."""
    return D * vx, Sx * vx + p, Sy * vx, Sz * vx, Sx - D * vx


def flux_y(D, Sx, Sy, Sz, tau, rho, p, vy):
    return D * vy, Sx * vy, Sy * vy + p, Sz * vy, Sy - D * vy


def flux_z(D, Sx, Sy, Sz, tau, rho, p, vz):
    return D * vz, Sx * vz, Sy * vz, Sz * vz + p, Sz - D * vz


def lax_friedrichs(U_L, U_R, F_L, F_R, alpha):
    """Componentwise local-Lax-Friedrichs flux."""
    return tuple(
        0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        for uL, uR, fL, fR in zip(U_L, U_R, F_L, F_R)
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class RelativisticEulerSolver3D:
    """3+1D relativistic Euler on a periodic / outflow uniform grid.

    `step()` advances by one RK4 step; `evolve()` runs to a target time.
    Setting `eta > 0` activates conformal viscous fluxes per direction
    (paper Modification 1).
    """
    Nx: int
    Ny: int
    Nz: int
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    Gamma: float = 4.0 / 3.0
    cfl: float = 0.4
    eta: float = 0.0
    boundary: str = "periodic"           # "periodic" or "outflow"

    D: np.ndarray = field(default=None, init=False)
    Sx: np.ndarray = field(default=None, init=False)
    Sy: np.ndarray = field(default=None, init=False)
    Sz: np.ndarray = field(default=None, init=False)
    tau: np.ndarray = field(default=None, init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        # Cell-centred coordinates.
        self.x = np.linspace(0.5 * self.dx, self.Lx - 0.5 * self.dx, self.Nx)
        self.y = np.linspace(0.5 * self.dy, self.Ly - 0.5 * self.dy, self.Ny)
        self.z = np.linspace(0.5 * self.dz, self.Lz - 0.5 * self.dz, self.Nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

    # ---- initial conditions -------------------------------------------------
    def initialise(self, rho_func, p_func, vx_func, vy_func, vz_func):
        rho = rho_func(self.X, self.Y, self.Z)
        p = p_func(self.X, self.Y, self.Z)
        vx = vx_func(self.X, self.Y, self.Z)
        vy = vy_func(self.X, self.Y, self.Z)
        vz = vz_func(self.X, self.Y, self.Z)
        self.D, self.Sx, self.Sy, self.Sz, self.tau = conservative_from_primitive_3d(
            rho, p, vx, vy, vz, self.Gamma,
        )
        self.t = 0.0

    # ---- boundary helpers ---------------------------------------------------
    def _shift(self, U: np.ndarray, axis: int, shift: int) -> np.ndarray:
        """Return U shifted by `shift` along axis 0=x, 1=y, 2=z."""
        if self.boundary == "periodic":
            return np.roll(U, -shift, axis=axis)
        elif self.boundary == "outflow":
            out = np.empty_like(U)
            slc_dst = [slice(None)] * U.ndim
            slc_src = [slice(None)] * U.ndim
            if shift > 0:
                slc_dst[axis] = slice(None, -shift)
                slc_src[axis] = slice(shift, None)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
                slc_dst[axis] = slice(-shift, None)
                slc_src[axis] = slice(-1, None)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
            elif shift < 0:
                slc_dst[axis] = slice(-shift, None)
                slc_src[axis] = slice(None, shift)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
                slc_dst[axis] = slice(None, -shift)
                slc_src[axis] = slice(None, 1)
                out[tuple(slc_dst)] = U[tuple(slc_src)]
            else:
                out = U.copy()
            return out
        raise ValueError(f"unknown boundary: {self.boundary}")

    # ---- per-direction LF flux ---------------------------------------------
    def _flux_direction(self, D, Sx, Sy, Sz, tau, rho, p, v_par, axis, flux_fn):
        """Compute LF flux at +1/2 interfaces along `axis`."""
        # Right-neighbour state.
        Dr = self._shift(D, axis, 1)
        Sxr = self._shift(Sx, axis, 1)
        Syr = self._shift(Sy, axis, 1)
        Szr = self._shift(Sz, axis, 1)
        taur = self._shift(tau, axis, 1)
        rho_r, p_r, vxr, vyr, vzr, _ = primitive_from_conservative_3d(
            Dr, Sxr, Syr, Szr, taur, self.Gamma,
        )
        v_par_r = (vxr, vyr, vzr)[axis]

        F_L = flux_fn(D, Sx, Sy, Sz, tau, rho, p, v_par)
        F_R = flux_fn(Dr, Sxr, Syr, Szr, taur, rho_r, p_r, v_par_r)
        U_L = (D, Sx, Sy, Sz, tau)
        U_R = (Dr, Sxr, Syr, Szr, taur)

        cs = sound_speed(rho, p, self.Gamma)
        cs_r = sound_speed(rho_r, p_r, self.Gamma)
        alpha = np.maximum(signal_speed(v_par, cs), signal_speed(v_par_r, cs_r))

        return lax_friedrichs(U_L, U_R, F_L, F_R, alpha), v_par_r

    # ---- RHS evaluation -----------------------------------------------------
    def rhs(self, D, Sx, Sy, Sz, tau,
             body_force=None):
        """Compute dU/dt.  Optional `body_force = (fx, fy, fz)` adds an
        external body force on the momentum equation, with the matching
        v.f work term added to the energy equation (so dt tau gets += v.f).
        """
        rho, p, vx, vy, vz, _ = primitive_from_conservative_3d(
            D, Sx, Sy, Sz, tau, self.Gamma,
        )

        # x-direction flux at i+1/2 interface
        flux_xp, vxr = self._flux_direction(D, Sx, Sy, Sz, tau, rho, p, vx, axis=0,
                                             flux_fn=flux_x)
        flux_yp, vyr = self._flux_direction(D, Sx, Sy, Sz, tau, rho, p, vy, axis=1,
                                             flux_fn=flux_y)
        flux_zp, vzr = self._flux_direction(D, Sx, Sy, Sz, tau, rho, p, vz, axis=2,
                                             flux_fn=flux_z)

        # Optional viscous corrections (conformal, per direction).
        if self.eta != 0.0:
            # x-direction: (4/3) eta dv_x/dx + eta (dv_y/dx + dv_x/dy)/2 ... etc.
            # Use the simplest: only diagonal stress contribution per direction.
            dvx_dx = (vxr - vx) / self.dx
            dvy_dy = (vyr - vy) / self.dy
            dvz_dz = (vzr - vz) / self.dz
            # F_S correction in each direction.
            # In x-direction flux for S_x: -(4/3) eta dvx/dx
            list_xp = list(flux_xp)
            list_xp[1] = list_xp[1] - (4.0 / 3.0) * self.eta * dvx_dx
            list_xp[4] = list_xp[4] - (4.0 / 3.0) * self.eta * dvx_dx * 0.5 * (vx + vxr)
            flux_xp = tuple(list_xp)
            list_yp = list(flux_yp)
            list_yp[2] = list_yp[2] - (4.0 / 3.0) * self.eta * dvy_dy
            list_yp[4] = list_yp[4] - (4.0 / 3.0) * self.eta * dvy_dy * 0.5 * (vy + vyr)
            flux_yp = tuple(list_yp)
            list_zp = list(flux_zp)
            list_zp[3] = list_zp[3] - (4.0 / 3.0) * self.eta * dvz_dz
            list_zp[4] = list_zp[4] - (4.0 / 3.0) * self.eta * dvz_dz * 0.5 * (vz + vzr)
            flux_zp = tuple(list_zp)

        # Differences (flux_{i+1/2} - flux_{i-1/2}) / d{axis}.
        def axis_div(flux_p, axis, dh):
            flux_m = tuple(self._shift(f, axis, -1) for f in flux_p)
            return tuple((fp - fm) / dh for fp, fm in zip(flux_p, flux_m))

        dx_terms = axis_div(flux_xp, 0, self.dx)
        dy_terms = axis_div(flux_yp, 1, self.dy)
        dz_terms = axis_div(flux_zp, 2, self.dz)

        # dU/dt = -(sum of axis divergences)
        rhs_D = -(dx_terms[0] + dy_terms[0] + dz_terms[0])
        rhs_Sx = -(dx_terms[1] + dy_terms[1] + dz_terms[1])
        rhs_Sy = -(dx_terms[2] + dy_terms[2] + dz_terms[2])
        rhs_Sz = -(dx_terms[3] + dy_terms[3] + dz_terms[3])
        rhs_tau = -(dx_terms[4] + dy_terms[4] + dz_terms[4])

        # Optional external body force on the momentum equation.
        if body_force is not None:
            fx, fy, fz = body_force
            rhs_Sx = rhs_Sx + fx
            rhs_Sy = rhs_Sy + fy
            rhs_Sz = rhs_Sz + fz
            # Power input to the energy equation: v . f.
            rhs_tau = rhs_tau + vx * fx + vy * fy + vz * fz

        return rhs_D, rhs_Sx, rhs_Sy, rhs_Sz, rhs_tau

    # ---- time step ----------------------------------------------------------
    def _max_signal(self):
        rho, p, vx, vy, vz, _ = primitive_from_conservative_3d(
            self.D, self.Sx, self.Sy, self.Sz, self.tau, self.Gamma,
        )
        cs = sound_speed(rho, p, self.Gamma)
        return float(np.max(np.maximum.reduce([
            signal_speed(vx, cs), signal_speed(vy, cs), signal_speed(vz, cs),
        ])))

    def step(self, dt: float = None, body_force=None):
        if dt is None:
            alpha = self._max_signal()
            dt_adv = self.cfl * min(self.dx, self.dy, self.dz) / max(alpha, 1e-12)
            if self.eta != 0.0:
                rho, _, _, _, _, _ = primitive_from_conservative_3d(
                    self.D, self.Sx, self.Sy, self.Sz, self.tau, self.Gamma,
                )
                nu_max = (4.0 / 3.0) * abs(self.eta) / max(float(np.min(rho)), 1e-12)
                dh2 = min(self.dx, self.dy, self.dz) ** 2
                dt_visc = self.cfl * dh2 / (2.0 * nu_max)
                dt = min(dt_adv, dt_visc)
            else:
                dt = dt_adv

        U0 = (self.D, self.Sx, self.Sy, self.Sz, self.tau)
        k1 = self.rhs(*U0, body_force=body_force)
        U2 = tuple(u + 0.5 * dt * k for u, k in zip(U0, k1))
        k2 = self.rhs(*U2, body_force=body_force)
        U3 = tuple(u + 0.5 * dt * k for u, k in zip(U0, k2))
        k3 = self.rhs(*U3, body_force=body_force)
        U4 = tuple(u + dt * k for u, k in zip(U0, k3))
        k4 = self.rhs(*U4, body_force=body_force)
        new = tuple(
            u + (dt / 6.0) * (a + 2 * b + 2 * c + d)
            for u, a, b, c, d in zip(U0, k1, k2, k3, k4)
        )
        self.D, self.Sx, self.Sy, self.Sz, self.tau = new
        self.t += dt
        return dt

    def evolve(self, t_end: float, callback=None):
        while self.t < t_end - 1e-12:
            self.step()
            if callback is not None:
                callback(self)

    # ---- diagnostics --------------------------------------------------------
    def primitives(self):
        return primitive_from_conservative_3d(
            self.D, self.Sx, self.Sy, self.Sz, self.tau, self.Gamma,
        )

    def cell_volume(self):
        return self.dx * self.dy * self.dz

    def total_mass(self) -> float:
        return float(np.sum(self.D)) * self.cell_volume()

    def total_energy(self) -> float:
        return float(np.sum(self.tau + self.D)) * self.cell_volume()

    def total_momentum(self) -> Tuple[float, float, float]:
        v = self.cell_volume()
        return (float(np.sum(self.Sx)) * v,
                float(np.sum(self.Sy)) * v,
                float(np.sum(self.Sz)) * v)
