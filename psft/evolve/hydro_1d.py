"""1+1D relativistic fluid solver -- numerical integration of the inviscid
PSFT master equation in the limit of a fluid on a flat Minkowski background.

Paper Theorem 12.1: in the inviscid limit, the v2 master equation reduces
to the relativistic Euler equations.  We discretise these on a uniform 1D
spatial grid with a Lax-Friedrichs flux, evolved in time by a 3+1
method-of-lines RK4 stepper.

State variables (conservative form):
    D    = gamma rho                       lab-frame rest-mass density
    S    = (rho + p + rho eps) gamma^2 v   momentum density
    tau  = (rho + p + rho eps) gamma^2 - p - D
                                            energy density (excess over D)

Equations of motion:
    dt D    + dx (D v)         = 0
    dt S    + dx (S v + p)     = 0
    dt tau  + dx (S - D v)     = 0

Equation of state (Gamma-law ideal gas):
    p = (Gamma - 1) rho eps
    h = 1 + Gamma eps                       (specific enthalpy)

A viscous extension (paper Modification 1: conformal viscosity, zeta = 0)
is provided by `viscous_flux_correction`; turning it on with non-zero
`eta` lets the solver evolve the simplest viscous shear configurations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Primitive recovery and EOS
# ---------------------------------------------------------------------------
def primitive_from_conservative(
    D: np.ndarray, S: np.ndarray, tau: np.ndarray,
    Gamma: float, max_iter: int = 80, tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Newton-Raphson recovery of (rho, p, v, W) from (D, S, tau).

    For Gamma-law p = (Gamma-1) rho eps, the variable x = tau + p + D = rho h W^2
    satisfies the closed-form algebraic relation
        F(x) = x^2 - Gamma A x + (Gamma-1) S^2 + (Gamma-1) D sqrt(x^2 - S^2) = 0,
    with A = tau + D.  Derivation: combine h = sqrt(x^2-S^2)/D and the
    Gamma-law eps = (h-1)/Gamma into p = (Gamma-1) rho eps, then use
    p = x - A.

    F'(x) = 2x - Gamma A + (Gamma-1) D x / sqrt(x^2 - S^2).

    Vectorised over the grid via numpy.  Returns rho, p, v, W (each same
    shape as D).
    """
    A = tau + D
    safe_A = np.where(A > 0, A, 1e-12)
    # Initial guess: non-relativistic limit x ~ A + S^2/(2 A).
    x = A + 0.5 * S * S / safe_A
    x = np.maximum(x, 1.0001 * np.abs(S))      # keep x > |S|

    for _ in range(max_iter):
        x2_minus_S2 = np.maximum(x * x - S * S, 1e-30)
        sqrt_term = np.sqrt(x2_minus_S2)
        F = (x * x - Gamma * A * x
             + (Gamma - 1.0) * S * S
             + (Gamma - 1.0) * D * sqrt_term)
        Fp = (2.0 * x - Gamma * A
              + (Gamma - 1.0) * D * x / sqrt_term)
        Fp = np.where(np.abs(Fp) < 1e-15, 1e-15, Fp)
        dx = -F / Fp
        x_new = x + dx
        bad = x_new * x_new <= S * S
        x_new = np.where(bad, 0.5 * (x + 1.0001 * np.abs(S)), x_new)
        x = x_new
        if np.max(np.abs(F)) < tol * max(float(np.max(np.abs(x * x))), 1.0):
            break

    p = x - A
    p = np.maximum(p, 1e-20)
    v = S / x
    W = x / np.sqrt(np.maximum(x * x - S * S, 1e-30))
    rho = D / W
    rho = np.maximum(rho, 1e-20)
    return rho, p, v, W


def conservative_from_primitive(
    rho: np.ndarray, p: np.ndarray, v: np.ndarray, Gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (D, S, tau) from primitive (rho, p, v).  Gamma-law EOS."""
    eps = p / ((Gamma - 1.0) * rho)
    h = 1.0 + Gamma * eps
    W = 1.0 / np.sqrt(1.0 - v * v)
    D = rho * W
    S = rho * h * W * W * v
    tau = rho * h * W * W - p - D
    return D, S, tau


# ---------------------------------------------------------------------------
# Fluxes
# ---------------------------------------------------------------------------
def euler_flux(D: np.ndarray, S: np.ndarray, tau: np.ndarray,
               rho: np.ndarray, p: np.ndarray, v: np.ndarray):
    """Physical flux (F_D, F_S, F_tau) at each grid point."""
    F_D = D * v
    F_S = S * v + p
    F_tau = S - D * v
    return F_D, F_S, F_tau


def signal_speed(v: np.ndarray, c_s: np.ndarray) -> np.ndarray:
    """Maximum relativistic signal speed |v +/- c_s| / (1 +/- v c_s)."""
    plus = np.abs((v + c_s) / (1.0 + v * c_s))
    minus = np.abs((v - c_s) / (1.0 - v * c_s))
    return np.maximum(plus, minus)


def sound_speed(rho: np.ndarray, p: np.ndarray, Gamma: float) -> np.ndarray:
    """Relativistic sound speed c_s^2 = Gamma p / (rho h).

    For Gamma-law: h = 1 + Gamma p / [(Gamma-1) rho].
    """
    eps = p / ((Gamma - 1.0) * np.maximum(rho, 1e-30))
    h = 1.0 + Gamma * eps
    cs2 = Gamma * p / (rho * h)
    return np.sqrt(np.maximum(cs2, 0.0))


def lax_friedrichs_flux(
    U_L: Tuple[np.ndarray, ...], U_R: Tuple[np.ndarray, ...],
    F_L: Tuple[np.ndarray, ...], F_R: Tuple[np.ndarray, ...],
    alpha: np.ndarray,
):
    """Local Lax-Friedrichs (Rusanov) flux between cells.

    F = (F_L + F_R) / 2 - (alpha / 2) (U_R - U_L)
    where alpha is the max signal speed at the interface.
    """
    return tuple(
        0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        for fL, fR, uL, uR in zip(F_L, F_R, U_L, U_R)
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class RelativisticEulerSolver1D:
    """1+1D relativistic Euler on a periodic uniform grid.

    Use `step()` to advance by one RK4 step; `evolve()` to advance to a
    target time.  Setting `eta > 0` activates a simple conformal viscous
    flux correction (paper Modification 1).
    """
    N: int                                  # number of cells
    L: float = 1.0                          # domain length
    Gamma: float = 4.0 / 3.0                # adiabatic index
    cfl: float = 0.4                        # CFL safety factor
    eta: float = 0.0                        # shear viscosity (conformal)
    boundary: str = "periodic"              # "periodic" or "outflow"

    # State (set by initialise / step).
    D: np.ndarray = field(default=None, init=False)
    S: np.ndarray = field(default=None, init=False)
    tau: np.ndarray = field(default=None, init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.linspace(0.5 * self.dx, self.L - 0.5 * self.dx, self.N)

    # ---- initial conditions -------------------------------------------------
    def initialise(self, rho_func, p_func, v_func):
        """Set initial state from user-supplied primitive functions of x."""
        rho = rho_func(self.x)
        p = p_func(self.x)
        v = v_func(self.x)
        self.D, self.S, self.tau = conservative_from_primitive(
            rho, p, v, self.Gamma,
        )
        self.t = 0.0

    # ---- boundary conditions ------------------------------------------------
    def _bc(self, U: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            # numpy.roll handles this implicitly in finite-volume flux loops.
            return U
        elif self.boundary == "outflow":
            # Copy first / last interior to ghost (zero gradient).
            return U
        raise ValueError(f"unknown boundary: {self.boundary}")

    def _neighbour(self, U: np.ndarray, shift: int) -> np.ndarray:
        """Return U shifted by `shift` (positive = forward) with chosen BC."""
        if self.boundary == "periodic":
            return np.roll(U, -shift)        # U[i] -> U[i+shift]
        elif self.boundary == "outflow":
            out = np.empty_like(U)
            if shift > 0:
                out[:-shift] = U[shift:]
                out[-shift:] = U[-1]
            elif shift < 0:
                out[-shift:] = U[:shift]
                out[:-shift] = U[0]
            else:
                out = U.copy()
            return out
        raise ValueError(f"unknown boundary: {self.boundary}")

    # ---- RHS evaluation -----------------------------------------------------
    def rhs(self, D, S, tau):
        rho, p, v, W = primitive_from_conservative(D, S, tau, self.Gamma)
        cs = sound_speed(rho, p, self.Gamma)
        F_D, F_S, F_tau = euler_flux(D, S, tau, rho, p, v)

        # Build right-neighbour arrays for interface values.
        Drp, Srp, taurp = self._neighbour(D, 1), self._neighbour(S, 1), self._neighbour(tau, 1)
        rho_R, p_R, v_R, _ = primitive_from_conservative(
            Drp, Srp, taurp, self.Gamma,
        )
        cs_R = sound_speed(rho_R, p_R, self.Gamma)
        F_DR, F_SR, F_tauR = euler_flux(Drp, Srp, taurp, rho_R, p_R, v_R)

        alpha = np.maximum(signal_speed(v, cs), signal_speed(v_R, cs_R))
        flux_D, flux_S, flux_tau = lax_friedrichs_flux(
            (D, S, tau), (Drp, Srp, taurp),
            (F_D, F_S, F_tau), (F_DR, F_SR, F_tauR),
            alpha,
        )
        # Optional viscous correction (conformal, 1D: tau^visc_{xx} = (4/3) eta dv/dx).
        if self.eta != 0.0:
            dv_dx = (v_R - v) / self.dx          # right-difference (interface)
            flux_S = flux_S - (4.0 / 3.0) * self.eta * dv_dx
            # Energy gets viscous-heating contribution (Euler-equation form):
            # tau equation receives v · viscous-stress at the interface.
            v_iface = 0.5 * (v + v_R)
            flux_tau = flux_tau - (4.0 / 3.0) * self.eta * dv_dx * v_iface

        # Now flux_{i+1/2}.  dU/dt = -(flux_{i+1/2} - flux_{i-1/2}) / dx.
        flux_D_im = self._neighbour(flux_D, -1)
        flux_S_im = self._neighbour(flux_S, -1)
        flux_tau_im = self._neighbour(flux_tau, -1)
        dDdt = -(flux_D - flux_D_im) / self.dx
        dSdt = -(flux_S - flux_S_im) / self.dx
        dtdt = -(flux_tau - flux_tau_im) / self.dx
        return dDdt, dSdt, dtdt

    # ---- time step ----------------------------------------------------------
    def _max_signal(self) -> float:
        rho, p, v, _ = primitive_from_conservative(self.D, self.S, self.tau, self.Gamma)
        cs = sound_speed(rho, p, self.Gamma)
        return float(np.max(signal_speed(v, cs)))

    def step(self, dt: float = None):
        """Advance state by one RK4 step.  If dt is None, choose via CFL.

        Two stability limits apply:
          * Advective:  dt < cfl * dx / max_signal_speed
          * Viscous:    dt < cfl * dx^2 / (2 * (4/3) * eta / rho_min)

        The viscous bound is active whenever `self.eta > 0`.
        """
        if dt is None:
            alpha = self._max_signal()
            dt_adv = self.cfl * self.dx / max(alpha, 1e-12)
            if self.eta != 0.0:
                rho_now, _, _, _ = primitive_from_conservative(
                    self.D, self.S, self.tau, self.Gamma,
                )
                nu_max = (4.0 / 3.0) * abs(self.eta) / max(float(np.min(rho_now)), 1e-12)
                dt_visc = self.cfl * self.dx * self.dx / (2.0 * nu_max)
                dt = min(dt_adv, dt_visc)
            else:
                dt = dt_adv

        D0, S0, tau0 = self.D, self.S, self.tau
        k1 = self.rhs(D0, S0, tau0)
        k2 = self.rhs(D0 + 0.5 * dt * k1[0], S0 + 0.5 * dt * k1[1], tau0 + 0.5 * dt * k1[2])
        k3 = self.rhs(D0 + 0.5 * dt * k2[0], S0 + 0.5 * dt * k2[1], tau0 + 0.5 * dt * k2[2])
        k4 = self.rhs(D0 + dt * k3[0], S0 + dt * k3[1], tau0 + dt * k3[2])
        self.D = D0 + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.S = S0 + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.tau = tau0 + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.t += dt
        return dt

    def evolve(self, t_end: float, snapshot_callback: Callable = None,
               n_snapshots: int = 0):
        """Evolve to t_end, optionally recording snapshots at regular intervals."""
        snapshots = []
        next_snap_time = self.t
        if n_snapshots > 0:
            snap_dt = (t_end - self.t) / n_snapshots
            next_snap_time = self.t + snap_dt
        while self.t < t_end - 1e-12:
            dt = self.step()
            if n_snapshots > 0 and self.t >= next_snap_time - 1e-12:
                if snapshot_callback is not None:
                    snapshots.append(snapshot_callback(self))
                next_snap_time += snap_dt
        return snapshots

    # ---- diagnostics --------------------------------------------------------
    def primitives(self):
        return primitive_from_conservative(self.D, self.S, self.tau, self.Gamma)

    def total_energy(self) -> float:
        return float(np.sum(self.tau + self.D)) * self.dx

    def total_momentum(self) -> float:
        return float(np.sum(self.S)) * self.dx

    def total_mass(self) -> float:
        return float(np.sum(self.D)) * self.dx
