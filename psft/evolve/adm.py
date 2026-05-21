"""3+1 ADM evolution of the spatial geometry on a Cartesian grid (Step 5.1
of the simulation roadmap: curved-background extension).

Implements the standard BSSN reformulation of the ADM equations
(Baumgarte-Shapiro 2nd ed., chapter 11; Alcubierre, chapter 5).
BSSN is preferred over vanilla ADM because vanilla ADM is only weakly
hyperbolic in standard gauges and is unstable for any non-trivial
evolution; BSSN with the moving-puncture gauge has been the de-facto
standard for binary-black-hole simulations since 2005.

State variables (24 components per grid cell):

    chi             conformal factor, chi = (det gamma)^{-1/3};
                    physical metric gamma_ij = (1/chi) gammabar_ij.
    gammabar_ij     conformal 3-metric (det = 1), 6 indep. symmetric.
    K               trace of extrinsic curvature, K = gamma^ij K_ij.
    Abar_ij         conformal trace-free extrinsic curvature
                    (Abar_ij = chi (K_ij - (1/3) gamma_ij K)), 6 components.
    Gammabar^i      conformal connection functions,
                    Gammabar^i = gammabar^jk Gammabar^i_jk.
    alpha           lapse (gauge).
    beta^i          shift vector (gauge).
    B^i             Gamma-driver shift auxiliary.

In this Phase 1 implementation we use:
  * 4th-order centred finite differences for spatial derivatives.
  * Standard RK4 from `psft.evolve.integrators`.
  * Periodic boundary conditions (suitable for Minkowski + small
    perturbations; radiative outer BCs are deferred to Phase 1.3 once
    Schwarzschild puncture runs cleanly).
  * 1+log lapse + Gamma-driver shift (the moving-puncture gauge).
  * Algebraic constraint projections every step:
        tr(Abar) = 0   (subtract trace of Abar_ij)
        det(gammabar) = 1   (rescale by det^{-1/3})
  * 6th-order Kreiss-Oliger dissipation on every BSSN field.

The Hamiltonian and momentum constraints are NOT actively enforced
(monitored only); BSSN with the algebraic projections above is known to
be stable for the test problems of Phase 1.

References (in order of utility for this file):
  Baumgarte & Shapiro, _Numerical Relativity_ (2nd ed.), Cambridge (2010),
    eqs. 11.49-11.54 + gauge eqs.
  Alcubierre, _Introduction to 3+1 Numerical Relativity_, Oxford (2008).
  Brügmann et al., Phys. Rev. D 77, 024027 (2008) -- moving-puncture
    gauge upwinding details.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math
import numpy as np


# ---------------------------------------------------------------------------
# Component indexing
# ---------------------------------------------------------------------------
# Single flat (NCOMP, Nx, Ny, Nz) array layout.  Symmetric 3x3 tensors
# are stored as the six independent components in the order xx, xy, xz,
# yy, yz, zz.  Indices into the flat axis:
#
# Layout below is BSSN + Z4c-Theta (Hilditch et al. 2013), 25 components.
# Field 24 (Theta) is the Z4c Hamiltonian-constraint scalar; with
# kappa1 = 0 (Z4c damping switched off in the evolver) it stays at 0
# identically and the system reduces to plain BSSN.
NCOMP = 25
IDX_CHI         = 0
IDX_GBAR        = slice(1, 7)    # gammabar: xx, xy, xz, yy, yz, zz
IDX_K           = 7
IDX_ABAR        = slice(8, 14)   # Abar:     xx, xy, xz, yy, yz, zz
IDX_GAMMABAR_U  = slice(14, 17)  # Gammabar^i: x, y, z
IDX_ALPHA       = 17
IDX_BETA        = slice(18, 21)  # beta^i:   x, y, z
IDX_B           = slice(21, 24)  # B^i (Gamma-driver aux): x, y, z
IDX_THETA       = 24             # Z4c Hamiltonian-constraint scalar

# Symmetric-tensor packing helper: a (6,) flat -> (3,3) matrix.
_SYM_IJ = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_PACK_MAT = np.array([
    [0, 1, 2],
    [1, 3, 4],
    [2, 4, 5],
])

def _sym_to_3x3(sym6):
    """Convert symmetric tensor (6, ...) to full (3, 3, ...)."""
    return sym6[_PACK_MAT]


def _3x3_to_sym(m33):
    """Convert symmetric (3, 3, ...) to packed (6, ...)."""
    return np.stack([m33[i, j] for i, j in _SYM_IJ], axis=0)


# ---------------------------------------------------------------------------
# Spatial derivatives (4th order centred on a periodic grid)
# ---------------------------------------------------------------------------
def _d1(F: np.ndarray, axis: int, dh: float) -> np.ndarray:
    """4th-order centred first derivative on a periodic grid."""
    return (
        -np.roll(F, 2, axis=axis) + 8 * np.roll(F, 1, axis=axis)
        - 8 * np.roll(F, -1, axis=axis) + np.roll(F, -2, axis=axis)
    ) / (12 * dh)


def _d2(F: np.ndarray, axis: int, dh: float) -> np.ndarray:
    """4th-order centred second derivative on a periodic grid."""
    return (
        -np.roll(F, 2, axis=axis) + 16 * np.roll(F, 1, axis=axis)
        - 30 * F
        + 16 * np.roll(F, -1, axis=axis) - np.roll(F, -2, axis=axis)
    ) / (12 * dh * dh)


def _d_mixed(F: np.ndarray, axis_a: int, axis_b: int, dh_a: float, dh_b: float) -> np.ndarray:
    """4th-order centred mixed second derivative d^2 F / (dx_a dx_b)."""
    if axis_a == axis_b:
        return _d2(F, axis_a, dh_a)
    return _d1(_d1(F, axis_a, dh_a), axis_b, dh_b)


def _kreiss_oliger(F: np.ndarray, axis: int, dh: float) -> np.ndarray:
    """6th-order Kreiss-Oliger dissipation.

    Adds the term -(epsilon * dh^5 / 64) * d^6 F / dx^6 to suppress
    grid-frequency modes.  We return the unweighted d^6 F; callers
    multiply by epsilon * dh^5 / 64.
    """
    return (
        np.roll(F, 3, axis=axis) - 6 * np.roll(F, 2, axis=axis)
        + 15 * np.roll(F, 1, axis=axis) - 20 * F
        + 15 * np.roll(F, -1, axis=axis) - 6 * np.roll(F, -2, axis=axis)
        + np.roll(F, -3, axis=axis)
    )


# ---------------------------------------------------------------------------
# BSSN state container
# ---------------------------------------------------------------------------
@dataclass
class BSSNState:
    """Wrapper around a (24, Nx, Ny, Nz) numpy array exposing named
    accessors.  Mutations go through the underlying array.
    """
    data: np.ndarray   # shape (24, Nx, Ny, Nz)
    dx: float
    dy: float
    dz: float

    @classmethod
    def empty(cls, Nx: int, Ny: int, Nz: int, dx: float, dy: float, dz: float):
        return cls(data=np.zeros((NCOMP, Nx, Ny, Nz)), dx=dx, dy=dy, dz=dz)

    @classmethod
    def flat_minkowski(cls, Nx: int, Ny: int, Nz: int,
                       dx: float, dy: float, dz: float):
        """Flat-Minkowski initial data: gamma_ij = delta_ij, K_ij = 0,
        alpha = 1, beta^i = 0.  Phase 1 acceptance test 1."""
        s = cls.empty(Nx, Ny, Nz, dx, dy, dz)
        s.data[IDX_CHI] = 1.0
        # gammabar = delta_ij  ->  diagonal entries (xx, yy, zz) = 1
        s.data[IDX_GBAR.start + 0] = 1.0   # xx
        s.data[IDX_GBAR.start + 3] = 1.0   # yy
        s.data[IDX_GBAR.start + 5] = 1.0   # zz
        # K = 0, Abar = 0, Gammabar^i = 0
        s.data[IDX_ALPHA] = 1.0
        # beta = 0, B = 0
        return s

    @classmethod
    def static_dust_ball(cls, Nx: int, Ny: int, Nz: int,
                          dx: float, dy: float, dz: float,
                          rho_rest: np.ndarray,
                          n_jacobi: int = 5000,
                          tol: float = 1e-10,
                          pre_collapsed_lapse: bool = True):
        """Static (K_ij = 0) self-gravitating dust ball.  Phase 1.3
        acceptance test.

        Solves the Lichnerowicz form of the Hamiltonian constraint
        for conformally flat, time-symmetric initial data:

            nabla^2 psi = -2 pi psi^5 rho_rest

        by Jacobi fixed-point iteration on a periodic grid.  The
        physical 3-metric is gamma_ij = psi^4 delta_ij; the conformal
        factor of BSSN is chi = psi^{-4}.

        Periodic-grid Jacobi requires the mean of the source to be
        subtracted (otherwise no solution exists with periodic BCs);
        we therefore solve for the perturbation around the box-mean
        gravitational potential.  This is an approximation valid when
        the dust is localised and the box is large enough that the
        boundary effects are negligible.

        Parameters
        ----------
        rho_rest : array
            Rest-mass density, shape (Nx, Ny, Nz).  Positive everywhere.
        n_jacobi : int
            Maximum Jacobi iterations.
        tol : float
            Convergence tolerance on max delta-psi per iteration.
        pre_collapsed_lapse : bool
            If True, initial alpha = psi^{-2} (pre-collapsed, the
            moving-puncture-style choice); else alpha = 1.

        Returns the constructed BSSNState plus the matter source
        rho_adm = rho_rest (since the matter is at rest, u^a = n^a
        and the ADM energy density equals the rest-mass density).
        """
        s = cls.empty(Nx, Ny, Nz, dx, dy, dz)
        s.data[IDX_GBAR.start + 0] = 1.0
        s.data[IDX_GBAR.start + 3] = 1.0
        s.data[IDX_GBAR.start + 5] = 1.0
        # K = 0, Abar = 0, Gammabar^i = 0
        # Jacobi iteration on psi satisfying:
        #   (lap psi) - (-2 pi psi^5 rho_rest) = 0
        # with mean-subtracted RHS (periodic BCs require zero mean).
        rho_mean = float(np.mean(rho_rest))
        rho_src = rho_rest - rho_mean
        psi = np.ones((Nx, Ny, Nz))
        h2 = dx * dx   # assume cubic cell for simplicity
        for it in range(n_jacobi):
            # Discrete Laplacian = (sum of 6 neighbours - 6 centre) / h^2
            # Solve for centre: centre = (sum_neighbours - h^2 * src) / 6
            neigh = (
                np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
                + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
                + np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2)
            )
            psi_new = (neigh + h2 * 2.0 * math.pi * psi ** 5 * rho_src) / 6.0
            # Normalize psi to have mean 1 (we removed the mean from
            # rho_src; psi mean is set by the global structure).
            psi_new /= float(np.mean(psi_new))
            delta = float(np.max(np.abs(psi_new - psi)))
            psi = psi_new
            if delta < tol:
                break
        else:
            print(f"  [static_dust_ball] Jacobi did not converge to tol={tol:.1e} "
                  f"(final delta = {delta:.2e})")
        s.data[IDX_CHI] = psi ** (-4)
        if pre_collapsed_lapse:
            s.data[IDX_ALPHA] = psi ** (-2)
        else:
            s.data[IDX_ALPHA] = 1.0
        return s, rho_rest

    @classmethod
    def schwarzschild_isotropic(cls, Nx: int, Ny: int, Nz: int,
                                 dx: float, dy: float, dz: float,
                                 M: float = 1.0,
                                 x0: float = None, y0: float = None,
                                 z0: float = None,
                                 r_floor: Optional[float] = None):
        """Schwarzschild puncture in isotropic coordinates.  Phase 1
        acceptance test 2.

        In isotropic coordinates the spatial metric is conformally flat:
            gamma_ij = psi^4 delta_ij    with    psi = 1 + M/(2 r),
        so the conformal metric gammabar_ij = delta_ij and the
        conformal factor chi = psi^{-4} = (1 + M/(2r))^{-4}.

        The extrinsic curvature K_ij = 0 (time-symmetric Brill-Lindquist
        initial slice), so K = 0 and Abar = 0.  The Conformal connection
        Gammabar^i = 0 (gammabar is flat).

        Lapse and shift are gauge: we use the "pre-collapsed" lapse
        alpha = psi^{-2} (which is positive at the puncture and is the
        standard choice for moving-puncture initial data), beta^i = 0,
        B^i = 0.

        This initial data satisfies the Hamiltonian and momentum
        constraints exactly in the continuum (vacuum,
        time-symmetric); on the grid the constraints are satisfied to
        finite-difference order.

        Parameters
        ----------
        M : float
            Schwarzschild mass.  The puncture sits at (x0, y0, z0) and
            the horizon is at r_iso = M/2 in isotropic coordinates.
        x0, y0, z0 : float
            Puncture location; defaults to box centre.
        r_floor : float
            Minimum coordinate radius (to avoid 1/r divergence at the
            puncture itself).  Default M/100.
        """
        L_x, L_y, L_z = Nx * dx, Ny * dy, Nz * dz
        if x0 is None: x0 = 0.5 * L_x
        if y0 is None: y0 = 0.5 * L_y
        if z0 is None: z0 = 0.5 * L_z
        if r_floor is None: r_floor = M / 100.0

        s = cls.empty(Nx, Ny, Nz, dx, dy, dz)
        x = np.linspace(0.5 * dx, L_x - 0.5 * dx, Nx)
        y = np.linspace(0.5 * dy, L_y - 0.5 * dy, Ny)
        z = np.linspace(0.5 * dz, L_z - 0.5 * dz, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
        r = np.maximum(r, r_floor)

        psi = 1.0 + M / (2.0 * r)
        # gammabar_ij = delta_ij (conformally flat)
        s.data[IDX_GBAR.start + 0] = 1.0
        s.data[IDX_GBAR.start + 3] = 1.0
        s.data[IDX_GBAR.start + 5] = 1.0
        # chi = psi^{-4}
        s.data[IDX_CHI] = psi ** (-4)
        # K = 0, Abar = 0, Gammabar^i = 0  (all already zero)
        # Pre-collapsed lapse: alpha = psi^{-2}
        s.data[IDX_ALPHA] = psi ** (-2)
        # beta = 0, B = 0
        return s

    @property
    def chi(self):     return self.data[IDX_CHI]
    @property
    def gbar(self):    return self.data[IDX_GBAR]    # (6, Nx, Ny, Nz)
    @property
    def K(self):       return self.data[IDX_K]
    @property
    def Abar(self):    return self.data[IDX_ABAR]
    @property
    def Gam_u(self):   return self.data[IDX_GAMMABAR_U]
    @property
    def alpha(self):   return self.data[IDX_ALPHA]
    @property
    def beta(self):    return self.data[IDX_BETA]
    @property
    def B(self):       return self.data[IDX_B]
    @property
    def theta(self):   return self.data[IDX_THETA]


# ---------------------------------------------------------------------------
# Algebraic constraint enforcement
# ---------------------------------------------------------------------------
def project_algebraic_constraints(y: np.ndarray) -> np.ndarray:
    """In-place enforcement of tr(Abar) = 0 and det(gammabar) = 1.

    Without these projections, BSSN evolutions develop unphysical drift
    in the trace of Abar and the determinant of gammabar within ~50
    steps; with them, the same evolution is stable.  This is the single
    most important practical detail of any BSSN implementation.
    """
    # det(gammabar) = 1 enforcement.  Compute determinant of the
    # symmetric 3x3 matrix from the six independent components.
    g = y[IDX_GBAR]    # (6, ...)
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = g[0], g[1], g[2], g[3], g[4], g[5]
    det = (
        g_xx * (g_yy * g_zz - g_yz * g_yz)
        - g_xy * (g_xy * g_zz - g_yz * g_xz)
        + g_xz * (g_xy * g_yz - g_yy * g_xz)
    )
    # Rescale gammabar -> gammabar / det^{1/3} so that the new det = 1.
    # Floor det to avoid numerical issues at coordinate singularities.
    det = np.where(np.abs(det) > 1e-30, det, 1e-30)
    rescale = det ** (-1.0 / 3.0)
    y[IDX_GBAR] *= rescale[np.newaxis, ...]

    # tr(Abar) = 0 enforcement.  trace = gammabar^ij Abar_ij; subtract
    # (1/3) gammabar_ij trace from Abar_ij.
    # Compute gammabar^ij (inverse of conformal metric, det = 1 so
    # adjugate works directly).
    gbar_inv = _inverse_sym3(y[IDX_GBAR])    # (6, ...)
    A = y[IDX_ABAR]    # (6, ...)
    # gammabar^ij Abar_ij = sum_{ij} gbar_inv_ij Abar_ij.
    # For symmetric tensors with the (xx, xy, xz, yy, yz, zz) packing:
    trace = (
        gbar_inv[0] * A[0]
        + 2 * gbar_inv[1] * A[1]
        + 2 * gbar_inv[2] * A[2]
        + gbar_inv[3] * A[3]
        + 2 * gbar_inv[4] * A[4]
        + gbar_inv[5] * A[5]
    )
    # Subtract (1/3) gammabar_ij * trace from Abar_ij.
    y[IDX_ABAR] -= (1.0 / 3.0) * y[IDX_GBAR] * trace[np.newaxis, ...]
    return y


def _inverse_sym3(sym6: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric 3x3 matrix given as 6 packed components.

    For BSSN we have det(gammabar) = 1 (after constraint projection),
    which lets us use the adjugate formula without explicitly dividing
    by the determinant.  We divide anyway for robustness.
    """
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = sym6[0], sym6[1], sym6[2], sym6[3], sym6[4], sym6[5]
    det = (
        g_xx * (g_yy * g_zz - g_yz * g_yz)
        - g_xy * (g_xy * g_zz - g_yz * g_xz)
        + g_xz * (g_xy * g_yz - g_yy * g_xz)
    )
    det = np.where(np.abs(det) > 1e-30, det, 1e-30)
    inv_det = 1.0 / det
    inv_xx = (g_yy * g_zz - g_yz * g_yz) * inv_det
    inv_xy = (g_xz * g_yz - g_xy * g_zz) * inv_det
    inv_xz = (g_xy * g_yz - g_xz * g_yy) * inv_det
    inv_yy = (g_xx * g_zz - g_xz * g_xz) * inv_det
    inv_yz = (g_xy * g_xz - g_xx * g_yz) * inv_det
    inv_zz = (g_xx * g_yy - g_xy * g_xy) * inv_det
    return np.stack([inv_xx, inv_xy, inv_xz, inv_yy, inv_yz, inv_zz], axis=0)


# ---------------------------------------------------------------------------
# BSSN RHS (vacuum)
# ---------------------------------------------------------------------------
def bssn_vacuum_rhs(t: float, y: np.ndarray, dx: float, dy: float, dz: float,
                     eta_shift: float = 1.0,
                     ko_epsilon: float = 0.5,
                     rho_adm: Optional[np.ndarray] = None,
                     S_i: Optional[np.ndarray] = None,
                     S_ij: Optional[np.ndarray] = None,
                     kappa1: float = 0.0,
                     kappa2: float = 0.0) -> np.ndarray:
    """BSSN(+Z4c-Theta) right-hand side dY/dt with optional matter sources.

    Implements equations (11.49-11.54) of Baumgarte-Shapiro 2nd ed.
    with 1+log lapse and Gamma-driver shift.  Matter enters through
    three sources:
        rho_adm : (Nx, Ny, Nz)  ADM energy density (T^{ab} n_a n_b).
        S_i     : (3, Nx, Ny, Nz)  ADM momentum density.
        S_ij    : (6, Nx, Ny, Nz)  ADM stress tensor (symmetric).

    Each defaults to zero (vacuum).

    Z4c constraint damping (Bernuzzi & Hilditch 2009/2010, PRD 81, 084003).
    When `kappa1 > 0`, three Z4c modifications relative to BSSN are
    enabled (the minimal Theta-only variant, dropping the spatial Z^i
    vector):

      1. Conformal-factor modification (eq. 14 of BH 2010):
             d_t chi += (4/3) alpha chi Theta
         The factor +2Theta inside (K + 2Theta) of the chi evolution.
         This is the critical structural piece that gives the constraint
         subsystem hyperbolic propagation.  Without it, the K-Theta
         back-coupling alone does not stabilise BSSN.

      2. K back-coupling (eq. 16 of BH 2010):
             d_t K += alpha * kappa1 * (1 - kappa2) * Theta
         Constraint-violation feedback from the Theta channel.

      3. Theta evolution (eq. 4 of BH 2010):
             d_t Theta = (alpha/2) * H + beta^k d_k Theta
                         - alpha * kappa1 * (2 + kappa2) * Theta
         Sources from the Hamiltonian-constraint violation H, and is
         damped at rate kappa1.  Here
             H = R + (2/3) K^2 - Abar:Abar - 16 pi rho_adm.

    Typical values: kappa1 ~ 0.02 - 0.1 / (light-crossing time), kappa2 = 0.

    With `kappa1 = 0` (default), Theta stays identically zero, all three
    Z4c modifications drop out, and the system is exactly pure BSSN.
    """
    dY = np.zeros_like(y)

    chi   = y[IDX_CHI]
    gbar  = y[IDX_GBAR]
    K     = y[IDX_K]
    Abar  = y[IDX_ABAR]
    Gam_u = y[IDX_GAMMABAR_U]
    alpha = y[IDX_ALPHA]
    beta  = y[IDX_BETA]
    Baux  = y[IDX_B]
    Theta = y[IDX_THETA]

    # ---- derivatives we will need
    # First derivatives of every BSSN scalar/component.  Individual
    # state components have shape (Nx, Ny, Nz) so spatial axis a in
    # {0, 1, 2} maps directly to numpy axis a.
    def d1(F, a):
        return _d1(F, a, (dx, dy, dz)[a])
    def d2(F, a):
        return _d2(F, a, (dx, dy, dz)[a])
    def dmix(F, a, b):
        return _d_mixed(F, a, b, (dx, dy, dz)[a], (dx, dy, dz)[b])

    # Inverse conformal metric (6-component symmetric).
    gbar_inv = _inverse_sym3(gbar)

    # Convenience: full 3x3 forms.
    gbar_m  = _sym_to_3x3(gbar)        # (3, 3, ...)
    gbar_iv = _sym_to_3x3(gbar_inv)    # (3, 3, ...)
    Abar_m  = _sym_to_3x3(Abar)

    # divergence of the shift  d_k beta^k
    div_beta = d1(beta[0], 0) + d1(beta[1], 1) + d1(beta[2], 2)

    # ---- Eq. (1): d_t chi = (2/3) chi (alpha K - d_k beta^k) + beta^k d_k chi
    # Z4c (BH 2010 eq. 14): replace K -> (K + 2 Theta) inside the bracket.
    # The extra term + (2/3) chi alpha (2 Theta) = +(4/3) alpha chi Theta is
    # the structural Z4c modification that makes the constraint subsystem
    # hyperbolic.  Without it, the K-Theta coupling alone does not stabilise
    # the BSSN constraint-violating mode.
    dchi_dx = [d1(chi, a) for a in range(3)]
    dY[IDX_CHI] = ((2.0 / 3.0) * chi * (alpha * K - div_beta)
                   + beta[0] * dchi_dx[0] + beta[1] * dchi_dx[1] + beta[2] * dchi_dx[2])
    if kappa1 != 0.0:
        dY[IDX_CHI] += (4.0 / 3.0) * alpha * chi * Theta

    # ---- Eq. (2): d_t gammabar_ij
    # = -2 alpha Abar_ij
    #   + beta^k d_k gammabar_ij
    #   + gammabar_ik d_j beta^k + gammabar_jk d_i beta^k
    #   - (2/3) gammabar_ij d_k beta^k
    dgbar_dk = np.stack([
        np.stack([d1(gbar[c], a) for a in range(3)], axis=0)
        for c in range(6)
    ], axis=0)    # shape (6, 3, ...)
    dbeta = np.stack([
        np.stack([d1(beta[c], a) for a in range(3)], axis=0)
        for c in range(3)
    ], axis=0)    # shape (3, 3, ...) where dbeta[c, a] = d_a beta^c

    for n, (i, j) in enumerate(_SYM_IJ):
        # Lie-derivative-along-beta term
        adv = sum(beta[k] * dgbar_dk[n, k] for k in range(3))
        # Stretch term: gammabar_ik d_j beta^k + gammabar_jk d_i beta^k
        stretch = sum(gbar_m[i, k] * dbeta[k, j] + gbar_m[j, k] * dbeta[k, i]
                      for k in range(3))
        # Trace removal
        trace = (2.0 / 3.0) * gbar_m[i, j] * div_beta
        dY[IDX_GBAR.start + n] = -2.0 * alpha * Abar_m[i, j] + adv + stretch - trace

    # ---- Physical metric: gamma_ij = gammabar_ij / chi
    # Christoffel symbols of gammabar (used in eq. 5 + Ricci).
    # Gambar^k_ij = (1/2) gammabar^kl ( d_i gammabar_jl + d_j gammabar_il - d_l gammabar_ij )
    Gambar_lower = np.zeros((3, 3, 3) + chi.shape)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                # d_i gammabar_jl
                n_jl = _PACK_MAT[j, l]
                Gambar_lower[i, j, l] = 0.5 * (
                    dgbar_dk[_PACK_MAT[j, l], i]
                    + dgbar_dk[_PACK_MAT[i, l], j]
                    - dgbar_dk[_PACK_MAT[i, j], l]
                )

    Gambar = np.einsum('km...,mij...->kij...', gbar_iv, Gambar_lower)
    # ^ shape (k, i, j, ...)
    # gbar_iv has shape (3, 3, ...); we converted via _sym_to_3x3.

    # ---- Eq. (3): d_t K = -D^i D_i alpha + alpha (Abar_ij Abar^ij + K^2/3)
    #                       + beta^i d_i K
    # We need D^i D_i alpha (Laplace-Beltrami wrt physical metric).
    # D_i D_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha   (physical Christoffel)
    #
    # Physical Christoffel relates to conformal via
    #   Gamma^k_ij = Gambar^k_ij - (1/(2 chi)) ( d_i chi delta^k_j
    #                + d_j chi delta^k_i - gammabar_ij gammabar^kl d_l chi )
    # (Baumgarte-Shapiro eq. 11.46 reformulated for chi.)
    dalpha = [d1(alpha, a) for a in range(3)]
    ddalpha = np.zeros((3, 3) + alpha.shape)
    for a in range(3):
        for b in range(3):
            ddalpha[a, b] = dmix(alpha, a, b)

    # Build physical-metric Christoffel from gammabar Christoffel and chi:
    # Gamma_phys^k_ij = Gambar^k_ij + C^k_ij where C^k_ij encodes chi terms.
    C_phys = np.zeros((3, 3, 3) + chi.shape)
    # -(1/(2 chi)) * ( d_i chi delta^k_j + d_j chi delta^k_i - gammabar_ij gammabar^kl d_l chi )
    half_over_chi = 0.5 / np.maximum(chi, 1e-30)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                term = 0.0
                if k == j:
                    term += dchi_dx[i]
                if k == i:
                    term += dchi_dx[j]
                # - gammabar_ij gammabar^kl d_l chi
                gtgg = sum(gbar_iv[k, l] * dchi_dx[l] for l in range(3))
                term -= gbar_m[i, j] * gtgg
                C_phys[k, i, j] = -half_over_chi * term

    Gamma_phys = Gambar + C_phys

    # gamma^ij = chi * gammabar^ij
    gamma_iv = chi * gbar_iv

    # Laplacian of alpha: gamma^ij ( d_i d_j alpha - Gamma^k_ij d_k alpha )
    lap_alpha = np.zeros_like(alpha)
    for i in range(3):
        for j in range(3):
            term = ddalpha[i, j]
            for k in range(3):
                term -= Gamma_phys[k, i, j] * dalpha[k]
            lap_alpha += gamma_iv[i, j] * term

    # Abar^ij = gammabar^ik gammabar^jl Abar_kl
    Abar_uu = np.einsum('ik...,jl...,kl...->ij...', gbar_iv, gbar_iv, Abar_m)
    # contract Abar_ij Abar^ij
    AA = np.einsum('ij...,ij...->...', Abar_m, Abar_uu)

    dK_dx = [d1(K, a) for a in range(3)]
    dY[IDX_K] = (-lap_alpha + alpha * (AA + K * K / 3.0)
                 + beta[0] * dK_dx[0] + beta[1] * dK_dx[1] + beta[2] * dK_dx[2])
    # Matter source: + 4 pi alpha (rho + S)
    if rho_adm is not None or S_ij is not None:
        S_trace = 0.0
        if S_ij is not None:
            S_ij_m = _sym_to_3x3(S_ij)
            S_trace = sum(gamma_iv[i, j] * S_ij_m[i, j]
                          for i in range(3) for j in range(3))
        rho_src = rho_adm if rho_adm is not None else 0.0
        dY[IDX_K] += 4.0 * math.pi * alpha * (rho_src + S_trace)
    # Z4c K back-coupling (BH 2010 eq. 16, last line).
    # In the Bernuzzi-Hilditch convention (Theta defined as Theta = -n_a Z^a
    # with positive (alpha/2) H source), the K back-coupling is +alpha kappa1
    # (1 - kappa2) Theta -- positive sign.  Combined with the chi modification
    # and the Theta evolution, this produces a hyperbolic, exponentially-damped
    # constraint subsystem.
    if kappa1 != 0.0:
        dY[IDX_K] += alpha * kappa1 * (1.0 - kappa2) * Theta

    # ---- Eq. (4): d_t Abar_ij
    # = chi (-D_i D_j alpha + alpha R_ij)^{TF}
    #   + alpha (K Abar_ij - 2 Abar_ik gammabar^kl Abar_lj)
    #   + beta^k d_k Abar_ij + Abar_ik d_j beta^k + Abar_jk d_i beta^k
    #   - (2/3) Abar_ij d_k beta^k
    #
    # For Phase 1 Minkowski + Schwarzschild (both vacuum), R_ij is the
    # physical Ricci.  We split: R_ij = R^chi_ij + R^gbar_ij where
    # R^chi_ij encodes the chi-dependent piece and R^gbar_ij is the
    # Ricci of gammabar.  Reference: Baumgarte-Shapiro eqs. 11.51-11.52.

    # Ricci of gammabar: R^gbar_ij.
    # Standard formula in terms of Gambar^k_ij + their derivatives, but
    # the BSSN reformulation uses Gammabar^i as an independent variable.
    # R^gbar_ij = -(1/2) gbar^kl d_k d_l gbar_ij
    #             + gbar_k(i d_j) Gammabar^k
    #             + Gammabar^k Gambar_(ij)k
    #             + gbar^kl ( 2 Gambar^m_k(i Gambar_j)ml + Gambar^m_ik Gambar_mjl )
    # We implement this in the form of Brown 2009.

    # d_k d_l gbar_ij  (for each ij)
    ddgbar = np.zeros((3, 3, 6) + chi.shape)
    for n in range(6):
        for a in range(3):
            for b in range(3):
                ddgbar[a, b, n] = dmix(gbar[n], a, b)

    # d_j Gammabar^i  (one derivative on each component)
    dGam_u = np.zeros((3, 3) + chi.shape)   # dGam_u[i, j] = d_j Gambar^i
    for i in range(3):
        for j in range(3):
            dGam_u[i, j] = d1(Gam_u[i], j)

    # Ricci of gammabar: standard formula (Baumgarte-Shapiro eq. 11.52).
    Ric_bar = np.zeros((3, 3) + chi.shape)
    for i in range(3):
        for j in range(3):
            # term A: -(1/2) gbar^kl d_k d_l gbar_ij
            n_ij = _PACK_MAT[i, j]
            term_a = -0.5 * sum(
                gbar_iv[k, l] * ddgbar[k, l, n_ij]
                for k in range(3) for l in range(3)
            )
            # term B: gbar_k(i d_j) Gambar^k
            term_b = 0.5 * sum(
                gbar_m[k, i] * dGam_u[k, j] + gbar_m[k, j] * dGam_u[k, i]
                for k in range(3)
            )
            # term C: Gambar^k Gambar_(ij)k = (1/2) Gambar^k ( Gambar_ijk + Gambar_jik )
            # where Gambar_ijk = gbar_il Gambar^l_jk = Gambar_lower's symmetrisation.
            # Actually Brown 2009 form:
            #   Gammabar^k * (1/2)( Gambar_kij + Gambar_kji )
            # with Gambar_kij = gbar_kl Gambar^l_ij  (lower-index Christoffel).
            term_c = 0.5 * sum(
                Gam_u[k] * (Gambar_lower[k, i, j] + Gambar_lower[k, j, i])
                for k in range(3)
            )
            # term D: gbar^kl ( 2 Gambar^m_k(i Gambar_j)ml + Gambar^m_ik Gambar_mjl )
            # (the trickiest piece; Brown 2009 eq. 16 has all four contractions)
            term_d = 0.0
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        # 2 Gambar^m_k(i Gambar_j)ml
                        # symmetrised over (i,j): use (1/2)(...+swap)
                        sym = (
                            Gambar[m, k, i] * Gambar_lower[j, m, l]
                            + Gambar[m, k, j] * Gambar_lower[i, m, l]
                        )
                        # Gambar^m_ik Gambar_mjl
                        last = Gambar[m, i, k] * Gambar_lower[m, j, l]
                        term_d += gbar_iv[k, l] * (sym + last)
            Ric_bar[i, j] = term_a + term_b + term_c + term_d

    # Ricci contribution from chi (Baumgarte-Shapiro eq. 11.51 rewritten
    # for chi instead of phi).
    # phi = -ln(chi)/4; d_i phi = -d_i chi / (4 chi); etc.
    # R^chi_ij = -2 D_i D_j phi - 2 gbar_ij gbar^kl D_k D_l phi
    #            + 4 D_i phi D_j phi - 4 gbar_ij gbar^kl D_k phi D_l phi
    # where D is the gammabar-compatible covariant derivative.
    # Working with chi directly:
    #   d_i phi = -(1/4) d_i chi / chi
    #   D_i D_j phi = -(1/4)[ (1/chi) d_i d_j chi - (1/chi^2) d_i chi d_j chi
    #                          - (1/chi) Gambar^k_ij d_k chi ]
    inv_chi = 1.0 / np.maximum(chi, 1e-30)
    ddchi = np.zeros((3, 3) + chi.shape)
    for a in range(3):
        for b in range(3):
            ddchi[a, b] = dmix(chi, a, b)

    DiDj_phi = np.zeros((3, 3) + chi.shape)
    for i in range(3):
        for j in range(3):
            DiDj_phi[i, j] = -0.25 * (
                inv_chi * ddchi[i, j]
                - inv_chi * inv_chi * dchi_dx[i] * dchi_dx[j]
                - inv_chi * sum(Gambar[k, i, j] * dchi_dx[k] for k in range(3))
            )

    # gbar^kl D_k phi D_l phi
    Dphi_squared = sum(
        gbar_iv[k, l] * (-0.25 * inv_chi * dchi_dx[k])
                       * (-0.25 * inv_chi * dchi_dx[l])
        for k in range(3) for l in range(3)
    )

    # gbar^kl D_k D_l phi
    Lap_phi = sum(
        gbar_iv[k, l] * DiDj_phi[k, l]
        for k in range(3) for l in range(3)
    )

    Ric_chi = np.zeros((3, 3) + chi.shape)
    for i in range(3):
        for j in range(3):
            Ric_chi[i, j] = (
                -2.0 * DiDj_phi[i, j]
                - 2.0 * gbar_m[i, j] * Lap_phi
                + 4.0 * (-0.25 * inv_chi * dchi_dx[i])
                      * (-0.25 * inv_chi * dchi_dx[j])
                - 4.0 * gbar_m[i, j] * Dphi_squared
            )

    Ric_phys = Ric_bar + Ric_chi

    # Trace-free part of (-D_i D_j alpha + alpha R_ij), using physical metric.
    # First: D_i D_j alpha = d_i d_j alpha - Gamma_phys^k_ij d_k alpha
    DiDj_alpha = np.zeros((3, 3) + chi.shape)
    for i in range(3):
        for j in range(3):
            DiDj_alpha[i, j] = ddalpha[i, j] - sum(
                Gamma_phys[k, i, j] * dalpha[k] for k in range(3)
            )
    M_ij = -DiDj_alpha + alpha * Ric_phys
    # Matter source: subtract 8 pi alpha S_ij from M_ij (acts inside the
    # trace-free projection so the trace contribution lands on K above).
    if S_ij is not None:
        S_ij_m = _sym_to_3x3(S_ij)
        M_ij = M_ij - 8.0 * math.pi * alpha * S_ij_m
    # Trace: gamma^ij M_ij
    M_trace = sum(
        gamma_iv[i, j] * M_ij[i, j] for i in range(3) for j in range(3)
    )
    # Trace-free: M_ij - (1/3) gamma_ij M_trace
    gamma_m = gbar_m / np.maximum(chi, 1e-30)
    M_TF = M_ij - (1.0 / 3.0) * gamma_m * M_trace

    # d_t Abar_ij
    dAbar_dk = np.stack([
        np.stack([d1(Abar[c], a) for a in range(3)], axis=0)
        for c in range(6)
    ], axis=0)
    for n, (i, j) in enumerate(_SYM_IJ):
        # advection
        adv = sum(beta[k] * dAbar_dk[n, k] for k in range(3))
        # K Abar_ij - 2 Abar_ik gbar^kl Abar_lj
        AbarAbar = sum(Abar_m[i, k] * gbar_iv[k, l] * Abar_m[l, j]
                       for k in range(3) for l in range(3))
        # stretch: Abar_ik d_j beta^k + Abar_jk d_i beta^k
        stretch = sum(Abar_m[i, k] * dbeta[k, j] + Abar_m[j, k] * dbeta[k, i]
                      for k in range(3))
        trace_term = (2.0 / 3.0) * Abar_m[i, j] * div_beta
        dY[IDX_ABAR.start + n] = (
            chi * M_TF[i, j]
            + alpha * (K * Abar_m[i, j] - 2.0 * AbarAbar)
            + adv + stretch - trace_term
        )

    # ---- Eq. (5): d_t Gammabar^i
    # = -2 Abar^ij d_j alpha
    #   + 2 alpha [ Gambar^i_jk Abar^jk
    #               - (2/3) gbar^ij d_j K
    #               + 6 Abar^ij d_j ln(1/sqrt(chi))   <- = -(1/2) Abar^ij d_j ln chi
    #             ]
    #   + beta^j d_j Gammabar^i
    #   - Gammabar^j d_j beta^i
    #   + (2/3) Gammabar^i d_k beta^k
    #   + gbar^jk d_j d_k beta^i
    #   + (1/3) gbar^ij d_j d_k beta^k
    ddbeta = np.zeros((3, 3, 3) + chi.shape)   # ddbeta[c, a, b] = d_a d_b beta^c
    for c in range(3):
        for a in range(3):
            for b in range(3):
                ddbeta[c, a, b] = dmix(beta[c], a, b)

    dGam_dt = np.zeros((3,) + chi.shape)
    for i in range(3):
        # -2 Abar^ij d_j alpha
        t1 = -2.0 * sum(Abar_uu[i, j] * dalpha[j] for j in range(3))
        # 2 alpha [ Gambar^i_jk Abar^jk - (2/3) gbar^ij d_j K - (1/2) Abar^ij d_j ln chi ]
        # Note: 6 Abar^ij d_j ln(1/sqrt(chi)) = 6 * Abar^ij * (-1/2) d_j ln(chi) = -3 Abar^ij d_j ln chi
        # Wait — Baumgarte-Shapiro 11.54 has 6 d_j phi where phi = ln(psi),
        # chi = psi^{-4}, so phi = -(1/4) ln chi, d_j phi = -(1/(4 chi)) d_j chi.
        # 6 Abar^ij d_j phi = -(3/(2 chi)) Abar^ij d_j chi.
        # We compute it directly:
        bracket = (
            sum(Gambar[i, j, k] * Abar_uu[j, k] for j in range(3) for k in range(3))
            - (2.0 / 3.0) * sum(gbar_iv[i, j] * dK_dx[j] for j in range(3))
            - 1.5 * inv_chi * sum(Abar_uu[i, j] * dchi_dx[j] for j in range(3))
        )
        t2 = 2.0 * alpha * bracket
        # Advection & shift-stretch
        t3 = sum(beta[j] * d1(Gam_u[i], j) for j in range(3))
        t4 = -sum(Gam_u[j] * dbeta[i, j] for j in range(3))
        t5 = (2.0 / 3.0) * Gam_u[i] * div_beta
        # gbar^jk d_j d_k beta^i
        t6 = sum(gbar_iv[j, k] * ddbeta[i, j, k] for j in range(3) for k in range(3))
        # (1/3) gbar^ij d_j d_k beta^k
        t7 = (1.0 / 3.0) * sum(
            gbar_iv[i, j] * ddbeta[k, j, k] for j in range(3) for k in range(3)
        )
        dGam_dt[i] = t1 + t2 + t3 + t4 + t5 + t6 + t7
        # Matter source: - 16 pi alpha gbar^ij S_j
        if S_i is not None:
            dGam_dt[i] -= 16.0 * math.pi * alpha * sum(
                gbar_iv[i, j] * S_i[j] for j in range(3)
            )
        dY[IDX_GAMMABAR_U.start + i] = dGam_dt[i]

    # ---- Eq. (6): d_t alpha = -2 alpha K + beta^k d_k alpha  (1+log slicing)
    dY[IDX_ALPHA] = -2.0 * alpha * K + sum(beta[k] * dalpha[k] for k in range(3))

    # ---- Eq. (7): d_t beta^i = (3/4) B^i + beta^k d_k beta^i  (Gamma-driver)
    for i in range(3):
        dY[IDX_BETA.start + i] = 0.75 * Baux[i] + sum(
            beta[k] * dbeta[i, k] for k in range(3)
        )

    # ---- Eq. (8): d_t B^i = d_t Gammabar^i - eta B^i + beta^k d_k B^i - beta^k d_k Gammabar^i
    for i in range(3):
        adv_B = sum(beta[k] * d1(Baux[i], k) for k in range(3))
        adv_Gam = sum(beta[k] * d1(Gam_u[i], k) for k in range(3))
        dY[IDX_B.start + i] = dGam_dt[i] - eta_shift * Baux[i] + adv_B - adv_Gam

    # ---- Z4c Eq.: d_t Theta = (alpha/2) H + beta^k d_k Theta
    #                          - alpha kappa1 (2 + kappa2) Theta
    # Where H = R + (2/3)K^2 - Abar:Abar - 16 pi rho_adm is the
    # Hamiltonian-constraint violation (Bernuzzi-Hilditch 2010 eq. 4,
    # dropping the spatial Z^k divergence which we set to zero in the
    # Theta-only minimal Z4c).  R = gamma^ij R_ij is computed from the
    # physical Ricci tensor already built above.
    if kappa1 != 0.0:
        R_scalar = sum(
            gamma_iv[i, j] * Ric_phys[i, j] for i in range(3) for j in range(3)
        )
        rho_src_Th = rho_adm if rho_adm is not None else 0.0
        H_constraint = (
            R_scalar + (2.0 / 3.0) * K * K - AA
            - 16.0 * math.pi * rho_src_Th
        )
        dTheta_dx = [d1(Theta, a) for a in range(3)]
        dY[IDX_THETA] = (
            0.5 * alpha * H_constraint
            - alpha * kappa1 * (2.0 + kappa2) * Theta
            + beta[0] * dTheta_dx[0]
            + beta[1] * dTheta_dx[1]
            + beta[2] * dTheta_dx[2]
        )
    # When kappa1 = 0, Z4c is off; Theta evolves trivially (dY[IDX_THETA]
    # stays at 0 from np.zeros_like(y) initialization above), and the
    # system is pure BSSN.

    # ---- Kreiss-Oliger dissipation on every field
    if ko_epsilon > 0.0:
        for c in range(NCOMP):
            ko = 0.0
            ko += _kreiss_oliger(y[c], axis=0, dh=dx) / (64.0 * dx)
            ko += _kreiss_oliger(y[c], axis=1, dh=dy) / (64.0 * dy)
            ko += _kreiss_oliger(y[c], axis=2, dh=dz) / (64.0 * dz)
            dY[c] -= ko_epsilon * ko

    return dY


# ---------------------------------------------------------------------------
# Evolver
# ---------------------------------------------------------------------------
@dataclass
class BSSNEvolver:
    """BSSN 3+1 ADM evolver on a periodic Cartesian grid.

    Usage:
        evolver = BSSNEvolver(state=BSSNState.flat_minkowski(...), cfl=0.25)
        evolver.step(dt=...)
        # or
        evolver.evolve(n_steps=100)

    Matter sources (rho_adm, S_i, S_ij) can be supplied as fixed
    arrays (held constant during the evolution) or as callables
    accepting the current time t and returning the appropriate array.
    """
    state: BSSNState
    cfl: float = 0.25
    eta_shift: float = 1.0
    ko_epsilon: float = 0.5
    t: float = 0.0
    rho_adm: Optional[np.ndarray] = None
    S_i: Optional[np.ndarray] = None
    S_ij: Optional[np.ndarray] = None
    # Z4c constraint damping (scaffold).  kappa1 = 0 (default)
    # reduces to plain BSSN identically.  Non-zero kappa1 enables a
    # partial Z4c implementation (Theta evolution + K back-coupling)
    # that, by itself, does NOT stabilise vanilla BSSN's
    # constraint-violating mode on noisy input.  The full Z4c also
    # needs the chi modification and Z^i (momentum-constraint)
    # damping; both deferred to Phase 1.5.  Leave kappa1 = 0 for
    # production runs until the full damping is wired in.
    kappa1: float = 0.0
    kappa2: float = 0.0

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        return bssn_vacuum_rhs(
            t, y, self.state.dx, self.state.dy, self.state.dz,
            eta_shift=self.eta_shift, ko_epsilon=self.ko_epsilon,
            rho_adm=self.rho_adm, S_i=self.S_i, S_ij=self.S_ij,
            kappa1=self.kappa1, kappa2=self.kappa2,
        )

    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dh_min = min(self.state.dx, self.state.dy, self.state.dz)
            dt = self.cfl * dh_min
        y = self.state.data
        # RK4
        k1 = self.rhs(self.t, y)
        k2 = self.rhs(self.t + dt / 2, y + dt / 2 * k1)
        k3 = self.rhs(self.t + dt / 2, y + dt / 2 * k2)
        k4 = self.rhs(self.t + dt, y + dt * k3)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Algebraic constraint projection: enforce tr(Abar) = 0 and
        # det(gammabar) = 1.  Without these, BSSN goes unstable.
        project_algebraic_constraints(y_new)
        self.state.data[:] = y_new
        self.t += dt
        return dt

    def evolve(self, n_steps: int, dt: Optional[float] = None,
               callback=None) -> float:
        for _ in range(n_steps):
            dt = self.step(dt)
            if callback is not None:
                callback(self)
        return self.t


# ---------------------------------------------------------------------------
# Constraint diagnostics
# ---------------------------------------------------------------------------
def hamiltonian_constraint(state: BSSNState, rho: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Hamiltonian constraint H = R + K^2 - K_ij K^ij - 16 pi rho.

    In vacuum H should vanish to FD precision.  Returns an array of
    shape (Nx, Ny, Nz).
    """
    # Build R from BSSN state.  This duplicates much of the RHS work
    # but is only called for diagnostics, so we keep it explicit.
    y = state.data
    chi = y[IDX_CHI]
    gbar = y[IDX_GBAR]
    K = y[IDX_K]
    Abar = y[IDX_ABAR]
    Gam_u = y[IDX_GAMMABAR_U]

    gbar_inv = _inverse_sym3(gbar)
    gbar_m = _sym_to_3x3(gbar)
    gbar_iv = _sym_to_3x3(gbar_inv)
    Abar_m = _sym_to_3x3(Abar)
    gamma_iv = chi * gbar_iv

    # Abar_ij Abar^ij
    Abar_uu = np.einsum('ik...,jl...,kl...->ij...', gbar_iv, gbar_iv, Abar_m)
    AA = np.einsum('ij...,ij...->...', Abar_m, Abar_uu)

    # Ricci scalar: R = gamma^ij R_ij.  We approximate via the BSSN-Ricci
    # formula, but for a diagnostic on near-flat data it suffices to
    # compute R via R = chi (R_bar - small chi terms).  Use a simpler
    # form: numerically compute R from gamma_ij via standard formula.
    # For vacuum Minkowski R = 0 exactly; for Schwarzschild outside
    # horizon R = 0 also.  This is just a sanity check.
    dx_grid = (state.dx, state.dy, state.dz)
    # Build gamma_ij = (1/chi) gammabar_ij as a (3,3,Nx,Ny,Nz).
    gamma_m = gbar_m / np.maximum(chi, 1e-30)
    R_scalar = _ricci_scalar_of_3metric(gamma_m, dx_grid)

    H = R_scalar + K * K - AA - 16 * math.pi * (rho if rho is not None else 0.0)
    return H


def _ricci_scalar_of_3metric(gamma_m: np.ndarray, dx_tuple) -> np.ndarray:
    """Direct numerical 3D Ricci scalar from a 3-metric gamma_ij.

    Diagnostic-quality only.  Computes Christoffel symbols and Ricci
    tensor by finite difference of the given metric on the grid.
    """
    dx, dy, dz = dx_tuple
    # Inverse metric
    g = gamma_m
    det = (
        g[0, 0] * (g[1, 1] * g[2, 2] - g[1, 2] * g[1, 2])
        - g[0, 1] * (g[0, 1] * g[2, 2] - g[1, 2] * g[0, 2])
        + g[0, 2] * (g[0, 1] * g[1, 2] - g[1, 1] * g[0, 2])
    )
    det = np.where(np.abs(det) > 1e-30, det, 1e-30)
    inv_det = 1.0 / det
    g_inv = np.zeros_like(g)
    g_inv[0, 0] = (g[1, 1] * g[2, 2] - g[1, 2] * g[1, 2]) * inv_det
    g_inv[0, 1] = (g[0, 2] * g[1, 2] - g[0, 1] * g[2, 2]) * inv_det
    g_inv[0, 2] = (g[0, 1] * g[1, 2] - g[0, 2] * g[1, 1]) * inv_det
    g_inv[1, 0] = g_inv[0, 1]
    g_inv[1, 1] = (g[0, 0] * g[2, 2] - g[0, 2] * g[0, 2]) * inv_det
    g_inv[1, 2] = (g[0, 1] * g[0, 2] - g[0, 0] * g[1, 2]) * inv_det
    g_inv[2, 0] = g_inv[0, 2]
    g_inv[2, 1] = g_inv[1, 2]
    g_inv[2, 2] = (g[0, 0] * g[1, 1] - g[0, 1] * g[0, 1]) * inv_det

    # Christoffel symbols
    def d1(F, a):
        return _d1(F, a, (dx, dy, dz)[a])

    dg = np.zeros((3, 3, 3) + g.shape[2:])    # dg[k, i, j] = d_k g_ij
    for k in range(3):
        for i in range(3):
            for j in range(3):
                dg[k, i, j] = d1(g[i, j], k)

    Gamma = np.zeros((3, 3, 3) + g.shape[2:])    # Gamma^k_ij
    for k in range(3):
        for i in range(3):
            for j in range(3):
                Gamma[k, i, j] = 0.5 * sum(
                    g_inv[k, l] * (dg[i, j, l] + dg[j, i, l] - dg[l, i, j])
                    for l in range(3)
                )

    # Ricci: R_ij = d_k Gamma^k_ij - d_j Gamma^k_ki + Gamma^k_kl Gamma^l_ij - Gamma^k_jl Gamma^l_ki
    dGamma = np.zeros((3, 3, 3, 3) + g.shape[2:])   # dGamma[m, k, i, j] = d_m Gamma^k_ij
    for m in range(3):
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    dGamma[m, k, i, j] = d1(Gamma[k, i, j], m)

    R_ij = np.zeros((3, 3) + g.shape[2:])
    for i in range(3):
        for j in range(3):
            t1 = sum(dGamma[k, k, i, j] for k in range(3))
            t2 = sum(dGamma[j, k, k, i] for k in range(3))
            t3 = sum(Gamma[k, k, l] * Gamma[l, i, j]
                     for k in range(3) for l in range(3))
            t4 = sum(Gamma[k, j, l] * Gamma[l, k, i]
                     for k in range(3) for l in range(3))
            R_ij[i, j] = t1 - t2 + t3 - t4

    R = sum(g_inv[i, j] * R_ij[i, j] for i in range(3) for j in range(3))
    return R


def momentum_constraint(state: BSSNState, S_i: Optional[np.ndarray] = None) -> np.ndarray:
    """Momentum constraint M^i = D_j (K^ij - gamma^ij K) - 8 pi S^i.

    In vacuum M^i should vanish to FD precision.  Returns (3, Nx, Ny, Nz).
    """
    y = state.data
    chi = y[IDX_CHI]
    K = y[IDX_K]
    Abar = y[IDX_ABAR]
    Gam_u = y[IDX_GAMMABAR_U]
    gbar = y[IDX_GBAR]

    gbar_inv = _inverse_sym3(gbar)
    gbar_iv = _sym_to_3x3(gbar_inv)
    Abar_m = _sym_to_3x3(Abar)
    Abar_uu = np.einsum('ik...,jl...,kl...->ij...', gbar_iv, gbar_iv, Abar_m)

    dh = (state.dx, state.dy, state.dz)
    # M^i = D_j (chi Abar^ij + (1/3) gamma^ij K - gamma^ij K)
    #     = D_j Abar^ij_phys - (2/3) gamma^ij D_j K
    # where Abar^ij_phys = chi Abar^ij  (raised with physical metric).
    # Simpler PT formula: M^i = d_j Abar^ij + Gammabar^j_jk Abar^ki + Gambar^i_jk Abar^jk
    #                            - (2/3) gbar^ij d_j K
    #                            - (3/(2 chi)) Abar^ij d_j chi
    # (Brown 2009 eq. 22 form)
    M = np.zeros((3,) + chi.shape)
    for i in range(3):
        # d_j Abar^ij
        t1 = sum(_d1(Abar_uu[i, j], axis=j, dh=dh[j]) for j in range(3))
        # - (2/3) gbar^ij d_j K
        t2 = -(2.0 / 3.0) * sum(
            gbar_iv[i, j] * _d1(K, axis=j, dh=dh[j]) for j in range(3)
        )
        # - (3/(2 chi)) Abar^ij d_j chi
        t3 = -1.5 / np.maximum(chi, 1e-30) * sum(
            Abar_uu[i, j] * _d1(chi, axis=j, dh=dh[j]) for j in range(3)
        )
        M[i] = t1 + t2 + t3
    if S_i is not None:
        M -= 8 * math.pi * S_i
    return M
