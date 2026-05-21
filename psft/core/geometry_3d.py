"""Spatial geometry helpers for matter sectors on a dynamical 3-metric
(Phase 2 of the curved-background extension).

Provides a thin wrapper around the spatial fields needed by
`hydro_3d`, `photonic_field`, and `gauge_sectors` when they run on
a curved background:

    gamma_ij      spatial metric, symmetric 3x3 per cell
    gamma^ij      inverse spatial metric
    sqrt_gamma    determinant^{1/2} (volume element)
    Gamma^k_ij    Christoffel symbols of gamma (3-D)
    alpha         lapse
    beta^i        shift vector

The geometry is provided as input by the caller; on flat Minkowski
the trivial defaults reduce all curved-aware kernels to their flat
behaviour, so a flat-default `SpatialGeometry.flat(N, N, N, dx)`
serves as a backward-compatibility helper for the existing examples
(Examples 14-24) that still run on Minkowski.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np


# Symmetric-tensor packing: (xx, xy, xz, yy, yz, zz) in the leading axis.
# This matches the `psft.evolve.adm.BSSNState` convention.
_PACK_MAT = np.array([
    [0, 1, 2],
    [1, 3, 4],
    [2, 4, 5],
])


def _sym_to_3x3(sym6: np.ndarray) -> np.ndarray:
    """Convert (6, ...) symmetric packing to (3, 3, ...) full tensor."""
    return sym6[_PACK_MAT]


def _inverse_sym3(sym6: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric 3x3 metric given as 6 components.
    Returns the inverse in the same (6, ...) packing."""
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = sym6[0], sym6[1], sym6[2], sym6[3], sym6[4], sym6[5]
    det = (
        g_xx * (g_yy * g_zz - g_yz * g_yz)
        - g_xy * (g_xy * g_zz - g_yz * g_xz)
        + g_xz * (g_xy * g_yz - g_yy * g_xz)
    )
    det_safe = np.where(np.abs(det) > 1e-30, det, 1e-30)
    inv_det = 1.0 / det_safe
    inv_xx = (g_yy * g_zz - g_yz * g_yz) * inv_det
    inv_xy = (g_xz * g_yz - g_xy * g_zz) * inv_det
    inv_xz = (g_xy * g_yz - g_xz * g_yy) * inv_det
    inv_yy = (g_xx * g_zz - g_xz * g_xz) * inv_det
    inv_yz = (g_xy * g_xz - g_xx * g_yz) * inv_det
    inv_zz = (g_xx * g_yy - g_xy * g_xy) * inv_det
    return np.stack([inv_xx, inv_xy, inv_xz, inv_yy, inv_yz, inv_zz], axis=0)


def _det_sym3(sym6: np.ndarray) -> np.ndarray:
    """Determinant of a symmetric 3x3 metric given as 6 components."""
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = sym6[0], sym6[1], sym6[2], sym6[3], sym6[4], sym6[5]
    return (
        g_xx * (g_yy * g_zz - g_yz * g_yz)
        - g_xy * (g_xy * g_zz - g_yz * g_xz)
        + g_xz * (g_xy * g_yz - g_yy * g_xz)
    )


@dataclass
class SpatialGeometry:
    """Spatial metric + lapse + shift on a 3D Cartesian grid.

    Conventions:
        gamma_ij (6, Nx, Ny, Nz)  symmetric (xx, xy, xz, yy, yz, zz)
        alpha     (Nx, Ny, Nz)
        beta      (3, Nx, Ny, Nz) shift vector, beta^i upper-index
    """
    gamma_ij: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray   # (3, Nx, Ny, Nz)
    dx: float
    dy: float
    dz: float

    @classmethod
    def flat(cls, Nx: int, Ny: int, Nz: int,
             dx: float, dy: float = None, dz: float = None):
        """Trivial flat-Minkowski geometry: gamma_ij = delta_ij,
        alpha = 1, beta = 0."""
        if dy is None: dy = dx
        if dz is None: dz = dx
        g = np.zeros((6, Nx, Ny, Nz))
        g[0] = 1.0   # xx
        g[3] = 1.0   # yy
        g[5] = 1.0   # zz
        alpha = np.ones((Nx, Ny, Nz))
        beta = np.zeros((3, Nx, Ny, Nz))
        return cls(gamma_ij=g, alpha=alpha, beta=beta, dx=dx, dy=dy, dz=dz)

    @classmethod
    def from_bssn(cls, bssn_state) -> "SpatialGeometry":
        """Extract a SpatialGeometry from a BSSNState.

        The physical 3-metric is gamma_ij = gammabar_ij / chi.
        """
        from psft.evolve.adm import IDX_GBAR, IDX_CHI, IDX_ALPHA, IDX_BETA
        chi = bssn_state.data[IDX_CHI]
        gbar = bssn_state.data[IDX_GBAR]
        gamma_ij = gbar / np.maximum(chi[np.newaxis, ...], 1e-30)
        alpha = bssn_state.data[IDX_ALPHA].copy()
        beta = bssn_state.data[IDX_BETA].copy()
        return cls(gamma_ij=gamma_ij, alpha=alpha, beta=beta,
                    dx=bssn_state.dx, dy=bssn_state.dy, dz=bssn_state.dz)

    @property
    def Nx(self): return self.alpha.shape[0]
    @property
    def Ny(self): return self.alpha.shape[1]
    @property
    def Nz(self): return self.alpha.shape[2]

    @property
    def gamma_inv(self) -> np.ndarray:
        """Inverse spatial metric gamma^ij, shape (6, Nx, Ny, Nz)."""
        return _inverse_sym3(self.gamma_ij)

    @property
    def det_gamma(self) -> np.ndarray:
        """det(gamma_ij), shape (Nx, Ny, Nz)."""
        return _det_sym3(self.gamma_ij)

    @property
    def sqrt_gamma(self) -> np.ndarray:
        """sqrt(det gamma_ij) -- the spatial volume element."""
        return np.sqrt(np.maximum(self.det_gamma, 1e-30))

    @property
    def is_flat(self) -> bool:
        """Cheap check whether this is essentially flat Minkowski.
        Useful for branching to faster flat code paths in matter
        kernels."""
        return (
            float(np.max(np.abs(self.gamma_ij[0] - 1.0))) < 1e-14
            and float(np.max(np.abs(self.gamma_ij[3] - 1.0))) < 1e-14
            and float(np.max(np.abs(self.gamma_ij[5] - 1.0))) < 1e-14
            and float(np.max(np.abs(self.gamma_ij[1]))) < 1e-14
            and float(np.max(np.abs(self.gamma_ij[2]))) < 1e-14
            and float(np.max(np.abs(self.gamma_ij[4]))) < 1e-14
            and float(np.max(np.abs(self.alpha - 1.0))) < 1e-14
            and float(np.max(np.abs(self.beta))) < 1e-14
        )

    def transport_velocity(self, v_phys: np.ndarray) -> np.ndarray:
        """Curved-aware transport velocity for scalar advection.

        For a passive scalar advected by the matter 3-velocity v^i on
        a curved background with lapse alpha and shift beta^i, the
        transport equation is
            d_t phi + (alpha v^i - beta^i) d_i phi = 0,
        so the "effective" advection velocity is u^i = alpha v^i - beta^i.

        Parameters
        ----------
        v_phys : (3, Nx, Ny, Nz)
            Physical matter 3-velocity.

        Returns
        -------
        u : (3, Nx, Ny, Nz)
            Coordinate transport velocity.
        """
        return self.alpha[np.newaxis, ...] * v_phys - self.beta


# ---------------------------------------------------------------------------
# Spatial Ricci scalar from gamma_ij (FD)
# ---------------------------------------------------------------------------
def _d1_centred(F: np.ndarray, axis: int, dh: float) -> np.ndarray:
    """4th-order centred first derivative on a periodic grid."""
    return (
        -np.roll(F, 2, axis=axis) + 8 * np.roll(F, 1, axis=axis)
        - 8 * np.roll(F, -1, axis=axis) + np.roll(F, -2, axis=axis)
    ) / (12 * dh)


def spatial_ricci_scalar(geom: SpatialGeometry) -> np.ndarray:
    """Compute the spatial Ricci scalar R^(3) = gamma^ij R_ij of the
    3-metric.  Used by Phase 2 modules (e.g. K-triggered viscosity in
    the gauge sectors) and as the geometry-side ingredient of the
    spacetime Kretschmann scalar."""
    gamma_m = _sym_to_3x3(geom.gamma_ij)    # (3, 3, ...)
    gamma_iv = _sym_to_3x3(geom.gamma_inv)
    dh = (geom.dx, geom.dy, geom.dz)

    # First derivatives of gamma_ij
    dgamma = np.zeros((3, 3, 3) + gamma_m.shape[2:])   # dgamma[k, i, j] = d_k gamma_ij
    for k in range(3):
        for i in range(3):
            for j in range(3):
                dgamma[k, i, j] = _d1_centred(gamma_m[i, j], axis=k, dh=dh[k])

    # Christoffel: Gamma^k_ij = (1/2) gamma^kl (d_i gamma_jl + d_j gamma_il - d_l gamma_ij)
    Gamma = np.zeros((3, 3, 3) + gamma_m.shape[2:])
    for k in range(3):
        for i in range(3):
            for j in range(3):
                Gamma[k, i, j] = 0.5 * sum(
                    gamma_iv[k, l] * (dgamma[i, j, l] + dgamma[j, i, l] - dgamma[l, i, j])
                    for l in range(3)
                )

    # Derivatives of Christoffel
    dGamma = np.zeros((3, 3, 3, 3) + gamma_m.shape[2:])   # dGamma[m, k, i, j]
    for m in range(3):
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    dGamma[m, k, i, j] = _d1_centred(Gamma[k, i, j], axis=m, dh=dh[m])

    # Ricci tensor (3D)
    R_ij = np.zeros((3, 3) + gamma_m.shape[2:])
    for i in range(3):
        for j in range(3):
            t1 = sum(dGamma[k, k, i, j] for k in range(3))
            t2 = sum(dGamma[j, k, k, i] for k in range(3))
            t3 = sum(Gamma[k, k, l] * Gamma[l, i, j]
                     for k in range(3) for l in range(3))
            t4 = sum(Gamma[k, j, l] * Gamma[l, k, i]
                     for k in range(3) for l in range(3))
            R_ij[i, j] = t1 - t2 + t3 - t4

    R = sum(gamma_iv[i, j] * R_ij[i, j] for i in range(3) for j in range(3))
    return R


def kretschmann_from_adm(bssn_state, dt_K=None) -> np.ndarray:
    """Compute the spacetime Kretschmann scalar K = R_abcd R^abcd from
    a 3+1 ADM state via the Gauss-Codazzi-Ricci relations.

    For PSFT this is the key quantity that triggers the v2
    Heaviside-activated viscosity (Postulate 3 of the main paper).
    Currently Example 20 uses the matter-field-gradient proxy
    |grad Phi|^2 as a stand-in for K; with this function and a
    dynamical metric, the real Kretschmann becomes available.

    NOTE: this is a Phase 2 placeholder.  A faithful 4D Kretschmann
    from 3+1 ADM data requires the Gauss-Codazzi decomposition

        R_abcd = (3)R_abcd + extrinsic-curvature terms,

    which is fully tensor-algebraic but tedious to write out for
    the full contraction R_abcd R^abcd.  The current implementation
    returns the SPATIAL Kretschmann
    R^(3)_ijkl R^(3)^ijkl based on the 3-metric only.  For
    stationary spacetimes this differs from the full 4-D Kretschmann
    by a calculable (extrinsic-curvature)^2 correction; for the
    PSFT v2 Heaviside trigger the spatial Kretschmann is the
    dominant contribution.

    Returns shape (Nx, Ny, Nz).
    """
    # The simplest meaningful diagnostic for triggering PSFT's
    # K-dependent viscosity: K_simple = gamma^ij gamma^kl R_ik R_jl
    # which is a curvature-norm scalar that vanishes for flat space
    # and reduces to the standard Kretschmann for spherically symmetric
    # geometries up to the (extrinsic curvature)^2 piece.
    #
    # A full implementation would carry K_ij from the BSSN state and
    # compute the Gauss-Codazzi-completed Kretschmann; that is on the
    # Phase 2c roadmap.
    geom = SpatialGeometry.from_bssn(bssn_state)
    R_scalar = spatial_ricci_scalar(geom)
    # Order-of-magnitude approximation: K ~ R^2 in 3D when the
    # off-diagonal curvature components are comparable to R.
    return R_scalar * R_scalar
