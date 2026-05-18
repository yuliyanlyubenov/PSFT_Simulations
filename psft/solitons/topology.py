"""Topological invariants of flow patterns.

Postulate 4 (Topological Charge) of PSFT identifies:
    electric charge   <-> U(1) winding along compact Killing direction
    colour charge     <-> SU(3) topological class
    weak isospin      <-> SU(2) topology

For matter solitons we use the standard topological tools:

    * 1D winding number  (electron-like vortex along axis)
    * 2D winding around a loop
    * Skyrme topological charge B (proton, neutron baryon number)
    * Hopf invariant proxy (knot energy)
"""
from __future__ import annotations
from typing import Callable
import numpy as np


def winding_number_1d(phi_values: np.ndarray) -> int:
    """Winding number of a U(1) phase field sampled on a closed loop.

    phi_values are phase values in radians (any range; unwrapping handles it).
    Closed loop assumption: phi_values[-1] is identified with phi_values[0].
    """
    unwrapped = np.unwrap(phi_values)
    delta = unwrapped[-1] - unwrapped[0]
    return int(np.round(delta / (2 * np.pi)))


def winding_number_2d_loop(field: np.ndarray, center: tuple, radius: float,
                            n_samples: int = 256) -> int:
    """Winding number of a 2D U(1) field around (cx, cy).

    `field` shape (..., Nx, Ny) of complex values -- last two axes are
    (x_index, y_index) in the same ij-ordered meshgrid convention used by
    `CartesianGrid.meshgrid`.  Samples a circular loop of radius `radius`
    (in grid units, counterclockwise) and counts the phase winding.
    """
    cx, cy = center
    angles = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    Nx, Ny = field.shape[-2], field.shape[-1]
    xi = np.clip(np.round(xs).astype(int), 0, Nx - 1)
    yi = np.clip(np.round(ys).astype(int), 0, Ny - 1)
    phases = np.angle(field[..., xi, yi])
    closed = np.concatenate([phases, phases[..., :1]], axis=-1)
    return winding_number_1d(closed)


def topological_charge_skyrme(n_field: np.ndarray, dx: float) -> float:
    """Skyrme baryon number for an S^3 field N^a (a = 0..3, |N|=1).

        B = (1/(2 pi^2)) int eps_{abcd} N^a d_x N^b d_y N^c d_z N^d  d^3x

    This is the contracted (single spatial ordering) form of the standard
    (1/12 pi^2) int eps^{ijk} eps_{abcd} N^a d_i N^b d_j N^c d_k N^d:
    the ijk-antisymmetric sum collapses to 6 equivalent terms by the
    antisymmetry of eps_{abcd} in (b,c,d), giving an overall 1/2 pi^2.

    `n_field` shape must be (4, Nx, Ny, Nz) with n_field[0] = sigma and
    n_field[1:4] = pion vector.  Returns approximate integer B.
    """
    n = n_field
    if n.shape[0] != 4:
        raise ValueError(
            "Skyrme topological charge requires the full O(4) field "
            "(sigma, n_vec) of shape (4, Nx, Ny, Nz)."
        )
    dn_x = np.gradient(n, dx, axis=1)
    dn_y = np.gradient(n, dx, axis=2)
    dn_z = np.gradient(n, dx, axis=3)
    eps4 = _eps4()
    density = np.einsum("abcd,a...,b...,c...,d...->...", eps4, n, dn_x, dn_y, dn_z)
    integral = density.sum() * (dx ** 3)
    return float(integral / (2 * np.pi ** 2))


def hopf_invariant_proxy(A_field: np.ndarray, F_field: np.ndarray, dx: float) -> float:
    """Approximate Hopf invariant H = (1/8 pi^2) int A . B  d^3x.

    For a U(1) connection A_i and magnetic field B_i = (1/2) eps_{ijk} F^{jk}.
    """
    if A_field.shape[0] != 3 or F_field.shape[0] != 3:
        raise ValueError("Hopf proxy requires 3-component A_i and B_i.")
    dens = np.einsum("i...,i...->...", A_field, F_field)
    return float(dens.sum() * dx**3 / (8 * np.pi**2))


def _eps3() -> np.ndarray:
    e = np.zeros((3, 3, 3))
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[0, 2, 1] = e[1, 0, 2] = e[2, 1, 0] = -1
    return e


def _eps4() -> np.ndarray:
    from itertools import permutations
    e = np.zeros((4, 4, 4, 4))
    for perm in permutations((0, 1, 2, 3)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        e[perm] = 1.0 if inv % 2 == 0 else -1.0
    return e
