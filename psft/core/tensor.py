"""Index gymnastics on (n,n,...) tensor fields living on a coordinate grid.

Convention: spacetime indices a,b,... = 0..3 with signature (-+++).
Tensors are stored as numpy arrays whose first axes are the tensor indices
and remaining axes are the grid dimensions.
"""
from __future__ import annotations
import numpy as np


def _is_grid_field(g_inv: np.ndarray) -> bool:
    return g_inv.ndim > 2


def raise_index(T: np.ndarray, g_inv: np.ndarray, index: int) -> np.ndarray:
    """Raise the given (covariant) index of T using g^{ab}.

    T has rank r tensor indices at the front; index in [0, r).
    g_inv shape is (4,4) or (4,4,*grid).
    """
    T = np.moveaxis(T, index, 0)              # (D, ...)
    if _is_grid_field(g_inv):
        # einsum over the contracted index, broadcast over grid axes.
        out = np.einsum("ab...,b...->a...", g_inv, T)
    else:
        out = np.einsum("ab,b...->a...", g_inv, T)
    return np.moveaxis(out, 0, index)


def lower_index(T: np.ndarray, g: np.ndarray, index: int) -> np.ndarray:
    T = np.moveaxis(T, index, 0)
    if _is_grid_field(g):
        out = np.einsum("ab...,b...->a...", g, T)
    else:
        out = np.einsum("ab,b...->a...", g, T)
    return np.moveaxis(out, 0, index)


def contract(T: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    """Trace T over its two tensor indices `axis_a` and `axis_b`."""
    return np.trace(T, axis1=axis_a, axis2=axis_b)


def sym(T: np.ndarray, i: int = 0, j: int = 1) -> np.ndarray:
    """Symmetrise tensor T over axes (i,j): T_{(ij)}."""
    return 0.5 * (T + np.swapaxes(T, i, j))


def antisym(T: np.ndarray, i: int = 0, j: int = 1) -> np.ndarray:
    """Antisymmetrise tensor T over axes (i,j): T_{[ij]}."""
    return 0.5 * (T - np.swapaxes(T, i, j))


def levi_civita_tensor(g: np.ndarray) -> np.ndarray:
    """Levi-Civita tensor in 4D, eps_{abcd} = sqrt(-det g) * eps_symbol.

    g may be (4,4) or (4,4,*grid).
    """
    eps = np.zeros((4, 4, 4, 4), dtype=float)
    for perm, sign in _perm4():
        a, b, c, d = perm
        eps[a, b, c, d] = sign
    if _is_grid_field(g):
        g4 = np.moveaxis(g, [0, 1], [-2, -1])      # (..., 4, 4)
        det = np.linalg.det(g4)                    # (...)
        sqrt_neg_g = np.sqrt(np.maximum(-det, 0.0))
        # broadcast: eps_grid[a,b,c,d,*grid] = eps[a,b,c,d] * sqrt(-g)(*grid)
        for _ in range(g.ndim - 2):
            eps = eps[..., np.newaxis]
        return eps * sqrt_neg_g
    det = float(np.linalg.det(g))
    return eps * np.sqrt(max(-det, 0.0))


def projector_h(g: np.ndarray, u_lower: np.ndarray) -> np.ndarray:
    """Spatial projector h_{ab} = g_{ab} + u_a u_b."""
    if g.ndim == 2:
        return g + np.einsum("a,b->ab", u_lower, u_lower)
    return g + np.einsum("a...,b...->ab...", u_lower, u_lower)


def _perm4():
    """All 24 permutations of (0,1,2,3) with signature (+1 even / -1 odd)."""
    from itertools import permutations
    out = []
    for perm in permutations((0, 1, 2, 3)):
        inversions = sum(
            1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j]
        )
        out.append((perm, 1 if inversions % 2 == 0 else -1))
    return out
