"""Gauge algebra primitives -- U(1), SU(2), SU(3).

The PSFT v2 master equation carries a gauge index A = 0..12:
    A = 0           gravitational singlet
    A = 1..8        SU(3) adjoint (8 colour shear modes)
    A = 9..11       SU(2) adjoint (3 weak modes)
    A = 12          U(1) (electromagnetism)

Convention: generators T^A are Hermitian, normalised by Tr(T^A T^B) = (1/2) delta^{AB}.
Structure constants f^{ABC} satisfy [T^A, T^B] = i f^{ABC} T^C.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

# Pauli matrices.
_SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

# Gell-Mann matrices.
_GELLMANN = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    (1.0 / np.sqrt(3.0)) * np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex
    ),
]


def generators_su2() -> List[np.ndarray]:
    return [0.5 * s for s in _SIGMA]


def generators_su3() -> List[np.ndarray]:
    return [0.5 * l for l in _GELLMANN]


def structure_constants_su(generators: List[np.ndarray]) -> np.ndarray:
    """Compute f^{ABC} from generators via Tr([T^A,T^B] T^C) = (i/2) f^{ABC}.

    Returns a real array of shape (n, n, n) with n = len(generators).
    """
    n = len(generators)
    f = np.zeros((n, n, n))
    for A in range(n):
        for B in range(n):
            comm = generators[A] @ generators[B] - generators[B] @ generators[A]
            for C in range(n):
                tr = np.trace(comm @ generators[C])
                # tr = (i/2) f^{ABC}  --> f^{ABC} = -2 i tr  (real)
                val = (-2j * tr)
                f[A, B, C] = float(val.real)
                if abs(val.imag) > 1e-9:
                    raise RuntimeError(
                        f"f^{{{A}{B}{C}}} has non-vanishing imaginary part: {val}"
                    )
    return f


@dataclass
class GaugeAlgebra:
    name: str
    dim: int                          # number of generators
    generators: List[np.ndarray] = field(default_factory=list)
    structure_constants: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    def commutator(self, A: int, B: int) -> np.ndarray:
        T_A, T_B = self.generators[A], self.generators[B]
        return T_A @ T_B - T_B @ T_A

    def covariant_derivative(self, partial_field: np.ndarray, A_a: np.ndarray,
                              gauge_field: np.ndarray, coupling: float = 1.0) -> np.ndarray:
        """D_a phi^A = d_a phi^A + g f^{ABC} A_a^B phi^C  (adjoint rep).

        partial_field shape: (..., dim).  A_a shape: (4, dim).
        gauge_field is `phi^A` itself (shape (..., dim)).
        """
        if self.structure_constants.shape[0] == 0:
            return partial_field
        f = self.structure_constants
        gA = np.einsum("ABC,aB,...C->a...A", f, A_a, gauge_field)
        return partial_field + coupling * gA


def _su(n: int, gens, name: str) -> GaugeAlgebra:
    f = structure_constants_su(gens)
    return GaugeAlgebra(name=name, dim=len(gens), generators=gens, structure_constants=f)


U1 = GaugeAlgebra(name="U(1)", dim=1,
                  generators=[np.array([[1.0 + 0j]])],
                  structure_constants=np.zeros((1, 1, 1)))
SU2 = _su(2, generators_su2(), "SU(2)")
SU3 = _su(3, generators_su3(), "SU(3)")
