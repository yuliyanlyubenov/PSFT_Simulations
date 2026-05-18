"""Photonic source tensor P_ab and the photonic 4-force (Section 4 of paper).

    P_{ab} = (1/4pi) [F_{ac} F^c_b - (1/4) g_{ab} F_{cd} F^{cd}]
             + Lambda_PSFT(l) g_{ab}
             + Q_{ab}

Classical limit (Q -> 0, Lambda_PSFT -> Lambda):
    P_{ab} -> T^(EM)_{ab} + Lambda g_{ab}

Photonic 4-force density (eq. 4.3):
    F_a^(P) = (8 pi G / c^4) h_a^b nabla_c P^c_b
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from psft.core.metric import Metric


@dataclass
class ElectromagneticField:
    """Maxwell 2-form F_{ab} = d_a A_b - d_b A_a."""
    F_dn: np.ndarray   # F_{ab} (4,4)
    g: np.ndarray
    g_inv: np.ndarray

    @classmethod
    def from_potential(
        cls,
        metric: Metric,
        x: np.ndarray,
        A_func: Callable[[np.ndarray], np.ndarray],
        h: float = 1e-5,
    ) -> "ElectromagneticField":
        g = metric.g(x)
        g_inv = metric.g_inv(x)
        dA = np.zeros((4, 4))  # dA[a,b] = d_a A_b
        for a in range(4):
            xp = x.copy(); xp[a] += h
            xm = x.copy(); xm[a] -= h
            dA[a] = (A_func(xp).astype(float) - A_func(xm).astype(float)) / (2 * h)
        F_dn = dA - dA.T
        return cls(F_dn=F_dn, g=g, g_inv=g_inv)

    def F_up(self) -> np.ndarray:
        return self.g_inv @ self.F_dn @ self.g_inv.T

    def stress_energy_em(self) -> np.ndarray:
        """T^EM_{ab} = (1/4pi)[F_{ac} F^c_b - (1/4) g_{ab} F_{cd} F^{cd}]."""
        F = self.F_dn
        F_up = self.g_inv @ F
        FF = F @ F_up                                     # F_{ac} F^c_b
        F2 = float(np.einsum("ab,ab->", F, self.F_up()))  # F_{cd} F^{cd}
        return (1.0 / (4 * np.pi)) * (FF - 0.25 * self.g * F2)


@dataclass
class PhotonicSource:
    """P_{ab} = T^EM_{ab} + Lambda g_{ab} + Q_{ab}.

    Q_{ab} (quantum deformation tensor) is supplied as an external field; it
    defaults to zero -- the classical limit used by Theorem 12.1.
    """
    P_dn: np.ndarray
    em: ElectromagneticField
    Lambda: float
    Q_dn: np.ndarray

    @classmethod
    def from_em(
        cls,
        em: ElectromagneticField,
        Lambda: float = 0.0,
        Q_dn: Optional[np.ndarray] = None,
    ) -> "PhotonicSource":
        T_em = em.stress_energy_em()
        Q = np.zeros_like(T_em) if Q_dn is None else Q_dn
        P = T_em + Lambda * em.g + Q
        return cls(P_dn=P, em=em, Lambda=Lambda, Q_dn=Q)

    def divergence(self, metric: Metric, x: np.ndarray, h: float = 1e-5,
                   regenerate: Optional[Callable[[np.ndarray], "PhotonicSource"]] = None) -> np.ndarray:
        """nabla_c P^c_b (lowered b index) at point x.

        For meaningful divergence the caller must provide `regenerate(x)` that
        returns a PhotonicSource at displaced x -- otherwise the result is
        identically zero (we cannot finite-difference a single point).
        """
        if regenerate is None:
            return np.zeros(4)
        g_inv = metric.g_inv(x)
        # P^c_b = g^{cd} P_{db}
        # Finite-difference d_c P^c_b at x.
        out = np.zeros(4)
        for c in range(4):
            xp = x.copy(); xp[c] += h
            xm = x.copy(); xm[c] -= h
            Pp = regenerate(xp)
            Pm = regenerate(xm)
            P_up_b_p = (metric.g_inv(xp) @ Pp.P_dn)[c]
            P_up_b_m = (metric.g_inv(xm) @ Pm.P_dn)[c]
            out += (P_up_b_p - P_up_b_m) / (2 * h)
        return out

    def force_density(self, metric: Metric, x: np.ndarray, h_proj: np.ndarray,
                      G: float, c_light: float,
                      regenerate: Optional[Callable] = None) -> np.ndarray:
        """F_a^{(P)} = (8 pi G / c^4) h_a^b nabla_c P^c_b."""
        div = self.divergence(metric, x, regenerate=regenerate)
        return (8 * np.pi * G / c_light**4) * (h_proj @ div)
