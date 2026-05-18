"""Viscosity functions for the PSFT master equation (Postulate 3, Section 6).

v1: eta(l) = eta_P * exp(-l^2 / lc^2)              Gaussian in length scale.
v2: eta(K) = eta_0 * Theta(K - Kc) * f(K/Kc)       Heaviside in Kretschmann.

The gauge-valued viscosity eta^A_B(K) inflates to a matrix in the adjoint
representation, with sector-dependent thresholds Kc^A.  Hall viscosity adds a
parity-odd, non-dissipative term in the weak sector.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np

from psft.core.constants import PSFTConstants


def profile_constant(_K_over_Kc: float) -> float:
    return 1.0


def profile_QCD_log(K_over_Kc: float, b0: float = 27.0 / (48 * np.pi)) -> float:
    """f(K/Kc) = 1 / (b0 ln(K/Kc)) -- asymptotic-freedom profile (Theorem 10.1(iii))."""
    if K_over_Kc <= 1.0:
        return 0.0
    return 1.0 / (b0 * np.log(K_over_Kc))


def profile_polynomial(K_over_Kc: float, power: float = 1.0) -> float:
    if K_over_Kc <= 1.0:
        return 0.0
    return (K_over_Kc - 1.0) ** power


class ViscosityProfile(ABC):
    @abstractmethod
    def eta(self, K: float) -> float: ...


@dataclass
class ScalarViscosity(ViscosityProfile):
    """v1 Gaussian profile (kept for reference / v1 master equation)."""
    eta_P: float
    l_c: float

    def eta(self, K: float, l: Optional[float] = None) -> float:
        if l is None:
            # Map K -> l via l_eff = K^{-1/4} (paper, Section 6).
            if K <= 0:
                return self.eta_P
            l = K ** (-0.25)
        return self.eta_P * float(np.exp(-(l ** 2) / (self.l_c ** 2)))


@dataclass
class HeavisideViscosity(ViscosityProfile):
    """v2 Heaviside profile: eta_0 * Theta(K - Kc) * f(K/Kc)."""
    eta_0: float
    K_c: float
    profile: Callable[[float], float] = profile_constant
    smoothing: float = 0.0  # if >0, replace Theta by tanh-smoothed version.

    def step(self, K: float) -> float:
        if self.smoothing <= 0.0:
            return 0.0 if K < self.K_c else 1.0
        return 0.5 * (1.0 + np.tanh((K - self.K_c) / self.smoothing))

    def eta(self, K: float) -> float:
        if self.eta_0 == 0.0:
            return 0.0
        if K <= self.K_c and self.smoothing == 0.0:
            return 0.0
        ratio = float("inf") if self.K_c == 0.0 else K / self.K_c
        return self.eta_0 * self.step(K) * float(self.profile(ratio))


@dataclass
class GaugeViscosity:
    """Gauge-matrix viscosity eta^A_B(K), with one HeavisideViscosity per sector.

    Sectors are keyed by string ("strong", "weak", "em", "gravity").  Each
    sector has dim (1, 3, 8, ...) gauge generators; we store one scalar
    eta(K) per sector and broadcast to a diagonal matrix in the sector's
    adjoint basis.
    """
    sectors: Dict[str, HeavisideViscosity] = field(default_factory=dict)

    def eta(self, sector: str, K: float) -> float:
        if sector not in self.sectors:
            return 0.0
        return self.sectors[sector].eta(K)

    def matrix(self, sector: str, K: float, dim: int) -> np.ndarray:
        """Diagonal eta^A_B(K) = eta(K) * I in the adjoint."""
        return self.eta(sector, K) * np.eye(dim)

    @classmethod
    def physical(cls, consts: PSFTConstants,
                 strong_profile: Callable[[float], float] = profile_QCD_log,
                 weak_profile: Callable[[float], float] = profile_constant) -> "GaugeViscosity":
        """Default gauge viscosity bundle with paper-suggested thresholds."""
        eta_p = consts.planck_viscosity()
        return cls(sectors={
            "strong": HeavisideViscosity(eta_0=eta_p,
                                          K_c=consts.Kc_strong,
                                          profile=strong_profile),
            "weak":   HeavisideViscosity(eta_0=eta_p * 1e-3,
                                          K_c=consts.Kc_weak,
                                          profile=weak_profile),
            "em":     HeavisideViscosity(eta_0=0.0,
                                          K_c=consts.Kc_em,
                                          profile=profile_constant),
            "gravity": HeavisideViscosity(eta_0=0.0,
                                          K_c=0.0,
                                          profile=profile_constant),
        })


@dataclass
class HallViscosity:
    """Parity-odd Hall viscosity coefficient for the weak sector (eq. 6.7).

    tau^Hall_{ab} = eta_odd^A(K) eps_b^{cde} u_c (D_d sigma^A) Theta(K - Kw).
    """
    eta_odd: float
    K_w: float
    smoothing: float = 0.0

    def coeff(self, K: float) -> float:
        if self.smoothing <= 0.0:
            return 0.0 if K < self.K_w else self.eta_odd
        return self.eta_odd * 0.5 * (1.0 + np.tanh((K - self.K_w) / self.smoothing))
