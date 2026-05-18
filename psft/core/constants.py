"""Physical and PSFT-specific constants.

Two unit systems are exposed:

    SI       - SI units, c and hbar restored.
    NATURAL  - c = hbar = 1, lengths in meters or in 1/GeV.

PSFT scales:
    eta_Planck   ~ c^3 / (16 pi G) Planck shear viscosity (kg/s in SI)
    Kc_strong    critical Kretschmann for SU(3) activation
    Kc_weak      critical Kretschmann for SU(2) activation
    Kc_EM        +infinity (U(1) sector never freezes)
    l_strong     ~ 10^-15 m (1 fm)
    l_weak       ~ 10^-18 m (electroweak)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PSFTConstants:
    """Snapshot of the physical constants used by the simulator.

    Choose `system` = 'SI' or 'natural'.  In 'natural' mode c = hbar = 1.
    """
    system: str = "SI"

    # Universal physical constants (SI numerical values).
    c: float = 2.99792458e8            # m/s
    hbar: float = 1.054571817e-34      # J s
    G: float = 6.67430e-11             # m^3 kg^-1 s^-2
    epsilon_0: float = 8.8541878128e-12  # F/m
    e: float = 1.602176634e-19         # C
    k_B: float = 1.380649e-23          # J/K
    m_electron: float = 9.1093837015e-31  # kg
    m_proton: float = 1.67262192369e-27   # kg

    # PSFT scales (paper conventions).
    l_strong: float = 1.0e-15           # m  (subnuclear)
    l_weak: float = 1.0e-18             # m  (electroweak)
    l_em: float = math.inf              # U(1) sector always inviscid

    # Critical Kretschmann scalars (m^-4) via Kc = 12 / lc^4 (eq. 6.10 in paper).
    Kc_strong: float = field(init=False)
    Kc_weak: float = field(init=False)
    Kc_em: float = field(init=False)

    # Standard Model couplings at relevant scales.
    alpha_em: float = 1.0 / 137.035999084
    alpha_strong: float = 0.1181          # at M_Z
    sin2_thetaW: float = 0.23121          # Weinberg

    # QCD scales.
    Lambda_QCD: float = 0.217e9 * 1.602176634e-19 / 1.054571817e-34  # ~rad/s
    string_tension_GeV2: float = 0.18
    R_tube_fm: float = 0.35

    # Higgs / W / Z masses (GeV converted to kg via E = m c^2).
    m_W_GeV: float = 80.379
    m_Z_GeV: float = 91.1876
    m_H_GeV: float = 125.10

    def __post_init__(self):
        object.__setattr__(self, "Kc_strong", 12.0 / self.l_strong**4)
        object.__setattr__(self, "Kc_weak", 12.0 / self.l_weak**4)
        object.__setattr__(self, "Kc_em", math.inf)

    def planck_length(self) -> float:
        return math.sqrt(self.hbar * self.G / self.c**3)

    def planck_mass(self) -> float:
        return math.sqrt(self.hbar * self.c / self.G)

    def planck_viscosity(self) -> float:
        """eta_Planck = c^3 / (16 pi G) -- Planck shear viscosity (kg/s)."""
        return self.c**3 / (16.0 * math.pi * self.G)

    def Kc(self, sector: str) -> float:
        sector = sector.lower()
        if sector in ("strong", "su3"):
            return self.Kc_strong
        if sector in ("weak", "su2"):
            return self.Kc_weak
        if sector in ("em", "u1", "electromagnetic"):
            return self.Kc_em
        if sector in ("gravity", "grav", "su0"):
            return 0.0
        raise ValueError(f"Unknown sector: {sector}")

    def to_natural(self) -> "PSFTConstants":
        """Return a copy whose dynamical SI factors are set to 1."""
        return PSFTConstants(
            system="natural",
            c=1.0, hbar=1.0, G=1.0, epsilon_0=1.0,
            e=self.e, k_B=self.k_B,
            m_electron=self.m_electron, m_proton=self.m_proton,
            l_strong=self.l_strong, l_weak=self.l_weak, l_em=self.l_em,
            alpha_em=self.alpha_em, alpha_strong=self.alpha_strong,
            sin2_thetaW=self.sin2_thetaW,
            Lambda_QCD=1.0,
            string_tension_GeV2=self.string_tension_GeV2,
            R_tube_fm=self.R_tube_fm,
            m_W_GeV=self.m_W_GeV, m_Z_GeV=self.m_Z_GeV, m_H_GeV=self.m_H_GeV,
        )


SI = PSFTConstants(system="SI")
NATURAL = SI.to_natural()
