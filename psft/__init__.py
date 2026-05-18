"""Photonic Spacetime Fluid Theory simulation library.

Numerical implementation of the v2 master equation from PSFT_paper.tex.
The library is organised around the paper's mathematical layers:

    core/     - manifold, metric, curvature, kinematics, photonic source
    gauge/    - U(1), SU(2), SU(3) algebra primitives
    sectors/  - viscosity functions and the master equation
    solitons/ - topological matter (vortex electron, knot proton/neutron, etc.)
    evolve/   - time integrators
    viz/      - plotting helpers
"""

from psft.core.constants import PSFTConstants, NATURAL, SI
from psft.core.metric import (
    Metric, MinkowskiMetric, SchwarzschildMetric, FLRWMetric, FunctionalMetric,
)
from psft.core.curvature import CurvatureBundle
from psft.core.kinematics import KinematicDecomposition
from psft.core.photonic import PhotonicSource, ElectromagneticField
from psft.gauge.algebra import U1, SU2, SU3, GaugeAlgebra
from psft.sectors.viscosity import (
    HeavisideViscosity, ScalarViscosity, GaugeViscosity, HallViscosity,
)
from psft.sectors.master_eq import MasterEquationV2, MasterEquationV1
from psft.solitons.topology import winding_number_1d, hopf_invariant_proxy
from psft.solitons.ansatz import VortexAnsatz, NielsenOlesenAnsatz
from psft.solitons.relax import GradientFlowRelaxer
from psft.evolve.integrators import RK4, AdaptiveRK45

__version__ = "0.1.0"

__all__ = [
    "PSFTConstants", "NATURAL", "SI",
    "Metric", "MinkowskiMetric", "SchwarzschildMetric", "FLRWMetric",
    "FunctionalMetric",
    "CurvatureBundle", "KinematicDecomposition",
    "PhotonicSource", "ElectromagneticField",
    "U1", "SU2", "SU3", "GaugeAlgebra",
    "HeavisideViscosity", "ScalarViscosity", "GaugeViscosity", "HallViscosity",
    "MasterEquationV2", "MasterEquationV1",
    "winding_number_1d", "hopf_invariant_proxy",
    "VortexAnsatz", "NielsenOlesenAnsatz", "GradientFlowRelaxer",
    "RK4", "AdaptiveRK45",
]
