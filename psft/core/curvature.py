"""Christoffel, Riemann, Ricci, scalar curvature, Kretschmann.

Pointwise: takes a Metric and a coordinate x = (4,) numpy array, returns
the curvature bundle at that point.  Derivatives default to finite-difference;
override Metric.dg() to supply analytic ones.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from psft.core.metric import Metric


@dataclass
class CurvatureBundle:
    """Holds curvature tensors at a single point.

    Conventions: Riemann is R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
                                       + Gamma^a_{ce} Gamma^e_{bd}
                                       - Gamma^a_{de} Gamma^e_{bc}.
    Ricci R_{bd} = R^a_{bad}. Scalar R = g^{bd} R_{bd}.
    Kretschmann K = R_{abcd} R^{abcd}.
    """
    g: np.ndarray            # (4,4)
    g_inv: np.ndarray        # (4,4)
    Gamma: np.ndarray        # (4,4,4) Gamma^a_{bc}
    Riemann: np.ndarray      # (4,4,4,4) R^a_{bcd}
    Riemann_dn: np.ndarray   # (4,4,4,4) R_{abcd}
    Ricci: np.ndarray        # (4,4)
    R_scalar: float
    K_scalar: float          # Kretschmann
    Einstein: np.ndarray     # G_{ab} = R_{ab} - 1/2 R g_{ab}

    @classmethod
    def from_metric(cls, metric: Metric, x: np.ndarray,
                    h: Optional[float] = None) -> "CurvatureBundle":
        if x.ndim != 1 or x.shape[0] != 4:
            raise ValueError("CurvatureBundle.from_metric expects pointwise x of shape (4,)")
        if h is None:
            h = 1e-5

        g = metric.g(x)
        g_inv = metric.g_inv(x)

        dg = metric.dg(x, h)  # shape (4,4,4): dg[c,a,b] = d_c g_{ab}

        # Christoffel of the second kind:
        # Gamma^a_{bc} = 1/2 g^{ad} ( d_b g_{dc} + d_c g_{db} - d_d g_{bc} )
        # dg_indexed[c,a,b] = d_c g_{ab}; we want d_b g_{dc}, d_c g_{db}, d_d g_{bc}.
        dg_cab = dg  # d_c g_{ab}
        # d_b g_{dc} -> shape (b, d, c). dg_cab is indexed [c,a,b]; transpose:
        d_b_gdc = np.transpose(dg_cab, (2, 1, 0))  # axes: b, a->d, c?  Let me redo with einsum names.
        # We rename axes: dg has shape (c,a,b). We want a (b,d,c)-array d_b g_{dc}.
        # If we relabel dg's index slots (c->b', a->d', b->c'), we have d_{b'} g_{d' c'}.
        # So d_b_gdc = transpose dg with axes (0->2 wrong). Let me use explicit indexing.
        Gamma = np.zeros((4, 4, 4))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    s = 0.0
                    for d in range(4):
                        s += g_inv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                    Gamma[a, b, c] = 0.5 * s

        # Derivative of Gamma via finite differences -- second-derivative-of-metric.
        # We compute d_c Gamma^a_{bd} numerically by perturbing x.
        dGamma = np.zeros((4, 4, 4, 4))  # axes: (c, a, b, d) -> d_c Gamma^a_{bd}
        for c in range(4):
            xp = x.copy(); xp[c] += h
            xm = x.copy(); xm[c] -= h
            Gp = _christoffel_at(metric, xp, h)
            Gm = _christoffel_at(metric, xm, h)
            dGamma[c] = (Gp - Gm) / (2 * h)

        # Riemann tensor R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
        #                          + Gamma^a_{ce} Gamma^e_{bd}
        #                          - Gamma^a_{de} Gamma^e_{bc}.
        Riemann = (
            np.transpose(dGamma, (1, 2, 0, 3))   # d_c Gamma^a_{bd}  -> (a, b, c, d)
            - np.transpose(dGamma, (1, 3, 2, 0))[..., ::-1, ::-1, :]  # placeholder, redo below
        )
        Riemann = np.zeros((4, 4, 4, 4))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        term = dGamma[c, a, b, d] - dGamma[d, a, b, c]
                        for e in range(4):
                            term += Gamma[a, c, e] * Gamma[e, b, d]
                            term -= Gamma[a, d, e] * Gamma[e, b, c]
                        Riemann[a, b, c, d] = term

        # Lower the first index for Kretschmann.
        Riemann_dn = np.einsum("ae,ebcd->abcd", g, Riemann)
        # And raise all for the contraction.
        Riemann_up = np.einsum(
            "ae,bf,cg,dh,efgh->abcd",
            g_inv, g_inv, g_inv, g_inv, Riemann_dn,
        )
        K_scalar = float(np.sum(Riemann_dn * Riemann_up))

        # Ricci: R_{bd} = R^a_{bad}
        Ricci = np.einsum("abad->bd", Riemann)
        R_scalar = float(np.einsum("bd,bd->", g_inv, Ricci))
        Einstein = Ricci - 0.5 * R_scalar * g

        return cls(
            g=g, g_inv=g_inv, Gamma=Gamma,
            Riemann=Riemann, Riemann_dn=Riemann_dn, Ricci=Ricci,
            R_scalar=R_scalar, K_scalar=K_scalar, Einstein=Einstein,
        )


def _christoffel_at(metric: Metric, x: np.ndarray, h: float) -> np.ndarray:
    g = metric.g(x)
    g_inv = metric.g_inv(x)
    dg = metric.dg(x, h)
    G = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                s = 0.0
                for d in range(4):
                    s += g_inv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                G[a, b, c] = 0.5 * s
    return G
