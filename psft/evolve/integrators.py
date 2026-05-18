"""Explicit time integrators used to evolve fields under the master equation.

Both integrators take a callable rhs(t, y) -> dy/dt; `y` may be any numpy
array.  RK4 has fixed step; AdaptiveRK45 uses Cash-Karp coefficients with
PI controller-style step size adjustment.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np


@dataclass
class RK4:
    rhs: Callable[[float, np.ndarray], np.ndarray]
    dt: float

    def step(self, t: float, y: np.ndarray) -> Tuple[float, np.ndarray]:
        h = self.dt
        k1 = self.rhs(t, y)
        k2 = self.rhs(t + h / 2, y + h / 2 * k1)
        k3 = self.rhs(t + h / 2, y + h / 2 * k2)
        k4 = self.rhs(t + h, y + h * k3)
        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t + h, y_next

    def run(self, t0: float, y0: np.ndarray, t_end: float):
        ts = [t0]; ys = [y0.copy()]
        t, y = t0, y0
        while t < t_end - 1e-15:
            t, y = self.step(t, y)
            ts.append(t); ys.append(y.copy())
        return np.array(ts), np.stack(ys, axis=0)


# Cash-Karp coefficients for the 4(5) embedded RK.
_CK_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [3/10, -9/10, 6/5],
    [-11/54, 5/2, -70/27, 35/27],
    [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
]
_CK_C = [0, 1/5, 3/10, 3/5, 1, 7/8]
_CK_B5 = [37/378, 0, 250/621, 125/594, 0, 512/1771]
_CK_B4 = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]


@dataclass
class AdaptiveRK45:
    rhs: Callable[[float, np.ndarray], np.ndarray]
    dt: float = 1e-3
    rtol: float = 1e-6
    atol: float = 1e-9
    dt_max: float = 1.0
    dt_min: float = 1e-12

    def step(self, t: float, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        h = self.dt
        while True:
            ks: List[np.ndarray] = []
            for i in range(6):
                yi = y.copy()
                for j, a in enumerate(_CK_A[i]):
                    yi = yi + h * a * ks[j]
                ks.append(self.rhs(t + _CK_C[i] * h, yi))
            y5 = y + h * sum(b * k for b, k in zip(_CK_B5, ks))
            y4 = y + h * sum(b * k for b, k in zip(_CK_B4, ks))
            err = np.max(np.abs(y5 - y4) / (self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))))
            if err <= 1.0 or h <= self.dt_min:
                self.dt = min(self.dt_max, h * min(5.0, max(0.1, 0.9 * err ** -0.2)) if err > 0 else h * 1.5)
                return t + h, y5, h
            h = max(self.dt_min, h * max(0.1, 0.9 * err ** -0.25))

    def run(self, t0: float, y0: np.ndarray, t_end: float):
        ts = [t0]; ys = [y0.copy()]
        t, y = t0, y0
        while t < t_end - 1e-15:
            t, y, _ = self.step(t, y)
            ts.append(t); ys.append(y.copy())
        return np.array(ts), np.stack(ys, axis=0)


def integrate_trajectory(rhs: Callable, t0: float, y0: np.ndarray, t_end: float,
                         method: str = "rk4", **kwargs):
    if method == "rk4":
        return RK4(rhs=rhs, dt=kwargs.get("dt", 1e-3)).run(t0, y0, t_end)
    elif method == "rk45":
        return AdaptiveRK45(rhs=rhs, **kwargs).run(t0, y0, t_end)
    raise ValueError(f"Unknown method: {method}")
