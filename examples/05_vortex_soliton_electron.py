"""Example 5: Construct a vortex soliton (electron candidate) and check
its U(1) topological charge.

By PSFT Postulate 4, electric charge = winding number along compact Killing
direction.  In flat space a U(1) line vortex with winding n carries n units
of U(1) charge.  We relax the field with the Mexican-hat potential

    E[Phi] = int [|grad Phi|^2 + (|Phi|^2 - 1)^2] d^3x

starting from the tanh ansatz and watch the energy decrease.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

from psft.core.manifold import CartesianGrid
from psft.solitons.ansatz import VortexAnsatz
from psft.solitons.topology import winding_number_2d_loop
from psft.solitons.relax import EnergyFunctional, GradientFlowRelaxer


def laplacian(f, dx):
    return (
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
        + np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2)
        - 6 * f
    ) / dx**2


def main():
    grid = CartesianGrid(shape=(48, 48, 8), bounds=((-4, 4), (-4, 4), (-1, 1)))
    dx = grid.deltas()[0]

    init = VortexAnsatz(winding=1, core_radius=0.4).evaluate(grid)
    Phi0 = init["phi_complex"]
    w0 = winding_number_2d_loop(Phi0[:, :, 4], center=(24, 24), radius=15)
    print(f"Initial vortex winding (z midplane): {w0}")

    # Pack real/imag parts for the relaxer (works on real arrays).
    def pack(Phi):  # complex (Nx,Ny,Nz) -> real (2, Nx, Ny, Nz)
        return np.stack([Phi.real, Phi.imag], axis=0)
    def unpack(state):
        return state[0] + 1j * state[1]

    def density(state, _grid):
        Phi = unpack(state)
        dPhi_x = np.gradient(Phi, dx, axis=0)
        dPhi_y = np.gradient(Phi, dx, axis=1)
        dPhi_z = np.gradient(Phi, dx, axis=2)
        grad2 = np.abs(dPhi_x)**2 + np.abs(dPhi_y)**2 + np.abs(dPhi_z)**2
        pot = (np.abs(Phi)**2 - 1.0)**2
        return grad2 + pot

    def grad(state, _grid):
        Phi = unpack(state)
        lap = laplacian(Phi, dx)
        # delta E / delta Phi*  =  -lap Phi + 2 (|Phi|^2 - 1) Phi
        var = -lap + 2.0 * (np.abs(Phi)**2 - 1.0) * Phi
        return np.stack([var.real, var.imag], axis=0) * 2.0

    energy = EnergyFunctional(density_func=density, grad_func=grad, dx=dx)
    relaxer = GradientFlowRelaxer(energy=energy, step=2e-3, max_iter=400,
                                  tol=1e-7, record_history=True)
    state0 = pack(Phi0)
    out = relaxer.relax(state0, grid)
    Phi_relaxed = unpack(out["phi"])
    w_relaxed = winding_number_2d_loop(Phi_relaxed[:, :, 4], center=(24, 24), radius=15)

    print(f"Initial energy: {energy.energy(state0, grid):.4f}")
    print(f"Relaxed energy: {out['energy']:.4f}  ({out['iters']} iters)")
    print(f"Relaxed vortex winding (z midplane): {w_relaxed}")
    if w_relaxed == w0:
        print("=> Topological charge conserved under gradient flow, as expected.")


if __name__ == "__main__":
    main()
