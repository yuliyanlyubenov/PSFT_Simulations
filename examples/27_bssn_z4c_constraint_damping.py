"""Example 27: Z4c constraint damping demonstration -- Phase 1.5 of the
curved-background extension (Step 5.1 of the simulation roadmap).

Implements the minimal Theta-only Z4c formulation of
Bernuzzi & Hilditch (PRD 81, 084003, 2010).  Z4c adds a single
scalar variable Theta on top of BSSN, with three structural
modifications:

    d_t chi += (4/3) alpha chi Theta             [BH 2010 eq. 14]
    d_t K   += alpha kappa1 (1 - kappa2) Theta   [BH 2010 eq. 16]
    d_t Theta = (alpha/2) H + beta . grad Theta
                - alpha kappa1 (2 + kappa2) Theta [BH 2010 eq. 4]

Here H = R + (2/3) K^2 - Abar:Abar - 16 pi rho_adm is the
Hamiltonian-constraint violation; R is the spatial Ricci scalar.

The chi modification is the critical structural piece: without it,
the K-Theta back-coupling alone does NOT stabilise BSSN's
constraint-violating mode.  With it, Z4c produces a hyperbolic
constraint subsystem with exponential damping at rate kappa1.

This example compares the Hamiltonian-constraint norm under three
formulations of the same dust-ball evolution:
  * vanilla BSSN (kappa1 = 0): the default Phase 1.3 setup
  * Z4c (kappa1 = 0.1)
  * Z4c (kappa1 = 0.5)

Why this matters for PSFT: high-curvature regions near matter
solitons (paper Postulate 3 + Theorem 10.1: K -> Kc^strong at the
fm scale) require a numerically stable BSSN-like formulation.
Vanilla BSSN's constraint violation grows polynomially (in mild
cases) or exponentially (in strong-field cases like punctures or
neutron-star surfaces), eventually drowning out physics.  Z4c
damps these violations on the timescale 1/kappa1 without
interfering with the physical dynamics.

Run:
    python3 examples/27_bssn_z4c_constraint_damping.py
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from psft.evolve.adm import (
    BSSNState, BSSNEvolver, hamiltonian_constraint,
)


def main():
    print("BSSN Phase 1.5: Z4c constraint damping demonstration")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    N = 32 if HIGH_RES else 20
    L = 4.0
    dh = L / N
    n_steps = 150 if HIGH_RES else 80

    print(f"  HIGH_RES = {HIGH_RES}")
    print(f"  grid: {N}^3, L = {L}, dh = {dh:.4f}, n_steps = {n_steps}")

    # Gaussian dust distribution (weak field).
    X = np.linspace(0.5 * dh, L - 0.5 * dh, N)
    Xg, Yg, Zg = np.meshgrid(X, X, X, indexing='ij')
    r = np.sqrt((Xg - L / 2) ** 2 + (Yg - L / 2) ** 2 + (Zg - L / 2) ** 2)
    sigma = 0.4
    rho_amp = 0.02
    rho_rest = rho_amp * np.exp(-r ** 2 / (2 * sigma ** 2))
    M_total = float(np.sum(rho_rest)) * dh ** 3
    print(f"  Gaussian dust: sigma={sigma}, amp={rho_amp}, M = {M_total:.4f}")

    # Compare three Z4c damping levels.
    configs = [
        ("vanilla BSSN", 0.0),
        ("Z4c kappa1=0.1", 0.1),
        ("Z4c kappa1=0.5", 0.5),
    ]

    histories = {}
    print()
    for label, kappa1 in configs:
        print(f"  Evolving '{label}' ...")
        state, rho_adm = BSSNState.static_dust_ball(
            N, N, N, dh, dh, dh, rho_rest=rho_rest, n_jacobi=3000,
        )
        evolver = BSSNEvolver(
            state=state, cfl=0.25, ko_epsilon=0.5,
            rho_adm=rho_adm, kappa1=kappa1, kappa2=0.0,
        )

        H_init = float(np.max(np.abs(hamiltonian_constraint(state, rho=rho_adm))))
        times = [0.0]
        H_history = [H_init]
        K_history = [0.0]
        Theta_history = [0.0]

        t_start = time.time()
        for step in range(n_steps):
            evolver.step()
            if (step + 1) % max(1, n_steps // 20) == 0:
                H = float(np.max(np.abs(hamiltonian_constraint(state, rho=rho_adm))))
                K_max = float(np.max(np.abs(state.K)))
                Theta_max = float(np.max(np.abs(state.theta)))
                times.append(evolver.t)
                H_history.append(H)
                K_history.append(K_max)
                Theta_history.append(Theta_max)
        elapsed = time.time() - t_start

        H_growth = H_history[-1] / H_init
        histories[label] = {
            'kappa1': kappa1,
            'times': times,
            'H': H_history,
            'K': K_history,
            'Theta': Theta_history,
            'elapsed': elapsed,
            'H_growth': H_growth,
        }
        print(f"     |H| init = {H_init:.2e}, final = {H_history[-1]:.2e} "
              f"(growth {H_growth:.1f}x, {elapsed:.1f}s)")

    # Damping effectiveness
    print()
    print("  Z4c damping factor:")
    H_bssn = histories['vanilla BSSN']['H'][-1]
    for label, _ in configs[1:]:
        H_z4c = histories[label]['H'][-1]
        damp = H_bssn / max(H_z4c, 1e-30)
        print(f"    {label}: H reduced {damp:.1f}x vs vanilla BSSN")

    # PASS criteria (thresholds depend on resolution and run length).
    if HIGH_RES:
        bssn_grow_target = 100.0
        z4c_damp_target = 5.0
    else:
        bssn_grow_target = 3.0   # low-res, short evolution: modest growth
        z4c_damp_target = 2.0
    pass_bssn_grows = histories['vanilla BSSN']['H_growth'] > bssn_grow_target
    pass_z4c_damps = histories['vanilla BSSN']['H'][-1] / max(histories['Z4c kappa1=0.5']['H'][-1], 1e-30) > z4c_damp_target
    pass_finite = all(
        not np.isnan(h[-1]) and not np.isinf(h[-1])
        for h in [histories[lbl]['H'] for lbl, _ in configs]
    )
    print(f"\n  PASS criteria (mode = {'HIGH_RES' if HIGH_RES else 'smoke'}):")
    print(f"    vanilla BSSN H grows >{bssn_grow_target:.0f}x        : "
          f"{'PASS' if pass_bssn_grows else 'FAIL'}")
    print(f"    Z4c k1=0.5 reduces H by >{z4c_damp_target:.0f}x       : "
          f"{'PASS' if pass_z4c_damps else 'FAIL'}")
    print(f"    all evolutions finite (no NaN)    : {'PASS' if pass_finite else 'FAIL'}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'vanilla BSSN': 'tab:blue', 'Z4c kappa1=0.1': 'tab:orange', 'Z4c kappa1=0.5': 'tab:green'}
    ax = axes[0]
    for label, _ in configs:
        h = histories[label]
        ax.semilogy(h['times'], h['H'], 'o-', color=colors[label], label=label)
    ax.set_xlabel('time t')
    ax.set_ylabel('max |H| (Hamiltonian constraint violation)')
    ax.set_title('Constraint violation: BSSN vs Z4c on dust ball')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for label, _ in configs:
        h = histories[label]
        ax.plot(h['times'], h['Theta'], 'o-', color=colors[label], label=label)
    ax.set_xlabel('time t')
    ax.set_ylabel('max |Theta|')
    ax.set_title('Z4c Theta absorbs constraint violation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), 'examples', 'out_bssn_z4c.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out}")

    print()
    print("  PSFT interpretation:")
    print("  PSFT requires accurate evolution in high-curvature regions")
    print("  (Postulate 3: K -> Kc^strong at the fm scale near matter")
    print("  solitons; Theorem 10.1: SU(3) viscosity activates above K_c).")
    print("  Vanilla BSSN's constraint-violating instability grows")
    print("  exponentially under high curvature -- the numerical analogue")
    print("  of the BSSN/ADM gauge mode that plagued NR codes pre-2005.")
    print("  Z4c (Bernuzzi-Hilditch 2010) damps these violations via the")
    print("  Theta channel: constraint violation -> Theta growth -> Theta")
    print("  damped by kappa1 -> constraint violation suppressed.  This")
    print("  example demonstrates the mechanism on a weak-field dust ball")
    print("  (where BSSN is only mildly unstable); for strong-field PSFT")
    print("  applications (neutron-star f_2 mode, soliton-on-soliton")
    print("  scattering, geon-style trapped-light configurations) Z4c is a")
    print("  prerequisite.")


if __name__ == "__main__":
    main()
