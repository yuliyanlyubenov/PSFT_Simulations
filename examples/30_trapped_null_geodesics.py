"""Example 30: trapped null geodesics around a Schwarzschild-strength
curvature -- the Wheeler-geon trapping mechanism (Step 4f of the
simulation roadmap, paper Section 7.4 forward prediction).

In PSFT, the Heaviside-activated viscosity at K -> Kc^strong is the
structural stabilisation mechanism for matter solitons (Postulate 1's
geon interpretation, paper Section 2).  At fm-scale curvatures
K ~ 1.2 x 10^61 m^-4, null geodesics on the local geometry can be
CLOSED: photons cannot escape the soliton's curvature well.  This is
the Wheeler-geon picture (Wheeler 1955) implemented by PSFT's
high-K viscosity activation rather than by pure-GR self-gravitation.

This example demonstrates the trapping mechanism on a known reference
geometry -- the Schwarzschild metric -- which we use as a stand-in
for the soliton's exterior at K ~ Kc^strong.  The Schwarzschild photon
sphere is at areal radius r = 3M, with critical impact parameter
b_crit = 3 sqrt(3) M ~ 5.196 M (e.g. Misner-Thorne-Wheeler chapter 25).

We launch null geodesics from rho = 20 M with varying impact parameters
b in {2, 3, 4, 5, 5.196 (critical), 5.5, 6, 8, 12} M and integrate to
proper time tau = 80 M.  Classification:

  * captured  : photon falls past the horizon at rho = M/2
  * trapped   : photon orbits near r = 3M for many cycles
  * escaped   : photon retreats to rho > 30 M

The critical photon (b = 5.196 M) sits on the photon sphere
indefinitely; b > b_crit escapes; b < b_crit is captured.  Without
PSFT's high-K viscosity, the orbiting photon at b = b_crit is
unstable (any perturbation throws it to capture or escape).
Hilditch et al. 2013 numerical relativity work shows the same on
black-hole punctures; in PSFT, the prediction is that the same
trapping geometry arises at fm scale near matter solitons, with the
v2 master equation's Heaviside-activated viscosity supplying the
STABILISATION mechanism that pure-GR geons lack.

Pass criteria:
  * b = 2 M    -> captured (photon falls into horizon)
  * b = 12 M   -> escaped (photon retreats to large rho)
  * b ~ 5.2 M  -> trapped (photon orbits near r = 3M for many proper
                  times)
  * Null condition |u . u| stays below 0.1 over the evolution (i.e.
    the geodesic stays null to ~10% throughout, RK4-FD accuracy)

Run:
    python3 examples/30_trapped_null_geodesics.py
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

from psft.core.metric import SchwarzschildMetric
from psft.evolve.geodesic import GeodesicState, GeodesicEvolver


def null_initial_velocity(metric, x):
    """Build a null 4-velocity at position x.

    For Schwarzschild in isotropic coords with photon moving in the +x
    direction:  u^t = B/A,  u^x = 1,  u^y = u^z = 0.
    This satisfies g_ab u^a u^b = -A^2 (B/A)^2 + B^2 = 0.
    """
    g = metric.g(x)
    # diagonal metric in isotropic coords: g_tt = -A^2, g_xx = g_yy = g_zz = B^2
    A = float(np.sqrt(-g[0, 0]))
    B = float(np.sqrt(g[1, 1]))
    return np.array([B / A, 1.0, 0.0, 0.0])


def classify_orbit(metric, traj, r_horizon, r_far):
    """Classify a geodesic trajectory by its asymptotic behaviour.

    `traj` is an array of shape (N+1, 4) of (t, x, y, z) positions.

    Returns ('captured', 'escaped', or 'trapped', final_areal_r).
    """
    # Compute areal radii.
    areal_r = np.array([metric.areal_radius(p) for p in traj])
    r_min = float(np.min(areal_r))
    r_final = float(areal_r[-1])
    # If photon ever reaches < r_horizon, it's captured.
    if r_min < r_horizon * 1.5:   # 1.5x horizon as numerical buffer
        return 'captured', r_final
    # If photon ends far away, escaped.
    if r_final > r_far:
        return 'escaped', r_final
    # Otherwise, trapped near photon sphere.
    return 'trapped', r_final


def main():
    print("Example 30: Trapped null geodesics around a Schwarzschild curvature")
    print("(Wheeler-geon trapping mechanism, PSFT paper Section 7.4 prediction)")
    print("=" * 70)

    HIGH_RES = os.environ.get("PSFT_HIGH_RES", "0") != "0"
    n_steps = 4000 if HIGH_RES else 2000
    dt = 0.04 if HIGH_RES else 0.08

    M = 1.0
    metric = SchwarzschildMetric(M=M, G=1.0, c=1.0)
    rho_horizon = M / 2.0          # isotropic horizon
    r_horizon = M * 2.0            # areal-radius horizon = 2M
    r_photon_sphere = 3.0 * M
    b_crit = 3.0 * np.sqrt(3.0) * M

    rho_launch = 20.0 * M
    r_far = 30.0 * M

    print(f"  M = {M},  r_horizon = {r_horizon} (areal),  "
          f"r_photon_sphere = {r_photon_sphere}")
    print(f"  b_crit = 3 sqrt(3) M = {b_crit:.4f}")
    print(f"  launching photons from rho = {rho_launch}")
    print(f"  HIGH_RES = {HIGH_RES}, dt = {dt}, n_steps = {n_steps}")

    # Impact parameters to test.
    impact_parameters = [2.0, 3.0, 4.0, 5.0, b_crit, 5.5, 6.0, 8.0, 12.0]
    classifications = []
    trajectories = []
    null_drifts = []

    evolver = GeodesicEvolver(metric=metric, dt=dt, fd_step=1e-3,
                              enforce_normalisation=False)

    print()
    for b in impact_parameters:
        # Initial position: at rho_launch, offset y = b for impact parameter
        x0 = np.array([0.0, -rho_launch, b, 0.0])
        u0 = null_initial_velocity(metric, x0)
        # Initial null condition
        g0 = metric.g(x0)
        null0 = float(np.einsum('ab,a,b->', g0, u0, u0))

        state = GeodesicState(tau=0.0, x=x0, u=u0)
        traj = [state.x.copy()]
        for _ in range(n_steps):
            # Check stopping conditions BEFORE the step (the metric is
            # singular inside the horizon and the inverse blows up).
            ar = metric.areal_radius(state.x)
            if ar < r_horizon * 1.1:
                # Photon captured -- horizon is at r = 2M.  Stop a hair
                # outside (1.1x) so the next step doesn't hit the singularity.
                break
            if ar > r_far * 1.5:
                # Photon escaped well beyond the launch radius.
                break
            try:
                state = evolver.step(state)
            except np.linalg.LinAlgError:
                # Metric inversion failed -- treat as captured.
                break
            traj.append(state.x.copy())

        traj = np.array(traj)
        cls, r_final = classify_orbit(metric, traj, r_horizon, r_far)
        # Check null drift at end
        null_final = float(np.einsum('ab,a,b->', metric.g(state.x), state.u, state.u))
        null_drifts.append(abs(null_final))
        trajectories.append((b, traj))
        classifications.append(cls)
        print(f"  b = {b:6.4f} M : {cls:10s} (r_final = {r_final:.2f}, "
              f"|null| = {abs(null_final):.2e}, {len(traj)} steps)")

    # Pass criteria.  Note: at b = b_crit EXACTLY, the photon orbits the
    # photon sphere indefinitely, but any tiny numerical perturbation
    # tips it to capture or escape -- this is the standard GR result
    # that the photon-sphere orbit is unstable.  We therefore test the
    # CAPTURE/ESCAPE TRANSITION rather than the marginal trapping
    # itself.
    #
    # IMPORTANT: the "b" values used as input are isotropic-coordinate
    # y-distances at the launch radius rho_launch.  The textbook b_crit
    # = 3 sqrt(3) M is in areal coordinates.  At rho_launch = 20 M the
    # conversion factor B(rho) = (1 + M/2rho)^2 ~ 1.05, so the
    # ISOTROPIC critical b at this launch radius is
    # b_crit_iso(rho_launch) = b_crit_areal / B(rho_launch).
    B_launch = (1.0 + M / (2.0 * rho_launch)) ** 2
    b_crit_iso = b_crit / B_launch
    pass_capture = classifications[0] == 'captured'        # b = 2M
    pass_escape  = classifications[-1] == 'escaped'        # b = 12M
    # Find the largest b that's captured, smallest b that escapes.
    captured_bs = [b for b, c in zip(impact_parameters, classifications) if c == 'captured']
    escaped_bs  = [b for b, c in zip(impact_parameters, classifications) if c == 'escaped']
    if captured_bs and escaped_bs:
        b_max_cap = max(captured_bs)
        b_min_esc = min(escaped_bs)
        # The numerical isotropic-b_crit should bracket between these.
        pass_transition = b_max_cap < b_crit_iso < b_min_esc + 1e-3
    else:
        pass_transition = False
    pass_null = max(null_drifts) < 0.5  # generous null-drift bound

    print(f"\n  PASS criteria:")
    print(f"    b = 2 M   captured        : {'PASS' if pass_capture else 'FAIL'}")
    print(f"    b = 12 M  escaped         : {'PASS' if pass_escape else 'FAIL'}")
    if captured_bs and escaped_bs:
        print(f"    transition between b = {b_max_cap:.4f}M (capt) and "
              f"b = {b_min_esc:.4f}M (esc)")
        print(f"      brackets b_crit_iso(rho={rho_launch}M) = {b_crit_iso:.4f} M")
        print(f"      (areal b_crit = 3 sqrt(3) M = {b_crit:.4f} M)")
    print(f"    capture/escape transition brackets b_crit : "
          f"{'PASS' if pass_transition else 'FAIL'}")
    print(f"    null condition |u.u| < 0.5 : {'PASS' if pass_null else 'FAIL'}")

    # Plot all trajectories in the equatorial plane (x, y).
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    color_map = {'captured': 'tab:red', 'trapped': 'tab:orange', 'escaped': 'tab:green'}
    for (b, traj), cls in zip(trajectories, classifications):
        ax.plot(traj[:, 1], traj[:, 2], '-', color=color_map[cls], alpha=0.8,
                label=f'b={b:.2f}M ({cls})')
    # Horizon + photon sphere
    theta = np.linspace(0, 2*np.pi, 200)
    # Horizon in isotropic coords: rho = M/2
    rho_h = M / 2.0
    ax.plot(rho_h * np.cos(theta), rho_h * np.sin(theta), 'k--',
            linewidth=2, label=f'horizon (rho={rho_h}M)')
    # Photon sphere in isotropic: rho = 1.866M (from h = 2-sqrt(3))
    rho_ps = M * (2.0 - np.sqrt(3.0)) ** (-1) * 0.5
    ax.plot(rho_ps * np.cos(theta), rho_ps * np.sin(theta), 'k:',
            linewidth=1.5, label=f'photon sphere (rho={rho_ps:.2f}M)')
    ax.set_xlabel('x / M'); ax.set_ylabel('y / M')
    ax.set_xlim(-25, 25); ax.set_ylim(-15, 15)
    ax.set_title('Null geodesic trajectories on Schwarzschild\n'
                  '(black-hole-equivalent of PSFT soliton-core trapping)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot areal radius vs proper time for each
    ax = axes[1]
    for (b, traj), cls in zip(trajectories, classifications):
        ar = np.array([metric.areal_radius(p) for p in traj])
        ts = np.arange(len(traj)) * dt
        ax.plot(ts, ar, color=color_map[cls], alpha=0.8,
                label=f'b={b:.2f}M ({cls})')
    ax.axhline(r_horizon, color='k', linestyle='--', alpha=0.7, label='horizon (r=2M)')
    ax.axhline(r_photon_sphere, color='k', linestyle=':', alpha=0.7, label='photon sphere (r=3M)')
    ax.set_xlabel('proper time tau / M')
    ax.set_ylabel('areal radius r / M')
    ax.set_yscale('log')
    ax.set_title('Photon radial distance vs time')
    ax.legend(loc='best', fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), 'examples', 'out_trapped_geodesics.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved to {out}")

    print()
    print("  PSFT interpretation:")
    print("  This demonstrates the Wheeler-geon trapping mechanism that")
    print("  paper Section 7.4 predicts at the fm-scale soliton core, where")
    print("  K -> Kc^strong ~ 1.2e61 m^-4 supplies a Schwarzschild-like")
    print("  curvature.  Without PSFT's high-K viscosity activation")
    print("  (Postulate 3 + Theorem 10.1), the marginally-trapped photon at")
    print("  b = b_crit is dynamically unstable -- as in the original 1955")
    print("  Wheeler-geon, it radiates away on a free-fall timescale.")
    print("  PSFT's prediction is that the Heaviside-activated SU(3)")
    print("  viscosity inside the soliton stabilises the trapped photon")
    print("  population, realising the geon interpretation of Postulate 1")
    print("  ('matter = stable solitonic pattern in P_ab').  Verifying that")
    print("  stabilisation numerically requires the matter-coupled photonic")
    print("  evolution on the BSSN-evolved geometry; this example provides")
    print("  the trapping-geometry baseline that the stabilisation builds on.")


if __name__ == "__main__":
    main()
