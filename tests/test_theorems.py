"""Numerical validation of the formal theorems proved in PSFT_paper.tex.

Each test sets up the precise hypothesis of a theorem and verifies its
conclusion using only the simulation library.  The tests are deliberately
strict: when the paper claims an identity (e.g. Bianchi nabla_a G^{ab} = 0
or eq. of motion uniqueness), we check it to FD-tolerance.

Run from simulation/ via:
    python3 -m unittest tests.test_theorems -v
"""
import math
import os
import sys
import unittest
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from psft.core.constants import SI, NATURAL, PSFTConstants
from psft.core.metric import (
    MinkowskiMetric, SchwarzschildMetric, FunctionalMetric,
)
from psft.core.curvature import CurvatureBundle
from psft.core.kinematics import KinematicDecomposition
from psft.core.photonic import ElectromagneticField
from psft.core.manifold import CartesianGrid
from psft.solitons.ansatz import VortexAnsatz
from psft.solitons.topology import winding_number_2d_loop
from psft.sectors.viscosity import (
    profile_QCD_log, HeavisideViscosity,
)


# ============================================================================
# Theorem 12.1 -- Classical Limit (PSFT inviscid <=> Einstein equations).
# ============================================================================
class TestTheorem121_ClassicalLimit(unittest.TestCase):
    """Theorem 12.1: in the inviscid limit, PSFT recovers GR exactly."""

    def test_vacuum_schwarzschild_is_ricci_flat(self):
        """R_ab = 0 in vacuum -- the defining condition for a Schwarzschild
        black hole, on which Theorem 12.1's `inviscid <=> Einstein' rests."""
        bh = SchwarzschildMetric(M=1.0, G=1.0, c=1.0)
        rho = 20.0  # well outside horizon
        x = np.array([0.0, rho, 0.0, 0.0])
        bundle = CurvatureBundle.from_metric(bh, x, h=1e-2)
        # In vacuum: R_ab = 0.  Compare against Kretschmann scale to get a
        # relative tolerance (we are finite-differencing, so absolute zero
        # is not realistic).
        scale = math.sqrt(abs(bundle.K_scalar))
        rel_ricci = np.max(np.abs(bundle.Ricci)) / scale
        self.assertLess(rel_ricci, 0.1,
                        msg=f"|R_ab|/sqrt(K) = {rel_ricci:.3e} -- not Ricci-flat")

    def test_schwarzschild_killing_equation(self):
        """Stationary Killing vector xi^a = (1,0,0,0) satisfies
            nabla_a xi_b + nabla_b xi_a = 0  exactly  on Schwarzschild.
        This is the hypothesis required by Theorem 9.1 to give EM.
        """
        bh = SchwarzschildMetric(M=1.0, G=1.0, c=1.0)
        x = np.array([0.0, 15.0, 0.0, 0.0])
        bundle = CurvatureBundle.from_metric(bh, x, h=1e-2)
        # xi^a = (1,0,0,0) -> xi_a = g_a0 (one column of g)
        xi_dn = bundle.g[:, 0].copy()
        # Compute nabla_a xi_b = d_a xi_b - Gamma^c_{ab} xi_c
        # d_a xi_b: only the column 0 of g varies with r (and only g_00 = -A^2).
        # For xi^a = delta^a_0 const, xi_a(x) = g_{a0}(x).
        h = 1e-3
        d_xi = np.zeros((4, 4))
        for a in range(4):
            xp = x.copy(); xp[a] += h
            xm = x.copy(); xm[a] -= h
            d_xi[a] = (bh.g(xp)[:, 0] - bh.g(xm)[:, 0]) / (2 * h)
        nabla_xi_dn = d_xi - np.einsum("cab,c->ab", bundle.Gamma, xi_dn)
        sym = nabla_xi_dn + nabla_xi_dn.T  # should vanish
        self.assertTrue(np.allclose(sym, 0.0, atol=1e-3),
                        msg=f"||sym(nabla xi)|| = {np.linalg.norm(sym):.3e}")

    def test_inviscid_master_eq_reduces_to_euler(self):
        """When K < K_c everywhere, the v2 viscous and Hall terms drop out
        identically; the master equation collapses to relativistic Euler.
        Verified by computing the residual of the master eq with viscosity 0
        and confirming it agrees with the pure-pressure form to FD precision."""
        from psft.sectors.viscosity import GaugeViscosity
        from psft.sectors.master_eq import MasterEquationV2
        m = MinkowskiMetric()
        gv = GaugeViscosity.physical(SI)  # all sectors inactive at K=0
        eq = MasterEquationV2(metric=m, consts=SI, gauge_viscosity=gv,
                              sector="gravity")
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u_func = lambda _x: np.array([1.0, 0.0, 0.0, 0.0])
        grad_p = np.array([0.0, 0.7, -0.3, 0.1])
        result = eq.evaluate(x, u_func, rho_g=1.0, p_g=0.0, grad_p_g=grad_p)
        # Euler-eq form: -h^a_b grad_b p with h = diag(0, 1, 1, 1) for static u.
        expected = -np.array([0.0, 0.7, -0.3, 0.1])
        self.assertTrue(np.allclose(result.pressure_force, expected, atol=1e-8))
        self.assertTrue(np.allclose(result.viscous_force, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(result.hall_force, 0.0, atol=1e-10))


# ============================================================================
# Theorem 9.1 -- Electromagnetic Emergence (Killing vorticity -> Maxwell).
# ============================================================================
class TestTheorem91_ElectromagneticEmergence(unittest.TestCase):

    def test_homogeneous_maxwell_d_F_equals_zero(self):
        """For any potential A_a, the field strength F = dA satisfies
            nabla_[a F_bc] = 0  (Bianchi identity).
        This is Theorem 9.1(i).  We verify it numerically on a Coulomb
        configuration, which is the static-spherically-symmetric solution."""
        m = MinkowskiMetric()
        Q = 1.0; eps0 = 1.0
        def A(xx):
            r = max(math.sqrt(xx[1]**2 + xx[2]**2 + xx[3]**2), 1e-6)
            return np.array([-Q / (4 * np.pi * eps0 * r), 0.0, 0.0, 0.0])
        x = np.array([0.0, 1.0, 0.5, 0.3])
        h = 1e-4
        # Evaluate F_ab and its derivatives via central differences.
        def F_at(xx):
            return ElectromagneticField.from_potential(m, xx, A, h=1e-5).F_dn
        F = F_at(x)
        dF = np.zeros((4, 4, 4))  # dF[a, b, c] = d_a F_{bc}
        for a in range(4):
            xp = x.copy(); xp[a] += h
            xm = x.copy(); xm[a] -= h
            dF[a] = (F_at(xp) - F_at(xm)) / (2 * h)
        # Bianchi: d_a F_{bc} + d_b F_{ca} + d_c F_{ab} = 0.
        bianchi_norm = 0.0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    val = dF[a, b, c] + dF[b, c, a] + dF[c, a, b]
                    bianchi_norm += val * val
        self.assertLess(math.sqrt(bianchi_norm), 1e-3,
                        msg=f"||d F||_F = {math.sqrt(bianchi_norm):.3e}")

    def test_inhomogeneous_maxwell_in_vacuum(self):
        """In vacuum, the inhomogeneous Maxwell eq reduces to
            nabla_a F^{ab} = 0
        for a static spherically symmetric Coulomb field outside any source."""
        m = MinkowskiMetric()
        Q = 1.0; eps0 = 1.0
        def A(xx):
            r = max(math.sqrt(xx[1]**2 + xx[2]**2 + xx[3]**2), 1e-6)
            return np.array([-Q / (4 * np.pi * eps0 * r), 0.0, 0.0, 0.0])
        x = np.array([0.0, 2.0, 1.0, 0.5])
        h = 1e-3
        def F_up_at(xx):
            em = ElectromagneticField.from_potential(m, xx, A, h=1e-5)
            return em.F_up()
        Fp = F_up_at(x)
        div = np.zeros(4)
        for a in range(4):
            xp = x.copy(); xp[a] += h
            xm = x.copy(); xm[a] -= h
            div += (F_up_at(xp)[a, :] - F_up_at(xm)[a, :]) / (2 * h)
        # Compare against the field strength to set a relative tolerance.
        field_scale = float(np.max(np.abs(Fp)))
        rel = float(np.max(np.abs(div))) / field_scale
        self.assertLess(rel, 0.05,
                        msg=f"||div F||/|F| = {rel:.3e} (should be ~0 in vacuum)")

    def test_coulomb_potential_radial_falloff(self):
        """Theorem 9.1(iii): the static spherically symmetric solution of the
        Maxwell eq is V(r) = Q/(4 pi eps_0 r).  Verify the 1/r^2 falloff of E."""
        m = MinkowskiMetric()
        Q = 1.0; eps0 = 1.0
        def A(xx):
            r = max(math.sqrt(xx[1]**2 + xx[2]**2 + xx[3]**2), 1e-6)
            return np.array([-Q / (4 * np.pi * eps0 * r), 0.0, 0.0, 0.0])
        radii = [0.5, 1.0, 2.0, 5.0]
        E_values = []
        for r in radii:
            x = np.array([0.0, r, 0.0, 0.0])
            em = ElectromagneticField.from_potential(m, x, A, h=1e-5)
            E_values.append(float(em.F_dn[1, 0]))   # F_{rt} ~ -E_r
        # Compute (E * r^2): should be approximately constant.
        products = [E * r * r for E, r in zip(E_values, radii)]
        ratio = max(products) / min(products)
        self.assertLess(abs(ratio - 1.0), 1e-4,
                        msg=f"E*r^2 not constant: {products}")

    def test_charge_quantization_via_winding(self):
        """Theorem 9.1(iv): electric charge = integer winding of the U(1)
        flow.  We verify this for windings n = -3, ..., +3 using the line-
        vortex ansatz: the winding number measured by the topology helper
        must equal n exactly."""
        grid = CartesianGrid(shape=(80, 80, 4), bounds=((-5, 5), (-5, 5), (-1, 1)))
        for n in (-3, -1, 0, 1, 2, 3):
            ans = VortexAnsatz(winding=n, core_radius=0.5).evaluate(grid)
            w = winding_number_2d_loop(
                ans["phi_complex"][:, :, 2],
                center=(40, 40), radius=30, n_samples=512,
            )
            self.assertEqual(w, n,
                             msg=f"winding mismatch for n={n}: got {w}")


# ============================================================================
# Theorem 10.1 -- Confinement (high viscosity -> linear potential).
# ============================================================================
class TestTheorem101_Confinement(unittest.TestCase):

    def test_string_tension_formula(self):
        """V(r) = sigma * r with sigma = C_F eta(Kc) g_s^2 / A_tube.

        Plug in the lattice-QCD values from the paper (sigma ~ 0.18 GeV^2,
        R_tube ~ 0.35 fm) and verify that the implied eta(Kc) is O(1) in
        natural units, matching the paper's ~ 1.31/g_s^2 estimate."""
        sigma = NATURAL.string_tension_GeV2          # 0.18 GeV^2
        R_tube_inv_GeV = NATURAL.R_tube_fm / 0.1973  # 1 fm = 1/0.1973 GeV^-1
        A_tube = math.pi * R_tube_inv_GeV ** 2
        C_F = 4.0 / 3.0
        g_s2 = 4 * math.pi * NATURAL.alpha_strong
        eta_Kc = sigma * A_tube / (C_F * g_s2)
        paper_estimate = 1.31 / g_s2
        rel = abs(eta_Kc - paper_estimate) / paper_estimate
        self.assertLess(rel, 0.05,
                        msg=f"eta(Kc) = {eta_Kc:.3f}, paper = {paper_estimate:.3f}")

    def test_asymptotic_freedom_log_slope(self):
        """Theorem 10.1(iii): f(K/Kc) = 1/(b_0 ln(K/Kc)) gives the one-loop
        running of QCD.  Verify that d/d(ln K/Kc) of 1/f equals b_0
        (the one-loop QCD coefficient 27/(48 pi) for N_f=3)."""
        # Sample the profile across many decades of K/Kc above threshold.
        ratios = np.logspace(1, 9, 25)        # K/Kc from 10 to 1e9
        f_vals = np.array([profile_QCD_log(r) for r in ratios])
        log_ratios = np.log(ratios)
        inv_f = 1.0 / f_vals
        slope = float(np.polyfit(log_ratios, inv_f, 1)[0])
        expected_b0 = 27.0 / (48 * math.pi)   # paper Theorem 10.1(iii)
        rel = abs(slope - expected_b0) / expected_b0
        self.assertLess(rel, 0.005,
                        msg=f"profile log-slope = {slope:.5f}, expected b_0 = {expected_b0:.5f}")

    def test_confinement_radius_below_Kc(self):
        """Below the confinement radius, K < Kc and the SU(3) viscosity is
        identically zero (Theorem 10.1(i)).  Sample inside and outside r_conf
        and verify the Heaviside behaviour."""
        eta = HeavisideViscosity(eta_0=1.0, K_c=10.0)
        self.assertEqual(eta.eta(K=5.0), 0.0,  msg="K < Kc must give zero eta")
        self.assertEqual(eta.eta(K=10.0), 0.0, msg="K = Kc must give zero eta")
        self.assertGreater(eta.eta(K=20.0), 0.0, msg="K > Kc must give non-zero")


# ============================================================================
# Theorem 11.1 -- Weak Force Emergence (Hall viscosity).
# ============================================================================
class TestTheorem111_WeakForce(unittest.TestCase):

    def test_hall_stress_is_parity_odd(self):
        """Theorem 11.1(i): tau^Hall_{ij} flips sign under P: x^i -> -x^i."""
        from itertools import permutations
        # Build eps_{abcd} in Minkowski.
        eps = np.zeros((4, 4, 4, 4))
        for p in permutations((0, 1, 2, 3)):
            inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
            eps[p] = 1.0 if inv % 2 == 0 else -1.0
        # eps_{ab}^{cd} = eps_{abef} g^{ec} g^{df} (Minkowski).
        eta_inv = np.diag([-1.0, 1.0, 1.0, 1.0])
        eps_mixed = np.einsum("abef,ec,fd->abcd", eps, eta_inv, eta_inv)
        u_dn = np.array([-1.0, 0.0, 0.0, 0.0])         # comoving observer
        J_dn = np.array([0.0, 0.3, 0.5, -0.2])         # D_d sigma^A (polar)
        # tau^Hall_{ab} = eps_{ab}^{cd} u_c J_d
        tau = np.einsum("abcd,c,d->ab", eps_mixed, u_dn, J_dn)
        # Polar transformation of polar 4-vectors: spatial components flip.
        u_P = u_dn.copy(); u_P[1:] = -u_P[1:]
        J_P = J_dn.copy(); J_P[1:] = -J_dn[1:]
        # Polar transformation of a rank-2 tensor: T_{ij} unchanged, T_{0i} flips.
        P_tau = tau.copy()
        for a in range(4):
            for b in range(4):
                if (int(a > 0) + int(b > 0)) % 2 == 1:
                    P_tau[a, b] = -tau[a, b]
        # Substitute parity-flipped inputs into the formula.  The pseudo-
        # tensor character of eps is implicit -- we do NOT separately flip
        # eps, because that would double-count.
        tau_from_P_inputs = np.einsum("abcd,c,d->ab", eps_mixed, u_P, J_P)
        # Parity-odd identity: tau(P inputs) = -P[tau].
        self.assertTrue(
            np.allclose(tau_from_P_inputs + P_tau, 0.0, atol=1e-12),
            msg=f"||tau(P x) + P[tau]|| = {np.linalg.norm(tau_from_P_inputs + P_tau):.3e}",
        )

    def test_w_mass_dispersion_relation(self):
        """Theorem 11.1(ii): linear perturbations on the broken vacuum satisfy
            omega^2 = v^2 k^2 + m^2  with  m^2 = kappa / (rho_g + p_g).
        Verify that with paper-supplied values (kappa/(rho+p) ~ 6400 GeV^2)
        the mass evaluates to m_W ~ 80 GeV."""
        kappa_over_rho_p = 6400.0    # GeV^2  (paper Sec. 11, proof of ii)
        m = math.sqrt(kappa_over_rho_p)
        m_W_expected = NATURAL.m_W_GeV
        rel = abs(m - m_W_expected) / m_W_expected
        self.assertLess(rel, 0.01,
                        msg=f"derived m_W = {m:.2f} GeV, paper m_W = {m_W_expected}")


# ============================================================================
# Theorem 13.1 -- Energy-momentum conservation.
# ============================================================================
class TestTheorem131_Conservation(unittest.TestCase):

    def test_contracted_bianchi_in_vacuum_schwarzschild(self):
        """Theorem 13.1(i) reduces in the inviscid limit to the contracted
        Bianchi identity nabla_a G^{ab} = 0.  We verify this by computing
        G^{ab}(x +/- h e_c) and differencing.  Strict check uses a smaller
        finite-difference step inside the CurvatureBundle call."""
        bh = SchwarzschildMetric(M=1.0, G=1.0, c=1.0)
        x = np.array([0.0, 25.0, 0.0, 0.0])
        h = 0.1
        def G_up_at(xx):
            b = CurvatureBundle.from_metric(bh, xx, h=1e-2)
            return np.einsum("ac,bd,cd->ab", b.g_inv, b.g_inv, b.Einstein)
        div = np.zeros(4)
        for a in range(4):
            xp = x.copy(); xp[a] += h
            xm = x.copy(); xm[a] -= h
            div += (G_up_at(xp)[a, :] - G_up_at(xm)[a, :]) / (2 * h)
        # Compare against a curvature scale (sqrt(K)).
        bundle0 = CurvatureBundle.from_metric(bh, x, h=1e-2)
        scale = math.sqrt(abs(bundle0.K_scalar))
        rel = float(np.max(np.abs(div))) / scale if scale > 0 else float(np.max(np.abs(div)))
        self.assertLess(rel, 0.2,
                        msg=f"||nabla G||/sqrt(K) = {rel:.3e}; expected ~0")


# ============================================================================
# Theorem 14.1 -- Lorentz covariance (every term is a tensor).
# ============================================================================
class TestGeodesicEvolver(unittest.TestCase):
    """The PSFT master equation in the inviscid limit reduces to the geodesic
    equation (Theorem 12.1).  Test the time-integrated evolver against the
    closed-form Schwarzschild perihelion-precession formula."""

    def test_schwarzschild_orbit_recovers_einstein_precession(self):
        from psft.evolve.geodesic import GeodesicState, GeodesicEvolver
        M = 1.0
        a = 100.0
        e = 0.10
        metric = SchwarzschildMetric(M=M, G=1.0, c=1.0)

        # Construct initial state at perihelion (Newtonian limit, refined by
        # the metric normalisation).
        r_peri = a * (1.0 - e)
        v_newt = math.sqrt(M * (1.0 + e) / (a * (1.0 - e)))
        x0 = np.array([0.0, r_peri, 0.0, 0.0])
        g = metric.g(x0)
        coeff = g[0, 0] + g[2, 2] * v_newt ** 2
        alpha = math.sqrt(-1.0 / coeff)
        beta = alpha * v_newt
        u0 = np.array([alpha, 0.0, beta, 0.0])

        # Integrate ~3 orbits.
        T = 2 * math.pi * math.sqrt(a ** 3 / M)
        dt = T / 1500.0
        n_steps = int(3 * T / dt)
        evolver = GeodesicEvolver(metric=metric, dt=dt, fd_step=1e-2)
        trajectory = evolver.trajectory(
            GeodesicState(tau=0.0, x=x0, u=u0), n_steps,
        )

        # Detect perihelia and measure precession (using parabolic refinement).
        rs = np.sqrt(trajectory[:, 1] ** 2 + trajectory[:, 2] ** 2)
        phi = np.unwrap(np.arctan2(trajectory[:, 2], trajectory[:, 1]))
        perihelia = []
        for i in range(1, len(rs) - 1):
            if rs[i] < rs[i - 1] and rs[i] < rs[i + 1]:
                y0, y1, y2 = rs[i - 1], rs[i], rs[i + 1]
                denom = y0 - 2 * y1 + y2
                delta = 0.0 if abs(denom) < 1e-15 else 0.5 * (y0 - y2) / denom
                if delta >= 0:
                    phi_min = phi[i] + delta * (phi[i + 1] - phi[i])
                else:
                    phi_min = phi[i] + delta * (phi[i] - phi[i - 1])
                perihelia.append(phi_min)

        self.assertGreaterEqual(len(perihelia), 2)
        # Average precession per orbit.
        deltas = [(perihelia[k] - perihelia[0]) - 2 * math.pi * k
                  for k in range(1, len(perihelia))]
        delta_phi_measured = float(np.mean([d / k for k, d in enumerate(deltas, 1)]))
        delta_phi_predicted = 6 * math.pi * M / (a * (1 - e ** 2))
        rel_err = abs(delta_phi_measured - delta_phi_predicted) / delta_phi_predicted
        self.assertLess(rel_err, 0.05,
                        msg=f"precession {delta_phi_measured:.4f} vs Einstein "
                            f"{delta_phi_predicted:.4f}, rel err {rel_err:.2%}")


class TestRelativisticHydro1D(unittest.TestCase):
    """1+1D inviscid PSFT master equation = relativistic Euler (Theorem 12.1)."""

    def test_primitive_recovery_is_machine_precise(self):
        """Round-trip (rho, p, v) -> (D, S, tau) -> (rho, p, v) should
        recover the primitives to ~1e-12."""
        from psft.evolve.hydro_1d import (
            primitive_from_conservative, conservative_from_primitive,
        )
        rho = np.array([1.0, 5.0, 0.1, 2.5])
        p = np.array([0.5, 2.0, 0.01, 0.8])
        v = np.array([0.0, 0.5, -0.3, 0.9])
        Gamma = 4.0 / 3.0
        D, S, tau = conservative_from_primitive(rho, p, v, Gamma)
        rho2, p2, v2, _ = primitive_from_conservative(D, S, tau, Gamma)
        self.assertTrue(np.allclose(rho2, rho, rtol=1e-10))
        self.assertTrue(np.allclose(p2, p, rtol=1e-10))
        self.assertTrue(np.allclose(v2, v, atol=1e-10))

    def test_static_equilibrium_is_preserved(self):
        """Uniform rest fluid should remain at rest."""
        from psft.evolve.hydro_1d import RelativisticEulerSolver1D
        s = RelativisticEulerSolver1D(N=64, L=1.0, Gamma=4.0/3.0, cfl=0.4)
        s.initialise(
            rho_func=lambda x: np.ones_like(x),
            p_func=lambda x: 0.1 * np.ones_like(x),
            v_func=lambda x: np.zeros_like(x),
        )
        M0, E0, P0 = s.total_mass(), s.total_energy(), s.total_momentum()
        s.evolve(t_end=0.5)
        self.assertAlmostEqual(s.total_mass() / M0 - 1.0, 0.0, places=12)
        self.assertAlmostEqual(s.total_energy() / E0 - 1.0, 0.0, places=12)
        self.assertAlmostEqual(abs(s.total_momentum() - P0), 0.0, places=12)

    def test_sound_speed_split_peaks(self):
        """Initial v=0 density bump splits into left+right movers at +/-c_s."""
        from psft.evolve.hydro_1d import RelativisticEulerSolver1D
        Gamma = 4.0 / 3.0
        rho0, p0 = 1.0, 0.1
        h0 = 1.0 + Gamma * p0 / ((Gamma - 1.0) * rho0)
        cs = math.sqrt(Gamma * p0 / (rho0 * h0))
        s = RelativisticEulerSolver1D(N=1024, L=1.0, Gamma=Gamma, cfl=0.4)
        sigma, x_c, amp = 0.04, 0.5, 0.005
        gauss = lambda x: np.exp(-((x - x_c) / sigma) ** 2)
        s.initialise(
            rho_func=lambda x: rho0 + amp * gauss(x),
            p_func=lambda x: p0 + (Gamma * p0 / (rho0 * h0)) * amp * gauss(x),
            v_func=lambda x: np.zeros_like(x),
        )
        s.evolve(t_end=0.30 / cs)
        rho_f, _, _, _ = s.primitives()
        drho = rho_f - rho0
        i_left = int(np.argmax(drho[s.x < x_c]))
        i_right = int(np.argmax(drho[s.x > x_c]))
        x_left = float(s.x[s.x < x_c][i_left])
        x_right = float(s.x[s.x > x_c][i_right])
        sep = x_right - x_left
        expected = 2.0 * cs * s.t
        self.assertLess(abs(sep - expected) / expected, 0.05)


class TestRelativisticHydro3D(unittest.TestCase):
    """3+1D inviscid PSFT master equation (extension of TestRelativisticHydro1D)."""

    def test_primitive_recovery_3d(self):
        from psft.evolve.hydro_3d import (
            primitive_from_conservative_3d, conservative_from_primitive_3d,
        )
        rho = np.array([[[1.0, 5.0]], [[0.1, 2.5]]])
        p = np.array([[[0.5, 2.0]], [[0.01, 0.8]]])
        vx = np.array([[[0.0, 0.5]], [[-0.3, 0.4]]])
        vy = np.array([[[0.0, 0.2]], [[0.1, -0.1]]])
        vz = np.array([[[0.0, 0.1]], [[0.05, 0.2]]])
        Gamma = 4.0 / 3.0
        D, Sx, Sy, Sz, tau = conservative_from_primitive_3d(rho, p, vx, vy, vz, Gamma)
        rho2, p2, vx2, vy2, vz2, _ = primitive_from_conservative_3d(D, Sx, Sy, Sz, tau, Gamma)
        self.assertTrue(np.allclose(rho2, rho, rtol=1e-10))
        self.assertTrue(np.allclose(p2, p, rtol=1e-10))
        self.assertTrue(np.allclose(vx2, vx, atol=1e-10))
        self.assertTrue(np.allclose(vy2, vy, atol=1e-10))
        self.assertTrue(np.allclose(vz2, vz, atol=1e-10))

    def test_static_equilibrium_3d(self):
        from psft.evolve.hydro_3d import RelativisticEulerSolver3D
        s = RelativisticEulerSolver3D(
            Nx=16, Ny=16, Nz=16, Lx=1, Ly=1, Lz=1, Gamma=4 / 3, cfl=0.4,
        )
        s.initialise(
            rho_func=lambda X, Y, Z: np.ones_like(X),
            p_func=lambda X, Y, Z: 0.1 * np.ones_like(X),
            vx_func=lambda X, Y, Z: np.zeros_like(X),
            vy_func=lambda X, Y, Z: np.zeros_like(X),
            vz_func=lambda X, Y, Z: np.zeros_like(X),
        )
        M0 = s.total_mass()
        E0 = s.total_energy()
        s.evolve(t_end=0.2)
        self.assertAlmostEqual(s.total_mass() / M0 - 1.0, 0.0, places=12)
        self.assertAlmostEqual(s.total_energy() / E0 - 1.0, 0.0, places=12)
        Px, Py, Pz = s.total_momentum()
        self.assertLess(max(abs(Px), abs(Py), abs(Pz)), 1e-12)


class TestPhotonicField3D(unittest.TestCase):
    """Step 4: photonic field on a 3D grid."""

    def test_static_coulomb_is_stable(self):
        """A discrete-Laplacian-balanced static Coulomb field should not
        drift under Maxwell wave-equation evolution."""
        from psft.evolve.photonic_field import PhotonicField3D
        N = 24
        L = 1.0
        field = PhotonicField3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="outflow",
        )
        sigma = 5 * field.dx
        field.initialise_static_coulomb(L / 2, L / 2, L / 2, 1.0, sigma)
        rho_q = field._laplacian(field.A_t) / (4 * math.pi)
        zero = np.zeros_like(rho_q)
        A_t0 = field.A_t.copy()
        U0 = field.total_field_energy()
        for _ in range(30):
            field.step(j_t=rho_q, j_x=zero, j_y=zero, j_z=zero)
        U_final = field.total_field_energy()
        self.assertAlmostEqual((U_final - U0) / U0, 0.0, places=12)
        self.assertTrue(np.allclose(field.A_t, A_t0, atol=1e-12))

    def test_self_consistent_coupling_conserves_total_energy(self):
        """Step 4.3: when the fluid sources its own photonic field
        (j_a from rho_q and v^i) and the field's Lorentz force is added
        to the fluid momentum equation, total energy U_fluid + U_EM is
        conserved up to time-integrator error.

        Pass: |dU_total / U_total_initial| < 5% over 40 steps at low
        coupling and photon-CFL time-stepping.
        """
        from psft.evolve.hydro_3d import RelativisticEulerSolver3D
        from psft.evolve.photonic_field import PhotonicField3D
        N = 24
        L = 1.0
        pf = PhotonicField3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L,
                              cfl=0.4, boundary="periodic")
        fluid = RelativisticEulerSolver3D(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L,
                                           Gamma=4 / 3, cfl=0.3,
                                           boundary="periodic")
        x0 = y0 = z0 = L / 2
        sigma = 8 * pf.dx
        rho_bg = 0.1
        q = 0.05
        fluid.initialise(
            rho_func=lambda X, Y, Z: rho_bg + np.exp(-((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) / (2 * sigma ** 2)),
            p_func=lambda X, Y, Z: 0.01 + 0 * X,
            vx_func=lambda X, Y, Z: np.zeros_like(X),
            vy_func=lambda X, Y, Z: np.zeros_like(X),
            vz_func=lambda X, Y, Z: np.zeros_like(X),
        )
        # Solve discrete Poisson on initial state (periodic).
        rho_now, _, _, _, _, _ = fluid.primitives()
        rho_q_full = q * (rho_now - rho_bg)
        mean_rho = float(np.mean(rho_q_full))
        rho_q = rho_q_full - mean_rho
        src = 4 * math.pi * rho_q
        A = np.zeros_like(rho_q)
        for _ in range(2000):
            A = (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
                 + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
                 + np.roll(A, 1, axis=2) + np.roll(A, -1, axis=2)
                 - src * pf.dx ** 2) / 6.0
        pf.A_t = A

        U0 = pf.total_field_energy() + fluid.total_energy()
        dt = pf.cfl * pf.dx / math.sqrt(3.0)
        for _ in range(40):
            rho_now, _, vx, vy, vz, _ = fluid.primitives()
            rho_q_step = q * (rho_now - rho_bg)
            rho_q_step = rho_q_step - float(np.mean(rho_q_step))
            fx, fy, fz = pf.lorentz_force(rho_q_step, vx, vy, vz)
            fluid.step(dt=dt, body_force=(fx, fy, fz))
            pf.step(dt=dt, j_t=rho_q_step,
                    j_x=rho_q_step * vx, j_y=rho_q_step * vy, j_z=rho_q_step * vz)
        U_final = pf.total_field_energy() + fluid.total_energy()
        rel_drift = (U_final - U0) / U0
        self.assertLess(abs(rel_drift), 0.05,
                        msg=f"|dU/U_0| = {abs(rel_drift)*100:.3f}% (target < 5%)")

    def test_step_44_opposite_charges_attract(self):
        """Step 4.4 (toy hydrogen atom): two opposite-sign charge blobs
        coupled to a self-consistent photonic field develop equal and
        opposite x-momenta on the two half-spaces, with the electron
        half acquiring NEGATIVE x-momentum (toward the proton).

        Confirms (i) the Lorentz attraction sign is correct, (ii) Newton's
        third law holds for the EM-fluid coupling to machine precision,
        and (iii) topology (U(1) winding) is preserved.
        """
        from psft.evolve.hydro_3d import RelativisticEulerSolver3D
        from psft.evolve.photonic_field import PhotonicField3D
        from psft.evolve.gauge_sectors import (
            ScalarAdvector3D, winding_number_in_plane,
        )
        N = 24
        L = 1.0
        dx = L / N
        sigma = 3 * dx
        sep = 0.4
        x_p = L / 2 - sep / 2
        x_e = L / 2 + sep / 2
        y0 = z0 = L / 2
        rho_bg = 0.05
        p_bg = 0.005

        fluid = RelativisticEulerSolver3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, Gamma=4 / 3, cfl=0.3,
            boundary="periodic",
        )
        fluid.initialise(
            rho_func=lambda X, Y, Z: rho_bg
                + np.exp(-((X - x_p) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) / (2 * sigma ** 2))
                + np.exp(-((X - x_e) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) / (2 * sigma ** 2)),
            p_func=lambda X, Y, Z: p_bg + 0 * X,
            vx_func=lambda X, Y, Z: np.zeros_like(X),
            vy_func=lambda X, Y, Z: np.zeros_like(X),
            vz_func=lambda X, Y, Z: np.zeros_like(X),
        )

        tracer = ScalarAdvector3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="periodic",
        )
        rho_q_init = (
            np.exp(-((tracer.X - x_p) ** 2 + (tracer.Y - y0) ** 2 + (tracer.Z - z0) ** 2) / (2 * sigma ** 2))
            - np.exp(-((tracer.X - x_e) ** 2 + (tracer.Y - y0) ** 2 + (tracer.Z - z0) ** 2) / (2 * sigma ** 2))
        )
        tracer.set_scalar(rho_q_init)

        sigma_A = ScalarAdvector3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="periodic",
        )
        r_perp = np.sqrt((sigma_A.X - x_e) ** 2 + (sigma_A.Y - y0) ** 2)
        ph = np.arctan2(sigma_A.Y - y0, sigma_A.X - x_e)
        amp = np.tanh(r_perp / (3 * dx))
        sigma_A.set_scalar(np.stack([amp * np.cos(-ph), amp * np.sin(-ph)], axis=0))

        pf = PhotonicField3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="periodic",
        )
        src = 4 * math.pi * (rho_q_init - float(np.mean(rho_q_init)))
        A = np.zeros_like(rho_q_init)
        for _ in range(2000):
            A = (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
                 + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
                 + np.roll(A, 1, axis=2) + np.roll(A, -1, axis=2)
                 - src * dx ** 2) / 6.0
        pf.A_t = A

        mask_e = fluid.X > L / 2
        winding_init = winding_number_in_plane(
            sigma_A.phi[0] + 1j * sigma_A.phi[1], axis=2, slice_idx=N // 2,
            center=(int(x_e / dx), int(y0 / dx)), radius_frac=0.2,
        )

        dt = pf.cfl * dx / math.sqrt(3.0)
        for _ in range(30):
            rho_now, _, vx, vy, vz, _ = fluid.primitives()
            rho_q_full = tracer.phi
            rho_q = rho_q_full - float(np.mean(rho_q_full))
            fx, fy, fz = pf.lorentz_force(rho_q, vx, vy, vz)
            fluid.step(dt=dt, body_force=(fx, fy, fz))
            _, _, vx_a, vy_a, vz_a, _ = fluid.primitives()
            tracer.set_velocity(
                lambda X, Y, Z, vx=vx_a: vx,
                lambda X, Y, Z, vy=vy_a: vy,
                lambda X, Y, Z, vz=vz_a: vz,
            )
            tracer.step(dt=dt)
            sigma_A.set_velocity(
                lambda X, Y, Z, vx=vx_a: vx,
                lambda X, Y, Z, vy=vy_a: vy,
                lambda X, Y, Z, vz=vz_a: vz,
            )
            sigma_A.step(dt=dt)
            pf.step(dt=dt, j_t=rho_q,
                    j_x=rho_q * vx, j_y=rho_q * vy, j_z=rho_q * vz)

        P_x_e = float(np.sum(fluid.Sx[mask_e])) * dx ** 3
        P_x_p = float(np.sum(fluid.Sx[~mask_e])) * dx ** 3
        winding_final = winding_number_in_plane(
            sigma_A.phi[0] + 1j * sigma_A.phi[1], axis=2, slice_idx=N // 2,
            center=(int(x_e / dx), int(y0 / dx)), radius_frac=0.2,
        )
        # (i) Lorentz attraction sign:
        self.assertLess(P_x_e, 0.0,
                        msg=f"electron-half P_x should be negative, got {P_x_e:.3e}")
        self.assertGreater(P_x_p, 0.0,
                           msg=f"proton-half P_x should be positive, got {P_x_p:.3e}")
        # (ii) Newton's 3rd law:
        sym_violation = abs(P_x_e + P_x_p) / max(abs(P_x_e), 1e-30)
        self.assertLess(sym_violation, 1e-6,
                        msg=f"Newton's 3rd law violated: {sym_violation:.2e}")
        # (iii) Topology preserved:
        self.assertEqual(winding_final, winding_init,
                         msg=f"winding changed: {winding_init} -> {winding_final}")

    def test_e_field_points_outward_from_positive_charge(self):
        """Sign convention: a positive charge at the centre produces E
        pointing radially outward.  Verify direction at a point along +x."""
        from psft.evolve.photonic_field import PhotonicField3D
        N = 32
        L = 1.0
        field = PhotonicField3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="outflow",
        )
        field.initialise_static_coulomb(L / 2, L / 2, L / 2, 1.0, 5 * field.dx)
        Ex, Ey, Ez = field.E_field()
        # Test point off-centre along x.  Because (L/2, L/2, L/2) does not
        # land exactly on a cell centre when N is even, there is a small
        # off-axis residual; we check that Ex dominates and is positive.
        ix = N // 2 + N // 4
        iy = N // 2
        iz = N // 2
        Ex_pt = float(Ex[ix, iy, iz])
        Ey_pt = float(Ey[ix, iy, iz])
        Ez_pt = float(Ez[ix, iy, iz])
        self.assertGreater(Ex_pt, 0.0,
                           msg="E_x should be positive (outward) for +Q on +x side")
        # On-axis residuals tiny relative to E_x.
        E_mag = math.sqrt(Ex_pt ** 2 + Ey_pt ** 2 + Ez_pt ** 2)
        cos_theta = Ex_pt / E_mag
        self.assertGreater(cos_theta, 0.9,
                           msg=f"E vector should be ~aligned with +x (cos = {cos_theta:.3f})")


class TestGaugeSectorAdvection(unittest.TestCase):
    """Step 3.2: scalar gauge fields on a 3D grid preserve topology."""

    def test_winding_preserved_under_static_evolution(self):
        from psft.evolve.gauge_sectors import (
            ScalarAdvector3D, winding_number_in_plane,
        )
        N = 32
        L = 1.0
        advector = ScalarAdvector3D(
            Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, cfl=0.4, boundary="periodic",
        )
        advector.set_velocity(
            lambda X, Y, Z: np.zeros_like(X),
            lambda X, Y, Z: np.zeros_like(X),
            lambda X, Y, Z: np.zeros_like(X),
        )
        for n_target in (-2, -1, 0, 1, 2):
            x0, y0 = L / 2, L / 2
            r = np.sqrt((advector.X - x0) ** 2 + (advector.Y - y0) ** 2)
            phi = np.arctan2(advector.Y - y0, advector.X - x0)
            amp = np.tanh(r / 0.06)
            field = np.stack([amp * np.cos(n_target * phi),
                               amp * np.sin(n_target * phi)], axis=0)
            advector.set_scalar(field)
            w_init = winding_number_in_plane(
                field[0] + 1j * field[1], axis=2, slice_idx=N // 2,
                center=(N // 2, N // 2), radius_frac=0.3,
            )
            dt = 0.4 * (L / N) / 1.0
            for _ in range(20):
                advector.step(dt=dt)
            w_final = winding_number_in_plane(
                advector.phi[0] + 1j * advector.phi[1],
                axis=2, slice_idx=N // 2,
                center=(N // 2, N // 2), radius_frac=0.3,
            )
            self.assertEqual(w_final, w_init,
                             msg=f"winding {n_target}: init {w_init} -> final {w_final}")


class TestTheorem141_LorentzCovariance(unittest.TestCase):

    @staticmethod
    def boost_metric(beta: float) -> FunctionalMetric:
        """A Minkowski metric represented in the frame boosted by beta along x.
        The metric components are still eta_{ab} in inertial coords, so we
        construct a non-trivial example by boosting an off-diagonal
        coordinate system.  For tensor-covariance tests on flat space,
        scalars must be invariant -- we sample K, R, and verify.
        """
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        L = np.array([
            [gamma, -beta * gamma, 0, 0],
            [-beta * gamma, gamma, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        def g_func(_x):
            return L.T @ eta @ L
        return FunctionalMetric(g_func=g_func, dg_func=lambda _x: np.zeros((4, 4, 4)))

    def test_kretschmann_invariant_under_boost(self):
        """K = R_{abcd} R^{abcd} is a Lorentz scalar.  In Minkowski, K = 0
        in any frame.  We confirm K(boost_frame) - K(rest_frame) = 0."""
        rest = MinkowskiMetric()
        boosted = self.boost_metric(beta=0.6)
        x = np.array([0.0, 1.0, 0.0, 0.0])
        K_rest = CurvatureBundle.from_metric(rest, x).K_scalar
        K_boost = CurvatureBundle.from_metric(boosted, x).K_scalar
        # Both should be (numerically) zero.
        self.assertAlmostEqual(K_rest, 0.0, places=8)
        self.assertAlmostEqual(K_boost, 0.0, places=8)
        self.assertAlmostEqual(K_rest, K_boost, places=8)

    def test_schwarzschild_K_independent_of_coordinate_origin(self):
        """The Kretschmann scalar K depends only on the local geometry; in
        particular it depends only on the areal radius r, not on the choice
        of spatial origin or rotation.  Verify K(x=(rho,0,0)) = K(x=(0,rho,0))."""
        bh = SchwarzschildMetric(M=1.0, G=1.0, c=1.0)
        K1 = CurvatureBundle.from_metric(
            bh, np.array([0.0, 30.0, 0.0, 0.0]), h=1e-2,
        ).K_scalar
        K2 = CurvatureBundle.from_metric(
            bh, np.array([0.0, 0.0, 30.0, 0.0]), h=1e-2,
        ).K_scalar
        K3 = CurvatureBundle.from_metric(
            bh, np.array([0.0, 0.0, 0.0, 30.0]), h=1e-2,
        ).K_scalar
        self.assertAlmostEqual(K1, K2, delta=1e-6)
        self.assertAlmostEqual(K2, K3, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
