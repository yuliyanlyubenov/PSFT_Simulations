"""Unit tests for the psft simulation library.

Run from the simulation/ directory:
    python -m unittest tests.test_basics -v
"""
import math
import os
import sys
import unittest
import numpy as np

# Make sibling psft package importable when running this file directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from psft.core.constants import SI, NATURAL
from psft.core.metric import (
    MinkowskiMetric, SchwarzschildMetric, FLRWMetric, FunctionalMetric,
)
from psft.core.curvature import CurvatureBundle
from psft.core.kinematics import KinematicDecomposition
from psft.core.photonic import ElectromagneticField, PhotonicSource
from psft.gauge.algebra import U1, SU2, SU3, structure_constants_su, generators_su2, generators_su3
from psft.sectors.viscosity import (
    HeavisideViscosity, ScalarViscosity, GaugeViscosity, HallViscosity,
    profile_QCD_log, profile_constant,
)
from psft.sectors.master_eq import MasterEquationV2
from psft.solitons.topology import winding_number_1d, winding_number_2d_loop, topological_charge_skyrme
from psft.solitons.ansatz import VortexAnsatz, SkyrmeHedgehogAnsatz
from psft.core.manifold import CartesianGrid
from psft.evolve.integrators import RK4


class TestConstants(unittest.TestCase):
    def test_planck_viscosity_kg_per_s(self):
        eta_p = SI.planck_viscosity()
        # c^3/(16 pi G) ~ 8e33 kg/s  (paper Appendix A).
        self.assertTrue(1e33 < eta_p < 1e35, f"eta_P = {eta_p:g}")

    def test_Kc_strong_matches_paper(self):
        # paper eq. 6.10: Kc ~ 1.2e61 m^-4 for lc = 1 fm.
        self.assertAlmostEqual(SI.Kc_strong, 12.0 / SI.l_strong**4, places=5)
        self.assertTrue(1e60 < SI.Kc_strong < 1e62)

    def test_natural_mode_sets_unit_constants(self):
        self.assertEqual(NATURAL.c, 1.0)
        self.assertEqual(NATURAL.hbar, 1.0)
        self.assertEqual(NATURAL.G, 1.0)


class TestMetricsAndCurvature(unittest.TestCase):
    def test_minkowski_is_flat(self):
        m = MinkowskiMetric()
        x = np.array([0.0, 1.0, 2.0, 3.0])
        bundle = CurvatureBundle.from_metric(m, x)
        self.assertAlmostEqual(bundle.R_scalar, 0.0, places=8)
        self.assertAlmostEqual(bundle.K_scalar, 0.0, places=8)
        self.assertTrue(np.allclose(bundle.Ricci, 0.0, atol=1e-7))

    def test_schwarzschild_kretschmann_scales_correctly(self):
        # K = 48 (GM)^2 / r^6  in areal radius; check at large rho where
        # rho ~ r so the answer is well-approximated.
        M = 1.0
        m = SchwarzschildMetric(M=M, G=1.0, c=1.0)
        rho = 50.0
        x = np.array([0.0, rho, 0.0, 0.0])
        bundle = CurvatureBundle.from_metric(m, x, h=1e-2)
        r_areal = m.areal_radius(x)
        expected = 48.0 * M**2 / r_areal**6
        rel_err = abs(bundle.K_scalar - expected) / expected
        self.assertLess(rel_err, 0.05, f"K={bundle.K_scalar:g}, expected={expected:g}")


class TestKinematics(unittest.TestCase):
    def test_static_observer_in_minkowski_has_zero_shear(self):
        m = MinkowskiMetric()
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u_func = lambda _x: np.array([1.0, 0.0, 0.0, 0.0])
        bundle = CurvatureBundle.from_metric(m, x)
        kin = KinematicDecomposition.from_velocity(m, x, u_func, Gamma=bundle.Gamma)
        self.assertTrue(kin.is_unit_timelike())
        self.assertAlmostEqual(kin.theta, 0.0, places=6)
        self.assertTrue(np.allclose(kin.sigma, 0.0, atol=1e-7))
        self.assertTrue(np.allclose(kin.omega, 0.0, atol=1e-7))

    def test_shear_with_simple_expansion(self):
        m = MinkowskiMetric()
        x = np.array([0.0, 0.0, 0.0, 0.0])

        def u_func(xx):
            # Uniform x-expansion: u^a = (1, alpha x, 0, 0) -- non-geodesic.
            alpha = 0.1
            return np.array([1.0, alpha * xx[1], 0.0, 0.0])
        bundle = CurvatureBundle.from_metric(m, x)
        kin = KinematicDecomposition.from_velocity(m, x, u_func, Gamma=bundle.Gamma)
        # At x=0 the velocity is just (1,0,0,0) so unit-timelike holds.
        self.assertTrue(kin.is_unit_timelike(tol=1e-6))
        # Expansion: theta = nabla_a u^a = d_1 u^1 = alpha = 0.1.
        self.assertAlmostEqual(kin.theta, 0.1, places=4)


class TestElectromagnetic(unittest.TestCase):
    def test_coulomb_potential_in_flat_space(self):
        # A_t = -Q/(4 pi eps0 r), rest zero.  F_{tr} = d_t A_r - d_r A_t = -d_r A_t.
        m = MinkowskiMetric()
        Q = 1.0; eps0 = 1.0
        def A(xx):
            r = max(math.sqrt(xx[1]**2 + xx[2]**2 + xx[3]**2), 1e-6)
            return np.array([-Q / (4 * np.pi * eps0 * r), 0.0, 0.0, 0.0])
        x = np.array([0.0, 1.0, 0.0, 0.0])
        em = ElectromagneticField.from_potential(m, x, A)
        # E_r = -d_r A_t -> F_{tr} = -d_r A_t = Q/(4 pi eps0 r^2)
        E_r = em.F_dn[0, 1]
        self.assertAlmostEqual(E_r, -Q / (4 * np.pi * eps0 * 1.0**2), places=2)


class TestGauge(unittest.TestCase):
    def test_su2_structure_constants_are_levi_civita(self):
        f = SU2.structure_constants
        # f^{ABC} = epsilon^{ABC} for SU(2).
        self.assertAlmostEqual(f[0, 1, 2], 1.0, places=8)
        self.assertAlmostEqual(f[1, 2, 0], 1.0, places=8)
        self.assertAlmostEqual(f[2, 0, 1], 1.0, places=8)
        self.assertAlmostEqual(f[0, 2, 1], -1.0, places=8)

    def test_su3_structure_constants_match_standard_values(self):
        # f^{123} = 1, f^{147} = 1/2, f^{458} = sqrt(3)/2  (standard table).
        f = SU3.structure_constants
        self.assertAlmostEqual(f[0, 1, 2], 1.0, places=8)
        self.assertAlmostEqual(f[0, 3, 6], 0.5, places=8)
        self.assertAlmostEqual(f[3, 4, 7], math.sqrt(3) / 2, places=8)

    def test_generators_are_hermitian(self):
        for g in generators_su3():
            self.assertTrue(np.allclose(g, g.conj().T, atol=1e-12))


class TestViscosity(unittest.TestCase):
    def test_heaviside_below_threshold_is_zero(self):
        eta = HeavisideViscosity(eta_0=1.0, K_c=10.0)
        self.assertEqual(eta.eta(5.0), 0.0)
        self.assertEqual(eta.eta(10.0), 0.0)
        self.assertEqual(eta.eta(20.0), 1.0)

    def test_qcd_log_profile_asymptotic_freedom(self):
        # f(K/Kc) -> 0 as K -> infinity (paper Theorem 10.1(iii)).
        f_low = profile_QCD_log(2.0)
        f_high = profile_QCD_log(1e6)
        self.assertGreater(f_low, f_high)
        self.assertGreater(f_high, 0.0)

    def test_gauge_viscosity_sectors(self):
        gv = GaugeViscosity.physical(SI)
        K = SI.Kc_strong * 10.0
        eta_strong = gv.eta("strong", K)
        eta_weak = gv.eta("weak", K)
        eta_em = gv.eta("em", K)
        self.assertGreater(eta_strong, 0.0)
        # weak threshold is much higher (Kc_weak >> Kc_strong since l_weak << l_strong)
        # so eta_weak should also be > 0 at K_strong*10? Check thresholds.
        self.assertEqual(eta_em, 0.0)


class TestSolitons(unittest.TestCase):
    def test_vortex_winding_number_equals_n(self):
        grid = CartesianGrid(shape=(64, 64, 4), bounds=((-5, 5), (-5, 5), (-1, 1)))
        for n in (1, 2, -3):
            ans = VortexAnsatz(winding=n, core_radius=0.5).evaluate(grid)
            w = winding_number_2d_loop(ans["phi_complex"][:, :, 2],
                                        center=(32, 32), radius=20)
            self.assertEqual(w, n, f"winding mismatch for n={n}: got {w}")

    def test_skyrme_hedgehog_baryon_number_close_to_one(self):
        grid = CartesianGrid(shape=(60, 60, 60), bounds=((-6, 6), (-6, 6), (-6, 6)))
        ans = SkyrmeHedgehogAnsatz(core_radius=1.0).evaluate(grid)
        dx = (12.0 / 59.0)
        B = topological_charge_skyrme(ans["N4"], dx)
        # B = +/-1 by topology; the sign depends on orientation convention.
        self.assertGreater(abs(B), 0.7)
        self.assertLess(abs(B), 1.3)


class TestEvolution(unittest.TestCase):
    def test_rk4_recovers_harmonic_oscillator(self):
        # y'' + y = 0 -> y(t) = cos(t), starting from (1,0).
        def rhs(_t, y):
            return np.array([y[1], -y[0]])
        ts, ys = RK4(rhs=rhs, dt=1e-4).run(0.0, np.array([1.0, 0.0]), 2 * np.pi)
        self.assertAlmostEqual(ys[-1, 0], 1.0, places=4)
        self.assertAlmostEqual(ys[-1, 1], 0.0, places=4)


class TestHallStress(unittest.TestCase):
    """Hall stress (paper Sec. 6, eq. 6.7) is non-vanishing and antisymmetric;
    the algebraic form eps^{bcde} u_c sigma_{de} that appeared in earlier
    paper deposits is identically zero for any symmetric shear tensor."""

    def test_hall_stress_is_nonzero_and_antisymmetric(self):
        from psft.core.metric import MinkowskiMetric
        from psft.core.curvature import CurvatureBundle
        from psft.sectors.master_eq import _levi_civita_at
        m = MinkowskiMetric()
        x0 = np.array([0.0, 0.1, 0.2, 0.3])
        curv = CurvatureBundle.from_metric(m, x0)
        eps_lower = _levi_civita_at(curv.g)
        u_up = np.array([1.0, 0.0, 0.0, 0.0])
        # Gauge-algebra scalar with non-trivial gradient.
        d_sigma_up = np.array([0.0, 0.6, 1.5, -0.2])
        tau = 0.5 * np.einsum("abmn,m,n->ab", eps_lower, u_up, d_sigma_up)
        self.assertGreater(np.linalg.norm(tau), 1e-3)
        self.assertTrue(np.allclose(tau + tau.T, 0.0, atol=1e-12))

    def test_legacy_algebraic_form_vanishes_for_symmetric_shear(self):
        from psft.core.metric import MinkowskiMetric
        from psft.sectors.master_eq import _levi_civita_at
        eps = _levi_civita_at(MinkowskiMetric().g(np.array([0.0, 0.0, 0.0, 0.0])))
        u = np.array([1.0, 0.0, 0.0, 0.0])
        # Any symmetric trace-free spatial shear.
        sigma = np.array([[0, 0, 0, 0],
                          [0, 1.0, 0.3, 0.1],
                          [0, 0.3, -0.5, 0.2],
                          [0, 0.1, 0.2, -0.5]], dtype=float)
        force = np.einsum("bcde,c,de->b", eps, u, sigma)
        self.assertTrue(np.allclose(force, 0.0, atol=1e-12),
                        msg="Legacy Hall contraction must vanish for symmetric shear.")


class TestMasterEquationReductions(unittest.TestCase):
    """Theorem 12.1: in the inviscid limit (K < Kc), the master equation
    must reduce to the relativistic Euler equation."""

    def test_inviscid_in_minkowski_is_pure_pressure(self):
        m = MinkowskiMetric()
        consts = SI
        # Vanishing viscosity in every sector.
        gv = GaugeViscosity.physical(consts)
        gv.sectors["gravity"].eta_0 = 0.0
        eq = MasterEquationV2(metric=m, consts=consts, gauge_viscosity=gv,
                              sector="gravity", Lambda=0.0)
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u_func = lambda _x: np.array([1.0, 0.0, 0.0, 0.0])
        # Pressure with a uniform gradient in x.
        grad_p = np.array([0.0, 1.0, 0.0, 0.0])
        result = eq.evaluate(x, u_func, rho_g=1.0, p_g=0.0, grad_p_g=grad_p)
        # Pressure force = -h^a_b grad_b p; for static u^a=(1,0,0,0), h is the
        # spatial projector, so the time component is zero and spatial ones equal
        # -grad p.
        self.assertAlmostEqual(result.pressure_force[0], 0.0, places=8)
        self.assertAlmostEqual(result.pressure_force[1], -1.0, places=6)
        # Viscous, hall, photonic forces all zero in this limit.
        self.assertTrue(np.allclose(result.viscous_force, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(result.hall_force, 0.0, atol=1e-10))
        self.assertTrue(np.allclose(result.photonic_force, 0.0, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
