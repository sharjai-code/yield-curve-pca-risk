"""Numerical and data-integrity checks, using the standard library test runner."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from yield_curve_pca.curves import Bond, discount_factors, key_rate_dv01, price_portfolio
from yield_curve_pca.data import load_yields, parse_gsw, yield_changes_bp
from yield_curve_pca.pca import align_components, block_bootstrap, fit_pca, rolling_pca


class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        factors = rng.normal(size=(150, 3))
        basis = np.array([[1, 1, 1, 1], [-1, -.5, .5, 1], [1, -1, -1, 1]])
        self.changes = pd.DataFrame(factors @ basis + rng.normal(0, .05, (150, 4)),
                                    columns=[1, 2, 5, 10], index=pd.bdate_range("2020-01-01", periods=150))
        self.mats = np.array([1., 2., 5., 10.])

    def test_covariance_eigenvalues_and_orthogonality(self):
        model = fit_pca(self.changes)
        expected = np.linalg.eigvalsh(np.cov(self.changes.to_numpy(), rowvar=False))[::-1]
        np.testing.assert_allclose(model.eigenvalues, expected, atol=1e-12)
        np.testing.assert_allclose(model.components @ model.components.T, np.eye(4), atol=1e-12)
        scores = model.transform(self.changes, 4)
        np.testing.assert_allclose(np.cov(scores, rowvar=False), np.diag(expected), atol=1e-12)

    def test_both_modes_roundtrip_to_raw_basis_points(self):
        for mode in ("covariance", "correlation"):
            model = fit_pca(self.changes, mode)
            np.testing.assert_allclose(model.reconstruct(self.changes, 4), self.changes, atol=1e-12)
        model = fit_pca(self.changes, "correlation")
        self.assertAlmostEqual(model.eigenvalues.sum(), 4., places=10)

    def test_holdout_uses_training_parameters(self):
        model = fit_pca(self.changes.iloc[:100], "correlation")
        original_mean, original_scale = model.mean.copy(), model.scale.copy()
        shifted = self.changes.iloc[100:].to_numpy() + 50
        model.reconstruct(shifted)
        np.testing.assert_array_equal(model.mean, original_mean)
        np.testing.assert_array_equal(model.scale, original_scale)
        np.testing.assert_allclose(model.transform(shifted), ((shifted-original_mean)/original_scale) @ model.components[:3].T)

    def test_component_sign_and_order_matching(self):
        ref = fit_pca(self.changes).components[:3]
        candidate = ref[[2, 0, 1]] * np.array([-1, 1, -1])[:, None]
        aligned, _ = align_components(ref, candidate)
        np.testing.assert_allclose(ref, aligned, atol=1e-12)

    def test_bootstrap_reproducible_and_contains_valid_intervals(self):
        a, load_a = block_bootstrap(self.changes, repetitions=30, seed=19)
        b, load_b = block_bootstrap(self.changes, repetitions=30, seed=19)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(load_a, load_b)
        self.assertTrue(0 <= a.iloc[0].lower_95 <= a.iloc[0].upper_95 <= 1)

    def test_rolling_pca_is_causal(self):
        original = rolling_pca(self.changes, window=40)
        modified = self.changes.copy()
        modified.iloc[100:] *= 20
        revised = rolling_pca(modified, window=40)
        pd.testing.assert_frame_equal(original.loc[:self.changes.index[99]], revised.loc[:self.changes.index[99]])

    def test_zero_coupon_price_and_analytic_dv01(self):
        yields = np.full(4, 4.)
        bond = Bond(5, coupon_pct=0, face_value=1000)
        price = price_portfolio([bond], self.mats, yields)
        self.assertAlmostEqual(price, 1000*np.exp(-.04*5), places=10)
        dv01 = key_rate_dv01([bond], self.mats, yields, bump_bp=.01)
        np.testing.assert_allclose(dv01, [0, 0, price*5*1e-4, 0], rtol=1e-8, atol=1e-9)

    def test_regular_coupon_bond_price(self):
        rate = .04
        times = np.arange(.5, 5.5, .5)
        expected = 20*np.exp(-rate*times).sum() + 1000*np.exp(-rate*5)
        self.assertAlmostEqual(price_portfolio([Bond(5, 4, 1000)], self.mats, np.full(4, 4)), expected, places=9)

    def test_log_discount_interpolation_and_origin(self):
        got = discount_factors(self.mats, np.full(4, 4), np.array([0, .5, 3.5, 10]))
        np.testing.assert_allclose(got, np.exp(-.04*np.array([0, .5, 3.5, 10])))
        with self.assertRaises(ValueError):
            discount_factors(self.mats, np.full(4, 4), np.array([11]))

    def test_data_header_missing_observations_and_bp_units(self):
        dates = pd.bdate_range("2020-01-01", periods=12)
        rows = ["Notes before header", "Date,SVENY01,SVENY02"]
        for i, d in enumerate(dates):
            rows.append(f"{d.date()},{'NA' if i == 4 else 1+i*.01},{2+i*.01}")
        with TemporaryDirectory() as folder:
            path = Path(folder)/"data.csv"
            path.write_text("\n".join(rows))
            yields = load_yields(path, maturities=(1, 2))
            self.assertEqual(len(yields), 11)
            self.assertNotIn(dates[4], yields.index)
            changes = yield_changes_bp(yields)
            self.assertAlmostEqual(changes.loc[dates[5], 1], 2.)
        with self.assertRaises(ValueError):
            parse_gsw("Date,SVENY01\n2020-01-01,1\n2020-01-01,2")

    def test_invalid_or_constant_pca_input(self):
        with self.assertRaises(ValueError):
            fit_pca(np.ones((20, 4)))
        with self.assertRaises(ValueError):
            fit_pca(self.changes, "unrecognized")
        with self.assertRaises(ValueError):
            fit_pca(self.changes.to_numpy(), maturities=np.array([1, 2, 2, 5]))


if __name__ == "__main__":
    unittest.main()
