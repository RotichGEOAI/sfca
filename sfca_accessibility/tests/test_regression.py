# -*- coding: utf-8 -*-
"""Validates regression_core against scipy/numpy reference values (scipy is a
TEST-only dependency). Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regression_core as rc  # noqa: E402

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


class Dist(unittest.TestCase):
    @unittest.skipIf(stats is None, "scipy not installed")
    def test_p_values_match_scipy(self):
        for t, df in ((2.1, 10), (0.3, 5), (-4.5, 200), (1.96, 3000)):
            self.assertAlmostEqual(rc.t_two_sided_p(t, df), 2 * stats.t.sf(abs(t), df), places=9)
        for f, d1, d2 in ((3.2, 3, 40), (12.0, 1, 500), (0.8, 4, 20)):
            self.assertAlmostEqual(rc.f_sf(f, d1, d2), stats.f.sf(f, d1, d2), places=9)
        for x, k in ((3.8, 1), (7.5, 3), (25.0, 4), (0.4, 2)):
            self.assertAlmostEqual(rc.chi2_sf(x, k), stats.chi2.sf(x, k), places=8)
        self.assertAlmostEqual(rc.t_crit(30), stats.t.ppf(0.975, 30), places=6)


class OLS(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 300
        self.X = rng.normal(size=(n, 3))
        self.X[:, 2] += 0.5 * self.X[:, 0]
        self.y = 2.0 + 1.5 * self.X[:, 0] - 0.7 * self.X[:, 1] + rng.normal(0, 1.2, n)

    def test_matches_numpy_and_scipy(self):
        r = rc.ols(self.y, self.X, ["a", "b", "c"])
        A = np.column_stack([np.ones(len(self.y)), self.X])
        ref = np.linalg.lstsq(A, self.y, rcond=None)[0]
        np.testing.assert_allclose(r["beta"], ref, rtol=1e-9)
        if stats is not None:
            lr = stats.linregress(self.X[:, 0], self.y)   # sanity on a 1-var fit
            one = rc.ols(self.y, self.X[:, :1], ["a"])
            self.assertAlmostEqual(one["beta"][1], lr.slope, places=9)
            self.assertAlmostEqual(one["p"][1], lr.pvalue, places=9)
            self.assertAlmostEqual(one["r2"], lr.rvalue ** 2, places=9)

    def test_recovers_truth_and_fit_stats(self):
        r = rc.ols(self.y, self.X, ["a", "b", "c"])
        self.assertAlmostEqual(r["beta"][1], 1.5, delta=0.25)
        self.assertAlmostEqual(r["beta"][2], -0.7, delta=0.25)
        self.assertLess(r["p"][1], 0.001)
        self.assertGreater(r["p"][3], 0.01)           # c has no true effect
        self.assertTrue(0 < r["adj_r2"] < r["r2"] < 1)
        self.assertEqual(r["df"], 300 - 4)

    def test_vif_detects_collinearity(self):
        x = np.column_stack([self.X[:, 0], self.X[:, 0] + np.random.default_rng(2).normal(0, .05, 300)])
        r = rc.ols(self.y, x, ["u", "v"])
        self.assertGreater(r["vif"][1], 100)

    def test_perfect_collinearity_rejected(self):
        x = np.column_stack([self.X[:, 0], 2 * self.X[:, 0]])
        with self.assertRaises(ValueError):
            rc.ols(self.y, x, ["u", "v"])

    def test_robust_se_differs_under_heteroskedasticity(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(1, 10, 400)
        y = 1 + 0.5 * x + rng.normal(0, 0.3 * x)
        a, b = rc.ols(y, x[:, None], ["x"]), rc.ols(y, x[:, None], ["x"], robust=True)
        self.assertNotAlmostEqual(a["se"][1], b["se"][1], places=3)
        self.assertLess(a["bp_p"], 0.05)               # BP flags the problem

    def test_dummy_coding(self):
        cols, names, ref = rc.dummy_code(["u", "u", "r", "r", "r", "x"], "Rurality")
        self.assertEqual(ref, "r")
        self.assertEqual(names, ["Rurality = u", "Rurality = x"])
        self.assertEqual(cols[0].tolist(), [1, 1, 0, 0, 0, 0])


class Moran(unittest.TestCase):
    def test_detects_clustering_and_randomness(self):
        g = np.array([(i, j) for i in range(20) for j in range(20)], dtype=float)
        idx = rc.knn_indices(g, 8)
        self.assertEqual(idx.shape, (400, 8))
        clustered = g[:, 0] + g[:, 1]                   # smooth surface
        i1, _z, p1 = rc.morans_i(clustered, idx, 199)
        self.assertGreater(i1, 0.8)
        self.assertLess(p1, 0.01)
        rnd = np.random.default_rng(5).normal(size=400)
        i2, _z, p2 = rc.morans_i(rnd, idx, 199)
        self.assertLess(abs(i2), 0.15)
        self.assertGreater(p2, 0.05)


if __name__ == "__main__":
    unittest.main()
