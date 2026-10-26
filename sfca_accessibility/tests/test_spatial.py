# -*- coding: utf-8 -*-
"""Tests for spatial_models: simulate data with KNOWN spatial structure and check recovery.
Run: python -m unittest discover -s tests -v   (no QGIS needed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regression_core as rc  # noqa: E402
import spatial_models as sm  # noqa: E402


def grid_xy(m=20, jitter=0.15, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array([(i, j) for i in range(m) for j in range(m)], dtype=float)
    return g + rng.uniform(-jitter, jitter, g.shape)


def simulate(kind, rho=0.6, n_side=20, seed=1):
    xy = grid_xy(n_side)
    n = len(xy)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    W = sm.KnnWeights(xy, 8)
    Wd = W.dense()
    I = np.eye(n)
    eps = rng.normal(0, 1.0, n)
    xb = 1.0 + 1.5 * X[:, 0] - 1.0 * X[:, 1]
    if kind == "lag":
        y = np.linalg.solve(I - rho * Wd, xb + eps)
    elif kind == "error":
        y = xb + np.linalg.solve(I - rho * Wd, eps)
    else:
        y = xb + eps
    return y, X, xy, W


class Weights(unittest.TestCase):
    def test_row_standardised_and_trace(self):
        xy = grid_xy(10)
        W = sm.KnnWeights(xy, 6)
        D = W.dense()
        np.testing.assert_allclose(D.sum(axis=1), 1.0)
        self.assertAlmostEqual(W.trace_w2(), float(np.trace(D @ D)), places=9)
        self.assertAlmostEqual(W.trace_t(), float(np.trace(D @ D + D.T @ D)), places=9)
        v = np.arange(len(xy), dtype=float)
        np.testing.assert_allclose(W.wy(v), D @ v)

    def test_logdet_exact_matches_slogdet_and_estimate_close(self):
        xy = grid_xy(15)
        W = sm.KnnWeights(xy, 8)
        D = W.dense()
        ex = sm.LogDet(W, exact_limit=10 ** 6)
        ap = sm.LogDet(W, exact_limit=0, order=40, probes=60)
        for rho in (-0.5, 0.3, 0.7):
            truth = np.linalg.slogdet(np.eye(len(xy)) - rho * D)[1]
            self.assertAlmostEqual(ex.ld(rho), truth, places=6)
            self.assertLess(abs(ap.ld(rho) - truth), 0.04 * max(1.0, abs(truth)) + 1.5)
        h = 1e-4
        num = (ex.ld(0.4 + h) - 2 * ex.ld(0.4) + ex.ld(0.4 - h)) / h ** 2
        self.assertAlmostEqual(ex.d2(0.4), num, delta=1e-2 * abs(num))


class LagError(unittest.TestCase):
    def test_lag_recovery_and_residuals_clean(self):
        y, X, xy, W = simulate("lag", rho=0.6)
        ld = sm.LogDet(W)
        m = sm.fit_lag(y, X, W, ld, ["x1", "x2"], perms=199)
        self.assertAlmostEqual(m["rho"], 0.6, delta=0.12)
        self.assertAlmostEqual(m["coef"][1], 1.5, delta=0.3)
        self.assertAlmostEqual(m["coef"][2], -1.0, delta=0.3)
        self.assertLess(m["p"][-1], 0.001)                       # rho significant
        self.assertTrue(all(np.isfinite(m["se"])))
        ols = rc.ols(y, X, ["x1", "x2"])
        mi_ols = rc.morans_i(ols["resid"], W.idx, 199)[0]
        self.assertGreater(mi_ols, 0.1)                          # OLS leaves spatial residuals
        self.assertLess(abs(m["moran_i"]), abs(mi_ols))          # lag model removes most
        self.assertLess(m["aic"], ols["aic"])                    # and fits better
        # total effect = beta / (1 - rho)
        self.assertAlmostEqual(m["total_effect"][1], m["coef"][1] / (1 - m["rho"]))

    def test_error_recovery(self):
        y, X, xy, W = simulate("error", rho=0.6, seed=3)
        ld = sm.LogDet(W)
        m = sm.fit_error(y, X, W, ld, ["x1", "x2"], perms=199)
        self.assertAlmostEqual(m["lambda"], 0.6, delta=0.15)
        self.assertAlmostEqual(m["coef"][1], 1.5, delta=0.3)
        self.assertLess(m["p"][-1], 0.001)
        ols = rc.ols(y, X, ["x1", "x2"])
        self.assertLess(m["aic"], ols["aic"])
        self.assertLess(abs(m["moran_i"]), 0.1)

    def test_no_spatial_structure_gives_small_rho(self):
        y, X, xy, W = simulate("none", seed=5)
        m = sm.fit_lag(y, X, W, sm.LogDet(W), ["x1", "x2"], perms=99)
        self.assertLess(abs(m["rho"]), 0.2)


class LM(unittest.TestCase):
    def test_lm_picks_the_right_model(self):
        y, X, xy, W = simulate("lag", rho=0.6, seed=2)
        res, rec = sm.lm_tests(y, X, W)
        self.assertLess(res["LM-lag"][1], 0.001)
        self.assertLess(res["Robust LM-lag"][1], res["Robust LM-error"][1])
        self.assertIn("lag", rec)
        y, X, xy, W = simulate("error", rho=0.6, seed=4)
        res, rec = sm.lm_tests(y, X, W)
        self.assertLess(res["LM-error"][1], 0.001)
        self.assertLess(res["Robust LM-error"][1], res["Robust LM-lag"][1])
        self.assertIn("error", rec)

    def test_lm_quiet_without_dependence(self):
        y, X, xy, W = simulate("none", seed=8)
        res, rec = sm.lm_tests(y, X, W)
        self.assertGreater(res["LM-error"][1], 0.01)
        self.assertGreater(res["LM-lag"][1], 0.01)


class GWR(unittest.TestCase):
    def test_recovers_varying_coefficient(self):
        m = 18
        xy = grid_xy(m, seed=11)
        n = len(xy)
        rng = np.random.default_rng(12)
        X = rng.normal(size=(n, 1))
        b1 = 0.2 + 0.25 * xy[:, 0]                 # slope grows west -> east
        y = 1.0 + b1 * X[:, 0] + rng.normal(0, 0.4, n)
        g = sm.gwr(y, X, xy, ["x"])
        self.assertLess(g["aicc"], g["ols_aicc"] - 10)           # GWR clearly better
        corr = np.corrcoef(g["betas"][:, 1], b1)[0, 1]
        self.assertGreater(corr, 0.9)
        self.assertGreater(g["r2"], rc.ols(y, X, ["x"])["r2"])
        self.assertTrue(np.isfinite(g["se"]).all())
        self.assertTrue(20 <= g["bandwidth"] <= n - 1)
        self.assertEqual(g["betas"].shape, (n, 2))

    def test_stationary_process_prefers_large_bandwidth(self):
        xy = grid_xy(18, seed=13)
        n = len(xy)
        rng = np.random.default_rng(14)
        X = rng.normal(size=(n, 1))
        y = 1.0 + 0.8 * X[:, 0] + rng.normal(0, 0.5, n)
        g = sm.gwr(y, X, xy, ["x"])
        self.assertGreater(g["bandwidth"], 0.5 * n)
        self.assertLess(np.ptp(g["betas"][:, 1]), 0.5)           # local slopes stay near 0.8

    def test_fixed_bandwidth_and_cancel(self):
        xy = grid_xy(15, seed=15)
        rng = np.random.default_rng(16)
        X = rng.normal(size=(len(xy), 1))
        y = X[:, 0] + rng.normal(size=len(xy))
        g = sm.gwr(y, X, xy, ["x"], bandwidth=60)
        self.assertEqual(g["bandwidth"], 60)
        with self.assertRaises(sm.Cancelled):
            sm.gwr(y, X, xy, ["x"], progress=lambda f, m: False)


if __name__ == "__main__":
    unittest.main()
