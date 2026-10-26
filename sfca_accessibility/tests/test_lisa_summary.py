# -*- coding: utf-8 -*-
"""Tests: Local Moran's I (LISA) and the interpretable regression summary tables."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regression_core as rc  # noqa: E402
import spatial_models as sm  # noqa: E402
from test_spatial import grid_xy  # noqa: E402


class LISA(unittest.TestCase):
    def setUp(self):
        self.xy = grid_xy(20, seed=3)
        self.W = sm.KnnWeights(self.xy, 8)
        self.rng = np.random.default_rng(5)

    def test_mean_of_local_equals_global(self):
        x = self.xy[:, 0] + self.rng.normal(0, 3, len(self.xy))
        r = sm.local_moran(x, self.W, perms=99)
        self.assertAlmostEqual(r["global_I"], r["global_perm_I"], places=9)

    def test_finds_hot_and_cold_spots(self):
        x = self.rng.normal(0, 0.5, len(self.xy))
        hot = (self.xy[:, 0] < 5) & (self.xy[:, 1] < 5)
        cold = (self.xy[:, 0] > 14) & (self.xy[:, 1] > 14)
        x[hot] += 4.0
        x[cold] -= 4.0
        r = sm.local_moran(x, self.W, perms=499)
        self.assertGreater((r["code"][hot] == "HH").mean(), 0.7)
        self.assertGreater((r["code"][cold] == "LL").mean(), 0.7)
        mid = (self.xy[:, 0] > 8) & (self.xy[:, 0] < 11) & (self.xy[:, 1] > 8) & (self.xy[:, 1] < 11)
        self.assertLess((r["code"][mid] != "NS").mean(), 0.2)
        self.assertEqual(sum(r["counts"].values()), len(x))

    def test_outlier_is_flagged(self):
        x = self.rng.normal(0, 0.3, len(self.xy))
        i = 210
        x[self.W.idx[i]] -= 1.0                        # its neighbours are low ...
        x[i] = 6.0                                     # ... and it is very high
        r = sm.local_moran(x, self.W, perms=499, fdr=False)
        self.assertEqual(r["quad"][i], "HL")
        self.assertEqual(r["code"][i], "HL")           # and the contrast is significant
        self.assertTrue(r["p"][i] < 0.05)

    def test_random_data_has_few_false_positives_with_fdr(self):
        x = self.rng.normal(size=len(self.xy))
        r = sm.local_moran(x, self.W, perms=199, fdr=True)
        self.assertLessEqual(r["counts"]["NS"], len(x))
        self.assertGreater(r["counts"]["NS"], 0.97 * len(x))
        r2 = sm.local_moran(x, self.W, perms=199, fdr=False)   # naive: ~5% by chance
        self.assertGreater(len(x) - r2["counts"]["NS"], len(x) - r["counts"]["NS"])

    def test_constant_and_cancel(self):
        with self.assertRaises(ValueError):
            sm.local_moran(np.ones(len(self.xy)), self.W, perms=9)
        with self.assertRaises(sm.Cancelled):
            sm.local_moran(self.rng.normal(size=len(self.xy)), self.W, perms=99,
                           progress=lambda f, m: False)


class Summary(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        n = 400
        x1 = rng.uniform(1, 10, n)
        x2 = x1 * 0.8 + rng.normal(0, 0.5, n)             # collinear with x1
        y = 1 + 0.5 * x1 + rng.normal(0, 0.3 * x1)         # heteroscedastic
        X = np.column_stack([x1, x2])
        self.c = rc.ols(y, X, ["Density", "Rurality = Remote"])
        self.r = rc.ols(y, X, ["Density", "Rurality = Remote"], robust=True)
        self.head, self.rows, self.fh, self.fr = rc.summary_tables(
            self.c, self.r, "access", moran=(0.2, 3.0, 0.002))

    def test_shape_and_columns(self):
        self.assertEqual(len(self.rows), 3)
        for r in self.rows:
            self.assertEqual(len(r), len(self.head))
        self.assertEqual(self.head[2], "HC3 robust SE")

    def test_hc3_and_classical_both_present_and_differ(self):
        i = self.head.index("HC3 robust SE")
        j = self.head.index("Classical SE")
        self.assertNotEqual(self.rows[1][i], self.rows[1][j])
        self.assertAlmostEqual(self.rows[1][i], self.r["se"][1], delta=abs(self.r["se"][1]) * 1e-3)
        self.assertAlmostEqual(self.rows[1][j], self.c["se"][1], delta=abs(self.c["se"][1]) * 1e-3)

    def test_vif_strength_and_text(self):
        k = self.head.index("Collinearity")
        self.assertIn(self.rows[1][k].split()[0], ("OK", "Moderate", "High"))
        self.assertEqual(rc.vif_flag(3), "OK (< 5)")
        self.assertEqual(rc.vif_flag(7), "Moderate (5 to 10)")
        self.assertEqual(rc.vif_flag(12), "High (> 10)")
        self.assertEqual([rc.strength_label(v) for v in (0.05, 0.2, 0.4, 0.7)],
                         ["Negligible", "Weak", "Moderate", "Strong"])
        interp = self.rows[1][-1]
        self.assertTrue(interp.startswith("Significant") or interp.startswith("Not significant"))
        self.assertIn("than the reference group", self.rows[2][-1])    # dummy wording

    def test_fit_table_flags_heteroscedasticity_and_clustering(self):
        d = dict((r[0], r) for r in self.fr)
        self.assertIn("Rely on the HC3", d["Heteroscedasticity (Breusch-Pagan, studentised)"][2])
        self.assertIn("spatially clustered", d["Moran's I of residuals"][2])
        self.assertIn("R-squared", d)
        self.assertIn("%", d["R-squared"][2])

    def test_log_outcome_wording(self):
        _h, rows, _fh, fr = rc.summary_tables(self.c, self.r, "access", log_outcome=True)
        self.assertIn("%", rows[1][-1])
        self.assertTrue(any(r[0] == "Outcome transformation" for r in fr))


if __name__ == "__main__":
    unittest.main()
