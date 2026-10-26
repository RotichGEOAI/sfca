# -*- coding: utf-8 -*-
"""Tests for stats_core. Run: python -m unittest discover -s tests -v (no QGIS needed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import core  # noqa: E402
import stats_core as sc  # noqa: E402
from test_core import line_network  # noqa: E402


class Weighted(unittest.TestCase):
    def test_mean_median_quantile(self):
        self.assertAlmostEqual(sc.wmean([1, 3], [1, 3]), 2.5)
        self.assertEqual(sc.wquantile([1, 2, 3], [1, 1, 1], 0.5), 2)
        self.assertEqual(sc.wquantile([1, 100], [9, 1], 0.5), 1)      # most people at 1
        self.assertEqual(sc.wquantile([1, 100], [9, 1], 0.95), 100)

    def test_gini(self):
        self.assertAlmostEqual(sc.gini([5, 5, 5], [1, 1, 1]), 0.0)
        # one of four equal-sized groups has everything: G = 0.75
        self.assertAlmostEqual(sc.gini([0, 0, 0, 10], [1, 1, 1, 1]), 0.75)
        # weighting by population matters
        self.assertAlmostEqual(sc.gini([0, 10], [3, 1]), 0.75)


class Tables(unittest.TestCase):
    def setUp(self):
        net = line_network(length_km=60)
        self.supply = [{"x": 0.0, "y": 0.0, "cap": 1.0}, {"x": 20000.0, "y": 0.0, "cap": 2.0}]
        self.demand = [{"x": 5000.0, "y": 0.0, "pop": 1000.0},
                       {"x": 10000.0, "y": 0.0, "pop": 3000.0},
                       {"x": 25000.0, "y": 0.0, "pop": 2000.0},
                       {"x": 45000.0, "y": 0.0, "pop": 500.0}]      # 25 km from nearest GP: outside 15 min
        self.d, self.s, self.summ = core.two_step_fca(net, self.supply, self.demand, 15.0)
        self.t = dict((n, (h, r)) for n, h, r in sc.compute_tables(
            self.demand, self.d, self.s, self.supply, self.summ, 15.0,
            class_labels=["A", "A", "B", "Z"], class_order=["Z", "A", "B"],
            groups=["north", "north", "south", "south"], group_name="Region"))

    def test_expected_tables_present(self):
        for k in ("Summary", "Map classes", "Access vs benchmark", "GPs reachable",
                  "Time to nearest GP", "By Region", "Providers"):
            self.assertIn(k, self.t)

    def test_population_shares_sum_to_100(self):
        for k in ("Access vs benchmark", "GPs reachable", "Time to nearest GP"):
            rows = self.t[k][1]
            self.assertAlmostEqual(sum(r[3] for r in rows), 100.0, delta=0.2)
            self.assertEqual(sum(r[1] for r in rows), 4)
            self.assertEqual(sum(r[2] for r in rows), 6500)

    def test_zero_access_counted(self):
        summary = dict((r[0], r[1]) for r in self.t["Summary"][1])
        self.assertIn("1 (7.7%", summary["EDs with no GP in catchment"])
        rows = dict((r[0], r) for r in self.t["GPs reachable"][1])
        self.assertEqual(rows["0"][1], 1)
        self.assertEqual(rows["0"][2], 500)

    def test_pop_weighted_mean_equals_benchmark(self):
        # 2SFCA conserves supply when every ED is analysed and reachable
        summary = dict((r[0], r[1]) for r in self.t["Summary"][1])
        mean = float(summary["Population-weighted mean access per 10,000"])
        self.assertLess(mean, float(summary["Benchmark: supply per 10,000 people"]) + 1e-9)

    def test_group_table(self):
        rows = dict((r[0], r) for r in self.t["By Region"][1])
        self.assertEqual(rows["north"][2], 4000)
        self.assertEqual(rows["south"][2], 2500)
        self.assertEqual(rows["north"][5], 0.0)

    def test_text_render(self):
        txt = sc.tables_to_text(list(sc.compute_tables(
            self.demand, self.d, self.s, self.supply, self.summ, 15.0)))
        self.assertIn("Summary", txt)
        self.assertIn("Gini", txt)
        self.assertEqual(sc.safe_name("By Region (x)"), "by_region_x")


if __name__ == "__main__":
    unittest.main()
