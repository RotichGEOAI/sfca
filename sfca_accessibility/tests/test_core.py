# -*- coding: utf-8 -*-
"""Unit tests for core.py - run with:  python -m unittest discover -s tests -v
(no QGIS needed)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import core  # noqa: E402


def line_network(length_km=30, step_km=1, kmh=60.0, oneway=0):
    net = core.Network()
    pts = [(i * step_km * 1000.0, 0.0) for i in range(int(length_km / step_km) + 1)]
    net.add_polyline(pts, kmh, oneway)
    return net


class TwoStepFCA(unittest.TestCase):
    def setUp(self):
        self.supply = [{"x": 0.0, "y": 0.0, "cap": 1.0},
                       {"x": 20000.0, "y": 0.0, "cap": 2.0}]
        self.demand = [{"x": 5000.0, "y": 0.0, "pop": 1000.0},
                       {"x": 10000.0, "y": 0.0, "pop": 3000.0},
                       {"x": 25000.0, "y": 0.0, "pop": 2000.0}]

    def test_hand_calculated_example(self):
        # 1 minute per km, 15 minute threshold = 15 km, boundary inclusive.
        d, s, summ = core.two_step_fca(line_network(), self.supply, self.demand, 15.0)
        self.assertAlmostEqual(s[0]["R"], 1 / 4000.0)          # D1 + D2
        self.assertAlmostEqual(s[1]["R"], 2 / 6000.0)          # D1 + D2 + D3
        self.assertAlmostEqual(d[0]["per10k"], 10000 * (1 / 4000 + 2 / 6000))
        self.assertAlmostEqual(d[1]["per10k"], 10000 * (1 / 4000 + 2 / 6000))
        self.assertAlmostEqual(d[2]["per10k"], 10000 * (2 / 6000))
        self.assertEqual([x["n_gp"] for x in d], [2, 2, 1])
        self.assertAlmostEqual(d[0]["min_time"], 5.0)
        self.assertAlmostEqual(d[2]["min_time"], 5.0)
        # benchmark = total capacity / total population
        self.assertAlmostEqual(summ["benchmark_per10k"], 10000 * 3 / 6000.0)

    def test_outside_threshold_gets_zero(self):
        d, _s, summ = core.two_step_fca(line_network(), self.supply, self.demand, 4.0)
        self.assertEqual([x["n_gp"] for x in d], [0, 0, 0])
        self.assertEqual(summ["n_zero"], 3)
        self.assertAlmostEqual(summ["pct_pop_zero"], 100.0)

    def test_one_way_direction_matters(self):
        # Road one-way towards +x. Patients west of a GP can reach it; east cannot.
        net = line_network(oneway=1)
        sup = [{"x": 10000.0, "y": 0.0, "cap": 1.0}]
        dem = [{"x": 5000.0, "y": 0.0, "pop": 100.0},    # west: can drive to GP
               {"x": 15000.0, "y": 0.0, "pop": 100.0}]   # east: cannot (against one-way)
        d, _s, _ = core.two_step_fca(net, sup, dem, 15.0)
        self.assertEqual(d[0]["n_gp"], 1)
        self.assertEqual(d[1]["n_gp"], 0)

    def test_unsnapped_points_are_excluded_not_zeroed(self):
        dem = self.demand + [{"x": 5000.0, "y": 50000.0, "pop": 999.0}]  # 50 km off-network
        d, _s, summ = core.two_step_fca(line_network(), self.supply, dem, 15.0,
                                        snap_max=2000.0)
        self.assertFalse(d[3]["snapped"])
        self.assertIsNone(d[3]["per10k"])
        self.assertEqual(summ["n_demand_snapped"], 3)
        self.assertAlmostEqual(summ["total_pop"], 6000.0)  # unsnapped pop not in R

    def test_snap_penalty_counts_towards_threshold(self):
        # demand 1.5 km off the road at 10 km/h access speed = 9 min penalty
        net = line_network()
        sup = [{"x": 0.0, "y": 0.0, "cap": 1.0}]
        dem = [{"x": 7000.0, "y": 1500.0, "pop": 100.0}]   # 7 min on road + 9 min = 16
        d, _s, _ = core.two_step_fca(net, sup, dem, 15.0, access_kmh=10.0, snap_max=2000)
        self.assertEqual(d[0]["n_gp"], 0)
        d, _s, _ = core.two_step_fca(net, sup, dem, 17.0, access_kmh=10.0, snap_max=2000)
        self.assertEqual(d[0]["n_gp"], 1)

    def test_disconnected_fragment_is_not_snapped_to(self):
        net = line_network()
        net.add_polyline([(5000.0, 400.0), (5100.0, 400.0)], 60.0)   # tiny island
        sup = [{"x": 0.0, "y": 0.0, "cap": 1.0}]
        dem = [{"x": 5050.0, "y": 400.0, "pop": 100.0}]
        d, _s, summ = core.two_step_fca(net, sup, dem, 15.0, snap_max=2000.0)
        # island ignored; point snaps to the main road 400 m away instead
        self.assertTrue(d[0]["snapped"])
        self.assertEqual(summ["n_components"], 2)

    def test_gaussian_decay_reduces_access(self):
        b, _, _ = core.two_step_fca(line_network(), self.supply, self.demand, 15.0)
        g, _, _ = core.two_step_fca(line_network(), self.supply, self.demand, 15.0,
                                    decay="gaussian")
        self.assertEqual(core.decay_weight(15.0, 15.0, "gaussian"), 0.0 + 0.0)
        self.assertAlmostEqual(core.decay_weight(0.0, 15.0, "gaussian"), 1.0)
        self.assertTrue(all(gi["per10k"] > 0 for gi in g[:3]))
        self.assertNotAlmostEqual(g[0]["per10k"], b[0]["per10k"])

    def test_cancel(self):
        with self.assertRaises(core.Cancelled):
            core.two_step_fca(line_network(), self.supply, self.demand, 15.0,
                              progress=lambda f, m: False)


class Speeds(unittest.TestCase):
    def test_driving_class_table_and_field(self):
        self.assertEqual(core.edge_speed_kmh("driving", None, "motorway"), 100.0)
        self.assertEqual(core.edge_speed_kmh("driving", 70, "residential"), 70.0)
        self.assertEqual(core.edge_speed_kmh("driving", 70, None), 70.0)
        self.assertEqual(core.edge_speed_kmh("driving", None, None, default_kmh=45), 45.0)
        self.assertIsNone(core.edge_speed_kmh("driving", None, "footway"))

    def test_overrides(self):
        ov = core.parse_speed_overrides("motorway=120, primary = 70")
        self.assertEqual(core.edge_speed_kmh("driving", None, "Motorway", overrides=ov), 120.0)

    def test_walking_and_cycling_caps_and_exclusions(self):
        self.assertEqual(core.edge_speed_kmh("walking", 80, "primary"), 5.0)
        self.assertEqual(core.edge_speed_kmh("cycling", 80, None), 15.0)
        self.assertIsNone(core.edge_speed_kmh("walking", None, "motorway"))
        self.assertIsNone(core.edge_speed_kmh("cycling", None, "trunk"))

    def test_oneway_parsing(self):
        self.assertEqual(core.parse_oneway("yes"), 1)
        self.assertEqual(core.parse_oneway("-1"), -1)
        self.assertEqual(core.parse_oneway("no"), 0)
        self.assertEqual(core.parse_oneway(None), 0)


class Breaks(unittest.TestCase):
    def test_quantile_and_equal(self):
        vals = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        q = core.class_breaks(vals, 5, "quantile")
        e = core.class_breaks(vals, 5, "equal")
        self.assertEqual(q[0], 1)
        self.assertEqual(q[-1], 10)
        self.assertEqual(len(q), 6)
        self.assertAlmostEqual(e[1] - e[0], 1.8)

    def test_degenerate(self):
        self.assertEqual(core.class_breaks([0, 0], 5), [0.0, 1.0])
        self.assertEqual(core.class_breaks([3, 3, 3], 5), [3, 3])


if __name__ == "__main__":
    unittest.main()


class Capacity(unittest.TestCase):
    def test_parse_capacity(self):
        self.assertEqual(core.parse_capacity(2), (2.0, "ok"))
        self.assertEqual(core.parse_capacity("1,5"), (1.5, "ok"))
        self.assertEqual(core.parse_capacity(" 0.8 "), (0.8, "ok"))
        self.assertEqual(core.parse_capacity(0), (0.0, "ok"))
        self.assertEqual(core.parse_capacity(None), (None, "blank"))
        self.assertEqual(core.parse_capacity("  "), (None, "blank"))
        self.assertEqual(core.parse_capacity("n/a"), (None, "invalid"))
        self.assertEqual(core.parse_capacity(float("nan")), (None, "invalid"))
        self.assertEqual(core.parse_capacity(-1), (None, "negative"))
        self.assertEqual(core.parse_capacity(-1, "one"), (None, "negative"))
        self.assertEqual(core.parse_capacity(None, "one"), (1.0, "blank"))
        self.assertEqual(core.parse_capacity("x", "zero"), (0.0, "invalid"))

    def test_capacity_changes_scores_proportionally(self):
        net = line_network()
        dem = [{"x": 5000.0, "y": 0.0, "pop": 1000.0}]
        one = core.two_step_fca(net, [{"x": 0.0, "y": 0.0, "cap": 1.0}], dem, 15.0)[0][0]
        fte = core.two_step_fca(net, [{"x": 0.0, "y": 0.0, "cap": 2.5}], dem, 15.0)[0][0]
        self.assertAlmostEqual(fte["per10k"], 2.5 * one["per10k"])
        zero = core.two_step_fca(net, [{"x": 0.0, "y": 0.0, "cap": 0.0}], dem, 15.0)[0][0]
        self.assertEqual(zero["per10k"], 0.0)

    def test_capacity_shifts_supply_between_gps(self):
        # Two GPs share one demand point; the benchmark equals total capacity / population.
        net = line_network()
        sup = [{"x": 0.0, "y": 0.0, "cap": 3.0}, {"x": 10000.0, "y": 0.0, "cap": 1.0}]
        dem = [{"x": 5000.0, "y": 0.0, "pop": 2000.0}]
        d, s, summ = core.two_step_fca(net, sup, dem, 15.0)
        self.assertAlmostEqual(s[0]["R"], 3 / 2000.0)
        self.assertAlmostEqual(s[1]["R"], 1 / 2000.0)
        self.assertAlmostEqual(d[0]["per10k"], 10000 * 4 / 2000.0)
        self.assertAlmostEqual(summ["benchmark_per10k"], 10000 * 4 / 2000.0)
