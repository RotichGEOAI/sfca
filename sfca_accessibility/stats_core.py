# -*- coding: utf-8 -*-
"""
Tabular statistics for 2SFCA results. Pure Python (no QGIS / Qt): unit-testable.

All "per 10k" values are supply units per 10,000 people (the choropleth measure).
Population weighting is used throughout, so a statistic describes PEOPLE, not areas.
Every table is (title, headers, rows) with rows of plain strings/numbers.
"""
import math


# --------------------------------------------------------------------------
# Weighted helpers
# --------------------------------------------------------------------------
def wmean(vals, w):
    tw = sum(w)
    return sum(v * x for v, x in zip(vals, w)) / tw if tw > 0 else float("nan")


def wquantile(vals, w, q):
    """Population-weighted quantile (lower interpolation)."""
    pairs = sorted(zip(vals, w))
    tw = sum(x for _v, x in pairs)
    if tw <= 0:
        return float("nan")
    target, cum = q * tw, 0.0
    for v, x in pairs:
        cum += x
        if cum >= target:
            return v
    return pairs[-1][0]


def wstd(vals, w):
    tw = sum(w)
    if tw <= 0:
        return float("nan")
    m = wmean(vals, w)
    return math.sqrt(sum(x * (v - m) ** 2 for v, x in zip(vals, w)) / tw)


def gini(vals, w):
    """Population-weighted Gini coefficient of access (0 = equal, 1 = all supply to one person)."""
    pairs = sorted(zip(vals, w))
    tw = sum(x for _v, x in pairs)
    tv = sum(v * x for v, x in pairs)
    if tw <= 0 or tv <= 0:
        return float("nan")
    p_prev = l_prev = area = 0.0
    for v, x in pairs:
        p = p_prev + x / tw
        l = l_prev + v * x / tv
        area += (p - p_prev) * (l + l_prev)
        p_prev, l_prev = p, l
    return 1.0 - area


def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


def _band_table(title, head, bands, items, total_pop, with_cum=False):
    """bands: list of (label, predicate). items: list of (value, pop). Each item goes in the
    first matching band. Returns a table (label, EDs, population, % of pop[, cumulative %])."""
    counts = [[0, 0.0] for _ in bands]
    for val, pop in items:
        for k, (_lab, pred) in enumerate(bands):
            if pred(val):
                counts[k][0] += 1
                counts[k][1] += pop
                break
    rows, cum = [], 0.0
    for (lab, _p), (n, pop) in zip(bands, counts):
        share = pct(pop, total_pop)
        cum += share
        row = [lab, n, round(pop), round(share, 1)]
        if with_cum:
            row.append(round(cum, 1))
        rows.append(row)
    headers = [head, "EDs", "Population", "% of population"] + (["Cumulative %"] if with_cum else [])
    return (title, headers, rows)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def compute_tables(demand, drs, srs, supply, summary, threshold, class_labels=None,
                   class_order=None, groups=None, group_name="Group", unit="GPs"):
    """
    demand/drs  : per-ED dicts from the engine (pop; per10k, index, n_gp, min_time, snapped)
    srs/supply  : per-GP dicts (R, demand, n_demand, snapped; cap)
    summary     : engine summary dict (benchmark_per10k etc.)
    class_labels: per-ED choropleth class label (same order), optional
    groups      : per-ED group value (e.g. county / urban-rural), optional
    """
    tables = []
    ok = [i for i, r in enumerate(drs) if r["snapped"]]
    pop = [demand[i]["pop"] for i in ok]
    val = [drs[i]["per10k"] for i in ok]
    idx = [drs[i]["index"] if drs[i]["index"] is not None else 0.0 for i in ok]
    tot = sum(pop)
    bench = summary["benchmark_per10k"]
    zero_pop = sum(p for p, v in zip(pop, val) if v <= 0)
    below_half = sum(p for p, x in zip(pop, idx) if x < 0.5)
    above = sum(p for p, x in zip(pop, idx) if x >= 1.0)

    # ---- 1. headline summary ----
    r = []
    r.append(["EDs analysed / total", "%d / %d" % (len(ok), len(drs))])
    r.append(["Population analysed", "%d" % round(tot)])
    r.append(["Total supply (%s)" % unit, "%.2f" % summary["total_capacity"]])
    r.append(["Catchment threshold (min)", "%g" % threshold])
    r.append(["Benchmark: supply per 10,000 people", "%.3f" % bench])
    r.append(["Population-weighted mean access per 10,000", "%.3f" % wmean(val, pop)])
    r.append(["Population-weighted median", "%.3f" % wquantile(val, pop, 0.5)])
    r.append(["Population-weighted std. deviation", "%.3f" % wstd(val, pop)])
    r.append(["10th percentile (people)", "%.3f" % wquantile(val, pop, 0.10)])
    r.append(["25th percentile", "%.3f" % wquantile(val, pop, 0.25)])
    r.append(["75th percentile", "%.3f" % wquantile(val, pop, 0.75)])
    r.append(["90th percentile", "%.3f" % wquantile(val, pop, 0.90)])
    p10, p90 = wquantile(val, pop, 0.10), wquantile(val, pop, 0.90)
    r.append(["90th / 10th percentile ratio", "%.2f" % (p90 / p10) if p10 > 0 else "n/a (P10 = 0)"])
    r.append(["Minimum / maximum ED value", "%.3f / %.3f" % (min(val), max(val)) if val else "n/a"])
    r.append(["Gini coefficient of access (0 equal - 1 unequal)", "%.3f" % gini(val, pop)])
    r.append(["EDs with no GP in catchment", "%d (%.1f%% of population)"
              % (sum(1 for v in val if v <= 0), pct(zero_pop, tot))])
    r.append(["Population with access < 50% of benchmark", "%d (%.1f%%)"
              % (round(below_half), pct(below_half, tot))])
    r.append(["Population with access >= benchmark", "%d (%.1f%%)"
              % (round(above), pct(above, tot))])
    mt = [(drs[i]["min_time"], demand[i]["pop"]) for i in ok if drs[i]["min_time"] is not None]
    if mt:
        r.append(["Pop-weighted mean minutes to nearest GP (within threshold)",
                  "%.1f" % wmean([m for m, _p in mt], [p for _m, p in mt])])
    n_unsnapped = len(drs) - len(ok)
    if n_unsnapped:
        r.append(["EDs excluded (no road within snap distance)", "%d" % n_unsnapped])
    tables.append(("Summary", ["Measure", "Value"], r))

    # ---- 2. choropleth classes ----
    if class_labels is not None:
        order = class_order or sorted(set(class_labels))
        acc = {c: [0, 0.0, 0.0] for c in order}
        for i, c in enumerate(class_labels):
            if c not in acc:
                acc[c] = [0, 0.0, 0.0]
            acc[c][0] += 1
            acc[c][1] += demand[i]["pop"]
            if drs[i]["snapped"]:
                acc[c][2] += demand[i]["pop"] * drs[i]["per10k"]
        rows = []
        for c in list(order) + [c for c in acc if c not in order]:
            n, p, wv = acc[c]
            rows.append([c, n, round(p), round(pct(p, tot + sum(
                demand[i]["pop"] for i in range(len(drs)) if not drs[i]["snapped"])), 1),
                round(wv / p, 3) if p > 0 and wv > 0 else ""])
        tables.append(("Map classes", ["Map class", "EDs", "Population", "% of population",
                                       "Mean access per 10,000"], rows))

    # ---- 3. relative to benchmark ----
    items = list(zip(idx, pop))
    tables.append(_band_table(
        "Access vs benchmark", "Access relative to benchmark",
        [("None (0)", lambda x: x <= 0),
         ("Very low (< 0.5)", lambda x: x < 0.5),
         ("Low (0.5 - 0.8)", lambda x: x < 0.8),
         ("About average (0.8 - 1.2)", lambda x: x <= 1.2),
         ("High (1.2 - 2)", lambda x: x <= 2.0),
         ("Very high (> 2)", lambda x: True)], items, tot, with_cum=True))

    # ---- 4. number of GPs reachable ----
    tables.append(_band_table(
        "GPs reachable", "GPs within catchment",
        [("0", lambda n: n == 0), ("1", lambda n: n == 1), ("2 - 3", lambda n: n <= 3),
         ("4 - 5", lambda n: n <= 5), ("6 - 10", lambda n: n <= 10), ("11 or more", lambda n: True)],
        [(drs[i]["n_gp"], demand[i]["pop"]) for i in ok], tot))

    # ---- 5. travel time to nearest GP ----
    t = float(threshold)
    a, b = t / 3.0, 2.0 * t / 3.0
    tables.append(_band_table(
        "Time to nearest GP", "Minutes to nearest GP",
        [("Up to %.0f" % a, lambda m: m is not None and m <= a),
         ("%.0f - %.0f" % (a, b), lambda m: m is not None and m <= b),
         ("%.0f - %.0f" % (b, t), lambda m: m is not None),
         ("More than %.0f (none in catchment)" % t, lambda m: True)],
        [(drs[i]["min_time"], demand[i]["pop"]) for i in ok], tot, with_cum=True))

    # ---- 6. by group ----
    if groups is not None:
        g = {}
        for i in ok:
            key = "(blank)" if groups[i] in (None, "") else str(groups[i])
            g.setdefault(key, []).append(i)
        rows = []
        for key in sorted(g):
            ii = g[key]
            gp_ = [demand[i]["pop"] for i in ii]
            gv = [drs[i]["per10k"] for i in ii]
            gx = [drs[i]["index"] or 0.0 for i in ii]
            gtot = sum(gp_)
            rows.append([key, len(ii), round(gtot), round(wmean(gv, gp_), 3),
                         round(wquantile(gv, gp_, 0.5), 3),
                         round(pct(sum(p for p, v in zip(gp_, gv) if v <= 0), gtot), 1),
                         round(pct(sum(p for p, x in zip(gp_, gx) if x < 0.5), gtot), 1),
                         round(gini(gv, gp_), 3) if gtot > 0 else ""])
        rows.sort(key=lambda r_: r_[3])
        tables.append(("By %s" % group_name, [group_name, "EDs", "Population",
                                              "Mean access per 10,000", "Median",
                                              "% pop no GP", "% pop < 50% of benchmark", "Gini"],
                       rows))

    # ---- 7. provider side ----
    sj = [(srs[j], supply[j]) for j in range(len(supply)) if srs[j]["snapped"]]
    if sj:
        R = [s["R"] * 10000.0 for s, _p in sj]
        dem = [s["demand"] for s, _p in sj]
        cap = [p["cap"] for _s, p in sj]
        rows = [
            ["GPs used / supplied", "%d / %d" % (len(sj), len(supply))],
            ["Total capacity", "%.2f" % sum(cap)],
            ["Mean capacity per GP point", "%.2f" % (sum(cap) / len(cap))],
            ["Provider-to-population ratio per 10,000: mean", "%.3f" % (sum(R) / len(R))],
            ["  median", "%.3f" % wquantile(R, [1.0] * len(R), 0.5)],
            ["  minimum / maximum", "%.3f / %.3f" % (min(R), max(R))],
            ["Population in catchment (demand): mean", "%d" % round(sum(dem) / len(dem))],
            ["  maximum (busiest catchment)", "%d" % round(max(dem))],
            ["GPs with no population in catchment", "%d" % sum(1 for d in dem if d <= 0)],
            ["GPs with ratio below half the benchmark (heavily loaded)",
             "%d" % (sum(1 for s, _p in sj if bench > 0 and s["R"] * 10000.0 < bench / 2.0 and s["demand"] > 0))],
        ]
        tables.append(("Providers", ["Provider measure", "Value"], rows))
    return tables


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def tables_to_text(tables, title="2SFCA accessibility statistics"):
    out = [title, "=" * len(title), ""]
    for name, head, rows in tables:
        out.append(name)
        out.append("-" * len(name))
        cols = list(zip(*([head] + [[str(c) for c in r] for r in rows])))
        wd = [max(len(str(c)) for c in col) for col in cols]
        out.append("  ".join(str(h).ljust(w) for h, w in zip(head, wd)))
        for r in rows:
            out.append("  ".join(str(c).ljust(w) for c, w in zip(r, wd)))
        out.append("")
    return "\n".join(out)


def safe_name(s):
    import re
    return re.sub(r"_+", "_", "".join(ch if ch.isalnum() else "_" for ch in s)).strip("_").lower()


def write_csvs(tables, prefix):
    """Write each table to <prefix>_<name>.csv; return the list of paths."""
    import csv
    paths = []
    for name, head, rows in tables:
        path = "%s_%s.csv" % (prefix, safe_name(name))
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)
        paths.append(path)
    return paths


def write_text(tables, path, title="2SFCA accessibility statistics"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tables_to_text(tables, title) + "\n")
    return path
