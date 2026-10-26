# -*- coding: utf-8 -*-
"""
Pure-Python engine for the 2SFCA plugin.

Nothing in this module imports QGIS or Qt, so it is independent of the
QGIS 3 -> 4 / PyQt5 -> PyQt6 API changes and can be unit-tested outside QGIS.

Units: coordinates in metres (projected CRS), travel cost in MINUTES.
"""
import heapq
import math
from collections import defaultdict

INF = float("inf")


class Cancelled(Exception):
    """Raised by a progress callback to abort a run."""


# --------------------------------------------------------------------------
# Transport-mode profiles
# --------------------------------------------------------------------------
# class_kmh: speed by road class (OSM "highway" style tags, case-insensitive).
#            A value of None means the class is NOT usable in that mode.
# cap_kmh:   upper bound applied to any speed read from a speed field.
MODE_PROFILES = {
    "driving": {
        "label": "Driving (car)",
        "default_kmh": 50.0,
        "cap_kmh": None,
        "use_oneway": True,
        "class_kmh": {
            "motorway": 100.0, "motorway_link": 60.0,
            "trunk": 80.0, "trunk_link": 50.0,
            "primary": 60.0, "primary_link": 40.0,
            "secondary": 50.0, "secondary_link": 40.0,
            "tertiary": 40.0, "tertiary_link": 30.0,
            "unclassified": 40.0, "residential": 30.0,
            "living_street": 15.0, "service": 20.0,
            "track": 15.0, "path": None, "footway": None,
            "cycleway": None, "pedestrian": None, "steps": None,
        },
    },
    "cycling": {
        "label": "Cycling",
        "default_kmh": 15.0,
        "cap_kmh": 15.0,
        "use_oneway": False,
        "class_kmh": {
            "motorway": None, "motorway_link": None,
            "trunk": None, "trunk_link": None,
            "steps": None, "pedestrian": 8.0, "footway": 8.0,
            "path": 12.0, "track": 12.0,
        },
    },
    "walking": {
        "label": "Walking",
        "default_kmh": 5.0,
        "cap_kmh": 5.0,
        "use_oneway": False,
        "class_kmh": {
            "motorway": None, "motorway_link": None,
            "trunk": None, "trunk_link": None,
        },
    },
}

_FORWARD = {"yes", "true", "1", "y", "t", "fw", "forward", "ft"}
_REVERSE = {"-1", "reverse", "tf", "rev", "backward"}


def parse_speed_overrides(text):
    """'motorway=110, primary=70' -> {'motorway': 110.0, 'primary': 70.0}"""
    out = {}
    for part in (text or "").replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip().lower()] = float(v)
        except ValueError:
            continue
    return out


def edge_speed_kmh(mode, speed_value=None, class_value=None,
                   default_kmh=None, overrides=None):
    """Resolve the speed (km/h) of one road feature, or None if unusable.

    Priority: class overrides/table  >  numeric speed field  >  default.
    Unusable classes (None in the mode table) are always excluded.
    """
    prof = MODE_PROFILES[mode]
    overrides = overrides or {}
    cap = prof["cap_kmh"]
    base_default = default_kmh if default_kmh and default_kmh > 0 else prof["default_kmh"]

    key = str(class_value).strip().lower() if class_value not in (None, "") else None
    if key is not None:
        if key in overrides:
            v = overrides[key]
            return min(v, cap) if cap else v
        if key in prof["class_kmh"]:
            v = prof["class_kmh"][key]
            if v is None:
                return None
            # an explicit numeric speed on the feature still wins if sane
            if _valid_speed(speed_value):
                v = float(speed_value)
            return min(v, cap) if cap else v
    if _valid_speed(speed_value):
        v = float(speed_value)
        return min(v, cap) if cap else v
    return min(base_default, cap) if cap else base_default


def _valid_speed(v):
    try:
        return v is not None and float(v) > 0 and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def parse_oneway(value):
    """Return 1 (forward only), -1 (reverse only) or 0 (both ways)."""
    if value is None:
        return 0
    s = str(value).strip().lower()
    if s in _FORWARD:
        return 1
    if s in _REVERSE:
        return -1
    return 0


def parse_capacity(value, missing="skip"):
    """Interpret a provider-capacity attribute (GP count, FTE, panel size...).

    Accepts numbers or numeric text ('1.5', '1,5', ' 2 '). Returns (capacity, status)
    with status 'ok', 'blank', 'invalid' or 'negative'.
    For blank/invalid values `missing` decides: 'skip' -> capacity None (GP dropped),
    'one' -> 1.0, 'zero' -> 0.0. Negative values are always dropped.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        status, num = "blank", None
    else:
        try:
            num = float(str(value).strip().replace(",", ".")) if isinstance(value, str) \
                else float(value)
            if math.isnan(num) or math.isinf(num):
                status, num = "invalid", None
            elif num < 0:
                return None, "negative"
            else:
                return num, "ok"
        except (TypeError, ValueError):
            status, num = "invalid", None
    fallback = {"one": 1.0, "zero": 0.0}.get(missing)
    return fallback, status


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class Network:
    """Directed, time-weighted graph built from noded polylines.

    Roads must be split at junctions (shared vertices) to be connected; the
    builder does not create intersections where lines merely cross.
    """

    def __init__(self, node_precision=0.1):
        self.prec = node_precision
        self._key2id = {}
        self.xs = []
        self.ys = []
        self.rev = []        # rev[v] = [(u, minutes)] for every directed edge u->v
        self.n_edges = 0
        self.component = None
        self._grid = None
        self._grid_cell = None

    # -- construction ------------------------------------------------------
    def _node(self, x, y):
        key = (round(x / self.prec), round(y / self.prec))
        nid = self._key2id.get(key)
        if nid is None:
            nid = len(self.xs)
            self._key2id[key] = nid
            self.xs.append(x)
            self.ys.append(y)
            self.rev.append([])
        return nid

    def add_polyline(self, pts, speed_kmh, oneway=0):
        """pts: [(x, y), ...] in metres. oneway: 0 both, 1 forward, -1 reverse."""
        if speed_kmh is None or speed_kmh <= 0 or len(pts) < 2:
            return
        m_per_min = speed_kmh * 1000.0 / 60.0
        prev = self._node(*pts[0])
        px, py = pts[0]
        for (x, y) in pts[1:]:
            cur = self._node(x, y)
            seg = math.hypot(x - px, y - py)
            px, py = x, y
            if cur == prev or seg == 0:
                prev = cur
                continue
            t = seg / m_per_min
            # edge u->v is stored in rev[v] so Dijkstra from a GP walks backwards,
            # giving the time for patients to travel TO the GP.
            if oneway >= 0:
                self.rev[cur].append((prev, t))
                self.n_edges += 1
            if oneway <= 0:
                self.rev[prev].append((cur, t))
                self.n_edges += 1
            prev = cur

    @property
    def n_nodes(self):
        return len(self.xs)

    # -- connectivity ------------------------------------------------------
    def label_components(self):
        """Weakly connected components (ignores edge direction)."""
        n = self.n_nodes
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for v in range(n):
            for (u, _t) in self.rev[v]:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
        comp = [find(i) for i in range(n)]
        counts = defaultdict(int)
        for c in comp:
            counts[c] += 1
        self.component = comp
        if counts:
            self.largest_component = max(counts, key=counts.get)
            self.largest_share = counts[self.largest_component] / float(n)
        else:
            self.largest_component, self.largest_share = None, 0.0
        self.n_components = len(counts)
        return self.n_components, self.largest_share

    # -- snapping ----------------------------------------------------------
    def build_snap_index(self, cell, only_largest=True):
        self._grid_cell = float(cell)
        grid = defaultdict(list)
        for i in range(self.n_nodes):
            if only_largest and self.component is not None \
                    and self.component[i] != self.largest_component:
                continue
            grid[(int(self.xs[i] // cell), int(self.ys[i] // cell))].append(i)
        self._grid = grid

    def snap(self, x, y, max_dist):
        """Nearest indexed node within max_dist -> (node_id, distance) or (None, None)."""
        cell = self._grid_cell
        cx, cy = int(x // cell), int(y // cell)
        reach = int(math.ceil(max_dist / cell))
        best, best_d = None, INF
        for gx in range(cx - reach, cx + reach + 1):
            for gy in range(cy - reach, cy + reach + 1):
                for i in self._grid.get((gx, gy), ()):
                    d = math.hypot(self.xs[i] - x, self.ys[i] - y)
                    if d < best_d:
                        best, best_d = i, d
        if best is not None and best_d <= max_dist:
            return best, best_d
        return None, None

    # -- shortest paths ----------------------------------------------------
    def reach_within(self, source, start_cost, cutoff):
        """Dijkstra over reversed edges; returns {node: minutes_to_source}."""
        rev = self.rev
        dist = {source: start_cost}
        heap = [(start_cost, source)]
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist.get(u, INF):
                continue
            for (v, w) in rev[u]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, INF):
                    dist[v] = nd
                    push(heap, (nd, v))
        return dist


# --------------------------------------------------------------------------
# Distance decay
# --------------------------------------------------------------------------
def decay_weight(t, threshold, kind="binary"):
    if t > threshold:
        return 0.0
    if kind == "gaussian":
        # Kwan (1998) / Dai (2010) Gaussian decay, 1 at t=0 and 0 at t=threshold
        e = math.exp(-0.5)
        return (math.exp(-0.5 * (t / threshold) ** 2) - e) / (1.0 - e)
    return 1.0  # classic binary catchment


# --------------------------------------------------------------------------
# Two-step floating catchment area
# --------------------------------------------------------------------------
def two_step_fca(net, supply, demand, threshold_min, access_kmh=10.0,
                 snap_max=2000.0, decay="binary", progress=None):
    """Run 2SFCA.

    supply: list of dicts {x, y, cap}
    demand: list of dicts {x, y, pop}
    Both are snapped to the network; the straight-line snap distance is
    converted to minutes with access_kmh (off-network access leg).

    Returns (demand_results, supply_results, summary)
    """
    def tick(frac, msg):
        if progress and progress(frac, msg) is False:
            raise Cancelled()

    m_per_min = access_kmh * 1000.0 / 60.0
    net.label_components()
    net.build_snap_index(max(snap_max, 1.0), only_largest=True)

    tick(0.0, "Snapping demand points…")
    d_node, d_pen = [], []
    by_node = defaultdict(list)
    for i, p in enumerate(demand):
        n, dist = net.snap(p["x"], p["y"], snap_max)
        d_node.append(n)
        d_pen.append(None if n is None else dist / m_per_min)
        if n is not None:
            by_node[n].append(i)
    tick(0.05, "Snapping supply points…")
    s_node, s_pen = [], []
    for p in supply:
        n, dist = net.snap(p["x"], p["y"], snap_max)
        s_node.append(n)
        s_pen.append(None if n is None else dist / m_per_min)

    # ---- catchments: one Dijkstra per supply point ----
    catch = [None] * len(supply)       # list of (demand_idx, minutes)
    n_s = max(len(supply), 1)
    for j, p in enumerate(supply):
        if j % 20 == 0:
            tick(0.1 + 0.7 * j / n_s, "Catchment %d of %d…" % (j + 1, len(supply)))
        if s_node[j] is None or s_pen[j] > threshold_min:
            catch[j] = []
            continue
        dist = net.reach_within(s_node[j], s_pen[j], threshold_min)
        pairs = []
        for node, d in dist.items():
            for i in by_node.get(node, ()):
                t = d + d_pen[i]
                if t <= threshold_min:
                    pairs.append((i, t))
        catch[j] = pairs

    # ---- step 1: provider-to-population ratio per supply point ----
    tick(0.85, "Step 1: provider-to-population ratios…")
    R = [0.0] * len(supply)
    dem_in = [0.0] * len(supply)
    for j, pairs in enumerate(catch):
        denom = 0.0
        for (i, t) in pairs:
            denom += demand[i]["pop"] * decay_weight(t, threshold_min, decay)
        dem_in[j] = denom
        R[j] = supply[j]["cap"] / denom if denom > 0 else 0.0

    # ---- step 2: sum of ratios per demand point ----
    tick(0.92, "Step 2: summing ratios per demand point…")
    A = [0.0] * len(demand)
    n_gp = [0] * len(demand)
    min_t = [None] * len(demand)
    for j, pairs in enumerate(catch):
        for (i, t) in pairs:
            A[i] += R[j] * decay_weight(t, threshold_min, decay)
            n_gp[i] += 1
            if min_t[i] is None or t < min_t[i]:
                min_t[i] = t

    # ---- benchmark & summary ----
    ok = [i for i in range(len(demand)) if d_node[i] is not None]
    tot_pop = sum(demand[i]["pop"] for i in ok)
    tot_cap = sum(supply[j]["cap"] for j in range(len(supply)) if s_node[j] is not None)
    bench = tot_cap / tot_pop if tot_pop > 0 else 0.0

    drs = []
    for i in range(len(demand)):
        snapped = d_node[i] is not None
        a = A[i] if snapped else None
        drs.append({
            "snapped": snapped,
            "access": a,                                   # GP per person
            "per10k": a * 10000.0 if snapped else None,    # GP per 10,000 people
            "n_gp": n_gp[i] if snapped else None,
            "min_time": min_t[i],
            "index": (a / bench) if (snapped and bench > 0) else None,
        })
    srs = []
    for j in range(len(supply)):
        srs.append({"snapped": s_node[j] is not None, "R": R[j],
                    "demand": dem_in[j], "n_demand": len(catch[j] or [])})

    pops = [demand[i]["pop"] for i in ok]
    vals = [A[i] for i in ok]
    zero_pop = sum(demand[i]["pop"] for i in ok if A[i] <= 0)
    summary = {
        "n_demand": len(demand), "n_demand_snapped": len(ok),
        "n_supply": len(supply),
        "n_supply_snapped": sum(1 for n in s_node if n is not None),
        "total_pop": tot_pop, "total_capacity": tot_cap,
        "benchmark_per10k": bench * 10000.0,
        "pop_weighted_mean_per10k": (sum(p * v for p, v in zip(pops, vals)) / tot_pop * 10000.0)
        if tot_pop > 0 else 0.0,
        "median_per10k": _median([v * 10000.0 for v in vals]),
        "n_zero": sum(1 for v in vals if v <= 0),
        "pop_zero": zero_pop,
        "pct_pop_zero": (100.0 * zero_pop / tot_pop) if tot_pop > 0 else 0.0,
        "n_nodes": net.n_nodes, "n_edges": net.n_edges,
        "n_components": net.n_components,
        "largest_component_share": net.largest_share,
    }
    tick(1.0, "Done")
    return drs, srs, summary


def _median(v):
    if not v:
        return 0.0
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# --------------------------------------------------------------------------
# Classification for the choropleth
# --------------------------------------------------------------------------
def class_breaks(values, n_classes=5, method="quantile"):
    """Break values for POSITIVE accessibility; zero is handled as its own class.

    Returns an ascending list [b0, b1, ..., bn] (n_classes intervals).
    """
    v = sorted(x for x in values if x is not None and x > 0)
    if not v:
        return [0.0, 1.0]
    lo, hi = v[0], v[-1]
    if lo == hi:
        return [lo, hi]
    k = max(1, min(n_classes, len(set(v))))
    if method == "equal":
        step = (hi - lo) / k
        br = [lo + i * step for i in range(k)] + [hi]
    else:  # quantile
        br = [lo]
        for i in range(1, k):
            pos = i * (len(v) - 1) / float(k)
            a = int(math.floor(pos))
            b = min(a + 1, len(v) - 1)
            br.append(v[a] + (v[b] - v[a]) * (pos - a))
        br.append(hi)
    out = [br[0]]
    for b in br[1:]:
        if b > out[-1]:
            out.append(b)
    return out
