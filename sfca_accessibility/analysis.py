# -*- coding: utf-8 -*-
"""
QGIS-side glue for the 2SFCA plugin (QGIS 4 / Qt6 / PyQt6).

Reads layers (project or file), converts to a projected working CRS, runs the
pure-Python engine in core.py, and builds the styled outputs.

Qt6 notes honoured here: scoped enums only (Qgis.GeometryType.Line,
QMetaType.Type.Double, ...), no QVariant field types, no exec_().
"""
import csv
import math
import os
from datetime import date
from urllib.parse import quote

from qgis.core import (
    Qgis, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature,
    QgsFeatureRequest, QgsField, QgsFillSymbol, QgsGeometry, QgsLayoutExporter,
    QgsLayoutItemLabel, QgsLayoutItemLegend, QgsLayoutItemMap, QgsLayoutItemPage,
    QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize, QgsMarkerSymbol,
    QgsPointXY, QgsPrintLayout, QgsProject, QgsRectangle, QgsRuleBasedRenderer,
    QgsSingleSymbolRenderer, QgsVectorFileWriter, QgsVectorLayer,
)
from qgis.PyQt.QtCore import QCoreApplication, QMetaType, QUrl
from qgis.PyQt.QtGui import QColor, QFont

from . import core, stats_core

# max plausible free-flow speed per mode, used only to size the road-loading window
_VMAX = {"driving": 130.0, "cycling": 15.0, "walking": 5.0}
_RAMP = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]   # YlGnBu
C_ZERO = "#d7301f"
C_NODATA = "#bdbdbd"


class AnalysisError(Exception):
    pass


def _pump():
    QCoreApplication.processEvents()


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Loading from files
# --------------------------------------------------------------------------
def csv_headers(path):
    """Return (delimiter, [header names]) for a CSV/TXT file."""
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
        headers = next(csv.reader(fh, delimiter=delim), [])
    return delim, [h.strip() for h in headers]


def guess_xy(headers):
    low = {h.lower(): h for h in headers}
    for xs, ys in (("longitude", "latitude"), ("lon", "lat"), ("long", "lat"),
                   ("lng", "lat"), ("x", "y"), ("easting", "northing"),
                   ("itm_e", "itm_n"), ("east", "north")):
        if xs in low and ys in low:
            return low[xs], low[ys]
    return None, None


def load_points_csv(path, xfield, yfield, crs_authid, delimiter=","):
    uri = ("%s?delimiter=%s&xField=%s&yField=%s&crs=%s&detectTypes=yes"
           "&spatialIndex=no&subsetIndex=no&watchFile=no") % (
        QUrl.fromLocalFile(path).toString(), quote(delimiter, safe=""),
        quote(xfield, safe=""), quote(yfield, safe=""), crs_authid)
    lyr = QgsVectorLayer(uri, os.path.splitext(os.path.basename(path))[0], "delimitedtext")
    if not lyr.isValid():
        raise AnalysisError("Could not read the CSV as points. Check the X/Y fields, "
                            "delimiter and CRS.")
    return lyr


def load_vector_file(path):
    lyr = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0], "ogr")
    if not lyr.isValid():
        raise AnalysisError("Could not open %s" % path)
    return lyr


# --------------------------------------------------------------------------
# Extraction (main thread, with event pumping so the UI stays responsive)
# --------------------------------------------------------------------------
def _transform(layer, work_crs, what):
    src = layer.crs()
    if not src.isValid():
        raise AnalysisError("%s layer has no CRS. Assign one (Layer > Set CRS) and retry." % what)
    return QgsCoordinateTransform(src, work_crs, QgsProject.instance())


def extract_supply(layer, work_crs, cap_field, name_field, progress, cap_missing="skip"):
    """Read GP points. With cap_field, each GP's supply = that attribute (e.g. FTE);
    otherwise every GP counts 1. Returns (supply, n_skipped, capacity_stats)."""
    ct = _transform(layer, work_crs, "GP")
    out, bad = [], 0
    stats = {"blank": 0, "invalid": 0, "negative": 0, "filled": 0, "zero": 0}
    n = max(layer.featureCount(), 1)
    for k, f in enumerate(layer.getFeatures()):
        if k % 500 == 0:
            _pump()
            if progress(0.02 * k / n, "Reading GP points…") is False:
                raise core.Cancelled()
        g = f.geometry()
        if g is None or g.isNull() or g.isEmpty():
            bad += 1
            continue
        pt = g.asMultiPoint()[0] if g.isMultipart() else g.asPoint()
        p = ct.transform(QgsPointXY(pt))
        cap = 1.0
        if cap_field:
            cap, status = core.parse_capacity(f[cap_field], cap_missing)
            if status != "ok":
                stats[status] += 1
                if cap is None:
                    bad += 1
                    continue
                stats["filled"] += 1
            if cap == 0:
                stats["zero"] += 1
        out.append({"x": p.x(), "y": p.y(), "cap": cap,
                    "name": str(f[name_field]) if name_field else "",
                    "geom": QgsGeometry(g)})
    if not out:
        raise AnalysisError("No usable GP points were found.%s" % (
            " Check the capacity field and the blank/invalid-value setting." if cap_field
            else ""))
    return out, bad, stats


def extract_demand(layer, work_crs, pop_field, use_surface, progress):
    ct = _transform(layer, work_crs, "Electoral division")
    out, feats, polys = [], [], []
    n = max(layer.featureCount(), 1)
    for k, f in enumerate(layer.getFeatures()):
        if k % 200 == 0:
            _pump()
            if progress(0.02 + 0.03 * k / n, "Reading electoral divisions…") is False:
                raise core.Cancelled()
        g = f.geometry()
        if g is None or g.isNull() or g.isEmpty():
            continue
        gw = QgsGeometry(g)
        gw.transform(ct)
        pt = (gw.pointOnSurface() if use_surface else gw.centroid()).asPoint()
        pop = _num(f[pop_field])
        out.append({"x": pt.x(), "y": pt.y(), "pop": pop if (pop and pop > 0) else 0.0})
        feats.append(QgsFeature(f))
        polys.append(gw)
    if not out:
        raise AnalysisError("No polygons with geometry were found in the population layer.")
    return out, feats, polys


def build_network(layer, work_crs, bbox, mode, speed_field, class_field, oneway_field,
                  default_kmh, overrides, progress):
    ct = _transform(layer, work_crs, "Roads")
    back = QgsCoordinateTransform(work_crs, layer.crs(), QgsProject.instance())
    rect = back.transformBoundingBox(bbox)
    req = QgsFeatureRequest().setFilterRect(rect)
    wanted = [x for x in (speed_field, class_field, oneway_field) if x]
    req.setSubsetOfAttributes(wanted, layer.fields())

    use_oneway = core.MODE_PROFILES[mode]["use_oneway"] and bool(oneway_field)
    net = core.Network()
    seen = used = 0
    for f in layer.getFeatures(req):
        seen += 1
        if seen % 1000 == 0:
            _pump()
            if progress(0.06 + 0.04 * min(seen / 2.0e5, 1.0), "Reading roads (%d)…" % seen) is False:
                raise core.Cancelled()
        g = f.geometry()
        if g is None or g.isNull() or g.type() != Qgis.GeometryType.Line:
            continue
        kmh = core.edge_speed_kmh(
            mode,
            f[speed_field] if speed_field else None,
            f[class_field] if class_field else None,
            default_kmh, overrides)
        if kmh is None:
            continue
        gw = QgsGeometry(g)
        gw.transform(ct)
        ow = core.parse_oneway(f[oneway_field]) if use_oneway else 0
        parts = gw.asMultiPolyline() if gw.isMultipart() else [gw.asPolyline()]
        for part in parts:
            net.add_polyline([(p.x(), p.y()) for p in part], kmh, ow)
        used += 1
    if net.n_edges == 0:
        raise AnalysisError(
            "No usable road segments were found around the data (read %d, used %d). "
            "Check the roads CRS, the road-class field and the chosen mode." % (seen, used))
    return net, seen, used


# --------------------------------------------------------------------------
# Edge-effect diagnostic
# --------------------------------------------------------------------------
def edge_flags(polys, pts, reach_m):
    """1 if a demand point lies within reach_m of the OUTER study-area boundary."""
    try:
        if len(polys) > 30000:
            return [None] * len(pts)
        outer = QgsGeometry.unaryUnion(polys).simplify(100.0)
        bnd = outer.convertToType(Qgis.GeometryType.Line, True)
        flags = []
        for k, p in enumerate(pts):
            if k % 300 == 0:
                _pump()
            d = bnd.distance(QgsGeometry.fromPointXY(QgsPointXY(p["x"], p["y"])))
            flags.append(1 if d <= reach_m else 0)
        return flags
    except Exception:
        return [None] * len(pts)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def ramp_colors(n):
    if n <= 1:
        return [_RAMP[-1]]
    stops = [QColor(c) for c in _RAMP]
    out = []
    for i in range(n):
        t = i * (len(stops) - 1) / float(n - 1)
        a = int(math.floor(t))
        b = min(a + 1, len(stops) - 1)
        w = t - a
        c = QColor(
            int(stops[a].red() + (stops[b].red() - stops[a].red()) * w),
            int(stops[a].green() + (stops[b].green() - stops[a].green()) * w),
            int(stops[a].blue() + (stops[b].blue() - stops[a].blue()) * w))
        out.append(c.name())
    return out


def class_labels(breaks, threshold):
    labels = ["No GP within %g min" % threshold]
    for i in range(len(breaks) - 1):
        labels.append("%.2f – %.2f" % (breaks[i], breaks[i + 1]))
    return labels


def classify(value, breaks, labels):
    if value is None:
        return "Not analysed (no road access)"
    if value <= 0:
        return labels[0]
    for i in range(len(breaks) - 1):
        if value <= breaks[i + 1] or i == len(breaks) - 2:
            return labels[i + 1]
    return labels[-1]


def _fill(color):
    return QgsFillSymbol.createSimple({"color": color, "outline_color": "#ffffff",
                                       "outline_width": "0.1"})


def style_choropleth(layer, field, breaks, threshold):
    labels = class_labels(breaks, threshold)
    cols = ramp_colors(len(breaks) - 1)
    root = QgsRuleBasedRenderer.Rule(None)

    def rule(color, expr, label):
        root.appendChild(QgsRuleBasedRenderer.Rule(_fill(color), 0, 0, expr, label))

    rule(C_NODATA, '"sf_snap" = 0', "Not analysed (no road access)")
    rule(C_ZERO, '"%s" = 0' % field, labels[0])
    for i in range(len(breaks) - 1):
        lo, hi = breaks[i], breaks[i + 1]
        op = ">=" if i == 0 else ">"
        rule(cols[i], '"%s" %s %.10g AND "%s" <= %.10g' % (field, op, lo, field, hi),
             labels[i + 1])
    layer.setRenderer(QgsRuleBasedRenderer(root))
    layer.triggerRepaint()


def style_supply(layer):
    sym = QgsMarkerSymbol.createSimple({"name": "circle", "color": "#0b6e4f",
                                        "outline_color": "#ffffff", "size": "2.2"})
    layer.setRenderer(QgsSingleSymbolRenderer(sym))


# --------------------------------------------------------------------------
# Output layers
# --------------------------------------------------------------------------
def _mem_layer(geom, name, crs):
    lyr = QgsVectorLayer("%s?crs=%s" % (geom, crs.authid() or "EPSG:4326"), name, "memory")
    lyr.setCrs(crs)
    return lyr


def _unique(name, existing):
    n = name
    while n in existing:
        n += "_"
    existing.add(n)
    return n


def build_ed_layer(ed_layer, feats, demand, results, edge, labels, breaks, name):
    lyr = _mem_layer("MultiPolygon", name, ed_layer.crs())
    existing = {f.name() for f in ed_layer.fields()}
    new_defs = [
        ("sf_pop", QMetaType.Type.Double), ("sf_nGP", QMetaType.Type.Int),
        ("sf_per10k", QMetaType.Type.Double), ("sf_mint", QMetaType.Type.Double),
        ("sf_idx", QMetaType.Type.Double), ("sf_edge", QMetaType.Type.Int),
        ("sf_snap", QMetaType.Type.Int), ("sf_class", QMetaType.Type.QString)]
    new_fields = [QgsField(_unique(n, existing), t) for n, t in new_defs]
    dp = lyr.dataProvider()
    dp.addAttributes(ed_layer.fields().toList() + new_fields)
    lyr.updateFields()

    out = []
    for f, d, r, e in zip(feats, demand, results, edge):
        g = QgsGeometry(f.geometry())
        g.convertToMultiType()
        nf = QgsFeature(lyr.fields())
        nf.setGeometry(g)
        mint = r["min_time"]
        nf.setAttributes(list(f.attributes()) + [
            d["pop"], r["n_gp"], r["per10k"],
            None if mint is None else round(mint, 2),
            r["index"], e, 1 if r["snapped"] else 0,
            classify(r["per10k"], breaks, labels)])
        out.append(nf)
    dp.addFeatures(out)
    lyr.updateExtents()
    return lyr


def build_supply_layer(gp_layer, supply, srs, name):
    lyr = _mem_layer("Point", name, gp_layer.crs())
    lyr.dataProvider().addAttributes([
        QgsField("name", QMetaType.Type.QString), QgsField("sf_cap", QMetaType.Type.Double),
        QgsField("sf_pop_in", QMetaType.Type.Double), QgsField("sf_R10k", QMetaType.Type.Double),
        QgsField("sf_nED", QMetaType.Type.Int), QgsField("sf_snap", QMetaType.Type.Int)])
    lyr.updateFields()
    feats = []
    for s, r in zip(supply, srs):
        nf = QgsFeature(lyr.fields())
        nf.setGeometry(s["geom"])
        nf.setAttributes([s["name"], s["cap"], r["demand"], r["R"] * 10000.0,
                          r["n_demand"], 1 if r["snapped"] else 0])
        feats.append(nf)
    lyr.dataProvider().addFeatures(feats)
    lyr.updateExtents()
    return lyr


def save_outputs(ed_mem, sup_mem, gpkg_path):
    """Write both layers into one GeoPackage + a CSV table; return reloaded layers."""
    ctx = QgsProject.instance().transformContext()

    def write(layer, path, driver, lname=None, overwrite_layer=False):
        o = QgsVectorFileWriter.SaveVectorOptions()
        o.driverName = driver
        o.fileEncoding = "UTF-8"
        if lname:
            o.layerName = lname
        if overwrite_layer:
            o.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        res = QgsVectorFileWriter.writeAsVectorFormatV3(layer, path, ctx, o)
        if res[0] != QgsVectorFileWriter.WriterError.NoError:
            raise AnalysisError("Could not write %s: %s" % (path, res[1]))

    base = os.path.splitext(gpkg_path)[0]
    write(ed_mem, gpkg_path, "GPKG", "ed_gp_accessibility")
    write(sup_mem, gpkg_path, "GPKG", "gp_supply", overwrite_layer=True)
    try:
        write(ed_mem, base + "_results.csv", "CSV")
    except Exception:
        pass
    ed = QgsVectorLayer("%s|layername=ed_gp_accessibility" % gpkg_path,
                        "GP accessibility by ED (2SFCA)", "ogr")
    sup = QgsVectorLayer("%s|layername=gp_supply" % gpkg_path, "GP supply (R_j)", "ogr")
    if not ed.isValid() or not sup.isValid():
        raise AnalysisError("Saved GeoPackage could not be re-opened.")
    return ed, sup


# --------------------------------------------------------------------------
# Print layout (A4 landscape) and PDF
# --------------------------------------------------------------------------
def make_layout(layer, title, subtitle, pdf_path=None, unit_label="GPs per 10,000 population"):
    project = QgsProject.instance()
    mm = Qgis.LayoutUnit.Millimeters
    lay = QgsPrintLayout(project)
    lay.initializeDefaults()
    lay.setName("GP accessibility 2SFCA - %s" % date.today().isoformat())
    lay.pageCollection().page(0).setPageSize("A4", QgsLayoutItemPage.Orientation.Landscape)

    def place(item, x, y, w=None, h=None):
        item.attemptMove(QgsLayoutPoint(x, y, mm))
        if w is not None:
            item.attemptResize(QgsLayoutSize(w, h, mm))
        lay.addLayoutItem(item)

    t = QgsLayoutItemLabel(lay)
    t.setText(title)
    f = QFont("Sans Serif", 18)
    f.setBold(True)
    t.setFont(f)
    place(t, 8, 6, 280, 10)

    s = QgsLayoutItemLabel(lay)
    s.setText(subtitle)
    s.setFont(QFont("Sans Serif", 9))
    place(s, 8, 15, 280, 6)

    m = QgsLayoutItemMap(lay)
    place(m, 8, 24, 200, 176)
    m.setFrameEnabled(True)
    m.setLayers([layer])
    m.setKeepLayerSet(True)
    ext = QgsCoordinateTransform(layer.crs(), project.crs(), project).transformBoundingBox(
        layer.extent())
    m.setExtent(ext)
    m.zoomToExtent(ext)

    leg = QgsLayoutItemLegend(lay)
    leg.setLinkedMap(m)
    leg.setAutoUpdateModel(False)
    root = leg.model().rootGroup()
    root.removeAllChildren()
    root.addLayer(layer)
    leg.setTitle(unit_label)
    place(leg, 214, 24)

    sb = QgsLayoutItemScaleBar(lay)
    sb.setLinkedMap(m)
    sb.setStyle("Single Box")
    sb.setUnits(Qgis.DistanceUnit.Kilometers)
    sb.setNumberOfSegments(2)
    sb.setNumberOfSegmentsLeft(0)
    sb.applyDefaultSize(Qgis.DistanceUnit.Kilometers)
    place(sb, 214, 188)

    project.layoutManager().addLayout(lay)
    if pdf_path:
        res = QgsLayoutExporter(lay).exportToPdf(pdf_path, QgsLayoutExporter.PdfExportSettings())
        if res != QgsLayoutExporter.ExportResult.Success:
            raise AnalysisError("PDF export failed (code %s)." % res)
    return lay


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_analysis(p, log, progress):
    """p: dict of parameters from the dialog. Returns dict of results."""
    work = p["work_crs"]
    if not work.isValid() or work.isGeographic():
        raise AnalysisError("Choose a projected (metre-based) analysis CRS, e.g. EPSG:2157 "
                            "(ITM) for Ireland.")
    mode = p["mode"]
    T = float(p["threshold"])

    log("Reading GP supply points…")
    supply, bad_s, cs = extract_supply(p["gp_layer"], work, p["gp_cap_field"],
                                       p["gp_name_field"], progress,
                                       p.get("gp_cap_missing", "skip"))
    log("  %d GP points (%d skipped)" % (len(supply), bad_s))
    if p["gp_cap_field"]:
        caps = [s["cap"] for s in supply]
        log("  Supply = field '%s': total %.2f, min %.2f, max %.2f, mean %.2f per point"
            % (p["gp_cap_field"], sum(caps), min(caps), max(caps), sum(caps) / len(caps)))
        if cs["blank"] or cs["invalid"] or cs["negative"]:
            log("  WARNING: capacity blank for %d, non-numeric for %d, negative for %d GPs "
                "(blank/invalid handled as '%s'; negatives always skipped)."
                % (cs["blank"], cs["invalid"], cs["negative"], p.get("gp_cap_missing", "skip")))
        if cs["zero"]:
            log("  Note: %d GPs have zero capacity and contribute no supply." % cs["zero"])
    else:
        log("  Supply = 1 per GP point (no capacity field chosen).")

    log("Converting electoral divisions to %s…" %
        ("points-on-surface" if p["use_surface"] else "centroids"))
    demand, feats, polys = extract_demand(p["ed_layer"], work, p["pop_field"],
                                          p["use_surface"], progress)
    log("  %d demand points, total population %.0f" % (len(demand),
                                                      sum(d["pop"] for d in demand)))

    xs = [q["x"] for q in demand + supply]
    ys = [q["y"] for q in demand + supply]
    pad = _VMAX[mode] * T / 60.0 * 1000.0 + p["snap_max"]
    bbox = QgsRectangle(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    log("Building %s network from roads…" % core.MODE_PROFILES[mode]["label"])
    net, seen, used = build_network(
        p["roads_layer"], work, bbox, mode, p["speed_field"], p["class_field"],
        p["oneway_field"], p["default_kmh"],
        core.parse_speed_overrides(p["speed_overrides"]), progress)
    log("  roads read: %d, usable: %d -> %d nodes, %d directed edges"
        % (seen, used, net.n_nodes, net.n_edges))

    def sub(frac, msg):
        return progress(0.10 + 0.82 * frac, msg)

    drs, srs, summ = core.two_step_fca(
        net, supply, demand, T, access_kmh=p["access_kmh"], snap_max=p["snap_max"],
        decay=p["decay"], progress=sub)
    log("  network components: %d (largest holds %.1f%% of nodes)"
        % (summ["n_components"], 100 * summ["largest_component_share"]))
    if summ["largest_component_share"] < 0.9:
        log("  WARNING: the road network is fragmented. Roads must be noded at "
            "junctions; points only snap to the largest connected component.")
    if summ["n_demand_snapped"] < summ["n_demand"]:
        log("  WARNING: %d demand points are further than %g m from a road and "
            "were excluded." % (summ["n_demand"] - summ["n_demand_snapped"], p["snap_max"]))
    if summ["n_supply_snapped"] < summ["n_supply"]:
        log("  WARNING: %d GP points could not be snapped to the road network."
            % (summ["n_supply"] - summ["n_supply_snapped"]))

    log("Edge-effect diagnostic…")
    reach = core.MODE_PROFILES[mode]["default_kmh"] * T / 60.0 * 1000.0
    edge = edge_flags(polys, demand, reach)

    vals = [r["per10k"] for r in drs if r["per10k"] is not None]
    breaks = core.class_breaks(vals, p["n_classes"], p["class_method"])
    labels = class_labels(breaks, T)

    ed_mem = build_ed_layer(p["ed_layer"], feats, demand, drs, edge, labels, breaks,
                            "GP accessibility by ED (2SFCA)")
    sup_mem = build_supply_layer(p["gp_layer"], supply, srs, "GP supply (R_j)")

    if p["out_gpkg"]:
        log("Writing %s…" % p["out_gpkg"])
        ed_out, sup_out = save_outputs(ed_mem, sup_mem, p["out_gpkg"])
    else:
        log("No output file chosen: results are temporary scratch layers.")
        ed_out, sup_out = ed_mem, sup_mem

    style_choropleth(ed_out, "sf_per10k", breaks, T)
    style_supply(sup_out)

    # ---- tabular statistics ----
    classes = [classify(r["per10k"], breaks, labels) for r in drs]
    groups = None
    if p.get("group_field"):
        groups = [f[p["group_field"]] for f in feats]
    tables = stats_core.compute_tables(
        demand, drs, srs, supply, summ, T, class_labels=classes,
        class_order=[labels[0]] + labels[1:] + ["Not analysed (no road access)"],
        groups=groups, group_name=p.get("group_field") or "Group",
        unit=("capacity (%s)" % p["gp_cap_field"]) if p["gp_cap_field"] else "GPs")
    stat_files = []
    if p["out_gpkg"]:
        prefix = os.path.splitext(p["out_gpkg"])[0] + "_stats"
        try:
            stat_files = stats_core.write_csvs(tables, prefix)
            stat_files.append(stats_core.write_text(tables, prefix + "_report.txt"))
            log("Statistics tables written: %d files (%s_*.csv, %s_report.txt)"
                % (len(stat_files), prefix, prefix))
        except OSError as e:
            log("WARNING: could not write statistics files: %s" % e)

    return {"tables": tables, "stat_files": stat_files,
            "ed_layer": ed_out, "supply_layer": sup_out, "summary": summ,
            "breaks": breaks, "labels": labels,
            "edge_n": sum(1 for e in edge if e == 1), "edge_known": edge[0] is not None,
            "mode_label": core.MODE_PROFILES[mode]["label"]}


def format_summary(summ, res, T):
    lines = [
        "=== 2SFCA summary (%s, %g-minute catchment) ===" % (res["mode_label"], T),
        "Demand points analysed: %d of %d   |   GP points used: %d of %d"
        % (summ["n_demand_snapped"], summ["n_demand"], summ["n_supply_snapped"], summ["n_supply"]),
        "Population analysed: %.0f   |   GP capacity: %.1f" % (summ["total_pop"],
                                                              summ["total_capacity"]),
        "Benchmark (total supply / population): %.2f per 10,000" % summ["benchmark_per10k"],
        "Population-weighted mean accessibility: %.2f per 10,000" % summ["pop_weighted_mean_per10k"],
        "Median ED accessibility: %.2f per 10,000" % summ["median_per10k"],
        "EDs with NO GP in catchment: %d (%.1f%% of population)" % (summ["n_zero"],
                                                                   summ["pct_pop_zero"]),
    ]
    if res["edge_known"]:
        lines.append("EDs within edge-effect reach of the study-area boundary: %d "
                     "(interpret with caution)" % res["edge_n"])
    return "\n".join(lines)
