# -*- coding: utf-8 -*-
"""
QGIS-side glue for the accessibility regression tool (QGIS 4 / Qt6).

Builds the analysis table from an accessibility polygon layer plus up to three
predictor shapefiles (population density, rurality, deprivation), each with a
user-chosen attribute field, runs regression_core, and builds the residual layer.
"""
import math

import numpy as np
from qgis.core import (
    Qgis, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature, QgsField,
    QgsFillSymbol, QgsGeometry, QgsPointXY, QgsProject, QgsRectangle,
    QgsRuleBasedRenderer, QgsSpatialIndex, QgsVectorLayer)
from qgis.PyQt.QtCore import QCoreApplication, QMetaType

from . import regression_core as rc
from . import spatial_models as sm


class RegressionError(Exception):
    pass


def _num(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _key(v):
    """Join key tolerant of 12 vs 12.0 vs ' 12 '."""
    if v is None:
        return None
    f = _num(v)
    if f is not None and f == int(f):
        return str(int(f))
    s = str(v).strip()
    return s or None


def _work_crs(layer):
    crs = layer.crs()
    if crs.isValid() and not crs.isGeographic():
        return crs
    wgs = QgsCoordinateReferenceSystem("EPSG:4326")
    c = QgsCoordinateTransform(crs, wgs, QgsProject.instance()).transform(
        layer.extent().center())
    zone = int((c.x() + 180) // 6) + 1
    return QgsCoordinateReferenceSystem("EPSG:%d" % ((32600 if c.y() >= 0 else 32700) + zone))


def _pump():
    QCoreApplication.processEvents()


# --------------------------------------------------------------------------
# Predictor lookup
# --------------------------------------------------------------------------
def _spatial_lookup(base_pts, base_crs, layer, field, progress_label):
    """For each base point (in base_crs) return the predictor value of the polygon
    containing it (None if none). base_pts: list of QgsPointXY."""
    ct = QgsCoordinateTransform(base_crs, layer.crs(), QgsProject.instance())
    index = QgsSpatialIndex(layer.getFeatures())
    geoms, vals = {}, {}
    for f in layer.getFeatures():
        geoms[f.id()] = QgsGeometry(f.geometry())
        vals[f.id()] = f[field]
    out = []
    for k, p in enumerate(base_pts):
        if k % 400 == 0:
            _pump()
        q = ct.transform(p)
        cand = index.intersects(QgsRectangle(q.x(), q.y(), q.x(), q.y()))
        g = QgsGeometry.fromPointXY(q)
        hit = None
        for fid in cand:
            if geoms[fid].intersects(g):
                hit = vals[fid]
                break
        out.append(hit)
    return out


def _id_lookup(base_ids, layer, field, id_field):
    table = {}
    for f in layer.getFeatures():
        k = _key(f[id_field])
        if k is not None:
            table[k] = f[field]
    return [table.get(_key(b)) if b is not None else None for b in base_ids]


# --------------------------------------------------------------------------
# Assemble + fit
# --------------------------------------------------------------------------
def assemble(spec):
    """spec keys:
        base (layer), outcome_field, base_id_field, log_outcome, drop_zero,
        predictors: list of dicts {label, layer (None = base), field, join ('spatial'|'id'),
                                   pred_id_field, log (bool)}
    Returns a dict with y, columns, names, xy, fids, notes.
    """
    base = spec["base"]
    if base is None or not spec["outcome_field"]:
        raise RegressionError("Choose the accessibility layer and its outcome field.")
    if not spec["predictors"]:
        raise RegressionError("Enable at least one predictor.")
    work = _work_crs(base)
    to_work = QgsCoordinateTransform(base.crs(), work, QgsProject.instance())

    fids, y_raw, pts_base, pts_work, base_ids = [], [], [], [], []
    skipped_geom = 0
    for k, f in enumerate(base.getFeatures()):
        if k % 400 == 0:
            _pump()
        g = f.geometry()
        if g is None or g.isNull() or g.isEmpty():
            skipped_geom += 1
            continue
        pos = g.pointOnSurface()
        if pos is None or pos.isNull():
            skipped_geom += 1
            continue
        p = pos.asPoint()
        fids.append(f.id())
        y_raw.append(f[spec["outcome_field"]])
        pts_base.append(QgsPointXY(p))
        pts_work.append(to_work.transform(QgsPointXY(p)))
        base_ids.append(f[spec["base_id_field"]] if spec.get("base_id_field") else None)
    if not fids:
        raise RegressionError("The accessibility layer has no usable polygons.")

    cols = {}            # label -> (list of raw values, spec)
    for sp in spec["predictors"]:
        lyr = sp["layer"] or base
        if sp["layer"] is None or sp["layer"].id() == base.id():
            if sp["join"] == "spatial":
                same = {f.id(): f[sp["field"]] for f in base.getFeatures()}
                vals = [same.get(fid) for fid in fids]
            else:
                vals = _id_lookup(base_ids, base, sp["field"], sp["pred_id_field"])
        elif sp["join"] == "id":
            if not spec.get("base_id_field") or not sp.get("pred_id_field"):
                raise RegressionError("'%s': choose the ID field on both layers for an "
                                      "attribute join." % sp["label"])
            vals = _id_lookup(base_ids, lyr, sp["field"], sp["pred_id_field"])
        else:
            vals = _spatial_lookup(pts_base, base.crs(), lyr, sp["field"], sp["label"])
        cols[sp["label"]] = (vals, sp)

    # ---- outcome ----
    y = [_num(v) for v in y_raw]

    keep = []
    reasons = {"missing outcome": 0, "zero outcome dropped": 0, "missing predictor": 0}
    pred_numeric = {}
    for label, (vals, sp) in cols.items():
        nums = [_num(v) for v in vals]
        usable = [n for n, v in zip(nums, vals) if v not in (None, "")]
        pred_numeric[label] = bool(usable) and len(usable) >= 0.9 * len(
            [v for v in vals if v not in (None, "")])
    for i in range(len(fids)):
        if y[i] is None:
            reasons["missing outcome"] += 1
            continue
        if spec["drop_zero"] and y[i] <= 0:
            reasons["zero outcome dropped"] += 1
            continue
        bad = False
        for label, (vals, sp) in cols.items():
            v = vals[i]
            if v is None or v == "" or (pred_numeric[label] and _num(v) is None):
                bad = True
                break
        if bad:
            reasons["missing predictor"] += 1
            continue
        keep.append(i)
    if len(keep) < 10:
        raise RegressionError(
            "Only %d complete rows remain after matching (missing outcome %d, zero outcome "
            "dropped %d, missing predictor %d). Check the join method, IDs, CRS and fields."
            % (len(keep), reasons["missing outcome"], reasons["zero outcome dropped"],
               reasons["missing predictor"]))

    yv = np.array([y[i] for i in keep], dtype=float)
    if spec["log_outcome"]:
        yv = np.log1p(np.clip(yv, 0, None))

    columns, names, notes = [], [], []
    for label, (vals, sp) in cols.items():
        sel = [vals[i] for i in keep]
        if pred_numeric[label]:
            arr = np.array([_num(v) for v in sel], dtype=float)
            nm = label
            if sp["log"]:
                if (arr < 0).any():
                    raise RegressionError("'%s' has negative values; cannot log-transform." % label)
                arr = np.log1p(arr)
                nm = "ln(1+%s)" % label
            if np.ptp(arr) == 0:
                raise RegressionError("'%s' is constant across the analysed EDs." % label)
            columns.append(arr)
            names.append(nm)
        else:
            dc, dn, ref = rc.dummy_code(sel, label)
            if not dc:
                raise RegressionError("'%s' has only one category." % label)
            columns += dc
            names += dn
            notes.append("%s is categorical: reference category = '%s'." % (label, ref))

    return {
        "y": yv, "columns": columns, "names": names, "fids": [fids[i] for i in keep],
        "xy": np.array([[pts_work[i].x(), pts_work[i].y()] for i in keep]),
        "y_raw": [y[i] for i in keep],
        "n_total": len(fids) + skipped_geom, "dropped": reasons, "notes": notes,
        "skipped_geom": skipped_geom,
    }


def fit(data, robust=True, k_neighbours=8, permutations=499):
    X = np.column_stack(data["columns"])
    res_c = rc.ols(data["y"], X, data["names"], robust=False)
    res_r = rc.ols(data["y"], X, data["names"], robust=True)
    res = res_r if robust else res_c
    res["res_c"], res["res_r"] = res_c, res_r          # both kept for the side-by-side table
    idx = rc.knn_indices(data["xy"], k_neighbours)
    mi, mz, mp = rc.morans_i(res["resid"], idx, permutations)
    res["moran_i"], res["moran_z"], res["moran_p"] = mi, mz, mp
    return res


# --------------------------------------------------------------------------
# Report text
# --------------------------------------------------------------------------
def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else \
        "." if p < 0.1 else ""


def make_report(spec, data, res):
    out_label = "ln(1+%s)" % spec["outcome_field"] if spec["log_outcome"] else spec["outcome_field"]
    L = []
    L.append("OLS regression of GP accessibility on area characteristics")
    L.append("Outcome: %s   |   Observations: %d of %d EDs   |   Std. errors: %s"
             % (out_label, res["n"], data["n_total"], "HC3 robust" if res["robust"] else "classical"))
    d = data["dropped"]
    L.append("Dropped: missing outcome %d, zero outcome %d, missing predictor %d, no geometry %d"
             % (d["missing outcome"], d["zero outcome dropped"], d["missing predictor"],
                data["skipped_geom"]))
    L.append("")
    w = max(len(n) for n in res["names"]) + 2
    L.append("%-*s %10s %9s %10s %8s %9s %20s %6s" % (w, "Variable", "Coef.", "Std.beta", "Std.err",
                                                       "t", "p", "95% CI", "VIF"))
    for i, nm in enumerate(res["names"]):
        sb = "" if i == 0 else "%9.3f" % res["std_beta"][i]
        vf = "" if i == 0 else "%6.2f" % res["vif"][i]
        L.append("%-*s %10.4g %9s %10.4g %8.2f %9.3g %s [%.3g, %.3g] %s" % (
            w, nm, res["beta"][i], sb, res["se"][i], res["t"][i], res["p"][i],
            _stars(res["p"][i]).ljust(3), res["ci_lo"][i], res["ci_hi"][i], vf))
    L.append("Signif.: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
    L.append("")
    L.append("R2 = %.3f   adj. R2 = %.3f   F(%d,%d) = %.2f (p = %.3g)   AIC = %.1f   BIC = %.1f"
             % (res["r2"], res["adj_r2"], res["k"], res["df"], res["F"], res["F_p"],
                res["aic"], res["bic"]))
    L.append("Breusch-Pagan heteroskedasticity: LM = %.2f, p = %.3g" % (res["bp"], res["bp_p"]))
    L.append("Moran's I of residuals (8 nearest neighbours): I = %.3f, z = %.2f, pseudo-p = %.3g"
             % (res["moran_i"], res["moran_z"], res["moran_p"]))
    L.append("")
    L.append("Interpretation (associations, not causal effects):")
    L += ["  - " + s for s in rc.interpret(res, spec["outcome_field"], spec["log_outcome"])]
    L.append("")
    L.append("Diagnostics:")
    flags = []
    hi = [n for n, v in zip(res["names"][1:], res["vif"][1:]) if v > 5]
    if hi:
        flags.append("High collinearity (VIF > 5): %s. Coefficients for these are unstable; "
                     "density and rurality are often strongly related." % ", ".join(hi))
    if res["moran_p"] < 0.05 and res["moran_i"] > 0:
        flags.append("Residuals are spatially clustered (Moran's I = %.2f). OLS standard errors "
                     "are likely too small. Tick 'Spatial lag', 'Spatial error' and/or 'GWR' "
                     "in the model options and re-run; the LM tests will say which fits best." %
                     res["moran_i"])
    if res["bp_p"] < 0.05 and not res["robust"]:
        flags.append("Heteroskedasticity detected; switch on robust (HC3) errors.")
    z0 = sum(1 for v in data["y_raw"] if v is not None and v <= 0)
    if z0 > 0.2 * res["n"] and not spec["drop_zero"]:
        flags.append("%.0f%% of outcomes are zero (no GP within the catchment). Linear models fit "
                     "zero-inflated outcomes poorly; consider a two-part (hurdle) model or Tobit."
                     % (100.0 * z0 / res["n"]))
    if res["r2"] < 0.1:
        flags.append("Low R2: these three factors explain little of the variation in access.")
    L += ["  - " + s for s in (flags or ["No major warnings."])]
    L += ["  - " + s for s in data["notes"]]
    return "\n".join(L)


def table_rows(res):
    rows = []
    for i, nm in enumerate(res["names"]):
        rows.append([nm, res["beta"][i], None if i == 0 else res["std_beta"][i], res["se"][i],
                     res["t"][i], res["p"][i], res["ci_lo"][i], res["ci_hi"][i],
                     None if i == 0 else res["vif"][i]])
    return rows


# --------------------------------------------------------------------------
# Residual layer
# --------------------------------------------------------------------------
def residual_layer(base, data, res, name="Regression residuals"):
    lyr = QgsVectorLayer("MultiPolygon?crs=%s" % (base.crs().authid() or "EPSG:4326"),
                         name, "memory")
    lyr.setCrs(base.crs())
    dp = lyr.dataProvider()
    dp.addAttributes([QgsField("reg_y", QMetaType.Type.Double),
                      QgsField("reg_fit", QMetaType.Type.Double),
                      QgsField("reg_res", QMetaType.Type.Double),
                      QgsField("reg_zres", QMetaType.Type.Double)])
    lyr.updateFields()
    feats = []
    for fid, y, fit_, r, z in zip(data["fids"], data["y"], res["fitted"], res["resid"],
                                  res["std_resid"]):
        src = base.getFeature(fid)
        g = QgsGeometry(src.geometry())
        g.convertToMultiType()
        nf = QgsFeature(lyr.fields())
        nf.setGeometry(g)
        nf.setAttributes([float(y), float(fit_), float(r), float(z)])
        feats.append(nf)
    dp.addFeatures(feats)
    lyr.updateExtents()

    def fill(c):
        return QgsFillSymbol.createSimple({"color": c, "outline_color": "#ffffff",
                                           "outline_width": "0.1"})
    root = QgsRuleBasedRenderer.Rule(None)
    for c, expr, lab in (
            ("#b2182b", '"reg_zres" < -2', "Much LESS access than predicted (z < -2)"),
            ("#ef8a62", '"reg_zres" >= -2 AND "reg_zres" < -1', "Less than predicted (-2 to -1)"),
            ("#f7f7f7", '"reg_zres" >= -1 AND "reg_zres" <= 1', "As predicted (-1 to 1)"),
            ("#67a9cf", '"reg_zres" > 1 AND "reg_zres" <= 2', "More than predicted (1 to 2)"),
            ("#2166ac", '"reg_zres" > 2', "Much MORE access than predicted (z > 2)")):
        root.appendChild(QgsRuleBasedRenderer.Rule(fill(c), 0, 0, expr, lab))
    lyr.setRenderer(QgsRuleBasedRenderer(root))
    return lyr


# --------------------------------------------------------------------------
# Spatial diagnostics and models
# --------------------------------------------------------------------------
def fit_spatial(data, opts, progress=None):
    """opts: k (neighbours), lag, error, gwr (bool), bandwidth (None = AICc), perms."""
    X = np.column_stack(data["columns"])
    y, names = data["y"], data["names"]
    W = sm.KnnWeights(data["xy"], opts["k"])
    out = {"W": W, "k": W.k}
    out["lm"], out["lm_rec"] = sm.lm_tests(y, X, W)
    ld = None
    if opts.get("lag") or opts.get("error"):
        ld = sm.LogDet(W, progress=progress)
    if opts.get("lag"):
        out["lag"] = sm.fit_lag(y, X, W, ld, names, perms=opts["perms"], progress=progress)
    if opts.get("error"):
        out["error"] = sm.fit_error(y, X, W, ld, names, perms=opts["perms"], progress=progress)
    if opts.get("gwr"):
        out["gwr"] = sm.gwr(y, X, data["xy"], names, bandwidth=opts.get("bandwidth"),
                            progress=progress)
    return out


def _model_rows(res, sp):
    ll_ols = (2.0 * (res["k"] + 1) - res["aic"]) / 2.0
    rows = [["OLS", "%.3f" % res["r2"], "%.1f" % ll_ols, "%.1f" % res["aic"], "-",
             "%.3f (p=%.3g)" % (res["moran_i"], res["moran_p"])]]
    for key, par in (("lag", "rho = %.3f"), ("error", "lambda = %.3f")):
        m = sp.get(key)
        if m:
            v = m["rho"] if key == "lag" else m["lambda"]
            rows.append([m["model"], "%.3f*" % m["pseudo_r2"], "%.1f" % m["loglik"],
                         "%.1f" % m["aic"], par % v,
                         "%.3f (p=%.3g)" % (m["moran_i"], m["moran_p"])])
    return rows


def spatial_report(res, sp, n_total=None):
    W = sp["W"]
    L = ["", "SPATIAL DIAGNOSTICS AND MODELS",
         "Weights: %d nearest neighbours, row-standardised." % W.k, "",
         "Lagrange-multiplier tests on the OLS residuals:"]
    for k, (v, p) in sp["lm"].items():
        L.append("  %-17s %9.3f   p = %-9.3g %s" % (k, v, p, _stars(p)))
    L.append("  -> " + sp["lm_rec"])
    L.append("")
    rows = _model_rows(res, sp)
    L.append("Model comparison (lower AIC is better; Moran's I should approach 0 if the model "
             "removes the clustering):")
    L.append("  %-14s %8s %10s %9s  %-14s %s" % ("Model", "R2", "Log-lik", "AIC", "Spatial par.",
                                                  "Moran's I of residuals"))
    for r in rows:
        L.append("  %-14s %8s %10s %9s  %-14s %s" % tuple(r))
    if sp.get("lag") or sp.get("error"):
        L.append("  * pseudo-R2 = squared correlation of observed and fitted values.")
    for key, title in (("lag", "SPATIAL LAG MODEL  (y = rho*W*y + X*beta + e)"),
                       ("error", "SPATIAL ERROR MODEL  (y = X*beta + u, u = lambda*W*u + e)")):
        m = sp.get(key)
        if not m:
            continue
        L += ["", title, "Log-determinant: %s." % m["logdet_mode"]]
        w = max(len(n) for n in m["names"]) + 2
        hdr = "%-*s %10s %10s %8s %9s" % (w, "Variable", "Coef.", "Std.err", "z", "p")
        if key == "lag":
            hdr += " %12s" % "Total effect"
        L.append(hdr)
        for i, nm in enumerate(m["names"]):
            line = "%-*s %10.4g %10.4g %8.2f %9.3g %s" % (
                w, nm, m["coef"][i], m["se"][i], m["z"][i], m["p"][i], _stars(m["p"][i]).ljust(3))
            if key == "lag" and m["total_effect"] and i not in (0, len(m["names"]) - 1):
                line += " %12.4g" % m["total_effect"][i]
            L.append(line)
        if key == "lag":
            L.append("In a lag model a coefficient is the DIRECT effect before spillovers; the "
                     "total effect (beta / (1 - rho)) includes feedback through neighbours.")
    g = sp.get("gwr")
    if g:
        n = g["n"]
        L += ["", "GEOGRAPHICALLY WEIGHTED REGRESSION  (%s kernel)" % g["kernel"],
              "Bandwidth: %d nearest neighbours (%.0f%% of EDs), %s."
              % (g["bandwidth"], 100.0 * g["bandwidth"] / n,
                 "chosen by AICc" if g["searched"] else "fixed by you"),
              "R2 = %.3f   adj. R2 = %.3f   effective parameters = %.1f"
              % (g["r2"], g["adj_r2"], g["tr_s"]),
              "AICc: GWR = %.1f vs OLS = %.1f  ->  %s"
              % (g["aicc"], g["ols_aicc"],
                 "GWR is clearly better (difference > 3): relationships vary across space."
                 if g["ols_aicc"] - g["aicc"] > 3 else
                 "no clear gain over OLS (difference <= 3): relationships look stable across space."),
              "", "Local coefficients (across %d EDs):" % n]
        w = max(len(nm) for nm in g["names"]) + 2
        L.append("%-*s %9s %9s %9s %9s %9s %9s %10s  %s" % (
            w, "Variable", "OLS", "Min", "Q1", "Median", "Q3", "Max", "% |t|>1.96", "Varies?"))
        q = g["beta_quantiles"]
        for j, nm in enumerate(g["names"]):
            iqr = q[3, j] - q[1, j]
            varies = "yes" if iqr > 2.0 * g["ols_se"][j] else "no"
            sig = 100.0 * float((np.abs(g["t"][:, j]) > 1.96).mean())
            L.append("%-*s %9.4g %9.4g %9.4g %9.4g %9.4g %9.4g %10.0f  %s" % (
                w, nm, g["ols_beta"][j], q[0, j], q[1, j], q[2, j], q[3, j], q[4, j], sig,
                varies if j > 0 else "-"))
        L.append("'Varies?' = yes when the interquartile range of local estimates exceeds twice the "
                 "OLS standard error (rule of thumb).")
        notes = ["Local t-values are pseudo-t statistics from many overlapping tests; treat the "
                 "significance map as exploratory."]
        if g["bandwidth"] >= min(n - 1, 2000) - 2:
            notes.append("The bandwidth hit its upper limit: the data show little spatial "
                         "variation, so GWR is close to OLS.")
        if g["singular_local"]:
            notes.append("%d local models were near-singular (often sparse dummy categories); "
                         "prefer numeric predictors for GWR." % g["singular_local"])
        L += ["Notes:"] + ["  - " + t for t in notes]
    return "\n".join(L)


def spatial_table_rows(model_key, sp):
    """(headers, rows) for the dialog table of a spatial model."""
    if model_key in ("lag", "error"):
        m = sp[model_key]
        head = ["Variable", "Coef.", "Std. err", "z", "p"] + (["Total effect"] if model_key == "lag" else [])
        rows = []
        for i, nm in enumerate(m["names"]):
            row = [nm, m["coef"][i], m["se"][i], m["z"][i], m["p"][i]]
            if model_key == "lag":
                te = m["total_effect"][i]
                row.append(te if te == te else None)
            rows.append(row)
        return head, rows
    g = sp["gwr"]
    q = g["beta_quantiles"]
    head = ["Variable", "OLS coef.", "GWR min", "Q1", "Median", "Q3", "Max", "% |t|>1.96"]
    rows = []
    for j, nm in enumerate(g["names"]):
        rows.append([nm, g["ols_beta"][j], q[0, j], q[1, j], q[2, j], q[3, j], q[4, j],
                     100.0 * float((np.abs(g["t"][:, j]) > 1.96).mean())])
    return head, rows


def _short(name, used):
    base = "".join(ch if ch.isalnum() else "_" for ch in name)[:7].strip("_") or "v"
    nm, k = base, 1
    while nm in used:
        k += 1
        nm = "%s%d" % (base[:6], k)
    used.add(nm)
    return nm


def gwr_layer(base, data, g, name="GWR local results"):
    """Polygon layer with local coefficients, pseudo-t and local R2 per ED.
    Returns (layer, [(label, coef_field, t_field), ...])."""
    lyr = QgsVectorLayer("MultiPolygon?crs=%s" % (base.crs().authid() or "EPSG:4326"),
                         name, "memory")
    lyr.setCrs(base.crs())
    used, fields, mapping = set(), [], []
    for j, nm in enumerate(g["names"]):
        short = _short(nm, used)
        bf, tf = "b_" + short, "t_" + short
        fields += [QgsField(bf, QMetaType.Type.Double), QgsField(tf, QMetaType.Type.Double)]
        mapping.append((nm, bf, tf))
    fields += [QgsField("loc_r2", QMetaType.Type.Double), QgsField("gwr_res", QMetaType.Type.Double)]
    dp = lyr.dataProvider()
    dp.addAttributes(fields)
    lyr.updateFields()
    feats = []
    for i, fid in enumerate(data["fids"]):
        src = base.getFeature(fid)
        geom = QgsGeometry(src.geometry())
        geom.convertToMultiType()
        nf = QgsFeature(lyr.fields())
        nf.setGeometry(geom)
        attrs = []
        for j in range(len(g["names"])):
            attrs += [float(g["betas"][i, j]), float(g["t"][i, j]) if g["t"][i, j] == g["t"][i, j] else None]
        attrs += [float(g["local_r2"][i]) if g["local_r2"][i] == g["local_r2"][i] else None,
                  float(g["resid"][i])]
        nf.setAttributes(attrs)
        feats.append(nf)
    dp.addFeatures(feats)
    lyr.updateExtents()
    return lyr, mapping


def style_gwr(layer, coef_field, t_field, label):
    """Four-class map: sign of the local coefficient x significance (|t| > 1.96)."""
    def fill(c):
        return QgsFillSymbol.createSimple({"color": c, "outline_color": "#ffffff",
                                           "outline_width": "0.1"})
    root = QgsRuleBasedRenderer.Rule(None)
    for c, expr, lab in (
            ("#b2182b", '"%s" < 0 AND abs("%s") > 1.96' % (coef_field, t_field),
             "Negative, significant"),
            ("#f4a582", '"%s" < 0 AND abs("%s") <= 1.96' % (coef_field, t_field),
             "Negative, not significant"),
            ("#92c5de", '"%s" >= 0 AND abs("%s") <= 1.96' % (coef_field, t_field),
             "Positive, not significant"),
            ("#2166ac", '"%s" >= 0 AND abs("%s") > 1.96' % (coef_field, t_field),
             "Positive, significant")):
        root.appendChild(QgsRuleBasedRenderer.Rule(fill(c), 0, 0, expr, lab))
    layer.setRenderer(QgsRuleBasedRenderer(root))
    layer.setName("GWR: %s" % label)
    layer.triggerRepaint()


# --------------------------------------------------------------------------
# Interpretable summary tables and CSV
# --------------------------------------------------------------------------
def summary_for(spec, res):
    """(coef_head, coef_rows, fit_head, fit_rows) for display and CSV."""
    return rc.summary_tables(res["res_c"], res["res_r"], spec["outcome_field"],
                             spec["log_outcome"],
                             (res["moran_i"], res["moran_z"], res["moran_p"]))


def write_summary_csv(path, tables, title="GP accessibility regression summary"):
    """One CSV with both blocks: coefficients (HC3 beside standardised beta, VIF ...) then
    model fit and diagnostics. UTF-8 with BOM so Excel opens it correctly."""
    import csv
    ch, cr, fh, fr = tables
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([title])
        w.writerow(["Coefficients: inference uses HC3 heteroscedasticity-consistent standard errors"])
        w.writerow(ch)
        for r in cr:
            w.writerow(["" if v is None else v for v in r])
        w.writerow([])
        w.writerow(["Model fit and diagnostics"])
        w.writerow(fh)
        for r in fr:
            w.writerow(["" if v is None else v for v in r])
        w.writerow([])
        w.writerow(["Notes: Sig.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns = not significant. "
                    "Strength thresholds for |std. beta| (0.1/0.3/0.5) and VIF thresholds (5/10) are "
                    "rules of thumb. Associations are not causal effects."])


# --------------------------------------------------------------------------
# Local Moran's I (LISA)
# --------------------------------------------------------------------------
def lisa_inputs_from_layer(base, field, id_field=None):
    """Values, work-CRS points and ids for every polygon of `base` with a numeric `field`."""
    work = _work_crs(base)
    ct = QgsCoordinateTransform(base.crs(), work, QgsProject.instance())
    vals, xy, fids, ids, skipped = [], [], [], [], 0
    for k, f in enumerate(base.getFeatures()):
        if k % 400 == 0:
            _pump()
        g = f.geometry()
        v = _num(f[field])
        if g is None or g.isNull() or g.isEmpty() or v is None:
            skipped += 1
            continue
        p = ct.transform(QgsPointXY(g.pointOnSurface().asPoint()))
        vals.append(v)
        xy.append([p.x(), p.y()])
        fids.append(f.id())
        ids.append(f[id_field] if id_field else f.id())
    if len(vals) < 20:
        raise RegressionError("Only %d usable polygons with a numeric '%s' value." % (len(vals), field))
    return {"values": np.array(vals), "xy": np.array(xy), "fids": fids, "ids": ids,
            "skipped": skipped, "label": field}


def lisa_inputs_from_residuals(base, data, res, id_field=None):
    ids = []
    for fid in data["fids"]:
        ids.append(base.getFeature(fid)[id_field] if id_field else fid)
    return {"values": np.asarray(res["resid"], float), "xy": data["xy"], "fids": list(data["fids"]),
            "ids": ids, "skipped": 0, "label": "OLS residuals"}


def run_lisa(inp, k, perms, alpha, fdr, progress=None):
    W = sm.KnnWeights(inp["xy"], k)
    r = sm.local_moran(inp["values"], W, perms=perms, alpha=alpha, fdr=fdr, progress=progress)
    r["k"] = W.k
    return r


def lisa_tables(inp, r):
    """(summary_head, summary_rows, detail_head, detail_rows)."""
    n = len(inp["values"])
    order = ["HH", "LL", "HL", "LH", "NS"]
    sh = ["Cluster type", "Code", "EDs", "% of EDs", "Mean value", "What it means"]
    meaning = {
        "HH": "Hot spot: high values surrounded by high values.",
        "LL": "Cold spot: low values surrounded by low values.",
        "HL": "Outlier: a high value among low neighbours.",
        "LH": "Outlier: a low value among high neighbours.",
        "NS": "No statistically significant local pattern."}
    rows = []
    for c in order:
        m = r["code"] == c
        cnt = int(m.sum())
        rows.append([sm.CLUSTER_LABELS[c], c, cnt, round(100.0 * cnt / n, 1),
                     rc._g(float(inp["values"][m].mean())) if cnt else None, meaning[c]])
    dh = ["ID", "Value", "Neighbour average (z)", "Local Moran's I", "p (pseudo, two-sided)",
          "Quadrant", "Cluster"]
    drows = []
    for i in range(n):
        drows.append([inp["ids"][i], rc._g(float(inp["values"][i])), rc._g(float(r["lag"][i]), 3),
                      rc._g(float(r["I"][i]), 3), rc._g(float(r["p"][i]), 3), str(r["quad"][i]),
                      str(r["label"][i])])
    return sh, rows, dh, drows


def lisa_report(inp, r):
    n = len(inp["values"])
    c = r["counts"]
    L = ["", "LOCAL MORAN'S I (LISA): %s" % inp["label"],
         "Weights: %d nearest neighbours, row-standardised. Inference: %d conditional permutations, "
         "two-sided pseudo p." % (r["k"], r["perms"]),
         "Significance: %s" % (
             "false discovery rate controlled at %.2f (Benjamini-Hochberg); a cluster counts when "
             "p <= %.4g" % (r["alpha"], r["p_crit"]) if r["fdr"] and r["p_crit"] > 0 else
             ("false discovery rate controlled at %.2f: no ED passes the correction." % r["alpha"]
              if r["fdr"] else "uncorrected p <= %.3g (expect about %.0f%% false positives)"
              % (r["alpha"], 100 * r["alpha"]))),
         "Global Moran's I = %.3f (z = %.2f, pseudo-p = %.3g): %s" % (
             r["global_perm_I"], r["global_z"], r["global_p"],
             "positive spatial autocorrelation: similar values cluster."
             if r["global_p"] < 0.05 and r["global_perm_I"] > 0 else
             "negative spatial autocorrelation: neighbours tend to differ."
             if r["global_p"] < 0.05 else "no significant global pattern."),
         "",
         "  %-38s %6s %8s" % ("Cluster type", "EDs", "% of EDs")]
    for code in ("HH", "LL", "HL", "LH", "NS"):
        L.append("  %-38s %6d %7.1f%%" % (sm.CLUSTER_LABELS[code], c[code], 100.0 * c[code] / n))
    resid = inp["label"] == "OLS residuals"
    L += ["", "How to read it:",
          "  - High-High / Low-Low are clusters of similar values (hot / cold spots).",
          "  - High-Low / Low-High are spatial outliers that differ from their surroundings."]
    if resid:
        L.append("  - For residuals: High-High = access consistently ABOVE what the model predicts; "
                 "Low-Low = consistently BELOW. Such pockets point to missing explanatory factors.")
    else:
        L.append("  - For accessibility: Low-Low pockets are the priority areas of poor access; "
                 "High-Low outliers are served places inside under-served areas.")
    return "\n".join(L)


def lisa_layer(base, inp, r, name=None):
    lyr = QgsVectorLayer("MultiPolygon?crs=%s" % (base.crs().authid() or "EPSG:4326"),
                         name or "LISA: %s" % inp["label"], "memory")
    lyr.setCrs(base.crs())
    dp = lyr.dataProvider()
    dp.addAttributes([QgsField("lisa_val", QMetaType.Type.Double),
                      QgsField("lisa_zlag", QMetaType.Type.Double),
                      QgsField("lisa_I", QMetaType.Type.Double),
                      QgsField("lisa_p", QMetaType.Type.Double),
                      QgsField("lisa_cl", QMetaType.Type.QString),
                      QgsField("lisa_lab", QMetaType.Type.QString)])
    lyr.updateFields()
    feats = []
    for i, fid in enumerate(inp["fids"]):
        src = base.getFeature(fid)
        g = QgsGeometry(src.geometry())
        g.convertToMultiType()
        nf = QgsFeature(lyr.fields())
        nf.setGeometry(g)
        nf.setAttributes([float(inp["values"][i]), float(r["lag"][i]), float(r["I"][i]),
                          float(r["p"][i]), str(r["code"][i]), str(r["label"][i])])
        feats.append(nf)
    dp.addFeatures(feats)
    lyr.updateExtents()
    style_lisa(lyr)
    return lyr


def style_lisa(layer):
    def fill(c):
        return QgsFillSymbol.createSimple({"color": c, "outline_color": "#ffffff",
                                           "outline_width": "0.1"})
    root = QgsRuleBasedRenderer.Rule(None)
    for code, color in (("HH", "#d7191c"), ("LL", "#2c7bb6"), ("HL", "#fdae61"),
                        ("LH", "#abd9e9"), ("NS", "#eeeeee")):
        root.appendChild(QgsRuleBasedRenderer.Rule(fill(color), 0, 0, '"lisa_cl" = \'%s\'' % code,
                                                   sm.CLUSTER_LABELS[code]))
    layer.setRenderer(QgsRuleBasedRenderer(root))
    layer.triggerRepaint()
