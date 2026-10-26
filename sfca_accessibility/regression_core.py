# -*- coding: utf-8 -*-
"""
Statistics engine for the accessibility regression tool.

Needs only numpy (bundled with QGIS). No scipy: p-values use a pure-Python
regularised incomplete beta. No QGIS / Qt imports, so it is unit-testable.
"""
import math

import numpy as np


# --------------------------------------------------------------------------
# Distribution functions (Numerical Recipes continued fraction for I_x(a,b))
# --------------------------------------------------------------------------
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-14:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def f_sf(f, d1, d2):
    """P(F >= f) for an F(d1, d2) variable."""
    if f <= 0 or d1 <= 0 or d2 <= 0 or not math.isfinite(f):
        return float("nan")
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def t_crit(df, alpha=0.05):
    """Two-sided critical value by bisection (for 95% CIs)."""
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# OLS
# --------------------------------------------------------------------------
def ols(y, X, names, robust=False):
    """OLS with intercept. X: (n, k) predictors WITHOUT the constant.

    robust=True uses HC3 standard errors (heteroskedasticity-consistent).
    Returns a dict of results.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    p = k + 1
    if n <= p:
        raise ValueError("Not enough observations (%d) for %d parameters." % (n, p))
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("The predictors are perfectly collinear (or one is constant). "
                         "Remove a predictor or a dummy category.")
    XtX_inv = np.linalg.inv(A.T @ A)
    beta = XtX_inv @ A.T @ y
    fitted = A @ beta
    resid = y - fitted
    df = n - p
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / df if sst > 0 else float("nan")
    sigma2 = sse / df

    if robust:
        h = np.einsum("ij,jk,ik->i", A, XtX_inv, A)           # leverage
        w = resid ** 2 / np.clip(1.0 - h, 1e-12, None) ** 2   # HC3
        cov = XtX_inv @ (A.T * w) @ A @ XtX_inv
    else:
        cov = sigma2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tval = np.divide(beta, se, out=np.full(p, np.nan), where=se > 0)
    pval = [t_two_sided_p(t, df) for t in tval]
    tc = t_crit(df)

    # global F test (classical)
    if k > 0 and sst > 0:
        fstat = ((sst - sse) / k) / sigma2
        fp = f_sf(fstat, k, df)
    else:
        fstat, fp = float("nan"), float("nan")

    # standardised betas (slopes only)
    sd_y = float(np.std(y, ddof=1))
    std_beta = [float("nan")]
    for j in range(k):
        sx = float(np.std(X[:, j], ddof=1))
        std_beta.append(beta[j + 1] * sx / sd_y if sd_y > 0 else float("nan"))

    # VIF
    vif = [float("nan")]
    for j in range(k):
        if k == 1:
            vif.append(1.0)
            continue
        others = np.delete(X, j, axis=1)
        B = np.column_stack([np.ones(n), others])
        coef = np.linalg.lstsq(B, X[:, j], rcond=None)[0]
        r = X[:, j] - B @ coef
        sst_j = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        r2j = 1.0 - float(r @ r) / sst_j if sst_j > 0 else float("nan")
        vif.append(1.0 / (1.0 - r2j) if r2j < 1 else float("inf"))

    # Breusch-Pagan (studentised / Koenker) for heteroskedasticity
    bp, bp_p = float("nan"), float("nan")
    if k > 0:
        u2 = resid ** 2
        g = np.linalg.lstsq(A, u2, rcond=None)[0]
        ss_u = float(((u2 - u2.mean()) ** 2).sum())
        if ss_u > 0:
            r2u = 1.0 - float(((u2 - A @ g) ** 2).sum()) / ss_u
            bp = n * r2u
            bp_p = chi2_sf(bp, k)

    # AIC / BIC (Gaussian)
    ll = -0.5 * n * (math.log(2 * math.pi * sse / n) + 1.0)
    return {
        "names": ["(Intercept)"] + list(names), "beta": beta.tolist(), "se": se.tolist(),
        "t": tval.tolist(), "p": pval, "ci_lo": (beta - tc * se).tolist(),
        "ci_hi": (beta + tc * se).tolist(), "std_beta": std_beta, "vif": vif,
        "n": n, "k": k, "df": df, "r2": r2, "adj_r2": adj, "F": fstat, "F_p": fp,
        "aic": 2 * p - 2 * ll, "bic": p * math.log(n) - 2 * ll,
        "bp": bp, "bp_p": bp_p, "robust": robust,
        "fitted": fitted, "resid": resid,
        "std_resid": resid / math.sqrt(sigma2) if sigma2 > 0 else resid,
    }


def chi2_sf(x, k):
    """Upper tail of chi-square via the regularised incomplete gamma (series/CF)."""
    if x <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:
        term = s = 1.0 / a
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


# --------------------------------------------------------------------------
# Spatial autocorrelation of residuals: Moran's I on k-nearest-neighbour weights
# --------------------------------------------------------------------------
def knn_indices(xy, k=8, chunk=500):
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    k = max(1, min(k, n - 1))
    out = np.empty((n, k), dtype=int)
    for s in range(0, n, chunk):
        blk = xy[s:s + chunk]
        d = ((blk[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        for r in range(len(blk)):
            d[r, s + r] = np.inf              # exclude self
        out[s:s + chunk] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return out


def morans_i(values, idx, permutations=499, seed=12345):
    """Moran's I with row-standardised KNN weights. Returns (I, z, pseudo_p)."""
    z = np.asarray(values, dtype=float)
    n = len(z)
    z = z - z.mean()
    denom = float(z @ z)
    if denom == 0 or n < 5:
        return float("nan"), float("nan"), float("nan")

    def stat(v):
        return float(v @ v[idx].mean(axis=1)) / denom if v is z else \
            float(v @ v[idx].mean(axis=1)) / float(v @ v)

    i_obs = stat(z)
    rng = np.random.default_rng(seed)
    sims = np.empty(permutations)
    for b in range(permutations):
        sims[b] = stat(rng.permutation(z))
    mu, sd = sims.mean(), sims.std(ddof=1)
    zsc = (i_obs - mu) / sd if sd > 0 else float("nan")
    p = (1 + int((np.abs(sims - mu) >= abs(i_obs - mu)).sum())) / (permutations + 1.0)
    return i_obs, zsc, p


# --------------------------------------------------------------------------
# Design-matrix helpers
# --------------------------------------------------------------------------
def dummy_code(values, prefix):
    """Treatment coding for a categorical column; reference = most frequent level.

    Returns (columns: list[np.ndarray], names: list[str], reference).
    """
    vals = ["" if v is None else str(v).strip() for v in values]
    levels = sorted(set(vals))
    counts = {lv: vals.count(lv) for lv in levels}
    ref = max(levels, key=lambda lv: counts[lv])
    cols, names = [], []
    for lv in levels:
        if lv == ref:
            continue
        cols.append(np.array([1.0 if v == lv else 0.0 for v in vals]))
        names.append("%s = %s" % (prefix, lv or "(blank)"))
    return cols, names, ref


def interpret(res, outcome_label, log_outcome):
    """Plain-language lines for each predictor."""
    lines = []
    for nm, b, p, sb in zip(res["names"][1:], res["beta"][1:], res["p"][1:],
                            res["std_beta"][1:]):
        sig = "significant" if p < 0.05 else "not statistically significant"
        direction = "higher" if b > 0 else "lower"
        if log_outcome:
            eff = "%.1f%% %s" % (abs(100 * (math.exp(b) - 1)), direction)
            msg = "%s: a one-unit increase is associated with %s %s (p=%.3g, %s)." % (
                nm, eff, outcome_label, p, sig)
        else:
            msg = "%s: a one-unit increase is associated with %.4g %s %s (p=%.3g, %s)." % (
                nm, abs(b), direction, outcome_label, p, sig)
        lines.append(msg)
    return lines


# --------------------------------------------------------------------------
# Interpretable summary tables (for display and CSV export)
# --------------------------------------------------------------------------
def _g(v, nd=4):
    """Round to nd significant figures, keeping NaN/None as None (clean CSV cells)."""
    if v is None:
        return None
    try:
        if v != v or math.isinf(v):
            return None
    except TypeError:
        return v
    return float("%.*g" % (nd, v))


def vif_flag(v):
    if v is None or v != v:
        return ""
    if v < 5:
        return "OK (< 5)"
    if v < 10:
        return "Moderate (5 to 10)"
    return "High (> 10)"


def strength_label(sb):
    """Rule-of-thumb size of a standardised beta (|beta| thresholds are conventions, not tests)."""
    if sb is None or sb != sb:
        return ""
    a = abs(sb)
    return "Negligible" if a < 0.1 else "Weak" if a < 0.3 else "Moderate" if a < 0.5 else "Strong"


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else \
        "." if p < 0.1 else "ns"


def summary_tables(res_c, res_r, outcome_label="access", log_outcome=False, moran=None):
    """Build the two tables shown in the regression window and saved to CSV.

    res_c / res_r : ols() results with classical and HC3 standard errors (same model)
    moran         : (I, z, p) of the residuals, optional
    Returns (coef_head, coef_rows, fit_head, fit_rows).
    """
    head = ["Variable", "Coefficient", "HC3 robust SE", "Robust t", "Robust p", "Sig.",
            "95% CI low (HC3)", "95% CI high (HC3)", "Std. beta", "Strength (|std. beta|)",
            "VIF", "Collinearity", "Classical SE", "Classical p", "Robust vs classical",
            "Interpretation"]
    rows = []
    for i, nm in enumerate(res_r["names"]):
        b, p_r, p_c = res_r["beta"][i], res_r["p"][i], res_c["p"][i]
        sig_r, sig_c = p_r < 0.05, p_c < 0.05
        if sig_r == sig_c:
            cmp_ = "Same conclusion"
        elif sig_r:
            cmp_ = "Significant only with HC3"
        else:
            cmp_ = "Significant only with classical SE"
        if i == 0:
            interp = "Baseline: predicted outcome when every predictor is 0 (or at its reference)."
            sb = vf = None
            stren = coll = ""
        else:
            sb, vf = res_r["std_beta"][i], res_r["vif"][i]
            stren, coll = strength_label(sb), vif_flag(vf)
            more = "higher" if b > 0 else "lower"
            if log_outcome:
                eff = "%.1f%% %s" % (abs(100.0 * (math.exp(b) - 1.0)), more)
            else:
                eff = "%s by %.4g" % (more, abs(b))
            if "=" in nm:
                core_ = "%s: %s %s than the reference group" % (nm, outcome_label, eff)
            else:
                core_ = "each 1-unit rise in %s goes with %s %s" % (nm, outcome_label, eff)
            interp = ("Significant (p < 0.05): " if sig_r else "Not significant (p >= 0.05): no clear "
                      "association; if real, ") + core_ + "."
        rows.append([nm, _g(b), _g(res_r["se"][i]), _g(res_r["t"][i], 3), _g(p_r, 3), _stars(p_r),
                     _g(res_r["ci_lo"][i]), _g(res_r["ci_hi"][i]), _g(sb, 3), stren,
                     _g(vf, 3), coll, _g(res_c["se"][i]), _g(p_c, 3), cmp_, interp])

    fh = ["Measure", "Value", "What it means"]
    r2, adj = res_r["r2"], res_r["adj_r2"]
    fr = [["Observations (EDs)", res_r["n"], "Rows used after matching and dropping missing values."]]
    fr.append(["R-squared", _g(r2, 3),
               "The predictors together explain %.0f%% of the variation in %s."
               % (100 * r2, outcome_label)])
    fr.append(["Adjusted R-squared", _g(adj, 3), "R-squared penalised for the number of predictors."])
    fr.append(["F-test (model as a whole)", "F(%d,%d) = %.2f, p = %.3g" % (
        res_r["k"], res_r["df"], res_r["F"], res_r["F_p"]),
        "The predictors jointly explain access better than chance." if res_r["F_p"] < 0.05
        else "No evidence the predictors jointly explain access."])
    fr.append(["AIC / BIC", "%.1f / %.1f" % (res_r["aic"], res_r["bic"]),
               "Lower is better when comparing models fitted to the same data."])
    bp, bpp = res_r["bp"], res_r["bp_p"]
    fr.append(["Heteroscedasticity (Breusch-Pagan, studentised)",
               "LM = %.2f, p = %.3g" % (bp, bpp),
               ("p < 0.05: error variance is not constant, so classical standard errors are unreliable. "
                "Rely on the HC3 robust errors shown above." if bpp < 0.05 else
                "No evidence of unequal variance; HC3 and classical errors should agree closely.")])
    fr.append(["Standard errors used for inference", "HC3 (MacKinnon-White)",
               "Heteroscedasticity-consistent; stays valid when error variance differs across EDs."])
    vifs = [(v, n_) for v, n_ in zip(res_r["vif"][1:], res_r["names"][1:]) if v == v]
    if vifs:
        v, n_ = max(vifs)
        fr.append(["Highest VIF", "%.2f (%s)" % (v, n_),
                   "Below 5 is fine; 5 to 10 is moderate and above 10 is high collinearity, which "
                   "makes individual coefficients unstable."])
    if moran is not None and moran[0] == moran[0]:
        mi, _mz, mp = moran
        fr.append(["Moran's I of residuals", "I = %.3f, p = %.3g" % (mi, mp),
                   ("Residuals are spatially clustered (I > 0): standard errors are too small even "
                    "with HC3. Use the spatial lag, spatial error or GWR options." if (mp < 0.05 and mi > 0)
                    else "No significant spatial clustering left in the residuals.")])
    if log_outcome:
        fr.append(["Outcome transformation", "ln(1 + y)",
                   "Coefficients are approximate % changes in access, not absolute changes."])
    return head, rows, fh, fr
