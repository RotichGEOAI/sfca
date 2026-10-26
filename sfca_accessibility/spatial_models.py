# -*- coding: utf-8 -*-
"""
Spatial regression engine: Lagrange-multiplier diagnostics, spatial lag (SAR) and spatial
error (SEM) models by maximum likelihood, and geographically weighted regression (GWR).

numpy only (no scipy / libpysal) and no QGIS imports, so it is unit-testable.
Spatial weights are row-standardised k-nearest-neighbour weights on point coordinates.

References: Anselin (1988) Spatial Econometrics; Anselin, Bera, Florax & Yoon (1996)
Simple diagnostic tests for spatial dependence; Fotheringham, Brunsdon & Charlton (2002)
Geographically Weighted Regression; Barry & Pace (1999) for the log-determinant estimate.
"""
import math

import numpy as np

try:
    from . import regression_core as rc
except ImportError:                      # running from tests (flat import path)
    import regression_core as rc


class Cancelled(Exception):
    """Raised when the progress callback asks to stop."""


def _tick(progress, frac, msg):
    if progress is not None and progress(frac, msg) is False:
        raise Cancelled()


def norm_p(z):
    """Two-sided normal p-value."""
    return math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else float("nan")


# --------------------------------------------------------------------------
# Spatial weights
# --------------------------------------------------------------------------
class KnnWeights:
    """Row-standardised k-nearest-neighbour weights (each row sums to 1)."""

    def __init__(self, xy, k=8):
        self.xy = np.asarray(xy, dtype=float)
        self.n = len(self.xy)
        self.idx = rc.knn_indices(self.xy, k)
        self.k = self.idx.shape[1]

    def wy(self, v):
        """W @ v for a vector (n,) or a matrix (n, p)."""
        return v[self.idx].mean(axis=1)

    def dense(self):
        W = np.zeros((self.n, self.n))
        rows = np.repeat(np.arange(self.n), self.k)
        W[rows, self.idx.ravel()] = 1.0 / self.k
        return W

    def trace_w2(self):
        """tr(W W): counts mutual neighbour pairs (exact, no dense matrix)."""
        n = self.n
        keys = (np.repeat(np.arange(n), self.k).astype(np.int64) * n + self.idx.ravel())
        rev = self.idx.ravel().astype(np.int64) * n + np.repeat(np.arange(n), self.k)
        return float(np.isin(rev, keys).sum()) / (self.k ** 2)

    def trace_t(self):
        """T = tr(W'W + W W) used by the LM tests; tr(W'W) = n / k for KNN weights."""
        return self.trace_w2() + self.n / float(self.k)


class LogDet:
    """ln|I - rho W| for any rho: exact via eigenvalues (n <= exact_limit) or the
    Barry-Pace Monte Carlo trace series for large n."""

    def __init__(self, W, exact_limit=2500, order=40, probes=30, seed=7, progress=None):
        self.n = W.n
        self.exact = W.n <= exact_limit
        if self.exact:
            _tick(progress, 0.0, "Eigenvalues of the weights matrix (n = %d)…" % W.n)
            self.eig = np.linalg.eigvals(W.dense())
        else:
            rng = np.random.default_rng(seed)
            t = np.zeros(order + 1)
            acc = np.zeros(order + 1)
            for p in range(probes):
                z = rng.choice([-1.0, 1.0], size=W.n)
                v = z.copy()
                for k in range(1, order + 1):
                    v = W.wy(v)
                    acc[k] += float(z @ v)
                _tick(progress, p / float(probes), "Estimating log-determinant…")
            t = acc / probes
            t[1] = 0.0                       # tr(W) = 0 (no self-neighbours), exact
            t[2] = W.trace_w2()              # exact
            self.t = t
            self.order = order

    def ld(self, rho):
        if self.exact:
            return float(np.log(np.abs(1.0 - rho * self.eig)).sum())
        ks = np.arange(1, self.order + 1)
        return float(-(rho ** ks * self.t[1:] / ks).sum())

    def d2(self, rho):
        """Second derivative of ln|I - rho W| with respect to rho."""
        if self.exact:
            lam = self.eig
            return float(-np.real(lam ** 2 / (1.0 - rho * lam) ** 2).sum())
        ks = np.arange(2, self.order + 1)
        return float(-((ks - 1) * rho ** (ks - 2) * self.t[2:]).sum())


def _maximize(f, lo, hi, grid=41, iters=60):
    xs = np.linspace(lo, hi, grid)
    vals = [f(x) for x in xs]
    j = int(np.argmax(vals))
    a, b = xs[max(j - 1, 0)], xs[min(j + 1, grid - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _design(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def _ll_gauss(sse, n):
    return -0.5 * n * (math.log(2.0 * math.pi * sse / n) + 1.0)


# --------------------------------------------------------------------------
# LM diagnostics on OLS residuals
# --------------------------------------------------------------------------
def lm_tests(y, X, W):
    """LM-error, LM-lag and their robust versions (Anselin et al. 1996).
    Returns dict {name: (statistic, p)} and a recommendation string."""
    A = _design(X)
    n = len(y)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b
    s2 = float(e @ e) / n
    We, Wy = W.wy(e), W.wy(y)
    T = W.trace_t()
    WXb = W.wy(A @ b)
    # M WXb : residual of WXb after regressing on A
    mw = WXb - A @ np.linalg.lstsq(A, WXb, rcond=None)[0]
    nJ = float(WXb @ mw) / s2 + T
    d_err = float(e @ We) / s2
    d_lag = float(e @ Wy) / s2
    out = {}
    out["LM-error"] = d_err ** 2 / T
    out["LM-lag"] = d_lag ** 2 / nJ
    den_e = T - T * T / nJ
    out["Robust LM-error"] = (d_err - T / nJ * d_lag) ** 2 / den_e if den_e > 0 else float("nan")
    den_l = nJ - T
    out["Robust LM-lag"] = (d_lag - d_err) ** 2 / den_l if den_l > 0 else float("nan")
    res = {k: (v, rc.chi2_sf(v, 1) if math.isfinite(v) else float("nan")) for k, v in out.items()}

    pl, pe = res["LM-lag"][1], res["LM-error"][1]
    rl, re_ = res["Robust LM-lag"][1], res["Robust LM-error"][1]
    if pl >= 0.05 and pe >= 0.05:
        rec = "Neither LM test is significant: no strong evidence of spatial dependence; OLS is adequate."
    elif pl < 0.05 and pe >= 0.05:
        rec = "Only LM-lag is significant: prefer the spatial lag model."
    elif pe < 0.05 and pl >= 0.05:
        rec = "Only LM-error is significant: prefer the spatial error model."
    else:
        if rl < 0.05 and re_ >= 0.05:
            rec = "Both LM tests significant; only the robust lag test is: prefer the spatial lag model."
        elif re_ < 0.05 and rl >= 0.05:
            rec = "Both LM tests significant; only the robust error test is: prefer the spatial error model."
        elif rl < re_:
            rec = "Both robust tests significant; the lag test is stronger: prefer the spatial lag model."
        else:
            rec = "Both robust tests significant; the error test is stronger: prefer the spatial error model."
    return res, rec


# --------------------------------------------------------------------------
# Spatial lag model:  y = rho W y + X b + e
# --------------------------------------------------------------------------
def fit_lag(y, X, W, ld, names, perms=499, progress=None):
    y = np.asarray(y, dtype=float)
    A = _design(X)
    n, p = A.shape
    Wy = W.wy(y)
    b0 = np.linalg.lstsq(A, y, rcond=None)[0]
    bL = np.linalg.lstsq(A, Wy, rcond=None)[0]
    e0, eL = y - A @ b0, Wy - A @ bL

    def conc(rho):
        r = e0 - rho * eL
        return -0.5 * n * math.log(float(r @ r) / n) + ld.ld(rho)

    _tick(progress, 0.2, "Spatial lag: maximising likelihood…")
    rho = _maximize(conc, -0.99, 0.99)
    beta = b0 - rho * bL
    r = e0 - rho * eL
    s2 = float(r @ r) / n
    ll = _ll_gauss(float(r @ r), n) + ld.ld(rho)

    # observed information for theta = (beta, rho, sigma2)
    XtX = A.T @ A
    XtWy = A.T @ Wy
    WyWy = float(Wy @ Wy)
    q = p + 2
    H = np.zeros((q, q))
    H[:p, :p] = -XtX / s2
    H[:p, p] = H[p, :p] = -XtWy / s2
    H[p, p] = ld.d2(rho) - WyWy / s2
    H[:p, p + 1] = H[p + 1, :p] = -(A.T @ r) / s2 ** 2
    H[p, p + 1] = H[p + 1, p] = -float(Wy @ r) / s2 ** 2
    H[p + 1, p + 1] = n / (2.0 * s2 ** 2) - float(r @ r) / s2 ** 3
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    est = np.concatenate([beta, [rho]])
    sev = se[:p + 1]
    z = est / sev
    pv = [norm_p(v) for v in z]

    # fitted trend incl. spatial multiplier: (I - rho W)^-1 A beta by Neumann series
    xb = A @ beta
    yhat, term = xb.copy(), xb.copy()
    for _ in range(200):
        term = rho * W.wy(term)
        yhat += term
        if np.abs(term).max() < 1e-10:
            break
    r2 = float(np.corrcoef(y, yhat)[0, 1] ** 2)
    mi, mz, mp = rc.morans_i(r, W.idx, perms)
    k_par = p + 1
    return {
        "model": "Spatial lag", "names": ["(Intercept)"] + list(names) + ["rho (spatial lag)"],
        "coef": est.tolist(), "se": sev.tolist(), "z": z.tolist(), "p": pv,
        "total_effect": [float("nan")] + [b / (1.0 - rho) for b in beta[1:]] + [float("nan")],
        "rho": rho, "sigma2": s2, "loglik": ll, "aic": 2 * k_par - 2 * ll,
        "bic": k_par * math.log(n) - 2 * ll, "pseudo_r2": r2, "n": n,
        "resid": r, "fitted": yhat, "moran_i": mi, "moran_z": mz, "moran_p": mp,
        "logdet_mode": "exact" if ld.exact else "Barry-Pace estimate",
    }


# --------------------------------------------------------------------------
# Spatial error model:  y = X b + u,  u = lambda W u + e
# --------------------------------------------------------------------------
def fit_error(y, X, W, ld, names, perms=499, progress=None):
    y = np.asarray(y, dtype=float)
    A = _design(X)
    n, p = A.shape
    Wy, WA = W.wy(y), W.wy(A)

    def solve(lam):
        ys, Xs = y - lam * Wy, A - lam * WA
        b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        return b, ys - Xs @ b, Xs

    def conc(lam):
        _b, r, _x = solve(lam)
        return -0.5 * n * math.log(float(r @ r) / n) + ld.ld(lam)

    _tick(progress, 0.6, "Spatial error: maximising likelihood…")
    lam = _maximize(conc, -0.99, 0.99)
    beta, r, Xs = solve(lam)
    s2 = float(r @ r) / n
    ll = _ll_gauss(float(r @ r), n) + ld.ld(lam)
    cov_b = s2 * np.linalg.inv(Xs.T @ Xs)
    se_b = np.sqrt(np.clip(np.diag(cov_b), 0, None))
    h = 1e-3
    lo, hi = max(lam - h, -0.995), min(lam + h, 0.995)
    mid = 0.5 * (lo + hi)
    d2 = (conc(hi) - 2.0 * conc(mid) + conc(lo)) / (0.5 * (hi - lo)) ** 2
    se_l = math.sqrt(-1.0 / d2) if d2 < 0 else float("nan")
    est = np.concatenate([beta, [lam]])
    sev = np.concatenate([se_b, [se_l]])
    z = est / sev
    pv = [norm_p(v) for v in z]
    fitted = A @ beta
    r2 = float(np.corrcoef(y, fitted)[0, 1] ** 2)
    mi, mz, mp = rc.morans_i(r, W.idx, perms)
    k_par = p + 1
    return {
        "model": "Spatial error", "names": ["(Intercept)"] + list(names) + ["lambda (spatial error)"],
        "coef": est.tolist(), "se": sev.tolist(), "z": z.tolist(), "p": pv,
        "total_effect": None, "lambda": lam, "sigma2": s2, "loglik": ll,
        "aic": 2 * k_par - 2 * ll, "bic": k_par * math.log(n) - 2 * ll,
        "pseudo_r2": r2, "n": n, "resid": r, "fitted": fitted,
        "moran_i": mi, "moran_z": mz, "moran_p": mp,
        "logdet_mode": "exact" if ld.exact else "Barry-Pace estimate",
    }


# --------------------------------------------------------------------------
# Geographically weighted regression (adaptive bisquare kernel)
# --------------------------------------------------------------------------
def _neighbours(xy, kmax):
    n = len(xy)
    nb = np.empty((n, kmax), dtype=np.int32)
    dist = np.empty((n, kmax))
    chunk = max(1, int(3e6 // n))
    for s in range(0, n, chunk):
        blk = xy[s:s + chunk]
        d2 = ((blk[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        for r in range(len(blk)):
            d2[r, s + r] = -1.0                      # self first
        order = np.argsort(d2, axis=1)[:, :kmax]
        nb[s:s + chunk] = order
        dist[s:s + chunk] = np.sqrt(np.clip(np.take_along_axis(d2, order, axis=1), 0, None))
    return nb, dist


def _solve_local(M, v):
    try:
        return np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M) @ v


def aicc_value(rss, n, tr_s):
    if n - 2 - tr_s <= 0:
        return float("inf")
    return n * math.log(2.0 * math.pi * rss / n) + n * (n + tr_s) / (n - 2.0 - tr_s)


def gwr(y, X, xy, names, bandwidth=None, progress=None, kmax=None):
    """Adaptive-bandwidth GWR. bandwidth = number of nearest neighbours (None = select by AICc)."""
    y = np.asarray(y, dtype=float)
    A = _design(X)
    xy = np.asarray(xy, dtype=float)
    n, p = A.shape
    kmin = max(2 * p + 10, 20)
    if n < kmin + 10:
        raise ValueError("Too few observations (%d) for GWR with %d parameters." % (n, p))
    kmax = min(n - 1, kmax or 2000)
    _tick(progress, 0.0, "GWR: ordering neighbours…")
    nb, dist = _neighbours(xy, kmax + 1 if kmax + 1 <= n else kmax)
    width = nb.shape[1]

    def fit_k(k, full=False):
        rss, tr_s = 0.0, 0.0
        if full:
            B = np.empty((n, p))
            SE = np.empty((n, p))
            LR2 = np.empty(n)
            fitted = np.empty(n)
            bad = 0
        for i in range(n):
            idx = nb[i, :k]
            d = dist[i, :k]
            h = dist[i, k] if k < width else dist[i, k - 1] * 1.0001
            h = h if h > 0 else 1.0
            w = (1.0 - (d / h) ** 2) ** 2
            Xi = A[idx]
            XtW = Xi.T * w
            M = XtW @ Xi
            b = _solve_local(M, XtW @ y[idx])
            yhat = float(A[i] @ b)
            rss += (y[i] - yhat) ** 2
            tr_s += float(A[i] @ _solve_local(M, A[i]))          # w_ii = 1 (self at distance 0)
            if full:
                fitted[i] = yhat
                B[i] = b
                res = y[idx] - Xi @ b
                sw = w.sum()
                ybar = float((w * y[idx]).sum() / sw)
                tss = float((w * (y[idx] - ybar) ** 2).sum())
                LR2[i] = 1.0 - float((w * res ** 2).sum()) / tss if tss > 0 else float("nan")
                try:
                    Minv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    Minv = np.linalg.pinv(M)
                    bad += 1
                C = Minv @ (XtW * w) @ Xi @ Minv                # (X'WX)^-1 X'W^2X (X'WX)^-1
                SE[i] = np.sqrt(np.clip(np.diag(C), 0, None))   # scaled by sigma2 below
        if full:
            return rss, tr_s, B, SE, LR2, fitted, bad
        return rss, tr_s

    cache = {}

    def score(k):
        if k not in cache:
            rss, tr_s = fit_k(k)
            cache[k] = (aicc_value(rss, n, tr_s), rss, tr_s)
        return cache[k][0]

    if bandwidth is None:
        lo, hi = kmin, min(kmax, n - 1)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = int(round(b - gr * (b - a))), int(round(a + gr * (b - a)))
        step = 0
        total = 14
        while b - a > 3 and step < total:
            _tick(progress, 0.1 + 0.7 * step / total, "GWR: selecting bandwidth (%d–%d)…" % (a, b))
            if score(c) < score(d):
                b, d = d, c
                c = int(round(b - gr * (b - a)))
            else:
                a, c = c, d
                d = int(round(a + gr * (b - a)))
            step += 1
        if not cache:
            score((lo + hi) // 2)
        bw = min(cache, key=lambda kk: cache[kk][0])
    else:
        bw = int(max(kmin, min(bandwidth, kmax)))
    _tick(progress, 0.85, "GWR: final fit with %d nearest neighbours…" % bw)
    rss, tr_s, B, SE, LR2, fitted, bad = fit_k(bw, full=True)
    sigma2 = rss / (n - tr_s)
    SE = SE * math.sqrt(sigma2)
    T = np.divide(B, SE, out=np.full_like(B, np.nan), where=SE > 0)
    sst = float(((y - y.mean()) ** 2).sum())
    r2g = 1.0 - rss / sst if sst > 0 else float("nan")
    ols_res = rc.ols(y, X, names)
    ols_aicc = aicc_value(float(ols_res["resid"] @ ols_res["resid"]), n, p)
    q = np.percentile(B, [0, 25, 50, 75, 100], axis=0)
    return {
        "model": "GWR", "names": ["(Intercept)"] + list(names), "bandwidth": bw, "kernel": "adaptive bisquare",
        "betas": B, "se": SE, "t": T, "local_r2": LR2, "fitted": fitted, "resid": y - fitted,
        "rss": rss, "tr_s": tr_s, "sigma2": sigma2, "r2": r2g,
        "adj_r2": 1.0 - (1.0 - r2g) * (n - 1) / (n - tr_s - 1) if sst > 0 else float("nan"),
        "aicc": aicc_value(rss, n, tr_s), "ols_aicc": ols_aicc, "n": n, "p": p,
        "beta_quantiles": q, "ols_beta": ols_res["beta"], "ols_se": ols_res["se"],
        "singular_local": bad, "searched": dict((k, v[0]) for k, v in sorted(cache.items())),
    }


# --------------------------------------------------------------------------
# Local Moran's I (LISA) with conditional permutation inference
# --------------------------------------------------------------------------
CLUSTER_LABELS = {
    "HH": "High-High (hot spot)", "LL": "Low-Low (cold spot)",
    "LH": "Low-High (low outlier among highs)", "HL": "High-Low (high outlier among lows)",
    "NS": "Not significant",
}


def local_moran(values, W, perms=999, alpha=0.05, fdr=True, seed=2024, progress=None):
    """Local Moran's I_i = z_i * sum_j w_ij z_j with z standardised (mean 0, SD 1).

    Inference: conditional permutation (z_i fixed, its k neighbours drawn at random without
    replacement from the other values), two-sided pseudo p. With fdr=True the Benjamini-Hochberg
    procedure controls the false discovery rate at `alpha` across all n tests.
    The mean of I_i equals the global Moran's I (row-standardised weights).
    """
    x = np.asarray(values, dtype=float)
    n, k = len(x), W.k
    if n < k + 3:
        raise ValueError("Too few observations (%d) for %d neighbours." % (n, k))
    sd = x.std()
    if sd == 0:
        raise ValueError("The variable is constant, so Local Moran's I is undefined.")
    z = (x - x.mean()) / sd
    lag = W.wy(z)
    I = z * lag
    rng = np.random.default_rng(seed)
    ge = np.zeros(n)
    le = np.zeros(n)
    rows = np.arange(n)[:, None]
    for b in range(perms):
        if b % 25 == 0:
            _tick(progress, 0.05 + 0.9 * b / perms, "Local Moran's I: permutation %d of %d…" % (b, perms))
        draw = rng.integers(0, n - 1, size=(n, k))
        while True:                                   # no repeated neighbour within a row
            s = np.sort(draw, axis=1)
            dup = (np.diff(s, axis=1) == 0).any(axis=1)
            if not dup.any():
                break
            draw[dup] = rng.integers(0, n - 1, size=(int(dup.sum()), k))
        draw = draw + (draw >= rows)                  # skip the focal ED itself
        sim = z * z[draw].mean(axis=1)
        ge += sim >= I
        le += sim <= I
    p_one = (np.minimum(ge, le) + 1.0) / (perms + 1.0)
    p = np.minimum(1.0, 2.0 * p_one)

    if fdr:
        order = np.sort(p)
        thresh = alpha * np.arange(1, n + 1) / n
        ok = np.nonzero(order <= thresh)[0]
        p_crit = float(order[ok.max()]) if len(ok) else 0.0
    else:
        p_crit = alpha
    sig = p <= p_crit if p_crit > 0 else np.zeros(n, bool)

    quad = np.where(z >= 0, np.where(lag >= 0, "HH", "HL"), np.where(lag >= 0, "LH", "LL"))
    code = np.where(sig, quad, "NS")
    counts = {c: int((code == c).sum()) for c in ("HH", "LL", "LH", "HL", "NS")}
    gi, gz, gp = rc.morans_i(x, W.idx, min(perms, 999))
    return {"x": x, "z": z, "lag": lag, "I": I, "p": p, "quad": quad, "code": code,
            "label": np.array([CLUSTER_LABELS[c] for c in code]), "sig": sig,
            "p_crit": p_crit, "alpha": alpha, "fdr": fdr, "counts": counts, "perms": perms,
            "global_I": float(I.mean()), "global_perm_I": gi, "global_z": gz, "global_p": gp}
