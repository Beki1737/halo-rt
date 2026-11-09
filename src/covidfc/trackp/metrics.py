"""Scores for Track P: MAE of the median, WIS (Bracher et al. 2021, K = 3 central intervals + median),
50% / 95% coverage; relative scores = geometric mean over locations of model / persistence."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

QS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
QC = [f"q{q}" for q in QS]
ALPHAS = (0.05, 0.2, 0.5)


def wis(Q, y):
    """Q: n x 7 quantiles (QS order), y: n truths."""
    med = Q[:, 3]
    s = 0.5 * np.abs(y - med)
    for k, a in enumerate(ALPHAS):
        lo, hi = Q[:, k], Q[:, 6 - k]
        iv = (hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)
        s = s + a / 2 * iv
    return s / (len(ALPHAS) + 0.5)


def score(F):
    """F: forecast frame with QC columns and 'truth'. Adds ae, wis, cov50, cov95."""
    Q = np.sort(F[QC].to_numpy(float), axis=1); y = F["truth"].to_numpy(float)
    F = F.copy()
    F["ae"] = np.abs(Q[:, 3] - y); F["wis"] = wis(Q, y)
    F["cov50"] = (Q[:, 2] <= y) & (y <= Q[:, 4]); F["cov95"] = (Q[:, 0] <= y) & (y <= Q[:, 6])
    return F


def relative(S, metric, ref="Persistence", by=("h",)):
    """Geometric mean over locations of (mean metric of model) / (mean metric of ref), on shared rows."""
    keys = ["loc", *by]
    m = S.groupby(["model", *keys])[metric].mean().unstack("model")
    out = {}
    for mod in m.columns:
        r = np.log((m[mod] + 1e-9) / (m[ref] + 1e-9)).dropna()
        out[mod] = r.groupby(level=list(range(1, len(keys)))).mean().pipe(np.exp) if len(by) else np.exp(r.mean())
    return pd.DataFrame(out)


def paired(S, a, b, metric="wis", hs=(1, 2, 3, 4)):
    """Wilcoxon across locations of log(mean metric a / mean metric b) per horizon; returns rows."""
    rows = []
    for h in hs:
        m = S[S["h"] == h].groupby(["model", "loc"])[metric].mean().unstack("model")[[a, b]].dropna()
        d = np.log((m[a] + 1e-9) / (m[b] + 1e-9))
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append({"h": h, "A": a, "B": b, "pct_change": 100 * (np.exp(d.median()) - 1), "p": p, "n": len(d)})
    return pd.DataFrame(rows)
