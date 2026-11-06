"""Real-time residual corrector (RT): learns only from real-time residuals, never from finalized history.
At origin s (vintage period), for each horizon h, a pooled ridge regression predicts the backbone's log-median error
from features computed on the data as available at s:
  backbone predicted change, own momentum, collective wave, ED, phase gap (halo_lite families) and, optionally,
  revision-state features (the location's median past first-release revision from the origin-safe Bank, a
  missing-last-week flag, and their interaction with the last observed growth).
Training rows: earlier real-time origins whose target week is at least `settle` weeks old at s (values from the
vintage at s), so labels are not themselves first releases. A soft gate alpha in [0, 1] is chosen on the last
`gate_n` settled origins with a model refitted without them (honest hold-out). Corrections are clipped."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import covidfc.trackp.revision as RV
from covidfc.trackp.halo_lite import Features
from covidfc.trackp.metrics import QC

FAM = ["own", "coll", "ed", "gap"]
AG = np.linspace(0, 1, 11)


def _feats(F, b, h, rev, fams=FAM):
    med = np.log1p(b["q0.5"].to_numpy(float))
    X = np.nan_to_num(np.array([[med[i] - F.y[-1, i]] + sum((F.block(f, i, h) for f in fams), [])
                                for i in range(len(F.loc))], float))
    if rev is not None:
        X = np.c_[X, rev[0], rev[1], rev[0] * np.nan_to_num(F.g[-1])]
    return X, med


def run(base, origins, hosp, ed, name, use_rev=True, settle=2, min_o=6, lam=5.0, gate_n=8, clip=0.5, fams=FAM):
    """base: backbone forecasts (loc, s, h, QC) covering `origins` (real-time origins in order)."""
    base = base.set_index(["s", "h", "loc"]).sort_index()
    rec, out = {}, []
    for k, s in enumerate(origins):
        Vh = hosp.asof(s)[0]; Ve = ed.asof(s)[0]; F = Features(Vh, Ve); locs = F.loc
        rev = None
        if use_rev:
            bank = RV.Bank(hosp, s + pd.Timedelta(days=4))
            r0 = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).sum() else 0.0 for v in bank.sorted[0]]) if bank.n else np.zeros(len(locs))
            rev = (np.nan_to_num(r0), Vh.iloc[-1].isna().to_numpy(float))
        Yl = np.log1p(Vh.clip(lower=0))
        for h in range(1, 5):
            if (s, h) not in base.index: continue
            b = base.loc[(s, h)].reindex(locs)
            X, med = _feats(F, b, h, rev, fams); rec[(s, h)] = (X, med)
            lq = np.log1p(b[QC].to_numpy(float))
            tr = [x for x in origins[:k] if (x, h) in rec and x + pd.Timedelta(weeks=h) <= s - pd.Timedelta(weeks=settle)]
            def xy(os_):
                Xs = np.vstack([rec[(x, h)][0] for x in os_])
                ys = np.concatenate([Yl.loc[x + pd.Timedelta(weeks=h)].reindex(locs).to_numpy(float) - rec[(x, h)][1]
                                     if x + pd.Timedelta(weeks=h) in Yl.index else np.full(len(locs), np.nan) for x in os_])
                ok = np.isfinite(ys); return Xs[ok], ys[ok]
            c = np.zeros(len(locs)); alpha = 0.0
            if len(tr) >= min_o:
                Xt, yt = xy(tr); c = np.clip(Ridge(alpha=lam).fit(Xt, yt).predict(X), -clip, clip)
                if len(tr) >= min_o + gate_n:
                    X2, y2 = xy(tr[:-gate_n]); Xg, yg = xy(tr[-gate_n:])
                    cg = np.clip(Ridge(alpha=lam).fit(X2, y2).predict(Xg), -clip, clip)
                    alpha = AG[int(np.argmin([np.abs(yg - a * cg).mean() for a in AG]))]
                else:
                    alpha = 0.5
            Q = np.sort(np.expm1(lq + alpha * c[:, None]).clip(0), 1)
            for i, l in enumerate(locs):
                out.append([name, l, s, h, *Q[i], alpha, alpha * c[i]])
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "alpha", "corr"])
