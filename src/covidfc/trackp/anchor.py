"""Horizon anchoring and online width recalibration for real-time correction (HALO-RT).
Motivation (cell28): a corrector learned per horizon from settled real-time residuals helps at h = 1..2 and hurts at
h = 3..4, because a label for horizon h is at least h + settle weeks old when it can be used. The freshest residual
(h = 1) mostly measures the current level error of the backbone (in real time, largely reporting revisions and
recent dynamics), and a level error carries over to later horizons. HALO-RT therefore learns only the h = 1
correction c1 and applies rho_h * c1 at horizon h:
  fixed  : rho_h = rho ** (h - 1)
  learned: rho_h on a grid in [0, 1] minimizing mean |e_h - rho * c1| over the last G settled origins (origin-safe)
Width: log-quantile deviations from the median are scaled by k_in (25-75%) and k_out (other levels), chosen per
horizon on the last G settled origins by WIS (values as available at s)."""
import numpy as np, pandas as pd
from covidfc.trackp.metrics import QC, wis

RG = np.linspace(0, 1, 11)
KG = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5])
INNER = np.array([False, False, True, False, True, False, False])


def run(base, c1, origins, hosp, name, rho="learned", width=False, settle=2, G=8):
    """base: backbone forecasts (loc, s, h, QC); c1: Series indexed (loc, s) with the issued h = 1 correction."""
    b = base.set_index(["s", "h", "loc"]).sort_index(); out = []
    c1 = c1.copy(); c1.index = c1.index.set_names(["loc", "s"])
    mem = {}                                             # (s, h) -> (locs, lq_corrected_before_width, c1)
    kin, kout = {h: 1.0 for h in range(1, 5)}, {h: 1.0 for h in range(1, 5)}
    for k, s in enumerate(origins):
        Yl = np.log1p(hosp.asof(s)[0].clip(lower=0)); locs = list(Yl.columns)
        cs = np.nan_to_num(c1.reindex(pd.MultiIndex.from_product([locs, [s]])).to_numpy(float))
        for h in range(1, 5):
            if (s, h) not in b.index: continue
            lq = np.log1p(b.loc[(s, h)].reindex(locs)[QC].to_numpy(float))
            past = [x for x in origins[:k] if (x, h) in mem and x + pd.Timedelta(weeks=h) <= s - pd.Timedelta(weeks=settle)][-G:]
            def ytrue(x):
                t = x + pd.Timedelta(weeks=h)
                return Yl.loc[t].reindex(locs).to_numpy(float) if t in Yl.index else np.full(len(locs), np.nan)
            if h == 1: r = 1.0
            elif rho == "learned":
                r = 0.0
                if len(past) >= 4:
                    e = np.concatenate([ytrue(x) - mem[(x, h)][0][:, 3] for x in past])
                    c = np.concatenate([mem[(x, h)][1] for x in past]); ok = np.isfinite(e)
                    r = RG[int(np.argmin([np.abs(e[ok] - q * c[ok]).mean() for q in RG]))]
            else:
                r = float(rho) ** (h - 1)
            lqc = lq + r * cs[:, None]
            if width and len(past) >= 4:
                P = np.vstack([mem[(x, h)][2] for x in past]); y = np.concatenate([ytrue(x) for x in past]); ok = np.isfinite(y)
                med = P[ok, 3:4]; dev = P[ok] - med; yt = np.expm1(y[ok])
                best, bk = np.inf, (1.0, 1.0)
                for a in KG:
                    for o in KG:
                        sc = np.where(INNER, a, o)
                        v = wis(np.sort(np.expm1(med + sc * dev).clip(0), 1), yt).mean()
                        if v < best: best, bk = v, (a, o)
                kin[h], kout[h] = bk
            mem[(s, h)] = (lq, cs, lqc)
            med = lqc[:, 3:4]; sc = np.where(INNER, kin[h], kout[h]) if width else 1.0
            Q = np.sort(np.expm1(med + sc * (lqc - med)).clip(0), 1)
            for i, l in enumerate(locs):
                out.append([name, l, s, h, *Q[i], r, kin[h], kout[h]])
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "rho", "k_in", "k_out"])
