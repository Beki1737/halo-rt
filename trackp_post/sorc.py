"""Simple online residual correction (SORC): the bias-correction baseline requested in review.
At origin s, for location i and horizon h, the correction is the mean log residual of the backbone over the last
`window` real-time origins whose label has settled (target week s' + h <= s - settle weeks):
    b_{i,s,h} = mean_{s'} [ log(1 + y^{(a_s)}_{i,s'+h}) - log(1 + median_{i,s',h}) ]
Labels are read from the same release as HALO-RT's labels (the latest release available at s), so SORC faces exactly
the same settlement delay and real-time information constraint. The correction is clipped to +-clip (HALO-RT uses the
same clip) and applied as a shift of all quantiles in log(1 + y) space. With `anchor_rho`, only the h = 1 bias is
used and applied at horizon h scaled by anchor_rho ** (h - 1), i.e. HALO-RT's horizon anchoring without features.
No features, no learning, no gate."""
import numpy as np, pandas as pd
from covidfc.trackp.metrics import QC


def run(base, origins, labels, name, window=8, settle=2, anchor_rho=None, clip=0.5, min_n=2):
    """base: backbone forecasts with columns loc, s, h and QC; origins: ordered real-time origins;
    labels(s): DataFrame (weeks x locations) of the values available at origin s."""
    b = base.set_index(["s", "h", "loc"]).sort_index()
    have = set(zip(base["s"], base["h"]))
    out = []
    for k, s in enumerate(origins):
        Yl = np.log1p(labels(s).clip(lower=0)); locs = list(Yl.columns)
        memo = {}

        def bias(h):
            if h in memo: return memo[h]
            past = [x for x in origins[:k] if (x, h) in have and x + pd.Timedelta(weeks=h) <= s - pd.Timedelta(weeks=settle)]
            past = past[-window:]
            c = np.zeros(len(locs))
            if len(past) >= min_n:
                E = []
                for x in past:
                    t = x + pd.Timedelta(weeks=h)
                    y = Yl.loc[t].reindex(locs).to_numpy(float) if t in Yl.index else np.full(len(locs), np.nan)
                    m = np.log1p(b.loc[(x, h)].reindex(locs)["q0.5"].to_numpy(float))
                    E.append(y - m)
                E = np.array(E); ok = np.isfinite(E); n = ok.sum(0)
                c = np.where(n > 0, np.where(ok, E, 0.0).sum(0) / np.maximum(n, 1), 0.0)
            memo[h] = np.clip(c, -clip, clip)
            return memo[h]

        for h in range(1, 5):
            if (s, h) not in have: continue
            lq = np.log1p(b.loc[(s, h)].reindex(locs)[QC].to_numpy(float))
            c = bias(1) * anchor_rho ** (h - 1) if anchor_rho is not None else bias(h)
            Q = np.sort(np.expm1(lq + c[:, None]).clip(0), 1)
            for i, l in enumerate(locs):
                if np.isfinite(Q[i]).all(): out.append([name, l, s, h, *Q[i], c[i]])
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "corr"])
