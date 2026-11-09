"""Reporting revisions of NHSN admissions (backfill): an origin-safe revision bank, revision augmentation of
finalized history, and a learned last-week nowcast. Every function uses only vintages released by its time stamp.

Revision of week w first released in vintage b, seen from the latest vintage T:  r = log1p(V_T[w]) - log1p(V_b[w]).
r > 0 means the first release was too low. A first release that is missing is stored as +inf."""
import numpy as np, pandas as pd

LAGS = 3            # weeks before the vintage's last week that are still revised (l = 0..LAGS-1)
GAP = 2             # a vintage enters the bank only if it is at least GAP weeks older than T


def _vint_at(p, t):
    """Latest vintage released by date t (at most 15 days old) or None."""
    a = p.vdates[(p.vdates <= t) & (p.vdates > t - pd.Timedelta(days=15))]
    return a[-1] if len(a) else None


class Bank:
    """Per-location empirical revisions at lags 0..LAGS-1, from vintages released by T."""

    def __init__(self, p, T):
        self.T = pd.Timestamp(T); self.locs = p.locs
        a = _vint_at(p, self.T)
        self.r = {l: [[] for _ in self.locs] for l in range(LAGS)}
        self.n = 0
        if a is None: return
        cur = p.vint[a].reindex(columns=self.locs)
        for b in p.vdates[p.vdates <= a - pd.Timedelta(weeks=GAP)]:
            Vb = p.vint[b].reindex(columns=self.locs); wb = Vb.index.max(); self.n += 1
            for l in range(LAGS):
                w = wb - pd.Timedelta(weeks=l)
                if w not in Vb.index or w not in cur.index: continue
                x, y = Vb.loc[w].to_numpy(float), cur.loc[w].to_numpy(float)
                for i in range(len(self.locs)):
                    if not np.isfinite(y[i]): continue
                    self.r[l][i].append(np.inf if not np.isfinite(x[i]) else np.log1p(max(y[i], 0)) - np.log1p(max(x[i], 0)))
        self.sorted = {l: [np.sort(np.array(v)) for v in self.r[l]] for l in range(LAGS)}
        self.pool = {l: np.sort(np.concatenate([v for v in self.sorted[l] if len(v)] or [np.zeros(1)])) for l in range(LAGS)}

    def draw(self, i, l, u, min_n=6):
        v = self.sorted[l][i]
        if len(v) < min_n: v = self.pool[l]
        return v[min(int(u * len(v)), len(v) - 1)]

    def augment(self, V, u):
        """Finalized panel V (weeks <= s) -> a plausible first-release panel. u: per-location uniforms (one draw per
        location shared across lags, so a late-reporting state is late at every lag)."""
        if self.n == 0: return V
        V = V.copy(); s = V.index.max()
        for l in range(LAGS):
            w = s - pd.Timedelta(weeks=l)
            if w not in V.index: continue
            x = V.loc[w].to_numpy(float).copy()
            for i in range(len(self.locs)):
                r = self.draw(i, l, u[i])
                x[i] = np.nan if not np.isfinite(r) else np.expm1(max(np.log1p(max(x[i], 0)) - r, 0.0)) if np.isfinite(x[i]) else np.nan
            V.loc[w] = x
        return V


class LastWeekNowcast:
    """Learned completion of the last reported week (pooled Huber regression across locations).
    Target: settled log value of the vintage's last week minus its lag-1 value; features: first-release growth
    (lag 0 vs 1, lag 1 vs 2), national first-release growth, the location's median past revision, a missing flag.
    Trained at each origin on vintages >= GAP weeks old, compared with the current vintage."""

    def __init__(self, p, K=20):
        self.p, self.K = p, K

    def _X(self, V, w, hist):
        y = np.log1p(V.clip(lower=0))
        v0 = y.loc[w].to_numpy(); v1 = y.loc[w - pd.Timedelta(weeks=1)].to_numpy(); v2 = y.loc[w - pd.Timedelta(weeks=2)].to_numpy()
        n0 = np.log(np.nansum(np.expm1(v0)) + 1); n1 = np.log(np.nansum(np.expm1(v1)) + 1)
        return np.nan_to_num(np.c_[v0 - v1, v1 - v2, np.full(len(v0), n0 - n1), hist, np.isnan(v0)]), v0, v1

    def apply(self, s, V):
        from sklearn.linear_model import HuberRegressor
        p = self.p; s = pd.Timestamp(s); a = _vint_at(p, s + pd.Timedelta(days=4))
        if a is None or s not in V.index: return V
        cur = p.vint[a].reindex(columns=p.locs)
        R, pairs = [], []
        for b in p.vdates[p.vdates <= a - pd.Timedelta(weeks=GAP)][-self.K:]:
            w = p.vint[b].index.max()
            if w not in cur.index: continue
            Vb = p.asof(w)[0]
            R.append(np.log1p(cur.loc[w].clip(lower=0)) - np.log1p(Vb.loc[w].clip(lower=0))); pairs.append((w, Vb))
        if len(pairs) < 4: return V
        hist = np.nan_to_num(np.nanmedian(np.array(R, float), 0))
        Xt, yt = [], []
        for w, Vb in pairs:
            X, v0, v1 = self._X(Vb, w, hist); t = np.log1p(cur.loc[w].clip(lower=0).to_numpy(float)) - v1
            ok = np.isfinite(t) & np.isfinite(v1); Xt.append(X[ok]); yt.append(t[ok])
        m = HuberRegressor(alpha=1e-2, max_iter=500).fit(np.vstack(Xt), np.concatenate(yt))
        X, v0, v1 = self._X(V, s, hist)
        V = V.copy(); V.loc[s] = np.where(np.isfinite(v1), np.expm1(v1 + m.predict(X)), V.loc[s].to_numpy(float))
        return V
