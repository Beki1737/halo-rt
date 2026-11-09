"""HALO-lite: a gated residual corrector on top of a frozen backbone, used in development to test WHICH form of
lag information carries signal beyond a strong forecaster. Feature families (all computed from data <= s):
  own   : the location's own weekly log growth (lags 0..3) and drawdown from its 8-week maximum
  coll  : collective wave: national log growth (lags 0..3), national drawdown, cross-location mean/std growth
  ed    : the location's ED-visit share: log growth (lags 0..2) and the phase gap z(ED) - z(admissions)
  gap   : collective phase gap z(national) - z(location) over a trailing 26-week window
  pair  : pairwise fixed-lag leaders in the TNNLS / CellAtt-LSTM style (top-5 other locations by training-window
          level correlation at lags 1..4 weeks, frozen at each refit; leader change ending at min(s, s+h-lag))
The corrector predicts the backbone's log-median error per horizon (pooled ridge across locations), refits every
4 origins on resolved origins only, and a skill gate switches it off when it did not help on the last 8 resolved
origins. Quantiles are shifted in log space."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from covidfc.trackp.metrics import QS, QC
from covidfc.trackp.data import H

FAMILIES = {"own": ["own"], "own+coll": ["own", "coll"], "own+coll+ed": ["own", "coll", "ed"],
            "own+coll+ed+gap": ["own", "coll", "ed", "gap"], "own+pair": ["own", "pair"]}


def _z(x, w=26):
    v = x[-w:]
    v = v[np.isfinite(v)]
    return (x[-1] - v.mean()) / (v.std() + 1e-6) if len(v) > 4 and np.isfinite(x[-1]) else 0.0


class Features:
    def __init__(self, Vh, Ved):
        self.loc = list(Vh.columns)
        Y = np.log1p(Vh.clip(lower=0)).ffill()
        self.y = Y.to_numpy(); self.g = Y.diff().to_numpy()
        tot = np.log1p(Vh.clip(lower=0).ffill().sum(1)).to_numpy(); self.nat = tot; self.gn = np.r_[np.nan, np.diff(tot)]
        if Ved is not None:
            E = np.log(Ved.reindex(index=Vh.index, columns=self.loc).clip(lower=1e-4)).ffill(limit=2)
            self.e = E.to_numpy(); self.ge = E.diff().to_numpy()
        else:
            self.e = self.ge = None

    def block(self, fam, i, h, pair_sel=None):
        y, g = self.y, self.g
        if fam == "own":
            return [g[-1 - k, i] for k in range(4)] + [y[-1, i] - np.nanmax(y[-9:, i])]
        if fam == "coll":
            oth = np.delete(g[-2:], i, axis=1)
            return [self.gn[-1 - k] for k in range(4)] + [self.nat[-1] - self.nat[-9:].max(),
                                                         np.nanmean(oth[-1]), np.nanstd(oth[-1]), np.nanmean(oth[-2])]
        if fam == "ed":
            if self.e is None or not np.isfinite(self.e[-1, i]):
                return [0.0, 0.0, 0.0, 0.0, 0.0]
            return [np.nan_to_num(self.ge[-1 - k, i]) for k in range(3)] + [_z(self.e[:, i]) - _z(y[:, i]), 1.0]
        if fam == "gap":
            return [_z(self.nat) - _z(y[:, i])]
        if fam == "pair":
            out = []
            for j, lag in (pair_sel or {}).get(self.loc[i], []):
                end = min(0, h - lag)                       # change of leader j ending at s + h - lag (<= s)
                a = len(y) - 1 + end
                out.append(y[a, j] - y[a - h, j] if a - h >= 0 else 0.0)
            return out + [0.0] * (5 - len(out))
        raise KeyError(fam)


def pair_select(Vh, maxlag=4, k=5):
    """TNNLS / CellAtt-style frozen pairwise lags from level correlation on all data <= s (origin-safe)."""
    Y = np.log1p(Vh.clip(lower=0)).ffill().to_numpy()
    loc = list(Vh.columns); sel = {}
    for i, l in enumerate(loc):
        best = []
        for j in range(len(loc)):
            if j == i: continue
            cs = [np.corrcoef(Y[maxlag:, i], Y[maxlag - L:len(Y) - L, j])[0, 1] for L in range(1, maxlag + 1)]
            L = int(np.nanargmax(cs)) + 1; best.append((np.nanmax(cs), j, L))
        best.sort(reverse=True); sel[l] = [(j, L) for _, j, L in best[:k]]
    return sel


def run(base, panels_at, families=FAMILIES, refit_every=4, gate_n=8, alpha=1.0, min_train_origins=20):
    """base: backbone forecasts (model, loc, s, h, QC). panels_at(s) -> (Vh, Ved) as available at s.
    Training targets are the realized values AS AVAILABLE at the refit origin (vintage-consistent), never the
    finalized values. Returns a DataFrame of corrected forecasts, model name 'HALO-lite[<family>]'."""
    base = base.sort_values(["s", "loc", "h"])
    S = sorted(base["s"].unique())
    out = []
    keys = [(f, h) for f in families for h in range(1, H + 1)]
    mem = {k: {"s": [], "li": [], "X": [], "med": [], "c": []} for k in keys}
    model = {k: None for k in keys}
    pair_sel = None
    for n, s in enumerate(S):
        Vh, Ved = panels_at(s)
        F = Features(Vh, Ved)
        Ylog = np.log1p(Vh.clip(lower=0)).to_numpy(); widx = {w: i for i, w in enumerate(Vh.index)}
        lidx = {l: i for i, l in enumerate(F.loc)}
        b = base[base["s"] == s]
        if any("pair" in v for v in families.values()) and (pair_sel is None or n % refit_every == 0):
            pair_sel = pair_select(Vh)
        for f, fams in families.items():
            for h in range(1, H + 1):
                key = (f, h); dt = pd.Timedelta(days=7 * h); M = mem[key]
                bh = b[b["h"] == h].set_index("loc").reindex(F.loc)
                med = np.log1p(bh["q0.5"].to_numpy(float))
                Xs = np.nan_to_num(np.array(
                    [[med[i] - F.y[-1, i]] + sum((F.block(fm, i, h, pair_sel) for fm in fams), [])
                     for i in range(len(F.loc))], float))
                # realized log values known at s for past records (vintage-consistent)
                if M["s"]:
                    s0 = np.array(M["s"]); li = np.array(M["li"])
                    tw = np.array([widx.get(x + dt, -1) for x in s0])
                    ok = (tw >= 0)
                    yr = np.full(len(s0), np.nan); yr[ok] = Ylog[tw[ok], li[ok]]
                    ok &= np.isfinite(yr)
                    Xr = np.array(M["X"])[ok]; er = (yr - np.array(M["med"]))[ok]; cr = np.array(M["c"])[ok]
                    sr = s0[ok]
                else:
                    sr = np.array([])
                if n % refit_every == 0 and len(np.unique(sr)) >= min_train_origins:
                    model[key] = Ridge(alpha=alpha).fit(Xr, er)
                c = model[key].predict(Xs) if model[key] is not None else np.full(len(F.loc), np.nan)
                gate = 0.0
                if len(sr):
                    recent = np.unique(sr)[-gate_n:]
                    m = np.isin(sr, recent) & np.isfinite(cr)
                    if len(recent) == gate_n and m.sum() >= gate_n * len(F.loc) // 2:
                        gate = 1.0 if np.mean(np.abs(er[m] - cr[m])) < np.mean(np.abs(er[m])) else 0.0
                Q = np.log1p(bh[QC].to_numpy(float)) + gate * np.nan_to_num(c)[:, None]
                Qc = np.sort(np.expm1(Q).clip(0), axis=1)
                for i, l in enumerate(F.loc):
                    out.append([f"HALO-lite[{f}]", l, s, h, *Qc[i], gate])
                M["s"] += [s] * len(F.loc); M["li"] += list(range(len(F.loc))); M["X"] += list(Xs)
                M["med"] += list(med); M["c"] += list(c)
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "gate"])
