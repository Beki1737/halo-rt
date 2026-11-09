#!/usr/bin/env bash
# cell24: Track P (HALO) step 1, development only. VAL and TEST origins are never scored here.
#   1. writes src/covidfc/trackp/ (data, metrics, backbones, gate, halo_lite, dev24) and configs/trackp_decisions.md
#   2. syncs the CDC COVID-19 Forecast Hub (target data with weekly vintages, locations, reference models)
#   3. data audit, pairwise-lag evidence gate, backbones (Persistence, Chronos-2 uni / group / ED covariate)
#      on 178 DEV origins, HALO-lite corrector families on Chronos-2, scores and paired tests
#   Needs internet for the git sync (github.com) and one GPU. Expect about 30-60 minutes. Run in tmux.
#   Rerunnable: finished backbone and corrector files in results/trackp are reused.
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp data/trackp src/covidfc/trackp configs; exec > >(tee -a logs/cell24.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell24 start"

echo "== 1. Write the Track P package and the decision log"
cat > src/covidfc/trackp/__init__.py <<'PYEOF'
PYEOF
cat > src/covidfc/trackp/data.py <<'PYEOF'
"""Track P data: weekly NHSN COVID-19 admissions and NSSP ED-visit share for 50 states + DC,
with real-time vintages from the CDC COVID-19 Forecast Hub target data (as_of, weekly from 2024-11-20).

Conventions (pre-registered in configs/trackp_decisions.md):
  * week index = epiweek ending Saturday (target_end_date).
  * origin s = the last week whose data are available at forecast time (week ending the Saturday before the
    Hub reference date R, i.e. s = R - 7 days). Horizon h = 1..4 targets week s + 7h  (Hub horizons 0..3).
  * inputs at origin s: the vintage published on the Wednesday after s (as_of = s + 4 days) when such a vintage
    exists, otherwise the finalized series truncated at s (dev period; stated limitation).
  * truth for scoring: the latest vintage (finalized values).
"""
import os, subprocess
import numpy as np, pandas as pd

PROJ = os.environ.get("PROJ", ".")
HUB = os.environ.get("TP_HUB", f"{PROJ}/data/trackp/hub")
OUT = os.environ.get("TP_OUT", f"{PROJ}/results/trackp")
HOSP, ED = "wk inc covid hosp", "wk inc covid prop ed visits"
EXCLUDE = ("US", "72")
SPLITS = {   # origin s (week-ending date) ranges; targets of every origin lie inside the same period
    "DEV":  ("2020-11-07", "2024-03-30"),   # targets <= 2024-04-27 (before the voluntary-reporting interval)
    "VAL":  ("2024-11-16", "2025-05-31"),   # mandatory reporting, partly versioned; targets <= 2025-06-28
    "TEST": ("2025-06-28", "2026-09-19"),   # sealed: fully versioned; targets <= 2026-10-17
}
VOLUNTARY = ("2024-05-01", "2024-10-31")
H = 4


def sync_hub():
    """Sparse clone (or update) of the CDC COVID-19 Forecast Hub: target data, locations, two reference models."""
    paths = ["target-data", "auxiliary-data/locations.csv", "model-output/CovidHub-ensemble",
             "model-output/CovidHub-baseline", "model-output/UMass-gbqr", "model-output/UM-DeepOutbreak"]
    if not os.path.isdir(f"{HUB}/.git"):
        os.makedirs(os.path.dirname(HUB), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/cdcgov/covid19-forecast-hub.git", HUB], check=True)
    else:
        subprocess.run(["git", "-C", HUB, "pull", "--ff-only"], check=True)
    subprocess.run(["git", "-C", HUB, "sparse-checkout", "set", "--no-cone", *paths], check=True)
    return subprocess.check_output(["git", "-C", HUB, "rev-parse", "HEAD"]).decode().strip()


def load_raw():
    t = pd.read_parquet(f"{HUB}/target-data/time-series.parquet")
    t["target_end_date"] = pd.to_datetime(t["target_end_date"]); t["as_of"] = pd.to_datetime(t["as_of"])
    t["location"] = t["location"].astype(str)
    t = t[~t["location"].isin(EXCLUDE)]
    return t


class Panel:
    """Week x location arrays for one target, finalized values plus vintage access."""

    def __init__(self, raw, target):
        d = raw[raw["target"] == target].sort_values("as_of")
        self.target = target
        fin = d.groupby(["location", "target_end_date"]).tail(1)
        P = fin.pivot(index="target_end_date", columns="location", values="observation").sort_index()
        full = pd.date_range(P.index.min(), raw["target_end_date"].max(), freq="7D")
        self.final = P.reindex(full)
        self.locs = list(self.final.columns)
        self.weeks = self.final.index
        self.vint = {a: g.pivot(index="target_end_date", columns="location", values="observation")
                     for a, g in d.groupby("as_of")}
        self.vdates = np.array(sorted(self.vint))

    def asof(self, s):
        """Values available at origin s (week-ending date): the Wednesday vintage, else finalized truncated at s."""
        s = pd.Timestamp(s); wed = s + pd.Timedelta(days=4)
        ok = self.vdates[(self.vdates <= wed) & (self.vdates > wed - pd.Timedelta(days=15))]   # latest vintage, at most 2 weeks old
        V = self.final.loc[:s].copy(); src = "final"
        if len(ok):   # the vintage overrides every week it covers; older weeks (rarely revised) come from final
            W = self.vint[ok[-1]].reindex(columns=self.locs)
            W = W[W.index <= s]
            V.loc[W.index] = W.where(W.notna(), V.loc[W.index])
            # weeks inside the vintage's span that it does not report yet are unknown at the origin
            V.loc[V.index > W.index.max()] = np.nan
            src = "vintage"
        assert len(V) == 0 or V.index.max() <= s, "provenance: a value dated after the origin"
        return V, src


def pop():
    loc = pd.read_csv(f"{HUB}/auxiliary-data/locations.csv", dtype={"location": str})
    return loc.set_index("location")["population"].astype(float)


def origins(split, panel=None):
    """Origins of a split. For VAL / TEST (versioned periods) an origin is kept only if a vintage at most two weeks
    old exists, which drops the Oct-Nov 2025 federal-shutdown weeks when no NHSN data were released and the Hub
    paused (reference dates 2025-10-04 to 2025-11-15)."""
    a, b = SPLITS[split]
    O = pd.date_range(a, b, freq="7D")
    if panel is not None and split != "DEV":
        O = pd.DatetimeIndex([s for s in O if panel.asof(s)[1] == "vintage"])
    return O


def audit(hosp, ed):
    """Coverage, reporting dropouts, vintage availability and revision size; returns a dict printed by the cell."""
    A = hosp.final
    rep = {}
    rep["weeks"] = (str(A.index.min().date()), str(A.index.max().date()), len(A))
    rep["locations"] = len(hosp.locs)
    vol = A.loc[VOLUNTARY[0]:VOLUNTARY[1]]
    pre = A.loc["2024-01-01":"2024-04-27"].median()
    drop = (vol.median() < 0.2 * pre).sum()
    rep["voluntary_interval_dropout_states"] = int(drop)
    rep["hosp_vintages"] = (str(hosp.vdates.min().date()), str(hosp.vdates.max().date()), len(hosp.vdates))
    rep["ed_weeks"] = (str(ed.final.dropna(how="all").index.min().date()), str(ed.final.index.max().date()))
    rep["ed_vintages"] = (str(ed.vdates.min().date()), str(ed.vdates.max().date()), len(ed.vdates))
    # first-release vs final relative revision of admissions (versioned period)
    revs = []
    for a in hosp.vdates[1:]:
        V = hosp.vint[a]
        last = V.index.max()
        if last not in A.index: continue
        f, v = A.loc[last, V.columns], V.loc[last]
        m = (f > 20)
        revs.append(((f[m] - v[m]) / f[m]).to_numpy())
    r = np.concatenate(revs) if revs else np.array([np.nan])
    rep["first_release_revision_median_pct"] = round(100 * float(np.nanmedian(r)), 1)
    rep["first_release_revision_p90_pct"] = round(100 * float(np.nanpercentile(r, 90)), 1)
    for sp in SPLITS:
        O0, O = origins(sp), origins(sp, hosp)
        rep[f"{sp}_origins"] = (len(O), f"{len(O0) - len(O)} dropped (no vintage within 2 weeks)")
    return rep
PYEOF
cat > src/covidfc/trackp/metrics.py <<'PYEOF'
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
PYEOF
cat > src/covidfc/trackp/backbones.py <<'PYEOF'
"""Base forecasters for Track P. Every function takes the data available at origin s only.
Rows returned: model, loc, s, h, q0.025..q0.975 (counts)."""
import os, numpy as np, pandas as pd
from covidfc.trackp.metrics import QS, QC
from covidfc.trackp.data import H

W7 = pd.Timedelta(days=7)


def _rows(model, s, loc, Q):
    out = []
    for i, l in enumerate(loc):
        for h in range(1, H + 1):
            out.append([model, l, s, h, *np.sort(Q[i, h - 1])])
    return out


def persistence(s, V):
    """Last observed value; quantiles from the location's trailing 52-week h-step log changes, centred at 0."""
    Y = np.log1p(V.clip(lower=0).ffill())
    last = Y.iloc[-1].to_numpy()
    Q = np.zeros((len(V.columns), H, len(QS)))
    for h in range(1, H + 1):
        d = (Y - Y.shift(h)).iloc[-52:]
        d = d - d.median()
        qd = np.nanquantile(d.to_numpy(), QS, axis=0).T            # loc x 7
        qd = np.nan_to_num(qd, nan=0.0)
        Q[:, h - 1] = np.expm1(last[:, None] + qd)
    return _rows("Persistence", s, list(V.columns), np.clip(Q, 0, None))


_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        d = os.path.join(os.environ["PROJ"], "models", "chronos-2")
        _PIPE = Chronos2Pipeline.from_pretrained(d, device_map="cuda")
    return _PIPE


def chronos(s, V, mode="uni", Ved=None, ctx_weeks=260):
    """Chronos-2 zero-shot on raw weekly counts. mode: 'uni' (series independent), 'group' (cross-learning over the
    51 locations of this one origin), 'edcov' (own ED share as a past-only covariate)."""
    V = V.iloc[-ctx_weeks:]
    loc = list(V.columns)
    frames = []
    for l in loc:
        f = pd.DataFrame({"id": l, "timestamp": V.index, "target": V[l].to_numpy(float)})
        if mode == "edcov":
            e = Ved.reindex(V.index)[l].to_numpy(float) if (Ved is not None and l in Ved.columns) else np.full(len(V), np.nan)
            f["ed"] = e
        frames.append(f)
    C = pd.concat(frames, ignore_index=True)
    if mode == "edcov" and not np.isfinite(C["ed"].to_numpy(float)).any():
        C = C.drop(columns="ed")          # no ED data yet at this origin: identical to the univariate call
    R = _pipe().predict_df(C, id_column="id", timestamp_column="timestamp", target="target", prediction_length=H,
                           quantile_levels=list(QS), batch_size=64, cross_learning=(mode == "group"))
    R = R.sort_values(["id", "timestamp"])
    Q = np.zeros((len(loc), H, len(QS)))
    for i, l in enumerate(loc):
        r = R[R["id"] == l]
        Q[i] = r[[str(q) for q in QS]].to_numpy(float)[:H]
    return _rows({"uni": "Chronos2", "group": "Chronos2-group", "edcov": "Chronos2-EDcov"}[mode], s, loc,
                 np.clip(Q, 0, None))


def damped(s, V, phi=0.8):
    """Offline stand-in backbone for local testing (damped log-trend), same quantile recipe as persistence."""
    Y = np.log1p(V.clip(lower=0).ffill())
    g = (Y.iloc[-1] - Y.iloc[-2]).to_numpy()
    Q = np.zeros((len(V.columns), H, len(QS)))
    for h in range(1, H + 1):
        mu = Y.iloc[-1].to_numpy() + g * sum(phi ** k for k in range(1, h + 1))
        d = (Y - Y.shift(h)).iloc[-52:]; d = d - d.median()
        qd = np.nan_to_num(np.nanquantile(d.to_numpy(), QS, axis=0).T)
        Q[:, h - 1] = np.expm1(mu[:, None] + qd)
    return _rows("Damped", s, list(V.columns), np.clip(Q, 0, None))
PYEOF
cat > src/covidfc/trackp/gate.py <<'PYEOF'
"""Evidence gate for pairwise fixed lags (the TNNLS / CellAtt-LSTM premise), development data only.
For target location r, horizon h, leader j and lead l >= h (leader change observed at the origin):
    rho = corr( target change y_r(s+h) - y_r(s) - base_h(s) ,  leader change y_j(s+h-l) - y_j(s-l) )
base 'persist' = 0 (raw change), base 'trend' = h * last weekly growth (target's own momentum).
The maximum over leaders and lags is compared with a circular-shift permutation null (leaders shifted by
26..T-26 weeks, target untouched), so the selection over ~400 candidates is accounted for."""
import numpy as np, pandas as pd

H, LMAX = (1, 2, 3, 4), 8


def _corrmat(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-12); B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return A.T @ B / len(A)          # R x R : target x leader


def best_rho(y, ylead, base):
    T, R = y.shape; s = np.arange(LMAX + 2, T - max(H))
    best = np.full((R, len(H)), -np.inf); arg = np.zeros((R, len(H)), int)
    for hi, h in enumerate(H):
        tgt = y[s + h] - y[s]
        if base == "trend": tgt = tgt - h * (y[s] - y[s - 1])
        for l in range(h, LMAX + 1):
            C = _corrmat(tgt, ylead[s + h - l] - ylead[s - l]); np.fill_diagonal(C, -np.inf)
            m = C.max(1); upd = m > best[:, hi]; best[upd, hi] = m[upd]; arg[upd, hi] = l
    return best, arg


def run(final, start="2020-08-08", end="2024-04-27", perms=200, seed=2026):
    A = final.loc[start:end]
    y = np.log1p(A.clip(lower=0)).interpolate(limit=2).bfill().ffill().to_numpy()
    T, R = y.shape; rng = np.random.default_rng(seed); out = []
    for base in ("persist", "trend"):
        best, arg = best_rho(y, y, base)
        null = np.stack([best_rho(y, np.stack([np.roll(y[:, j], k) for j, k in
                                               enumerate(rng.integers(26, T - 26, R))], 1), base)[0]
                         for _ in range(perms)])
        p = (null >= best[None]).mean(0)
        for hi, h in enumerate(H):
            out.append({"base": base, "h": h, "median_best_rho": np.median(best[:, hi]),
                        "median_null_rho": np.median(null[:, :, hi]), "share_p<0.05": (p[:, hi] < 0.05).mean(),
                        "modal_lag": int(np.bincount(arg[:, hi]).argmax())})
    return pd.DataFrame(out)
PYEOF
cat > src/covidfc/trackp/halo_lite.py <<'PYEOF'
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
PYEOF
cat > src/covidfc/trackp/dev24.py <<'PYEOF'
"""cell24 development run (Track P). Never touches VAL or TEST origins.
  1. sync the CDC Forecast Hub target data and audit it
  2. evidence gate: do pairwise fixed lags carry information beyond the target's own momentum?
  3. backbones on DEV origins: Persistence, Chronos-2 (uni / group / ED covariate)  [or Damped offline]
  4. HALO-lite corrector families on the Chronos-2 backbone: which kind of lag information helps?
  5. scores by window and horizon, paired tests, gate usage
usage: python -m covidfc.trackp.dev24 [--offline]"""
import os, sys, json, time
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo_lite as HL
import covidfc.trackp.gate as GT
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT; os.makedirs(OUT, exist_ok=True)
WINDOWS = {"DEV-A (Delta/Omicron, no ED)": ("2021-06-05", "2022-10-29"),
           "DEV-B (2023-24, with ED)": ("2023-01-07", "2024-03-30")}
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if not OFF and not os.environ.get("TP_NOSYNC"):
    log("hub commit", TD.sync_hub())
raw = TD.load_raw()
hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
log("== 1. audit"); aud = TD.audit(hosp, ed)
for k, v in aud.items(): print(f"   {k}: {v}")
json.dump(aud, open(f"{OUT}/audit.json", "w"), indent=1, default=str)

log("== 2. evidence gate for pairwise fixed lags (DEV data only, 200 permutations)")
G = GT.run(hosp.final, perms=int(os.environ.get("TP_PERMS", 200)))
print(G.round(3).to_string(index=False)); G.to_csv(f"{OUT}/gate_pairwise.csv", index=False)

log("== 3. backbones on DEV origins")
O = TD.origins("DEV")
cache = {}
def panels_at(s):
    if s not in cache:
        Vh = hosp.asof(s)[0]; Ve = ed.asof(s)[0]
        cache[s] = (Vh, Ve)
    return cache[s]
fpath = f"{OUT}/dev24_backbones.parquet"
if os.path.exists(fpath):
    B = pd.read_parquet(fpath); log("loaded", fpath, B.shape)
else:
    rows = []
    for k, s in enumerate(O):
        Vh, Ve = panels_at(s)
        rows += BB.persistence(s, Vh)
        if OFF:
            rows += BB.damped(s, Vh)
        else:
            rows += BB.chronos(s, Vh, "uni") + BB.chronos(s, Vh, "group") + BB.chronos(s, Vh, "edcov", Ve)
        if k % 25 == 0: log(f"   origin {k + 1}/{len(O)} {s.date()}")
    B = pd.DataFrame(rows, columns=["model", "loc", "s", "h", *QC])
    B.to_parquet(fpath)
backbone = "Damped" if OFF else "Chronos2"

log(f"== 4. HALO-lite families on the {backbone} backbone")
hpath = f"{OUT}/dev24_halolite_{backbone}.parquet"
if os.path.exists(hpath):
    HLF = pd.read_parquet(hpath)
else:
    HLF = HL.run(B[B["model"] == backbone], panels_at)
    HLF.to_parquet(hpath)
print("   share of forecasts with the skill gate open, by family and h:")
print(HLF.groupby(["model", "h"])["gate"].mean().unstack("h").round(2).to_string())

log("== 5. scores (truth = finalized values)")
F = pd.concat([B, HLF.drop(columns="gate")], ignore_index=True)
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
F = F.merge(tr, on=["t", "loc"], how="inner")
S = score(F)
for wname, (a, b) in WINDOWS.items():
    W = S[(S["s"] >= a) & (S["s"] <= b)]
    print(f"\n--- {wname}: {W['s'].nunique()} origins x {W['loc'].nunique()} locations")
    for met in ("wis", "ae"):
        print(f"relative {met.upper()} vs persistence (geometric mean over locations), by horizon in weeks:")
        print(relative(W, met).T.round(3).sort_values(4).to_string())
    print("coverage (95%):", W.groupby("model")["cov95"].mean().round(2).to_dict())
    best = [m for m in W["model"].unique() if m.startswith("HALO-lite")]
    pt = pd.concat([paired(W, m, backbone) for m in best] +
                   ([paired(W, m, "Chronos2-group") for m in best] if not OFF else []), ignore_index=True)
    print("paired Wilcoxon across locations (WIS):"); print(pt.round(4).to_string(index=False))
    pt.assign(window=wname).to_csv(f"{OUT}/dev24_paired_{wname[:5]}.csv", index=False)
S.to_parquet(f"{OUT}/dev24_scores.parquet")
log("done")
PYEOF
cat > configs/trackp_decisions.md <<'MDEOF'
# Track P (HALO) decisions and pre-registered conventions

Recorded on 22 Oct 2026, before any VAL or TEST origin was scored. Prof. Jang has the proposal (D1 to D6) and has not answered yet. These are provisional working decisions; any change he asks for will be logged as an amendment.

## Working decisions (provisional, pending Prof. Jang)
- **D1 Target.** The primary target is weekly confirmed COVID-19 admissions (NHSN, via CDC COVID-19 Forecast Hub target data), 50 states plus DC. Daily cases 2020 to 2022 (Track S) remain a secondary benchmark.
- **D2 Leaders.**
  - Regional admissions of the other 50 jurisdictions.
  - NSSP ED-visit share for COVID-19 (state).
  - NWSS wastewater, if a state-level series can be built.
  - Google Trends is deferred to a sensitivity analysis because of access risk (alpha API; website rescaling).
- **D3 Method framing.** HALO is a backbone-agnostic corrector, with linear and neural versions; the neural version is primary if it is at least as good on VAL.
- **D4 Lab baselines.** TNNLS (origin-safe, verified on its influenza data), CellAtt-LSTM, and lead-aware Chronos-2 from the rerun.
- **D5 Venue.** IEEE TNNLS; IEEE JBHI is the fallback.
- **D6 People.** Beka asks Inhwan Kim about the TNNLS code. The author list is to be confirmed with Prof. Jang.

## Conventions
- Week = epiweek ending Saturday.
- Origin s = the last week available at forecast time (Hub reference date minus 7 days). h = 1..4 targets s + 7h, which corresponds to Hub horizons 0..3.
- Inputs at s:
  - the Wednesday vintage (as_of = s + 4 days) when one exists (hospital admissions from 2024-11-20, ED from 2025-06-18);
  - otherwise the finalized series truncated at s (DEV only; stated limitation).
- Training targets of any learned component are the realized values as available at the refit origin. Finalized values are never used for training.
- Truth for scoring is the latest vintage.
- Splits (origins s):
  - DEV: 2020-11-07 to 2024-03-30.
  - VAL: 2024-11-16 to 2025-05-31.
  - TEST (sealed): 2025-06-28 to 2026-09-19.
  - The voluntary-reporting interval (May to Oct 2024) is excluded: 26 states dropped out.
  - VAL and TEST origins need a vintage at most two weeks old. This drops 5 TEST origins: the Oct to Nov 2025 federal shutdown, when NHSN released no data and the Hub paused. TEST keeps 60 origins.
- Metrics:
  - WIS (primary) and MAE, each relative to persistence (geometric mean over locations);
  - 50% and 95% coverage;
  - Wilcoxon across locations with Holm correction, plus bootstrap intervals.
- TEST stays sealed until the git tag `trackp-lock` exists. The lock freezes code, features, hyperparameters and the hypothesis family.

## Development evidence (DEV only, local analysis of 22 Oct 2026)
- **E1. Pairwise fixed lags (the TNNLS / CellAtt premise) carry no information beyond the target's own momentum at weekly resolution.**
  - The best single leader and lag (l >= h) against a circular-shift null was significant in 0 of 51 states at every horizon, once the target's own trend is removed.
  - Against raw change it was significant in 96%, 98%, 86% and 49% of states. That signal is shared momentum, not lead information.
- **E2. Collective cross-regional information does carry signal.**
  - National growth added to own momentum lowered out-of-sample MAE by 9.5%, 12.4%, 13.2% and 13.3% at h = 1..4 (pooled ridge, rolling origin).
- **E3. Explicit phase alignment to the national wave adds little at weekly resolution** (0.5 to 1%).
  - 61% of state-weeks are in phase with the national wave; per-state mean lags range from 0.2 to 1.8 weeks.
- **E4. ED-visit share is a strong leading indicator.**
  - Own + ED lowered MAE by 19 to 23% relative to own momentum (2022-11 to 2024-04).
  - ED adds 9 to 11% beyond own + national.
  - Phase-gap encodings (z-scored leader minus target level) add 1 to 2% at h = 3..4.

## Hypothesis ladder (pivot rules)
- **H-A (primary).** Collective and exogenous lead information, used through a gated corrector, improves a frozen foundation-model backbone (Chronos-2) on DEV-B and VAL.
- **F1.** If the corrector does not beat Chronos-2-group on DEV-B, move the lead information into the backbone:
  - Chronos-2 known-future covariates with lead-aware masking and phase-gap features;
  - LoRA fine-tuning with lead-aware masking (cell19 code).
- **F2.** If lag information is weak at weekly resolution, test it at daily resolution on Track S, where lags of days exist.
- **F3.** If the linear corrector saturates, use the neural version: lag-indexed cross-region attention with coverage masking, trained on residuals.
MDEOF
python -c "import covidfc.trackp.data, covidfc.trackp.halo_lite, covidfc.trackp.gate; print('package imports OK')"

echo "== 2-5. Development run"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.dev24

echo "== 6. Commit"
git add configs/trackp_decisions.md src/covidfc/trackp cells/cell24.sh results/trackp/*.csv results/trackp/audit.json && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell24: Track P data, evidence gate, DEV backbones and HALO-lite" && git log --oneline -1
echo "[$(date +%H:%M)] cell24 done. Upload logs/cell24.log."
