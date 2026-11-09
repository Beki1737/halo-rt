#!/usr/bin/env bash
# cell28: Track P, real-time diagnosis of the cell27 failure and first real-time method candidates.
#   VAL is now real-time development (amendment A1 in the decision log). TEST stays sealed (code guard).
#   1. adds revision.py (revision bank, learned last-week nowcast), rtcorr.py (real-time residual corrector),
#      val28.py; halo.py gains a 'settle' option (unused by default); updates the decision log
#   2. D1 oracle-input decomposition, D2 revision diagnostics, M1 Chronos2-group on nowcast inputs,
#      M2 real-time corrector on Chronos2-group, Chronos2-group-nowcast, CovidHub-ensemble, UMass-gbqr
#   3. writes results/trackp/trackp_val28_export.tar.gz (forecast files for local analysis): please upload it too
#   One GPU, no internet. Expect about 15-30 minutes. Run in tmux. Rerunnable (cached parquet files).
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp; exec > >(tee -a logs/cell28.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell28 start"
git tag -l trackp-lock | grep -q . && { echo "trackp-lock exists: this cell is pre-lock only"; exit 1; } || true
echo "== 1. Update the package and the decision log"
cat > src/covidfc/trackp/revision.py <<'PYEOF'
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
PYEOF
cat > src/covidfc/trackp/rtcorr.py <<'PYEOF'
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
                out.append([name, l, s, h, *Q[i], alpha])
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "alpha"])
PYEOF
cat > src/covidfc/trackp/halo.py <<'PYEOF'
"""HALO v1: lead-aware correction of a frozen backbone (development version).

At each origin s, using only data available at s:
  * features: the backbone's predicted log change, the location's own momentum, exogenous leading signals
    (ED-visit share growth and its phase gap to admissions), the collective phase gap, optionally collective
    growth and TNNLS-style pairwise leaders (ablations);
  * soft lags (option 'softlag'): for each exogenous source the lag profile l = 0..L is weighted by online
    residual-lead evidence  pi_l ~ exp(kappa * corr_w(resid_h, x_l))  computed on resolved origins with
    exponential forgetting, so the effective lag tracks drift;
  * robust per-horizon regression of the backbone's log-median error (Huber), refitted every 4 origins on
    realized values AS AVAILABLE at the refit origin;
  * soft skill gate: the correction is multiplied by alpha in [0, 1] that minimizes recent absolute error on the
    last G resolved origins (online shrinkage; alpha = 0 recovers the backbone);
  * correction clipped to +-clip in log units;
  * online width calibration: log-quantile spreads around the median are scaled by the factor in a grid that
    minimizes recent WIS on resolved origins.
"""
import os
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge, HuberRegressor
from covidfc.trackp.metrics import QS, QC, wis
from covidfc.trackp.data import H
from covidfc.trackp.halo_lite import Features, pair_select

AGRID = np.linspace(0, 1, 11)
KGRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4])
LAGS = 4            # exogenous lag profile l = 0..LAGS-1 weeks before the origin


@dataclass
class Spec:
    name: str
    fams: list = field(default_factory=list)       # own, coll, ed, gap, pair
    softlag: bool = False                           # ED enters through evidence-weighted lag profiles
    reg: str = "huber"
    gate: str = "soft"                              # soft | hard | none
    clip: float = 0.5
    conformal: bool = True
    correct: bool = True
    clip_long: float = None                         # optional tighter clip for h >= 3
    refit_every: int = 4                            # neural learners refit less often
    settle: int = 0                                 # use a resolved record only once its target week is >= settle
                                                    # weeks old at s (real-time: recent weeks are still revised)


def _fixed(F, fams, i, h, pair_sel):
    v = []
    for fm in fams:
        if fm == "ed" : continue          # handled below (fixed or soft)
        v += F.block(fm, i, h, pair_sel)
    return v


def _edprof(F, i):
    """ED lag profiles at the origin: growth and phase gap at l = 0..LAGS-1 (all dated <= s); plus availability."""
    if F.e is None or not np.isfinite(F.e[-1, i]):
        return np.zeros((2, LAGS)), 0.0
    from covidfc.trackp.halo_lite import _z
    g = np.array([np.nan_to_num(F.ge[-1 - l, i]) for l in range(LAGS)])
    gap = np.array([_z(F.e[:len(F.e) - l, i]) - _z(F.y[:len(F.y) - l, i]) for l in range(LAGS)])
    return np.vstack([g, gap]), 1.0


def _wcorr(x, y, w):
    w = w / w.sum(); mx, my = (w * x).sum(), (w * y).sum()
    cx, cy = x - mx, y - my
    d = np.sqrt((w * cx * cx).sum() * (w * cy * cy).sum())
    return (w * cx * cy).sum() / d if d > 0 else 0.0


ATTW = {}          # attention weights of the lead-attention learner: (spec, h) -> {origin: R x (R-1)}


def run(base, panels_at, specs, refit_every=4, gate_n=12, min_train_origins=20, lam=0.97, kappa=8.0, label=""):
    base = base.sort_values(["s", "loc", "h"])
    S = sorted(base["s"].unique()); out = []
    st = {(sp.name, h): {"s": [], "li": [], "Xf": [], "P": [], "av": [], "med": [], "dev": [], "c": [],
                         "model": None, "pi": None, "alpha": 0.0, "k": 1.0} for sp in specs for h in range(1, H + 1)}
    pair_sel = None
    need_pair = any("pair" in sp.fams for sp in specs)
    need_tok = any(sp.reg.startswith("attn") for sp in specs); tok = {}
    if need_tok:
        from covidfc.trackp.leadattn import tokens, LeadAttn
    for n, s in enumerate(S):
        Vh, Ved = panels_at(s)
        F = Features(Vh, Ved); R = len(F.loc)
        Ylog = np.log1p(Vh.clip(lower=0)).to_numpy(); widx = {w: k for k, w in enumerate(Vh.index)}
        b = base[base["s"] == s]
        if need_pair and (pair_sel is None or n % refit_every == 0): pair_sel = pair_select(Vh)
        prof = [_edprof(F, i) for i in range(R)]
        if need_tok: tok[s] = tokens(F)
        for sp in specs:
            for h in range(1, H + 1):
                M = st[(sp.name, h)]; dt = pd.Timedelta(days=7 * h)
                bh = b[b["h"] == h].set_index("loc").reindex(F.loc)
                lq = np.log1p(bh[QC].to_numpy(float)); med = lq[:, 3]; dev = lq - med[:, None]
                Xf = np.nan_to_num(np.array([[med[i] - F.y[-1, i]] + _fixed(F, sp.fams, i, h, pair_sel)
                                             for i in range(R)], float))
                P = np.array([p[0] for p in prof]); av = np.array([p[1] for p in prof])
                # ---- resolved records (realized values as available at s)
                if M["s"]:
                    s0 = np.array(M["s"]); li = np.array(M["li"])
                    tw = np.array([widx.get(x + dt, -1) for x in s0]); ok = tw >= 0
                    if sp.settle: ok &= (s0 + dt) <= (s - pd.Timedelta(days=7 * sp.settle))
                    yr = np.full(len(s0), np.nan); yr[ok] = Ylog[tw[ok], li[ok]]; ok &= np.isfinite(yr)
                    er = (yr - np.array(M["med"]))[ok]; sr = s0[ok]
                    Xr = np.array(M["Xf"])[ok]; Pr = np.array(M["P"])[ok]; avr = np.array(M["av"])[ok]
                    cr = np.array(M["c"])[ok]; devr = np.array(M["dev"])[ok]; medr = np.array(M["med"])[ok]
                    yrr = yr[ok]; lir = li[ok]
                else:
                    sr = np.array([])
                def design(Xf_, P_, av_, pi):
                    if "ed" not in sp.fams: return Xf_
                    if sp.softlag:   # evidence-weighted lag profile per source
                        return np.c_[Xf_, (P_ * pi[None]).sum(2), av_]
                    return np.c_[Xf_, P_[:, 0, :3], P_[:, 1, :1], av_]
                # ---- refit
                if sp.correct and n % sp.refit_every == 0 and len(np.unique(sr)) >= min_train_origins:
                    pi = np.ones((2, LAGS)) / LAGS
                    if sp.softlag and "ed" in sp.fams:
                        w = lam ** ((s - sr) / np.timedelta64(7, "D")).astype(float); m = avr > 0
                        for k in range(2):
                            z = np.array([_wcorr(Pr[m, k, l], er[m], w[m]) for l in range(LAGS)])
                            # sign-agnostic evidence: the regression learns the sign; pi weights the lags
                            e = np.exp(kappa * np.abs(z)); pi[k] = e / e.sum()
                    M["pi"] = pi
                    Xd = design(Xr, Pr, avr, pi)
                    if sp.reg.startswith("attn"):
                        Tr = np.stack([tok[x][i] for x, i in zip(sr, lir)])
                        M["model"] = LeadAttn(mode="attn" if sp.reg == "attn" else "mean", epochs=int(os.environ.get("TP_ATTN_EPOCHS", 40)),
                                              seeds=(0, 1)).fit(Xd, Tr, er)
                    else:
                      try:
                        M["model"] = (HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500) if sp.reg == "huber"
                                      else Ridge(alpha=1.0)).fit(Xd, er)
                      except Exception:
                        M["model"] = Ridge(alpha=1.0).fit(Xd, er)
                c = np.zeros(R)
                if sp.correct and M["model"] is not None:
                    cl = sp.clip_long if (sp.clip_long is not None and h >= 3) else sp.clip
                    if sp.reg.startswith("attn"):
                        if sp.reg == "attn":
                            c, wts = M["model"].predict(design(Xf, P, av, M["pi"]), tok[s], return_w=True)
                            ATTW.setdefault((sp.name, h), {})[s] = wts.astype(np.float16)
                        else:
                            c = M["model"].predict(design(Xf, P, av, M["pi"]), tok[s])
                        c = np.clip(c, -cl, cl)
                    else:
                        c = np.clip(M["model"].predict(design(Xf, P, av, M["pi"])), -cl, cl)
                # ---- gate (on corrections that were actually issued for the resolved origins)
                if sp.correct and len(sr):
                    recent = np.unique(sr)[-gate_n:]; m = np.isin(sr, recent)
                    if len(recent) == gate_n:
                        if sp.gate == "soft":
                            M["alpha"] = AGRID[np.argmin([np.abs(er[m] - a * cr[m]).mean() for a in AGRID])]
                        elif sp.gate == "hard":
                            M["alpha"] = float(np.abs(er[m] - cr[m]).mean() < np.abs(er[m]).mean())
                        else:
                            M["alpha"] = 1.0
                # ---- online width calibration on resolved origins
                if sp.conformal and len(sr):
                    recent = np.unique(sr)[-gate_n:]; m = np.isin(sr, recent)
                    if len(recent) == gate_n:
                        mc = medr[m] + M["alpha"] * cr[m]; yt = np.expm1(yrr[m])
                        sc = [wis(np.expm1(mc[:, None] + k * devr[m]).clip(0), yt).mean() for k in KGRID]
                        M["k"] = KGRID[int(np.argmin(sc))]
                cm = M["alpha"] * c
                Q = np.sort(np.expm1(med[:, None] + cm[:, None] + M["k"] * dev).clip(0), axis=1)
                for i, l in enumerate(F.loc):
                    out.append([f"HALO[{sp.name}]{label}", l, s, h, *Q[i], M["alpha"], M["k"]])
                M["s"] += [s] * R; M["li"] += list(range(R)); M["Xf"] += list(Xf); M["P"] += list(P)
                M["av"] += list(av); M["med"] += list(med); M["dev"] += list(dev); M["c"] += list(c)
    return pd.DataFrame(out, columns=["model", "loc", "s", "h", *QC, "alpha", "kscale"])
PYEOF
cat > src/covidfc/trackp/val28.py <<'PYEOF'
"""cell28: real-time diagnosis of the cell27 failure and the first real-time method candidates (VAL origins only;
VAL is now real-time development, amendment A1; TEST stays sealed).
  D1  oracle-input decomposition: Chronos2-group and HALO[v1+pair-cliplong] rerun on VAL with FINALIZED inputs
      (an infeasible counterfactual) -> how much of the VAL failure is data revision vs regime change.
  D2  revision diagnostics: first-release vs final revisions by lag, and the median bias of each model.
  M1  Chronos2-group on inputs whose last week is completed by an origin-safe learned nowcast.
  M2  RT: a corrector trained only on settled real-time residuals (never on finalized history), applied to
      Chronos2-group, Chronos2-group with nowcast inputs, the CDC Hub ensemble and UMass-gbqr; ablations of the
      feature families (bias only, own, own+ED, full) and of the revision-state features.
  Exports the backbone and corrector forecasts for local analysis (results/trackp/trackp_val28_export.tar.gz).
usage: python -m covidfc.trackp.val28 [--offline]"""
import os, sys, time, json, tarfile
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo as HA
import covidfc.trackp.hub as HB
import covidfc.trackp.rtcorr as RT
from covidfc.trackp.halo import Spec
from covidfc.trackp.revision import LastWeekNowcast
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
OV = list(TD.origins("VAL", hosp)); WARM = OV[10]
TEST0 = pd.Timestamp(TD.SPLITS["TEST"][0])
assert max(OV) < TEST0, "TEST guard"
BBK = "Damped" if OFF else "Chronos2-group"
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
def sc(F):
    F = F.copy(); F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
    S = score(F.merge(tr, on=["t", "loc"], how="inner")); S["lbias"] = np.log1p(S["q0.5"]) - np.log1p(S["truth"]); return S
def table(S, ref, title):
    print(f"\n{title} (relative WIS vs {ref}; all VAL | after warm-up from {WARM.date()})")
    a = relative(S, "wis", ref=ref).T.round(3); b = relative(S[S["s"] >= WARM], "wis", ref=ref).T.round(3)
    print(pd.concat([a.add_prefix("all_h"), b.add_prefix("warm_h")], axis=1).sort_values("warm_h1").to_string())

log("== D2. revisions of first releases on VAL (log first release - log final)")
rows = []
for s in OV:
    V = hosp.asof(s)[0]
    for l in range(4):
        w = s - pd.Timedelta(weeks=l); f = hosp.final.loc[w]; v = V.loc[w]; m = (f > 20) & v.notna()
        d = np.log1p(v[m]) - np.log1p(f[m]); rows.append((l, d.median(), d.abs().mean(), (d < -0.2).mean(), v.isna().mean()))
print(pd.DataFrame(rows, columns=["lag", "median", "mean_abs", "share_below_-20%", "share_missing"]).groupby("lag").mean().round(3).to_string())

log("== D1/M1. backbones on VAL with finalized inputs (oracle) and nowcast-completed inputs")
p = f"{OUT}/val28_backbones.parquet"
if os.path.exists(p): BX = pd.read_parquet(p)
else:
    nc = LastWeekNowcast(hosp); rows = []
    for s in OV:
        Vh = hosp.asof(s)[0]; Vo = hosp.final.loc[:s]; Vn = nc.apply(s, Vh)
        for tag, VV in (("oracle", Vo), ("nowcast", Vn)):
            rr = BB.damped(s, VV) if OFF else BB.chronos(s, VV, "group")
            rr += BB.persistence(s, VV)
            rows += [[f"{r[0]}|{tag}", *r[1:]] for r in rr]
    BX = pd.DataFrame(rows, columns=["model", "loc", "s", "h", *QC]); BX.to_parquet(p)
BV = pd.read_parquet(f"{OUT}/val27_backbones.parquet"); BD = pd.read_parquet(f"{OUT}/dev24_backbones.parquet")

log("== D1. HALO[v1+pair-cliplong] in the oracle world (finalized inputs at every origin)")
q = f"{OUT}/val28_halo_oracle.parquet"
if os.path.exists(q): HO = pd.read_parquet(q)
else:
    def pa_or(s): return hosp.final.loc[:s].copy(), ed.final.loc[:s].copy()
    base = pd.concat([BD[BD["model"] == BBK], BX[BX["model"] == f"{BBK}|oracle"].assign(model=BBK)])
    SPEC = Spec("v1+pair-cliplong", ["own", "coll", "ed", "gap", "pair"], reg="ridge", gate="soft", clip=9, clip_long=0.5)
    HO = HA.run(base, pa_or, [SPEC], label="|oracle"); HO = HO[HO["s"].isin(OV)]; HO.to_parquet(q)
HR = pd.read_parquet(f"{OUT}/val27_halo_{BBK}.parquet"); HR = HR[HR["s"].isin(OV)]
S1 = sc(pd.concat([BV[BV["model"].isin([BBK, "Persistence"])], BX, HO.drop(columns=["alpha", "kscale"]),
                   HR.drop(columns=["alpha", "kscale"])], ignore_index=True))
print("\nD1 decomposition (relative WIS vs the same backbone in the same world):")
d = {}
for world, bk, hl in (("real-time", BBK, f"HALO[v1+pair-cliplong]@{BBK}"), ("oracle", f"{BBK}|oracle", "HALO[v1+pair-cliplong]|oracle")):
    W = S1[S1["model"].isin([bk, hl])]; r = relative(W, "wis", ref=bk); d[f"HALO/{world}"] = r[hl]
r = relative(S1, "wis", ref=BBK); d["backbone oracle/real-time"] = r[f"{BBK}|oracle"]; d["backbone nowcast/real-time"] = r[f"{BBK}|nowcast"]
d["persistence oracle/real-time"] = relative(S1, "wis", ref="Persistence")["Persistence|oracle"]
print(pd.DataFrame(d).T.round(3).to_string())
print("\nmedian log bias (forecast median vs final truth):")
print(S1.groupby(["model", "h"])["lbias"].mean().unstack().round(3).to_string())

log("== M2. real-time residual corrector (RT) on four backbones")
HUBF = HB.load(); HUBF = HUBF[HUBF["s"].isin(OV)]
backs = {BBK: BV[BV["model"] == BBK], f"{BBK}|nowcast": BX[BX["model"] == f"{BBK}|nowcast"]}
for hm in ("CovidHub-ensemble", "UMass-gbqr"):
    if hm in set(HUBF["model"]): backs[hm] = HUBF[HUBF["model"] == hm]
VARS = {"full": dict(), "full-norev": dict(use_rev=False), "own+ed": dict(fams=["own", "ed"]),
        "own": dict(fams=["own"]), "bias": dict(fams=[], use_rev=False)}
res = {}
for bk, bdf in backs.items():
    f = f"{OUT}/val28_rt_{bk.replace('|', '_')}.parquet"
    if os.path.exists(f): R = pd.read_parquet(f)
    else:
        R = pd.concat([RT.run(bdf, OV, hosp, ed, f"RT[{v}]@{bk}", **kw) for v, kw in VARS.items()], ignore_index=True)
        R.to_parquet(f)
    S = sc(pd.concat([bdf, R.drop(columns="alpha")], ignore_index=True))
    key = S.groupby(["loc", "s", "h"])["model"].transform("nunique") == S["model"].nunique(); S = S[key]
    table(S, bk, f"RT on {bk}")
    print("mean gate alpha after warm-up:"); print(R[R["s"] >= WARM].groupby(["model", "h"])["alpha"].mean().unstack().round(2).to_string())
    pt = pd.concat([paired(S[S["s"] >= WARM], f"RT[{v}]@{bk}", bk) for v in ("full", "own+ed")], ignore_index=True)
    print("paired Wilcoxon after warm-up:"); print(pt.round(4).to_string(index=False))
    res[bk] = pt.to_dict("records")

log("== cross-model table on shared rows (all real-time forecasts, VAL after warm-up)")
ALLF = [BV[BV["model"].isin([BBK, "Persistence"])], HUBF, HR.drop(columns=["alpha", "kscale"])]
for bk in backs:
    f = f"{OUT}/val28_rt_{bk.replace('|', '_')}.parquet"; R = pd.read_parquet(f)
    ALLF.append(R[R["model"].isin([f"RT[full]@{bk}", f"RT[own+ed]@{bk}"])].drop(columns="alpha"))
ALLF.append(BX[BX["model"] == f"{BBK}|nowcast"])
SA = sc(pd.concat(ALLF, ignore_index=True)); SA = SA[SA["s"] >= WARM]
key = SA.groupby(["loc", "s", "h"])["model"].transform("nunique") == SA["model"].nunique(); SA = SA[key]
print(relative(SA, "wis").T.round(3).sort_values(1).to_string())
print("coverage 50 / 95:"); print(SA.groupby("model")[["cov50", "cov95"]].mean().round(2).to_string())
json.dump(res, open(f"{OUT}/val28_results.json", "w"), indent=1, default=float)

log("== export for local analysis")
with tarfile.open(f"{OUT}/trackp_val28_export.tar.gz", "w:gz") as t:
    for f in ["dev24_backbones.parquet", "val27_backbones.parquet", "val28_backbones.parquet", "val28_halo_oracle.parquet",
              f"val27_halo_{BBK}.parquet", "val27_halo_Chronos2-group-EDcov.parquet"] + \
             [x for x in os.listdir(OUT) if x.startswith("val28_rt_")]:
        if os.path.exists(f"{OUT}/{f}"): t.add(f"{OUT}/{f}", arcname=f)
print("export:", f"{OUT}/trackp_val28_export.tar.gz", round(os.path.getsize(f"{OUT}/trackp_val28_export.tar.gz") / 1e6, 1), "MB")
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

## Results of cell24 (server, DEV only, Chronos-2 backbones)
- **E5. Chronos-2 with cross-learning over the 51 locations (Chronos2-group) is the strongest backbone.**
  - Relative WIS vs persistence: DEV-A 0.83, 0.79, 0.80, 0.78; DEV-B 1.04, 0.93, 0.88, 0.86 (h = 1..4).
  - It beats the independent-series Chronos-2 at every horizon. Collective information is confirmed at the level of the foundation model.
- **E6. The foundation model barely uses the ED leading signal.** Chronos-2 with ED as a past covariate changed relative WIS by about 1%. On the same data, an explicit corrector using ED lowered WIS by 17.8% at h = 1 relative to Chronos-2 (DEV-B). This is the gap HALO targets.
- **E7. HALO-lite (ridge corrector, hard gate) on the independent-series Chronos-2:**
  - DEV-B gains were significant at all horizons: -17.5%, -14.3%, -8.5%, -6.6% (own + coll + ED + gap).
  - Against Chronos2-group it was -9.6% and -6.4% at h = 1..2 (significant) and not significant at h = 3..4.
  - In DEV-A (Omicron) it hurt at h = 3..4 (+14% to +18%). Robustness at long horizons is the open problem.
- **E8.** Every Chronos-2 variant over-covers: 95% intervals hold 0.96 to 0.98. Width calibration should lower WIS.

## Pivot decided after cell24 (cell25)
- HALO now corrects Chronos2-group, the strongest backbone.
- Chronos2-group-EDcov tests whether the foundation model can use ED itself.
- v1 = ridge corrector, soft skill gate (online shrinkage alpha in [0, 1]) and online WIS-optimal width calibration.
- Ablations: Huber loss, clipping, soft lags, no ED, no gap, no collective features, plus pairwise leaders.
- Offline check (damped stand-in backbone): v1 beat the cell24 recipe in both windows; the largest gain was at DEV-A h = 4 (1.02 vs 1.10). Soft lags and clipping were worse than fixed lags and no clipping.

## Results of cell25 (server, DEV only, Chronos2-group backbone)
- **E9. HALO v1 (ridge, soft gate, width calibration) beats Chronos2-group.**
  - DEV-B (2023-24): -17.9%, -10.5%, -9.6% and -8.9% WIS at h = 1..4, all p < 1e-4 (Wilcoxon across 51 locations). Relative WIS vs persistence is 0.86, 0.82, 0.79, 0.78, against 1.04, 0.93, 0.88, 0.86 for the backbone.
  - DEV-A (2021-22): -7.2% at h = 1 (significant), -1.5% at h = 2..3 (not significant), and +2.8% at h = 4 (p = 0.025, worse).
- **E10. Chronos-2 given ED as a covariate (Chronos2-group-EDcov) gains only 1 to 3% in DEV-B.** The explicit correction gains 9 to 18%, so the foundation model's covariate interface underuses the ED leading signal.
- **E11. Ablations within v1** (positive = worse without the component):
  - Collective features: +3.3% to +5.9% in DEV-B and +2.1% to +9.5% in DEV-A (essential).
  - ED: +1.9% to +5.5% in DEV-B (essential).
  - Phase gap: +0.6% to +0.9% at h = 3..4 in DEV-B (small).
  - Soft lags: +2.0% to +3.0% in DEV-B (soft lags are worse; dropped).
  - Huber: mixed (better in DEV-A at h = 1..2, worse in DEV-B; ridge kept).
  - Tighter clip (0.5): -1.4% and -1.6% at DEV-A h = 3..4, neutral in DEV-B.
  - Pairwise leaders added to v1: -0.8% to -2.6% (small, partly significant).
  - Calibration alone without correction: +0.1% to +0.9% (no gain).
  - Independent-series backbone (v1@uni): +1.6% to +16% (backbone matters).

## cell26 plan and pre-registered selection rule
- New variants:
  - lead attention (a neural learner of soft, state-dependent leader locations and lags), and its uniform-pooling ablation;
  - v1 without calibration;
  - v1 with clip 0.5 at h >= 3;
  - v1 + pairwise + clip at h >= 3;
  - v1 on the Chronos2-group-EDcov backbone.
- Selection for VAL (decided before cell26 runs):
  - Pick the variant with the lowest mean log relative WIS over DEV-A and DEV-B and h = 1..4.
  - It must not be worse than the backbone by more than 2% at any horizon in either window. If none qualifies, take the best overall and log a deviation.
  - VAL then confirms that one variant against Chronos2-group and the other baselines. TEST stays sealed until trackp-lock.

## Results of cell26 (server, DEV only, Chronos2-group backbone)
- **E12. Neural lead attention does not beat the linear corrector.** Learned soft attention over the other 50 locations' lag profiles was 3.7% to 6.7% worse than v1 in DEV-B. Its uniform-pooling ablation was mixed. Both are dropped as the main method and reported as a negative result: at weekly resolution and 51 locations, a learned leader selection overfits relative to fixed collective summaries. This is consistent with E1 (no pairwise lead information beyond own momentum).
- **E13. Width calibration helps inside v1.** v1 without calibration was 1% to 2% worse than v1 in DEV-B.
- **E14. v1 on Chronos2-group-EDcov is the strongest variant in DEV-B** (-18.8%, -14.3%, -12.1%, -11.5% vs Chronos2-group) but is 2.9% worse than the backbone at DEV-A h = 4 (worst ratio 1.025), so it fails the 2% guard.
- **E15. Selected for VAL by the pre-registered rule: HALO[v1+pair-cliplong]** (own + coll + ED + gap + pairwise leaders, ridge, soft gate, width calibration, clip 0.5 at h >= 3). Mean log relative WIS -0.2451; worst ratio to backbone 0.9999.
  - DEV-A: -11.2% (p < 1e-4), -2.8% (n.s.), -4.8% (p = 0.004), +0.8% (n.s.).
  - DEV-B: -17.6%, -12.1%, -9.8%, -8.1% (all p < 1e-4).

## VAL pre-registration (cell27, fixed before any VAL origin is scored)
- VAL origins: 29 (2024-11-16 to 2025-05-31), all with real-time vintages. TEST origins are not built (code guard).
- HALO state is carried through DEV into VAL (continuous rolling refits every 4 origins); only VAL rows are scored.
- **V1 (primary confirmation).** HALO[v1+pair-cliplong]@Chronos2-group vs Chronos2-group, WIS, h = 1..4, Wilcoxon across 51 locations, Holm over the four horizons. Confirmed if at least 2 horizons improve significantly and none is significantly worse.
- **V2 (external benchmark, secondary).** The same HALO vs real-time CDC Hub submissions (CovidHub-ensemble, CovidHub-baseline, UMass-gbqr, UM-DeepOutbreak) on the forecasts both issued. Caveat recorded in advance: Hub teams forecast at the reference date with the data then available, which matches our vintage rule, but their inputs and revisions differ. V2 is reported descriptively plus Wilcoxon, with no pass/fail rule.
- **Selection for TEST.** A = HALO[v1+pair-cliplong]@Chronos2-group; B = the same on Chronos2-group-EDcov. B is chosen only if its mean log relative WIS on VAL is lower and it is never more than 2% worse than A at any horizon.
- **Pivots if V1 fails** (decided now, so a failure is not followed by search):
  - P1. If V1 fails but the h = 1..2 gains hold, the claim narrows to short horizons, and the long-horizon gate becomes the open problem for the method section.
  - P2. If V1 fails overall, check whether the backbone itself changed regime (VAL is post-voluntary reporting). The fallback contribution becomes the diagnostic study (E1, E6/E10, E12: where lag information lives and which interface uses it), with HALO as one probe among several backbones.

## Results of cell27 (server, VAL, 22 Oct 2026): pre-registered confirmation FAILED
- **E16. V1 not confirmed. HALO[v1+pair-cliplong] is worse than Chronos2-group at every horizon on VAL:** +4.6%, +15.9%, +11.3%, +17.7% WIS (Holm p < 1e-4 each). Relative WIS vs persistence: backbone 0.975, 0.871, 0.787, 0.699; HALO 1.020, 1.017, 0.895, 0.822. The EDcov variant also loses. Point errors are hurt (relative AE 1.11 to 1.15), so the median correction is wrong, not just the width.
- **E17. The real-time CDC Hub ensemble is far stronger than our backbone on VAL:** relative WIS 0.76, 0.74, 0.71, 0.65 vs 0.97, 0.87, 0.79, 0.71 for Chronos2-group on shared rows. Our HALO beats CovidHub-baseline only at h = 3..4 (n.s.) and UM-DeepOutbreak at h = 3..4.
- Pre-registered P2 applies: the DEV gains do not carry to real time.

## Local diagnosis after cell27 (persistence and Hub-ensemble proxies; Chronos not available locally)
- **E18. Reporting revisions dominate real-time error at short horizons.** On VAL the first release of the last week is low by a median of 8.5% (mean |log error| 0.164; 24% of state-weeks low by more than 20%; 10% missing); lag 1: -2.8%, lag 2: -1.4%. Revisions are state-specific (split-half correlation of state means 0.53) but only weakly persistent week to week (r = 0.28). Persistence on finalized inputs is 25% better at h = 1 than on real-time inputs.
- **E19. Two causes of the HALO failure, separated with an oracle-input counterfactual** (HALO on persistence): DEV-B -17% to -20%; VAL with finalized inputs -7.7%, -0.8%, -0.1%, -2.5%; VAL real time +8%, +18%, 0%, -7%. (a) Train/serve skew: the corrector learned on finalized history meets revised real-time inputs. (b) Regime change: even with finalized inputs, the lead-information gain shrinks sharply in 2024-25. Settled-target training and a per-state completion factor did not repair it.
- **E20. A corrector trained only on real-time residuals (RT) helps at h = 1 and hurts at h = 3..4.** On the Hub ensemble after a 10-origin warm-up: -9.3% at h = 1 (p < 1e-4), -1% at h = 2, +5% to +9% at h = 3..4. Feature ablation at h = 1: backbone change only -3.4%; own -6.6%; own + collective -4.9%; own + ED -9.9%; full -9.3%. The ED leading signal survives real time; collective lag information does not.
- **Literature check (perplexity_search, 22 Oct 2026).** Revision-aware refinement exists: Back2Future (Kamarthi et al., ICLR 2022); Delphi-RF (PLOS Comput Biol 2025); STS-Residual (2026) corrects the FluSight ensemble with residual learning, -11.6% at h0 down to -2.7% at h3, robust to backfill because it is trained on archived real-time forecasts; MACROCAST (2026) names revision bias in time-series foundation models for macroeconomics. Claims must be positioned against these.

## Amendment A1 (22 Oct 2026): VAL becomes real-time development
- VAL was used for confirmation and failed. It is now real-time development data. TEST (60 origins) is the only confirmation set and stays sealed until `trackp-lock`. The TEST hypothesis, method, variant and analysis will be fixed in this file before the lock.
- **New primary hypothesis H-RT.** In real-time operation, lead information helps only where it is fresh: (i) the exogenous ED signal through a corrector trained on settled real-time residuals, at short horizons; (ii) revision-aware input completion for the backbone. Finalized-history lead correction (HALO v1) is reported as a negative result with the oracle decomposition.
- **cell28 (VAL, exploratory):** oracle-input decomposition for Chronos2-group (D1), revision diagnostics (D2), Chronos2-group on nowcast-completed inputs (M1), the RT corrector on Chronos2-group, Chronos2-group-nowcast, the Hub ensemble and UMass-gbqr with feature ablations (M2), and an export of all forecasts for local analysis.
- **Paper framing if H-RT holds on TEST.** (1) Where lag information lives: pairwise leads carry nothing (E1), collective information helps only on finalized data, ED survives real time. (2) Finalized-data development reverses sign in real time, decomposed into revision and regime parts. (3) A horizon-aware real-time corrector that is backbone-agnostic (foundation model and the operational Hub ensemble). If H-RT fails on TEST, the paper is the diagnostic study (1) + (2).
MDEOF
python -c "import covidfc.trackp.revision, covidfc.trackp.rtcorr, covidfc.trackp.halo; print('imports OK')"
echo "== 2. Real-time diagnosis and candidates"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.val28
echo "== 3. Commit (forecast files stay out of git)"
git add configs/trackp_decisions.md src/covidfc/trackp cells/cell28.sh results/trackp/val28_results.json && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell28: real-time diagnosis (oracle decomposition, revisions), RT corrector candidates (VAL as real-time DEV)" && git log --oneline -1
ls -lh results/trackp/trackp_val28_export.tar.gz
echo "[$(date +%H:%M)] cell28 done. Upload logs/cell28.log and results/trackp/trackp_val28_export.tar.gz."
