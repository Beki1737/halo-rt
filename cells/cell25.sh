#!/usr/bin/env bash
# cell25: Track P, DEV only (VAL and TEST untouched). Pivot after cell24:
#   1. updates backbones.py (adds Chronos2-group-EDcov), adds halo.py (HALO v1) and dev25.py; updates the decision log
#   2. Chronos2-group-EDcov on the 178 DEV origins (does the foundation model use the ED lead itself?)
#   3. HALO v1 and 9 ablations on the Chronos2-group backbone, HALO v1 on independent Chronos-2; scores, tests
#   One GPU, no internet needed. Expect about 15-30 minutes. Run in tmux. Rerunnable (cached parquet files).
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp; exec > >(tee -a logs/cell25.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell25 start"
echo "== 1. Update the package and the decision log"
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
        if mode in ("edcov", "group-edcov"):
            e = Ved.reindex(V.index)[l].to_numpy(float) if (Ved is not None and l in Ved.columns) else np.full(len(V), np.nan)
            f["ed"] = e
        frames.append(f)
    C = pd.concat(frames, ignore_index=True)
    if mode in ("edcov", "group-edcov") and not np.isfinite(C["ed"].to_numpy(float)).any():
        C = C.drop(columns="ed")          # no ED data yet at this origin: identical to the univariate call
    R = _pipe().predict_df(C, id_column="id", timestamp_column="timestamp", target="target", prediction_length=H,
                           quantile_levels=list(QS), batch_size=64, cross_learning=mode.startswith("group"))
    R = R.sort_values(["id", "timestamp"])
    Q = np.zeros((len(loc), H, len(QS)))
    for i, l in enumerate(loc):
        r = R[R["id"] == l]
        Q[i] = r[[str(q) for q in QS]].to_numpy(float)[:H]
    return _rows({"uni": "Chronos2", "group": "Chronos2-group", "edcov": "Chronos2-EDcov",
                  "group-edcov": "Chronos2-group-EDcov"}[mode], s, loc,
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


def run(base, panels_at, specs, refit_every=4, gate_n=12, min_train_origins=20, lam=0.97, kappa=8.0, label=""):
    base = base.sort_values(["s", "loc", "h"])
    S = sorted(base["s"].unique()); out = []
    st = {(sp.name, h): {"s": [], "li": [], "Xf": [], "P": [], "av": [], "med": [], "dev": [], "c": [],
                         "model": None, "pi": None, "alpha": 0.0, "k": 1.0} for sp in specs for h in range(1, H + 1)}
    pair_sel = None
    need_pair = any("pair" in sp.fams for sp in specs)
    for n, s in enumerate(S):
        Vh, Ved = panels_at(s)
        F = Features(Vh, Ved); R = len(F.loc)
        Ylog = np.log1p(Vh.clip(lower=0)).to_numpy(); widx = {w: k for k, w in enumerate(Vh.index)}
        b = base[base["s"] == s]
        if need_pair and (pair_sel is None or n % refit_every == 0): pair_sel = pair_select(Vh)
        prof = [_edprof(F, i) for i in range(R)]
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
                    yr = np.full(len(s0), np.nan); yr[ok] = Ylog[tw[ok], li[ok]]; ok &= np.isfinite(yr)
                    er = (yr - np.array(M["med"]))[ok]; sr = s0[ok]
                    Xr = np.array(M["Xf"])[ok]; Pr = np.array(M["P"])[ok]; avr = np.array(M["av"])[ok]
                    cr = np.array(M["c"])[ok]; devr = np.array(M["dev"])[ok]; medr = np.array(M["med"])[ok]
                    yrr = yr[ok]
                else:
                    sr = np.array([])
                def design(Xf_, P_, av_, pi):
                    if "ed" not in sp.fams: return Xf_
                    if sp.softlag:   # evidence-weighted lag profile per source
                        return np.c_[Xf_, (P_ * pi[None]).sum(2), av_]
                    return np.c_[Xf_, P_[:, 0, :3], P_[:, 1, :1], av_]
                # ---- refit
                if sp.correct and n % refit_every == 0 and len(np.unique(sr)) >= min_train_origins:
                    pi = np.ones((2, LAGS)) / LAGS
                    if sp.softlag and "ed" in sp.fams:
                        w = lam ** ((s - sr) / np.timedelta64(7, "D")).astype(float); m = avr > 0
                        for k in range(2):
                            z = np.array([_wcorr(Pr[m, k, l], er[m], w[m]) for l in range(LAGS)])
                            # sign-agnostic evidence: the regression learns the sign; pi weights the lags
                            e = np.exp(kappa * np.abs(z)); pi[k] = e / e.sum()
                    M["pi"] = pi
                    Xd = design(Xr, Pr, avr, pi)
                    try:
                        M["model"] = (HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500) if sp.reg == "huber"
                                      else Ridge(alpha=1.0)).fit(Xd, er)
                    except Exception:
                        M["model"] = Ridge(alpha=1.0).fit(Xd, er)
                c = np.zeros(R)
                if sp.correct and M["model"] is not None:
                    c = np.clip(M["model"].predict(design(Xf, P, av, M["pi"])), -sp.clip, sp.clip)
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
cat > src/covidfc/trackp/dev25.py <<'PYEOF'
"""cell25 development run (Track P, DEV origins only). Pivot after cell24:
  * the strongest backbone is Chronos-2 with cross-learning over the 51 locations (Chronos2-group), so HALO now
    corrects that backbone; Chronos2-group-EDcov tests whether the foundation model itself can use the ED lead;
  * HALO v1: robust (Huber) correction, soft skill gate, clipping, online width calibration, soft ED lags;
  * ablations isolate: calibration alone, ED, phase gaps, soft lags, collective growth, pairwise leaders.
usage: python -m covidfc.trackp.dev25 [--offline]"""
import os, sys, time
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo as HA
from covidfc.trackp.halo import Spec
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT
WINDOWS = {"DEV-A": ("2021-06-05", "2022-10-29"), "DEV-B": ("2023-01-07", "2024-03-30")}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
cache = {}
def panels_at(s):
    if s not in cache: cache[s] = (hosp.asof(s)[0], ed.asof(s)[0])
    return cache[s]

B = pd.read_parquet(f"{OUT}/dev24_backbones.parquet")
PRIMARY = "Damped" if OFF else "Chronos2-group"
if not OFF and "Chronos2-group-EDcov" not in set(B["model"]):
    log("== Chronos2-group-EDcov on DEV origins")
    rows = []
    for s in TD.origins("DEV"):
        Vh, Ve = panels_at(s); rows += BB.chronos(s, Vh, "group-edcov", Ve)
    B = pd.concat([B, pd.DataFrame(rows, columns=B.columns)], ignore_index=True)
    B.to_parquet(f"{OUT}/dev24_backbones.parquet")

FULL = ["own", "coll", "ed", "gap"]
SPECS = [Spec("cal", correct=False),                                                     # width calibration only
         Spec("lite", FULL, reg="ridge", gate="hard", clip=9, conformal=False),       # cell24 recipe, new backbone
         Spec("v1", FULL, reg="ridge", gate="soft", clip=9),                          # primary candidate
         Spec("v1-huber", FULL, reg="huber", gate="soft", clip=9),
         Spec("v1-clip", FULL, reg="ridge", gate="soft", clip=0.5),
         Spec("v1-softlag", FULL, softlag=True, reg="ridge", gate="soft", clip=9),
         Spec("v1-noED", ["own", "coll", "gap"], reg="ridge", gate="soft", clip=9),
         Spec("v1-nogap", ["own", "coll", "ed"], reg="ridge", gate="soft", clip=9),
         Spec("v1-nocoll", ["own", "ed", "gap"], reg="ridge", gate="soft", clip=9),
         Spec("v1+pair", FULL + ["pair"], reg="ridge", gate="soft", clip=9)]
log(f"== HALO v1 and ablations on {PRIMARY}")
p = f"{OUT}/dev25_halo_{PRIMARY}.parquet"
if os.path.exists(p): HF = pd.read_parquet(p)
else:
    HF = HA.run(B[B["model"] == PRIMARY], panels_at, SPECS); HF.to_parquet(p)
second = "Chronos2" if not OFF else None
if second:
    p2 = f"{OUT}/dev25_halo_{second}.parquet"
    if os.path.exists(p2): H2 = pd.read_parquet(p2)
    else:
        H2 = HA.run(B[B["model"] == second], panels_at, [Spec("v1", FULL, reg="ridge", gate="soft", clip=9)], label="@uni")
        H2.to_parquet(p2)
    HF = pd.concat([HF, H2], ignore_index=True)
print("mean soft-gate alpha and width factor by model and horizon:")
print(HF.groupby(["model", "h"])[["alpha", "kscale"]].mean().unstack("h").round(2).to_string())

F = pd.concat([B, HF.drop(columns=["alpha", "kscale"])], ignore_index=True)
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
S = score(F.merge(tr, on=["t", "loc"], how="inner"))
S.to_parquet(f"{OUT}/dev25_scores.parquet")
for w, (a, b) in WINDOWS.items():
    W = S[(S["s"] >= a) & (S["s"] <= b)]
    print(f"\n--- {w}: {W['s'].nunique()} origins")
    for met in ("wis", "ae"):
        print(f"relative {met.upper()} vs persistence:"); print(relative(W, met).T.round(3).sort_values(4).to_string())
    print("coverage 50% / 95%:"); print(W.groupby("model")[["cov50", "cov95"]].mean().round(2).T.to_string())
    halo = sorted(m for m in W["model"].unique() if m.startswith("HALO"))
    pt = pd.concat([paired(W, m, PRIMARY) for m in halo] +
                   [paired(W, m, "HALO[v1]") for m in halo if m != "HALO[v1]"], ignore_index=True)
    print("paired Wilcoxon across locations, WIS (pct_change = median change of A relative to B):")
    print(pt.round(4).to_string(index=False)); pt.assign(window=w).to_csv(f"{OUT}/dev25_paired_{w}.csv", index=False)
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
MDEOF
python -c "import covidfc.trackp.halo; print('halo imports OK')"
echo "== 2-3. Development run"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.dev25
echo "== 4. Commit"
git add configs/trackp_decisions.md src/covidfc/trackp cells/cell25.sh results/trackp/dev25_paired_*.csv && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell25: HALO v1 on Chronos2-group, ablations (DEV)" && git log --oneline -1
echo "[$(date +%H:%M)] cell25 done. Upload logs/cell25.log."
