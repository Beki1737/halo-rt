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
