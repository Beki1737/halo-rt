#!/usr/bin/env bash
# cell26: Track P, DEV only (VAL and TEST untouched).
#   1. adds leadattn.py (neural lead attention), updates halo.py, adds dev26.py; updates the decision log
#   2. HALO variants on Chronos2-group: lead attention, uniform-pooling ablation, v1 without calibration,
#      tighter long-horizon clip, v1 + pairwise + clip; HALO v1 on Chronos2-group-EDcov
#   3. pre-registered selection of the single variant carried to VAL (results/trackp/dev26_selection.json)
#   One GPU, no internet. Expect about 20-40 minutes. Run in tmux. Rerunnable (cached parquet files).
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp; exec > >(tee -a logs/cell26.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell26 start"
echo "== 1. Update the package and the decision log"
cat > src/covidfc/trackp/leadattn.py <<'PYEOF'
"""Lead attention: a small neural corrector that learns, at every origin, which other locations (and which part of
their recent lag profile) lead the target. It generalizes the two hand-built forms of cross-regional lag
information tested in cell24/25:
  * uniform pooling over locations ("coll" features)     -> mode="mean"  (attention weights fixed and equal)
  * hard, frozen pairwise leaders at fixed lags ("pair")  -> replaced by soft, state-dependent attention
Tokens for location j (seen from target i), all dated <= s:
  growth lag profile g_j(s-l), l = 0..5 ; level gap z(y_j) - z(y_i) ; drawdown of j ; ED growth of j and availability
Query: the target's own features (backbone predicted change, own momentum, ED profile, gaps) and own lag profile.
Output: the backbone's log-median error at one horizon. Inputs are standardized on the training rows."""
import numpy as np, torch, torch.nn as nn

LP = 6          # lag profile length (weeks)


def tokens(F):
    """R x (R-1) x 10 token array for one origin (Features object of halo_lite)."""
    from covidfc.trackp.halo_lite import _z
    y, g = F.y, F.g; R = y.shape[1]
    prof = np.nan_to_num(np.stack([g[-1 - l] for l in range(LP)], 1))             # R x LP
    zl = np.array([_z(y[:, j]) for j in range(R)])
    dd = y[-1] - np.nanmax(y[-9:], 0)
    if F.e is not None:
        eg = np.nan_to_num(F.ge[-1]); ea = np.isfinite(F.e[-1]).astype(float)
    else:
        eg = np.zeros(R); ea = np.zeros(R)
    T = np.zeros((R, R - 1, LP + 4), np.float32)
    for i in range(R):
        js = [j for j in range(R) if j != i]
        T[i] = np.c_[prof[js], (zl[js] - zl[i])[:, None], dd[js][:, None], eg[js][:, None], ea[js][:, None]]
    return np.nan_to_num(T)


class _Net(nn.Module):
    def __init__(self, ft, fk, d=32, heads=4, mode="attn"):
        super().__init__()
        self.mode = mode
        self.q = nn.Sequential(nn.Linear(ft, d), nn.GELU(), nn.Linear(d, d))
        self.k = nn.Sequential(nn.Linear(fk, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.lin = nn.Linear(ft, 1)                                   # linear skip (ridge-like part)

    def forward(self, xt, xk, return_w=False):
        q = self.q(xt); k = self.k(xk)
        if self.mode == "attn":
            ctx, w = self.att(q[:, None], k, k, need_weights=True, average_attn_weights=True)
            ctx = ctx[:, 0]
        else:
            ctx, w = k.mean(1), None
        y = self.out(torch.cat([q, ctx], 1)).squeeze(-1) + self.lin(xt).squeeze(-1)
        return (y, w) if return_w else y


class LeadAttn:
    def __init__(self, mode="attn", epochs=80, lr=3e-3, wd=1e-3, seeds=(0, 1, 2), device=None):
        self.mode, self.epochs, self.lr, self.wd, self.seeds = mode, epochs, lr, wd, seeds
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, Xt, Tk, y):
        def ms(A):
            m, s = A.mean(0), A.std(0)
            const = s < 1e-3                    # constant on the training rows (e.g. ED not yet available)
            return np.where(const, 0.0, m), np.where(const, 1.0, s)
        self.mt, self.st = ms(Xt)
        self.mk, self.sk = ms(Tk.reshape(-1, Tk.shape[-1]))
        xt = torch.tensor(np.clip((Xt - self.mt) / self.st, -8, 8), dtype=torch.float32, device=self.dev)
        xk = torch.tensor(np.clip((Tk - self.mk) / self.sk, -8, 8), dtype=torch.float32, device=self.dev)
        yy = torch.tensor(y, dtype=torch.float32, device=self.dev)
        self.nets = []
        n = len(y); bs = 256
        for sd in self.seeds:
            torch.manual_seed(sd); net = _Net(Xt.shape[1], Tk.shape[-1], mode=self.mode).to(self.dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            g = torch.Generator(device="cpu").manual_seed(sd)
            for ep in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for b in range(0, n, bs):
                    idx = perm[b:b + bs].to(self.dev)
                    loss = nn.functional.huber_loss(net(xt[idx], xk[idx]), yy[idx], delta=0.3)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, Xt, Tk, return_w=False):
        xt = torch.tensor(np.clip((Xt - self.mt) / self.st, -8, 8), dtype=torch.float32, device=self.dev)
        xk = torch.tensor(np.clip((Tk - self.mk) / self.sk, -8, 8), dtype=torch.float32, device=self.dev)
        outs = [net(xt, xk, return_w=True) for net in self.nets]
        p = torch.stack([o[0] for o in outs]).mean(0).cpu().numpy()
        if return_w and outs[0][1] is not None:
            return p, torch.stack([o[1][:, 0] for o in outs]).mean(0).cpu().numpy()
        return p
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
cat > src/covidfc/trackp/dev26.py <<'PYEOF'
"""cell26 development run (Track P, DEV origins only).
  * lead attention (neural, learned soft leaders and lags) vs uniform pooling vs v1 (ridge, hand-built features)
  * v1 without width calibration; v1 with a tighter clip at h >= 3; v1 + pairwise leaders + tighter clip
  * v1 on the Chronos2-group-EDcov backbone
  * pre-registered selection rule for the single variant carried to VAL (see configs/trackp_decisions.md)
usage: python -m covidfc.trackp.dev26 [--offline]"""
import os, sys, time, json
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
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
FULL = ["own", "coll", "ed", "gap"]
SPECS = [Spec("v1-nocal", FULL, reg="ridge", gate="soft", clip=9, conformal=False),
         Spec("v1-cliplong", FULL, reg="ridge", gate="soft", clip=9, clip_long=0.5),
         Spec("v1+pair-cliplong", FULL + ["pair"], reg="ridge", gate="soft", clip=9, clip_long=0.5),
         Spec("attn", FULL, reg="attn", gate="soft", clip=1.0, refit_every=8),
         Spec("attn-mean", FULL, reg="attn-mean", gate="soft", clip=1.0, refit_every=8)]
p = f"{OUT}/dev26_halo_{PRIMARY}.parquet"
if os.path.exists(p): HF = pd.read_parquet(p)
else:
    log(f"== HALO variants on {PRIMARY} (lead attention trains on the GPU)")
    HF = HA.run(B[B["model"] == PRIMARY], panels_at, SPECS); HF.to_parquet(p)
    np.savez_compressed(f"{OUT}/dev26_attention_weights.npz",
                        **{f"{k[0]}|h{k[1]}|{s.date()}": w for k, d in HA.ATTW.items() for s, w in d.items()})
parts = [HF]
p25 = f"{OUT}/dev25_halo_{PRIMARY}.parquet"
if os.path.exists(p25):
    H25 = pd.read_parquet(p25); parts.append(H25[H25["model"].isin(["HALO[v1]", "HALO[v1+pair]", "HALO[lite]"])])
if not OFF:
    p3 = f"{OUT}/dev26_halo_Chronos2-group-EDcov.parquet"
    if os.path.exists(p3): H3 = pd.read_parquet(p3)
    else:
        log("== HALO v1 on Chronos2-group-EDcov")
        H3 = HA.run(B[B["model"] == "Chronos2-group-EDcov"], panels_at,
                    [Spec("v1", FULL, reg="ridge", gate="soft", clip=9)], label="@group-EDcov"); H3.to_parquet(p3)
    parts.append(H3)
HF = pd.concat(parts, ignore_index=True)
print("mean soft-gate alpha / width factor:")
print(HF.groupby(["model", "h"])[["alpha", "kscale"]].mean().unstack("h").round(2).to_string())

keep = [PRIMARY, "Persistence"] + ([] if OFF else ["Chronos2-group-EDcov"])
F = pd.concat([B[B["model"].isin(keep)], HF.drop(columns=["alpha", "kscale"])], ignore_index=True)
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
S = score(F.merge(tr, on=["t", "loc"], how="inner")); S.to_parquet(f"{OUT}/dev26_scores.parquet")
sel = {}
for w, (a, b) in WINDOWS.items():
    W = S[(S["s"] >= a) & (S["s"] <= b)]
    print(f"\n--- {w}: {W['s'].nunique()} origins")
    rw = relative(W, "wis"); print("relative WIS vs persistence:"); print(rw.T.round(3).sort_values(4).to_string())
    print("relative AE vs persistence:"); print(relative(W, "ae").T.round(3).sort_values(4).to_string())
    print("coverage 50% / 95%:"); print(W.groupby("model")[["cov50", "cov95"]].mean().round(2).T.to_string())
    halo = sorted(m for m in W["model"].unique() if m.startswith("HALO"))
    pt = pd.concat([paired(W, m, PRIMARY) for m in halo] +
                   [paired(W, m, "HALO[v1]") for m in halo if m != "HALO[v1]" and "HALO[v1]" in halo], ignore_index=True)
    print("paired Wilcoxon across locations, WIS:"); print(pt.round(4).to_string(index=False))
    pt.assign(window=w).to_csv(f"{OUT}/dev26_paired_{w}.csv", index=False)
    for m in halo: sel.setdefault(m, []).append(float(np.log(rw[m]).mean()))
# pre-registered rule: lowest mean log relative WIS over the two DEV windows and four horizons, among the
# variants that are not worse than the backbone at any horizon in either window by more than 2%
ok = {}
for m, v in sel.items():
    worst = max(S[(S["s"] >= a) & (S["s"] <= b)].pipe(relative, "wis")[m].div(
        S[(S["s"] >= a) & (S["s"] <= b)].pipe(relative, "wis")[PRIMARY]).max() for a, b in WINDOWS.values())
    ok[m] = {"mean_log_rel_wis": float(np.mean(v)), "worst_ratio_to_backbone": float(worst)}
R = pd.DataFrame(ok).T.sort_values("mean_log_rel_wis"); print("\nselection table:"); print(R.round(4).to_string())
elig = R[R["worst_ratio_to_backbone"] <= 1.02]
choice = (elig if len(elig) else R).index[0]
print("SELECTED for VAL:", choice, "(eligible)" if len(elig) else "(no variant met the 2% guard; best overall)")
json.dump({"selected": choice, "table": R.round(5).to_dict()}, open(f"{OUT}/dev26_selection.json", "w"), indent=1)
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
MDEOF
python -c "import covidfc.trackp.halo, covidfc.trackp.leadattn, torch; print('imports OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "== 2-3. Development run"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.dev26
echo "== 4. Commit"
git add configs/trackp_decisions.md src/covidfc/trackp cells/cell26.sh results/trackp/dev26_paired_*.csv results/trackp/dev26_selection.json && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell26: lead attention and HALO variants (DEV), VAL selection" && git log --oneline -1
echo "[$(date +%H:%M)] cell26 done. Upload logs/cell26.log."
