"""cell31 (post-lock, amendment A2 in configs/trackp_postlock.md, written before this run).
  1. SORC baseline (simple online residual correction, same real-time labels and settlement delay as HALO-RT):
     window 4, window 8, window 8 with HALO-RT's anchoring (rho = 0.8); plus HALO-RT[bias-only] (HALO-RT's learner,
     gate, clip and anchoring with no features beyond the backbone's predicted change).
  2. Correction ladder on VAL (after warm-up) and TEST: backbone < SORC < HALO-RT machinery without ED < HALO-RT.
     Primary: HALO-RT vs SORC-8 on TEST for Chronos2-group and CovidHub-ensemble (Wilcoxon, Holm over 8 tests,
     moving-block bootstrap over origins).
  3. Post-hoc sensitivity on TEST (reported as post hoc): settling delay delta in {1, 2, 3}, rho in {0.6, 0.8, 1.0}.
     delta = 2, rho = 0.8 must reproduce the locked HALO-RT forecasts (checked).
  4. Diagnostics for the Table III subset (shared forecasts of all leaderboard models).
The locked HALO-RT forecasts are read from cell30's files; no locked file is modified.
usage: python -m covidfc.trackp_post.post31"""
import os, time, json
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.hub as HB
import covidfc.trackp.rtcorr as RT
import covidfc.trackp.anchor as AN
from covidfc.trackp.metrics import QC, score, relative, paired
import covidfc.trackp_post.sorc as SO

OUT = TD.OUT
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def fn(bk): return bk.replace("|", "_")

raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
OV = list(TD.origins("VAL", hosp)); OT = list(TD.origins("TEST", hosp)); ORIG = OV + OT; WARM = OV[10]
BBK, ENS = "Chronos2-group", "CovidHub-ensemble"
print(f"VAL {len(OV)} origins (scored from {WARM.date()}), TEST {len(OT)} origins ({OT[0].date()} to {OT[-1].date()})")

BT = pd.read_parquet(f"{OUT}/test30_backbones.parquet")
BV = pd.concat([pd.read_parquet(f"{OUT}/val27_backbones.parquet"), pd.read_parquet(f"{OUT}/val28_backbones.parquet")])
BV = BV[BV["s"].isin(OV) & ~BV["model"].str.endswith("|oracle")]
BALL = pd.concat([BV, BT], ignore_index=True)
HUBF = HB.load(); HUBF = HUBF[HUBF["s"].isin(ORIG)]
backs = {BBK: BALL[BALL["model"] == BBK], f"{BBK}|nowcast": BALL[BALL["model"] == f"{BBK}|nowcast"],
         ENS: HUBF[HUBF["model"] == ENS], "UMass-gbqr": HUBF[HUBF["model"] == "UMass-gbqr"]}
backs = {k: v for k, v in backs.items() if len(v)}
LAB = {}
def labels(s):
    if s not in LAB: LAB[s] = hosp.asof(s)[0]
    return LAB[s]
COLS = ["model", "loc", "s", "h", *QC]

# ---------------------------------------------------------------- 1. SORC and HALO-RT[bias-only]
log("== 1. SORC baselines and HALO-RT[bias-only]")
F = []
for bk, bdf in backs.items():
    orig = [s for s in ORIG if s in set(bdf["s"])]
    p = f"{OUT}/post31_ladder_{fn(bk)}.parquet"
    if os.path.exists(p): R = pd.read_parquet(p)
    else:
        parts = [SO.run(bdf, orig, labels, f"SORC-4@{bk}", window=4),
                 SO.run(bdf, orig, labels, f"SORC-8@{bk}", window=8),
                 SO.run(bdf, orig, labels, f"SORC-8-anchored@{bk}", window=8, anchor_rho=0.8)]
        r = RT.run(bdf, orig, hosp, ed, "RT", fams=[], use_rev=False)
        c1 = r[r["h"] == 1].set_index(["loc", "s"])["corr"]
        parts.append(AN.run(bdf, c1, orig, hosp, f"HALO-RT[bias-only]@{bk}", rho=0.8))
        R = pd.concat([x[COLS] for x in parts], ignore_index=True); R.to_parquet(p)
    Lk = pd.read_parquet(f"{OUT}/test30_halort_{fn(bk)}.parquet")
    F += [bdf.assign(model=bk)[COLS], R[COLS], Lk[COLS]]
    print(f"{bk}: {len(orig)} origins; models: {sorted(set(R['model']) | set(Lk['model']))}")

# ---------------------------------------------------------------- 2. sensitivity (post hoc)
log("== 2. post-hoc sensitivity of HALO-RT to delta and rho (Chronos2-group, Hub ensemble)")
check = {}
for bk in (BBK, ENS):
    if bk not in backs: continue
    bdf = backs[bk]; orig = [s for s in ORIG if s in set(bdf["s"])]
    p = f"{OUT}/post31_sens_{fn(bk)}.parquet"
    if os.path.exists(p): SN = pd.read_parquet(p)
    else:
        parts = []
        for d in (1, 2, 3):
            r = RT.run(bdf, orig, hosp, ed, "RT", fams=["own", "ed"], settle=d)
            c1 = r[r["h"] == 1].set_index(["loc", "s"])["corr"]
            for rho in ((0.6, 0.8, 1.0) if d == 2 else (0.8,)):
                parts.append(AN.run(bdf, c1, orig, hosp, f"SENS[delta={d},rho={rho}]@{bk}", rho=rho, settle=d)[COLS])
        SN = pd.concat(parts, ignore_index=True); SN.to_parquet(p)
    F.append(SN)
    Lk = pd.read_parquet(f"{OUT}/test30_halort_{fn(bk)}.parquet")
    a = SN[SN["model"] == f"SENS[delta=2,rho=0.8]@{bk}"].merge(Lk[Lk["model"] == f"HALO-RT@{bk}"], on=["loc", "s", "h"], suffixes=("", "_L"))
    dmax = float(np.nanmax(np.abs(np.log1p(a["q0.5"]) - np.log1p(a["q0.5_L"])))) if len(a) else np.nan
    check[bk] = {"rows": int(len(a)), "max_abs_log_diff": dmax}
    print(f"reproduction check {bk}: {len(a)} rows, max |log diff| of medians vs locked HALO-RT = {dmax:.2e}",
          "OK" if dmax < 1e-6 else "WARNING: differs from the locked run")

# ---------------------------------------------------------------- 3. scoring
log("== 3. scoring")
FF = pd.concat(F, ignore_index=True); FF["s"] = pd.to_datetime(FF["s"]); FF = FF.dropna(subset=QC)
FF = FF.drop_duplicates(subset=["model", "loc", "s", "h"])
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
FF["t"] = FF["s"] + pd.to_timedelta(7 * FF["h"], unit="D")
S = score(FF.merge(tr, on=["t", "loc"], how="inner"))
S["win"] = np.where(S["s"].isin(OT), "TEST", np.where(S["s"] >= WARM, "VAL", "warm-up"))
S.to_parquet(f"{OUT}/post31_scores.parquet")

def shared(W):
    key = W.groupby(["loc", "s", "h"])["model"].transform("nunique") == W["model"].nunique()
    return W[key]

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.empty(m)
    a[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return a

def boot_ci(Sx, a, b, B=1000, block=4, seed=0):
    rng = np.random.default_rng(seed); Os = np.array(sorted(Sx["s"].unique())); n = len(Os); out = {}
    for h in range(1, 5):
        X = Sx[(Sx["h"] == h) & Sx["model"].isin([a, b])].pivot_table(index=["s", "loc"], columns="model", values="wis").dropna()
        A = X[a].unstack("loc").reindex(Os); Bm = X[b].unstack("loc").reindex(Os); est = []
        for _ in range(B):
            st = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(x, x + block) for x in st])[:n]
            r = np.log(A.iloc[idx].mean() / Bm.iloc[idx].mean()).dropna(); est.append(np.exp(r.mean()))
        out[h] = (round(100 * (np.percentile(est, 2.5) - 1), 1), round(100 * (np.percentile(est, 97.5) - 1), 1))
    return out

LAD = ["SORC-4", "SORC-8", "SORC-8-anchored", "HALO-RT[bias-only]", "HALO-RT[no-ED]", "RT[per-h]", "HALO-RT"]
res = {"reproduction": check, "ladder": {}, "primary": None, "secondary": [], "sensitivity": {}, "table3": {}}
for win in ("VAL", "TEST"):
    for bk in backs:
        ms = [bk] + [f"{m}@{bk}" for m in LAD if f"{m}@{bk}" in set(S["model"])]
        W = shared(S[(S["win"] == win) & S["model"].isin(ms)])
        if W.empty: continue
        rw = relative(W, "wis", ref=bk).T
        rw = rw.reindex([m for m in ms if m in rw.index])
        print(f"\n--- {win}: correction ladder on {bk} ({W['s'].nunique()} origins); relative WIS vs the backbone")
        print(rw.round(3).to_string())
        print("coverage 50 / 95:"); print(W.groupby("model")[["cov50", "cov95"]].mean().reindex(ms).round(2).T.to_string())
        res["ladder"][f"{win}|{bk}"] = rw.round(4).to_dict()

log("== 4. primary baseline comparison on TEST: HALO-RT vs SORC-8")
rows = []
for bk in (BBK, ENS):
    a, b = f"HALO-RT@{bk}", f"SORC-8@{bk}"
    W = shared(S[(S["win"] == "TEST") & S["model"].isin([a, b])])
    pt = paired(W, a, b); ci = boot_ci(W, a, b)
    for _, x in pt.iterrows():
        rows.append({"backbone": bk, "h": int(x["h"]), "pct_change": x["pct_change"], "ci95": ci[int(x["h"])], "p": x["p"], "n_loc": x["n"]})
P = pd.DataFrame(rows); P["p_holm8"] = holm(P["p"])
print("HALO-RT vs SORC-8 (median per-jurisdiction change in WIS; bootstrap interval of the geometric-mean ratio over origins):")
print(P.round(4).to_string(index=False))
res["primary"] = P.to_dict("records")

print("\nsecondary paired tests on TEST (median per-jurisdiction change, Wilcoxon p, unadjusted):")
for bk in backs:
    for a, b in ((f"HALO-RT@{bk}", f"SORC-4@{bk}"), (f"HALO-RT@{bk}", f"SORC-8-anchored@{bk}"),
                 (f"HALO-RT@{bk}", f"HALO-RT[bias-only]@{bk}"), (f"SORC-8@{bk}", bk), (f"SORC-4@{bk}", bk),
                 (f"SORC-8-anchored@{bk}", bk), (f"HALO-RT@{bk}", bk)):
        if a not in set(S["model"]) or b not in set(S["model"]): continue
        W = shared(S[(S["win"] == "TEST") & S["model"].isin([a, b])]); pt = paired(W, a, b)
        line = {"A": a, "B": b, **{f"h{int(x['h'])}": f"{x['pct_change']:+.1f}% (p={x['p']:.3g})" for _, x in pt.iterrows()}}
        res["secondary"].append(line); print(line)

log("== 5. post-hoc sensitivity on TEST (relative WIS vs the backbone)")
for bk in (BBK, ENS):
    ms = [bk] + sorted(m for m in S["model"].unique() if m.startswith("SENS[") and m.endswith(f"@{bk}"))
    W = shared(S[(S["win"] == "TEST") & S["model"].isin(ms)])
    if W.empty: continue
    rw = relative(W, "wis", ref=bk).T; print(f"\n{bk}:"); print(rw.round(3).to_string())
    res["sensitivity"][bk] = rw.round(4).to_dict()

log("== 6. Table III diagnostics (cell30 scores)")
S30 = pd.read_parquet(f"{OUT}/test30_scores.parquet")
LBm = ["Persistence", BBK, f"{BBK}|nowcast", *HB.MODELS] + [f"HALO-RT@{b}" for b in backs]
for lab, mods in (("all leaderboard models", LBm), ("without UM-DeepOutbreak", [m for m in LBm if m != "UM-DeepOutbreak"])):
    L = S30[S30["model"].isin(mods)]; Ls = shared(L)
    n_o = Ls.groupby("h")["s"].nunique().to_dict(); n_r = (Ls.groupby("h").size() / Ls["model"].nunique()).astype(int).to_dict()
    print(f"\n[{lab}] shared forecasts: origins per h {n_o}, location-origins per h {n_r}")
    print(relative(Ls, "wis").T.round(3).sort_values(1).to_string())
    res["table3"][lab] = {"origins": n_o, "rows": n_r, "relwis": relative(Ls, "wis").T.round(4).to_dict()}
E = S30[S30["model"].isin([ENS, f"HALO-RT@{ENS}"])]; Es = shared(E)
Ls = shared(S30[S30["model"].isin(LBm)]); keys = Ls[["loc", "s", "h"]].drop_duplicates()
Esub = Es.merge(keys, on=["loc", "s", "h"])
for lab, X in (("all ensemble forecasts", Es), ("Table III subset", Esub)):
    rw = relative(X, "wis", ref=ENS)[f"HALO-RT@{ENS}"].round(3).to_dict()
    pt = paired(X, f"HALO-RT@{ENS}", ENS)
    md = {int(x["h"]): round(x["pct_change"], 2) for _, x in pt.iterrows()}
    print(f"HALO-RT@ensemble vs ensemble, {lab} ({X['s'].nunique()} origins): geometric-mean relative WIS {rw}; median change % {md}")
    res["table3"][f"ensemble|{lab}"] = {"origins": int(X["s"].nunique()), "relwis": rw, "median_pct": md}
json.dump(res, open(f"{OUT}/post31_results.json", "w"), indent=1, default=str)
log("done")
