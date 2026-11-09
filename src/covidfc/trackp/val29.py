"""cell29: HALO-RT design selection on VAL (real-time development, amendment A1). TEST stays sealed.
HALO-RT = real-time corrector learned only at h = 1 from settled real-time residuals (own + ED + revision-state
features, honest hold-out gate), anchored to h = 2..4 (fixed decay rho^(h-1) or online-learned rho_h), optionally
with online two-scale width recalibration. Compared on four backbones: Chronos2-group, Chronos2-group on
nowcast-completed inputs, CovidHub-ensemble, UMass-gbqr.
Pre-registered selection (decision log, before this run): the variant with the lowest mean log relative WIS over
4 backbones x 4 horizons after warm-up, among variants never worse than the backbone by more than 2% in any cell.
usage: python -m covidfc.trackp.val29 [--offline]"""
import os, sys, time, json
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.hub as HB
import covidfc.trackp.rtcorr as RT
import covidfc.trackp.anchor as AN
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
OV = list(TD.origins("VAL", hosp)); WARM = OV[10]
assert max(OV) < pd.Timestamp(TD.SPLITS["TEST"][0]), "TEST guard"
BBK = "Damped" if OFF else "Chronos2-group"
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
def sc(F):
    F = F.copy(); F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
    return score(F.merge(tr, on=["t", "loc"], how="inner"))

BV = pd.read_parquet(f"{OUT}/val27_backbones.parquet"); BX = pd.read_parquet(f"{OUT}/val28_backbones.parquet")
HUBF = HB.load(); HUBF = HUBF[HUBF["s"].isin(OV)]
backs = {BBK: BV[BV["model"] == BBK], f"{BBK}|nowcast": BX[BX["model"] == f"{BBK}|nowcast"]}
for hm in ("CovidHub-ensemble", "UMass-gbqr"):
    if hm in set(HUBF["model"]): backs[hm] = HUBF[HUBF["model"] == hm]
VARS = {"RT[per-h]": None, "HA[rho=0.6]": dict(rho=0.6), "HA[rho=0.8]": dict(rho=0.8), "HA[rho=1]": dict(rho=1.0),
        "HA[learned]": dict(rho="learned"), "HA[learned]+W": dict(rho="learned", width=True),
        "HA[rho=0.8]+W": dict(rho=0.8, width=True), "W-only": dict(rho=0.0, width=True, zero=True)}
cells, allS = {}, []
for bk, bdf in backs.items():
    log("==", bk)
    f = f"{OUT}/val29_rt_{bk.replace('|', '_')}.parquet"
    if os.path.exists(f): R = pd.read_parquet(f)
    else: R = RT.run(bdf, OV, hosp, ed, "RT[per-h]", fams=["own", "ed"]); R.to_parquet(f)
    c1 = R[R["h"] == 1].set_index(["loc", "s"])["corr"]
    parts = [bdf.assign(model="backbone"), R.drop(columns=["alpha", "corr"])]
    for v, kw in VARS.items():
        if kw is None: continue
        kw = dict(kw); zero = kw.pop("zero", False)
        A = AN.run(bdf, c1 * (0 if zero else 1), OV, hosp, v, **kw)
        if v == "HA[learned]+W": print("mean rho / k_in / k_out after warm-up:"); print(A[A["s"] >= WARM].groupby("h")[["rho", "k_in", "k_out"]].mean().round(2).T.to_string())
        parts.append(A.drop(columns=["rho", "k_in", "k_out"]))
    S = sc(pd.concat(parts, ignore_index=True)); S = S[S["s"] >= WARM]
    key = S.groupby(["loc", "s", "h"])["model"].transform("nunique") == S["model"].nunique(); S = S[key]
    rw = relative(S, "wis", ref="backbone"); print("relative WIS vs the backbone (after warm-up):"); print(rw.T.round(3).sort_values(4).to_string())
    print("coverage 50 / 95:"); print(S.groupby("model")[["cov50", "cov95"]].mean().round(2).T.to_string())
    for v in VARS: cells[(bk, v)] = rw[v]
    allS.append(S.assign(model=S["model"].where(S["model"] != "backbone", bk)).assign(bb=bk))
log("== pre-registered selection")
rows = []
for v in VARS:
    r = pd.concat([cells[(bk, v)] for bk in backs], axis=1)
    rows.append({"variant": v, "mean_log_rel_wis": float(np.log(r).mean().mean()), "worst_cell": float(r.max().max())})
T = pd.DataFrame(rows).sort_values("mean_log_rel_wis"); print(T.round(4).to_string(index=False))
el = T[T["worst_cell"] <= 1.02]; choice = (el if len(el) else T).iloc[0]["variant"]
print("SELECTED:", choice, "(eligible)" if len(el) else "(none met the 2% guard; best overall, deviation logged)")
SA = pd.concat(allS, ignore_index=True)
print("\npaired Wilcoxon of the selected variant vs each backbone (after warm-up):")
pt = pd.concat([paired(SA[SA["bb"] == bk].replace({"model": {bk: "backbone"}}), choice, "backbone").assign(backbone=bk) for bk in backs], ignore_index=True)
print(pt.round(4).to_string(index=False))
print("\nreal-time leaderboard on shared rows (relative WIS vs persistence, after warm-up):")
P = BV[BV["model"] == "Persistence"]
L = [P, HUBF] + [SA[(SA["bb"] == bk) & (SA["model"] == choice)][["loc", "s", "h", *QC]].assign(model=f"{choice}@{bk}") for bk in backs] + [backs[BBK], backs[f"{BBK}|nowcast"]]
SL = sc(pd.concat(L, ignore_index=True)); SL = SL[SL["s"] >= WARM]
key = SL.groupby(["loc", "s", "h"])["model"].transform("nunique") == SL["model"].nunique(); SL = SL[key]
print(relative(SL, "wis").T.round(3).sort_values(1).to_string())
json.dump({"selection": T.to_dict("records"), "selected": choice, "paired": pt.to_dict("records")},
          open(f"{OUT}/val29_results.json", "w"), indent=1, default=float)
log("done")
