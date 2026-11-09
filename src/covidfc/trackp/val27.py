"""cell27: VAL confirmation (Track P). Origins 2024-11-16 .. 2025-05-31, inputs from real-time vintages.
TEST origins are never built, forecast or scored here (guard below).
Pre-registered in configs/trackp_decisions.md before this run:
  V1 (primary): HALO[v1+pair-cliplong] on Chronos2-group vs Chronos2-group, WIS, h = 1..4, Wilcoxon across
      locations with Holm over the four horizons. Confirmed if >= 2 horizons improve significantly and none is
      significantly worse.
  V2: the same HALO vs the real-time CDC Hub ensemble on the forecasts both issued (descriptive + Wilcoxon).
  Selection for TEST: HALO[v1+pair-cliplong] on Chronos2-group (A) or on Chronos2-group-EDcov (B), by mean log
      relative WIS on VAL, B only if it is never > 2% worse than A at any horizon.
usage: python -m covidfc.trackp.val27 [--offline]"""
import os, sys, time, json
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo as HA
import covidfc.trackp.hub as HB
from covidfc.trackp.halo import Spec
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return adj
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
OV = TD.origins("VAL", hosp); OD = TD.origins("DEV")
assert OV.max() < pd.Timestamp(TD.SPLITS["TEST"][0]), "TEST guard"
cache = {}
def panels_at(s):
    assert s < pd.Timestamp(TD.SPLITS["TEST"][0]), "TEST guard"
    if s not in cache: cache[s] = (hosp.asof(s)[0], ed.asof(s)[0])
    return cache[s]
print("VAL origins:", len(OV), OV.min().date(), "to", OV.max().date(),
      "| input sources:", pd.Series([hosp.asof(s)[1] for s in OV]).value_counts().to_dict())

log("== 1. backbones on VAL origins")
p = f"{OUT}/val27_backbones.parquet"
if os.path.exists(p): BV = pd.read_parquet(p)
else:
    rows = []
    for s in OV:
        Vh, Ve = panels_at(s)
        rows += BB.persistence(s, Vh)
        rows += BB.damped(s, Vh) if OFF else (BB.chronos(s, Vh, "group") + BB.chronos(s, Vh, "group-edcov", Ve) +
                                               BB.chronos(s, Vh, "uni"))
    BV = pd.DataFrame(rows, columns=["model", "loc", "s", "h", *QC]); BV.to_parquet(p)
BD = pd.read_parquet(f"{OUT}/dev24_backbones.parquet")
ALL = pd.concat([BD, BV], ignore_index=True)

log("== 2. HALO through DEV and VAL (state carried over; only VAL rows are scored)")
SPEC = Spec("v1+pair-cliplong", ["own", "coll", "ed", "gap", "pair"], reg="ridge", gate="soft", clip=9, clip_long=0.5)
bbs = ["Damped"] if OFF else ["Chronos2-group", "Chronos2-group-EDcov"]
HV = []
for bb in bbs:
    q = f"{OUT}/val27_halo_{bb}.parquet"
    if os.path.exists(q): Hq = pd.read_parquet(q)
    else:
        Hq = HA.run(ALL[ALL["model"] == bb], panels_at, [SPEC], label=f"@{bb}"); Hq.to_parquet(q)
    HV.append(Hq[Hq["s"].isin(OV)])
HV = pd.concat(HV, ignore_index=True)
print("mean alpha / width factor on VAL:"); print(HV.groupby(["model", "h"])[["alpha", "kscale"]].mean().unstack("h").round(2).to_string())

log("== 3. real-time Hub submissions on VAL origins")
HUBF = HB.load(); HUBF = HUBF[HUBF["s"].isin(OV)]
print("Hub rows on VAL:", HUBF.groupby("model").size().to_dict())

F = pd.concat([BV, HV.drop(columns=["alpha", "kscale"]), HUBF], ignore_index=True)
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
S = score(F.merge(tr, on=["t", "loc"], how="inner")); S.to_parquet(f"{OUT}/val27_scores.parquet")

A = f"HALO[{SPEC.name}]@{bbs[0]}"; BBK = bbs[0]
own = S[S["model"].isin([m for m in S["model"].unique() if not m.startswith(HB.MODELS)])]
print(f"\n--- VAL: {S['s'].nunique()} origins")
for met in ("wis", "ae"):
    print(f"relative {met.upper()} vs persistence (our models, all VAL rows):"); print(relative(own, met).T.round(3).sort_values(4).to_string())
print("coverage 50% / 95%:"); print(S.groupby("model")[["cov50", "cov95"]].mean().round(2).T.to_string())

v1 = paired(S, A, BBK); v1["p_holm"] = holm(v1["p"])
print(f"\nV1 primary: {A} vs {BBK}"); print(v1.round(4).to_string(index=False))
better = ((v1["p_holm"] < 0.05) & (v1["pct_change"] < 0)).sum(); worse = ((v1["p_holm"] < 0.05) & (v1["pct_change"] > 0)).sum()
print(f"V1 {'CONFIRMED' if better >= 2 and worse == 0 else 'NOT CONFIRMED'}: {better} horizons better, {worse} worse (Holm)")

rows = []
for hm in HB.MODELS:
    if hm not in set(S["model"]): continue
    shared = S[S["model"].isin([A, hm, BBK, "Persistence"])]
    key = shared.groupby(["loc", "s", "h"])["model"].transform("nunique") == 4
    sh = shared[key]
    if sh.empty: continue
    r = relative(sh, "wis").T
    pt = paired(sh, A, hm)
    for _, x in pt.iterrows():
        rows.append({"vs": hm, "h": x["h"], "rel_wis_HALO": r.loc[A, x["h"]], "rel_wis_backbone": r.loc[BBK, x["h"]],
                     "rel_wis_hub": r.loc[hm, x["h"]], "pct_change": x["pct_change"], "p": x["p"],
                     "n_loc": x["n"], "rows": int(len(sh) / 4)})
V2 = pd.DataFrame(rows)
print("\nV2: HALO vs real-time Hub submissions on shared forecasts (relative WIS vs persistence on those rows):")
print(V2.round(4).to_string(index=False))

sel = {"A": A, "B": None, "selected_for_TEST": A}
if not OFF:
    B = f"HALO[{SPEC.name}]@Chronos2-group-EDcov"
    rw = relative(own, "wis")
    la, lb = np.log(rw[A]).mean(), np.log(rw[B]).mean(); worst = (rw[B] / rw[A]).max()
    pick = B if (lb < la and worst <= 1.02) else A
    sel = {"A": A, "B": B, "mean_log_rel_wis_A": la, "mean_log_rel_wis_B": lb, "worst_ratio_B_to_A": worst,
           "selected_for_TEST": pick}
print("\nselection for TEST:", sel)
json.dump({"V1": v1.to_dict("records"), "V2": V2.to_dict("records"), "selection": sel},
          open(f"{OUT}/val27_results.json", "w"), indent=1, default=float)
log("done")
