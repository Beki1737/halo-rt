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
