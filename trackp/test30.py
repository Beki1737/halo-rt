"""cell30: the sealed TEST evaluation of HALO-RT (Track P). Runs only after the git tag `trackp-lock` exists.
Protocol: configs/trackp_decisions.md, section "TEST protocol (locked)". Nothing here is tuned on TEST.
Every learned component runs continuously from the first VAL origin through TEST (real-time vintages only);
only TEST origins are scored.
usage: python -m covidfc.trackp.test30            (server, after the lock)
       python -m covidfc.trackp.test30 --dry      (code check: VAL origins stand in for TEST, offline backbones)"""
import os, sys, time, json, subprocess
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo as HA
import covidfc.trackp.hub as HB
import covidfc.trackp.rtcorr as RT
import covidfc.trackp.anchor as AN
from covidfc.trackp.halo import Spec
from covidfc.trackp.revision import LastWeekNowcast
from covidfc.trackp.metrics import QC, score, relative, paired

DRY = "--dry" in sys.argv
OUT = TD.OUT
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

if not DRY:
    tags = subprocess.run(["git", "-C", os.environ["PROJ"], "tag", "-l", "trackp-lock"], capture_output=True, text=True).stdout
    assert "trackp-lock" in tags, "TEST is sealed: create the trackp-lock tag first"
    print("lock commit:", subprocess.run(["git", "-C", os.environ["PROJ"], "rev-list", "-n", "1", "trackp-lock"],
                                         capture_output=True, text=True).stdout.strip())
raw = TD.load_raw(); hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
OV = list(TD.origins("VAL", hosp))
OT = OV[12:] if DRY else list(TD.origins("TEST", hosp))
OV = OV[:12] if DRY else OV
ORIG = OV + OT
TAG = "dry30" if DRY else "test30"
BBK = "Damped" if DRY else "Chronos2-group"
print(f"{'DRY RUN: VAL origins stand in for TEST. ' if DRY else ''}scored origins: {len(OT)} ({OT[0].date()} to {OT[-1].date()}); "
      f"input sources: {pd.Series([hosp.asof(s)[1] for s in OT]).value_counts().to_dict()}; "
      f"ED sources: {pd.Series([ed.asof(s)[1] for s in OT]).value_counts().to_dict()}")

log("== 1. backbones on the scored origins (real-time vintages; nowcast-completed inputs)")
p = f"{OUT}/{TAG}_backbones.parquet"
if os.path.exists(p): BT = pd.read_parquet(p)
elif DRY:
    BT = pd.concat([pd.read_parquet(f"{OUT}/val27_backbones.parquet"), pd.read_parquet(f"{OUT}/val28_backbones.parquet")])
    BT = BT[BT["s"].isin(OT) & ~BT["model"].str.endswith("|oracle")]
else:
    nc = LastWeekNowcast(hosp); rows = []
    for s in OT:
        Vh = hosp.asof(s)[0]; Vn = nc.apply(s, Vh)
        rows += BB.persistence(s, Vh) + BB.chronos(s, Vh, "group")
        rows += [[f"{r[0]}|nowcast", *r[1:]] for r in BB.chronos(s, Vn, "group")]
    BT = pd.DataFrame(rows, columns=["model", "loc", "s", "h", *QC]); BT.to_parquet(p)
BV = pd.concat([pd.read_parquet(f"{OUT}/val27_backbones.parquet"), pd.read_parquet(f"{OUT}/val28_backbones.parquet")])
BV = BV[BV["s"].isin(OV) & ~BV["model"].str.endswith("|oracle")]
BALL = pd.concat([BV, BT], ignore_index=True)
HUBF = HB.load(); HUBF = HUBF[HUBF["s"].isin(ORIG)]
print("Hub forecasts on scored origins:", HUBF[HUBF["s"].isin(OT)].groupby("model")["s"].nunique().to_dict())

log("== 2. HALO-RT (locked: own + ED + revision features at h = 1, settled labels, hold-out gate, rho = 0.8)")
backs = {BBK: BALL[BALL["model"] == BBK], f"{BBK}|nowcast": BALL[BALL["model"] == f"{BBK}|nowcast"],
         "CovidHub-ensemble": HUBF[HUBF["model"] == "CovidHub-ensemble"], "UMass-gbqr": HUBF[HUBF["model"] == "UMass-gbqr"]}
backs = {k: v for k, v in backs.items() if len(v)}
OUTF = []
for bk, bdf in backs.items():
    f = f"{OUT}/{TAG}_halort_{bk.replace('|', '_')}.parquet"
    if os.path.exists(f): R = pd.read_parquet(f)
    else:
        orig = [s for s in ORIG if s in set(bdf["s"])]
        parts = []
        for lab, fams in (("HALO-RT", ["own", "ed"]), ("HALO-RT[no-ED]", ["own"])):
            if lab.endswith("[no-ED]") and bk not in (BBK, "CovidHub-ensemble"): continue
            r = RT.run(bdf, orig, hosp, ed, "RT", fams=fams); c1 = r[r["h"] == 1].set_index(["loc", "s"])["corr"]
            A = AN.run(bdf, c1, orig, hosp, f"{lab}@{bk}", rho=0.8)
            parts += [A.drop(columns=["rho", "k_in", "k_out"])]
            if lab == "HALO-RT": parts.append(r.drop(columns=["alpha", "corr"]).assign(model=f"RT[per-h]@{bk}"))
        R = pd.concat(parts, ignore_index=True); R.to_parquet(f)
    OUTF.append(R)

log("== 3. HALO v1 (finalized-history corrector) through DEV, VAL and the scored origins: does the reversal persist?")
f = f"{OUT}/{TAG}_halov1.parquet"
if os.path.exists(f): HV = pd.read_parquet(f)
else:
    cache = {}
    def pa(s):
        if s not in cache: cache[s] = (hosp.asof(s)[0], ed.asof(s)[0])
        return cache[s]
    BD = pd.read_parquet(f"{OUT}/dev24_backbones.parquet")
    base = pd.concat([BD[BD["model"] == BBK], BALL[BALL["model"] == BBK]], ignore_index=True)
    SPEC = Spec("v1+pair-cliplong", ["own", "coll", "ed", "gap", "pair"], reg="ridge", gate="soft", clip=9, clip_long=0.5)
    HV = HA.run(base, pa, [SPEC], label=f"@{BBK}"); HV = HV[HV["s"].isin(OT)].drop(columns=["alpha", "kscale"]); HV.to_parquet(f)

log("== 4. scoring (TEST origins only)")
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F = pd.concat([BT, HUBF, *OUTF, HV], ignore_index=True); F["s"] = pd.to_datetime(F["s"]); F = F[F["s"].isin(OT)]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
S = score(F.merge(tr, on=["t", "loc"], how="inner")); S.to_parquet(f"{OUT}/{TAG}_scores.parquet")
last = hosp.final.dropna(how="all").index.max()
print(f"truth vintage covers weeks to {last.date()}; targets within 3 weeks of it are not yet settled (sensitivity below)")

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.empty(m)
    a[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return a

def boot_ci(Sx, a, b, B=1000, block=4, seed=0):
    """Moving-block bootstrap over origins of the relative WIS a / b (geometric mean over locations), per h."""
    rng = np.random.default_rng(seed); Os = np.array(sorted(Sx["s"].unique())); n = len(Os); out = {}
    for h in range(1, 5):
        X = Sx[(Sx["h"] == h) & Sx["model"].isin([a, b])].pivot_table(index=["s", "loc"], columns="model", values="wis").dropna()
        A = X[a].unstack("loc"); Bm = X[b].unstack("loc"); A, Bm = A.reindex(Os), Bm.reindex(Os)
        est = []
        for _ in range(B):
            st = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(x, x + block) for x in st])[:n]
            r = np.log(A.iloc[idx].mean() / Bm.iloc[idx].mean()).dropna(); est.append(np.exp(r.mean()))
        out[h] = (round(100 * (np.percentile(est, 2.5) - 1), 1), round(100 * (np.percentile(est, 97.5) - 1), 1))
    return out

def primary(Sx, label):
    print(f"\n=== PRIMARY ({label}) ===")
    rows = []
    for name, a, b in (("T1", f"HALO-RT@{BBK}", BBK), ("T2", "HALO-RT@CovidHub-ensemble", "CovidHub-ensemble")):
        W = Sx[Sx["model"].isin([a, b])]; key = W.groupby(["loc", "s", "h"])["model"].transform("nunique") == 2; W = W[key]
        pt = paired(W, a, b); ci = boot_ci(W, a, b)
        for _, x in pt.iterrows(): rows.append({"test": name, "A": a, "B": b, "h": int(x["h"]), "pct_change": x["pct_change"],
                                                "ci95": ci[int(x["h"])], "p": x["p"], "n_loc": x["n"]})
    P = pd.DataFrame(rows); P["p_holm8"] = holm(P["p"]); print(P.round(4).to_string(index=False))
    verdict = {}
    for t, g in P.groupby("test"):
        bt = ((g["p_holm8"] < 0.05) & (g["pct_change"] < 0)).sum(); wt = ((g["p_holm8"] < 0.05) & (g["pct_change"] > 0)).sum()
        verdict[t] = f"{'CONFIRMED' if bt >= 2 and wt == 0 else 'NOT CONFIRMED'} ({bt} better, {wt} worse)"
    print("verdicts:", verdict); return P, verdict

P, V = primary(S, "all TEST targets")
P2, V2 = primary(S[S["t"] <= last - pd.Timedelta(weeks=3)], "sensitivity: settled targets only")

print("\n=== SECONDARY ===")
for bk in backs:
    W = S[S["model"].str.endswith(f"@{bk}") | (S["model"] == bk)]
    key = W.groupby(["loc", "s", "h"])["model"].transform("nunique") == W["model"].nunique(); W = W[key]
    print(f"\n{bk}: relative WIS vs the backbone"); print(relative(W, "wis", ref=bk).T.round(3).sort_values(4).to_string())
W = S[S["model"].isin([BBK, f"HALO[v1+pair-cliplong]@{BBK}"])]
print("\nreversal check, HALO v1 vs backbone:"); print(paired(W, f"HALO[v1+pair-cliplong]@{BBK}", BBK).round(4).to_string(index=False))
print("\nleaderboard on shared rows (relative WIS vs persistence):")
L = S[S["model"].isin(["Persistence", BBK, f"{BBK}|nowcast", *HB.MODELS] + [f"HALO-RT@{b}" for b in backs])]
key = L.groupby(["loc", "s", "h"])["model"].transform("nunique") == L["model"].nunique(); L = L[key]
print(relative(L, "wis").T.round(3).sort_values(1).to_string())
print("coverage 50 / 95:"); print(L.groupby("model")[["cov50", "cov95"]].mean().round(2).T.to_string())
PER = {"summer-fall 2025": ("2025-06-28", "2025-11-29"), "winter 2025-26": ("2025-12-06", "2026-04-25"),
       "spring-summer 2026": ("2026-05-02", "2026-09-19")}
if not DRY:
    print("\nby period (pre-registered), relative WIS of HALO-RT vs its backbone:")
    for nm, (a, b) in PER.items():
        W = S[(S["s"] >= a) & (S["s"] <= b)]
        r = {bk: relative(W[W["model"].isin([bk, f"HALO-RT@{bk}"])], "wis", ref=bk)[f"HALO-RT@{bk}"].round(3).to_dict() for bk in backs}
        print(nm, W["s"].nunique(), "origins:", r)
json.dump({"primary": P.to_dict("records"), "verdict": V, "sensitivity": P2.to_dict("records"), "verdict_settled": V2},
          open(f"{OUT}/{TAG}_results.json", "w"), indent=1, default=str)
log("done")
