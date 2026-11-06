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
