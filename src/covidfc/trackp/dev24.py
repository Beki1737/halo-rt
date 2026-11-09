"""cell24 development run (Track P). Never touches VAL or TEST origins.
  1. sync the CDC Forecast Hub target data and audit it
  2. evidence gate: do pairwise fixed lags carry information beyond the target's own momentum?
  3. backbones on DEV origins: Persistence, Chronos-2 (uni / group / ED covariate)  [or Damped offline]
  4. HALO-lite corrector families on the Chronos-2 backbone: which kind of lag information helps?
  5. scores by window and horizon, paired tests, gate usage
usage: python -m covidfc.trackp.dev24 [--offline]"""
import os, sys, json, time
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
import covidfc.trackp.backbones as BB
import covidfc.trackp.halo_lite as HL
import covidfc.trackp.gate as GT
from covidfc.trackp.metrics import QC, score, relative, paired

OFF = "--offline" in sys.argv
OUT = TD.OUT; os.makedirs(OUT, exist_ok=True)
WINDOWS = {"DEV-A (Delta/Omicron, no ED)": ("2021-06-05", "2022-10-29"),
           "DEV-B (2023-24, with ED)": ("2023-01-07", "2024-03-30")}
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if not OFF and not os.environ.get("TP_NOSYNC"):
    log("hub commit", TD.sync_hub())
raw = TD.load_raw()
hosp, ed = TD.Panel(raw, TD.HOSP), TD.Panel(raw, TD.ED)
log("== 1. audit"); aud = TD.audit(hosp, ed)
for k, v in aud.items(): print(f"   {k}: {v}")
json.dump(aud, open(f"{OUT}/audit.json", "w"), indent=1, default=str)

log("== 2. evidence gate for pairwise fixed lags (DEV data only, 200 permutations)")
G = GT.run(hosp.final, perms=int(os.environ.get("TP_PERMS", 200)))
print(G.round(3).to_string(index=False)); G.to_csv(f"{OUT}/gate_pairwise.csv", index=False)

log("== 3. backbones on DEV origins")
O = TD.origins("DEV")
cache = {}
def panels_at(s):
    if s not in cache:
        Vh = hosp.asof(s)[0]; Ve = ed.asof(s)[0]
        cache[s] = (Vh, Ve)
    return cache[s]
fpath = f"{OUT}/dev24_backbones.parquet"
if os.path.exists(fpath):
    B = pd.read_parquet(fpath); log("loaded", fpath, B.shape)
else:
    rows = []
    for k, s in enumerate(O):
        Vh, Ve = panels_at(s)
        rows += BB.persistence(s, Vh)
        if OFF:
            rows += BB.damped(s, Vh)
        else:
            rows += BB.chronos(s, Vh, "uni") + BB.chronos(s, Vh, "group") + BB.chronos(s, Vh, "edcov", Ve)
        if k % 25 == 0: log(f"   origin {k + 1}/{len(O)} {s.date()}")
    B = pd.DataFrame(rows, columns=["model", "loc", "s", "h", *QC])
    B.to_parquet(fpath)
backbone = "Damped" if OFF else "Chronos2"

log(f"== 4. HALO-lite families on the {backbone} backbone")
hpath = f"{OUT}/dev24_halolite_{backbone}.parquet"
if os.path.exists(hpath):
    HLF = pd.read_parquet(hpath)
else:
    HLF = HL.run(B[B["model"] == backbone], panels_at)
    HLF.to_parquet(hpath)
print("   share of forecasts with the skill gate open, by family and h:")
print(HLF.groupby(["model", "h"])["gate"].mean().unstack("h").round(2).to_string())

log("== 5. scores (truth = finalized values)")
F = pd.concat([B, HLF.drop(columns="gate")], ignore_index=True)
tr = hosp.final.stack().rename("truth").reset_index(); tr.columns = ["t", "loc", "truth"]
F["t"] = F["s"] + pd.to_timedelta(7 * F["h"], unit="D")
F = F.merge(tr, on=["t", "loc"], how="inner")
S = score(F)
for wname, (a, b) in WINDOWS.items():
    W = S[(S["s"] >= a) & (S["s"] <= b)]
    print(f"\n--- {wname}: {W['s'].nunique()} origins x {W['loc'].nunique()} locations")
    for met in ("wis", "ae"):
        print(f"relative {met.upper()} vs persistence (geometric mean over locations), by horizon in weeks:")
        print(relative(W, met).T.round(3).sort_values(4).to_string())
    print("coverage (95%):", W.groupby("model")["cov95"].mean().round(2).to_dict())
    best = [m for m in W["model"].unique() if m.startswith("HALO-lite")]
    pt = pd.concat([paired(W, m, backbone) for m in best] +
                   ([paired(W, m, "Chronos2-group") for m in best] if not OFF else []), ignore_index=True)
    print("paired Wilcoxon across locations (WIS):"); print(pt.round(4).to_string(index=False))
    pt.assign(window=wname).to_csv(f"{OUT}/dev24_paired_{wname[:5]}.csv", index=False)
S.to_parquet(f"{OUT}/dev24_scores.parquet")
log("done")
