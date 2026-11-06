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
