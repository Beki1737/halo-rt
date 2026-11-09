#!/usr/bin/env bash
# cell27: Track P, VAL confirmation (TEST stays sealed; a code guard stops any TEST origin from being built).
#   1. adds hub.py (real-time CDC Hub submissions) and val27.py; updates the decision log with cell26 results and
#      the VAL pre-registration (written before this run scores anything)
#   2. Chronos-2 backbones on the 29 VAL origins (real-time vintages)
#   3. HALO[v1+pair-cliplong] run continuously through DEV into VAL on Chronos2-group (A) and Chronos2-group-EDcov (B)
#   4. V1 primary test, V2 comparison with the Hub's own real-time forecasts, A/B selection for TEST
#   One GPU, no internet (uses the Hub clone from cell24). Expect about 15-30 minutes. Run in tmux. Rerunnable.
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp; exec > >(tee -a logs/cell27.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell27 start"
echo "== 1. Update the package and the decision log"
cat > src/covidfc/trackp/hub.py <<'PYEOF'
"""Real-time CDC COVID-19 Forecast Hub submissions in the Track P forecast schema.
Hub reference date R, horizon k (0..3)  ->  origin s = R - 7 days, h = k + 1."""
import glob, os
import numpy as np, pandas as pd
import covidfc.trackp.data as TD
from covidfc.trackp.metrics import QS, QC

MODELS = ("CovidHub-ensemble", "CovidHub-baseline", "UMass-gbqr", "UM-DeepOutbreak")


def load(models=MODELS):
    out = []
    want = {f"{q:g}": c for q, c in zip(QS, QC)}
    for m in models:
        for f in sorted(glob.glob(f"{TD.HUB}/model-output/{m}/*")):
            if not f.endswith((".csv", ".parquet")): continue
            d = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, dtype={"location": str, "output_type_id": str})
            d = d[(d["target"] == "wk inc covid hosp") & (d["output_type"] == "quantile") & d["horizon"].between(0, 3)]
            d = d[d["output_type_id"].astype(str).map(lambda x: f"{float(x):g}").isin(want)].copy()
            if d.empty: continue
            d["q"] = d["output_type_id"].astype(str).map(lambda x: want[f"{float(x):g}"])
            w = d.pivot_table(index=["reference_date", "location", "horizon"], columns="q", values="value").reset_index()
            if not set(QC) <= set(w.columns): continue
            w["s"] = pd.to_datetime(w["reference_date"]) - pd.Timedelta(days=7); w["h"] = w["horizon"] + 1
            w["model"] = m; w = w.rename(columns={"location": "loc"})
            out.append(w[["model", "loc", "s", "h", *QC]])
    if not out: return pd.DataFrame(columns=["model", "loc", "s", "h", *QC])
    F = pd.concat(out, ignore_index=True)
    F = F[~F["loc"].isin(TD.EXCLUDE)]
    F[QC] = np.sort(F[QC].to_numpy(float), axis=1)
    return F
PYEOF
cat > src/covidfc/trackp/val27.py <<'PYEOF'
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

## Results of cell26 (server, DEV only, Chronos2-group backbone)
- **E12. Neural lead attention does not beat the linear corrector.** Learned soft attention over the other 50 locations' lag profiles was 3.7% to 6.7% worse than v1 in DEV-B. Its uniform-pooling ablation was mixed. Both are dropped as the main method and reported as a negative result: at weekly resolution and 51 locations, a learned leader selection overfits relative to fixed collective summaries. This is consistent with E1 (no pairwise lead information beyond own momentum).
- **E13. Width calibration helps inside v1.** v1 without calibration was 1% to 2% worse than v1 in DEV-B.
- **E14. v1 on Chronos2-group-EDcov is the strongest variant in DEV-B** (-18.8%, -14.3%, -12.1%, -11.5% vs Chronos2-group) but is 2.9% worse than the backbone at DEV-A h = 4 (worst ratio 1.025), so it fails the 2% guard.
- **E15. Selected for VAL by the pre-registered rule: HALO[v1+pair-cliplong]** (own + coll + ED + gap + pairwise leaders, ridge, soft gate, width calibration, clip 0.5 at h >= 3). Mean log relative WIS -0.2451; worst ratio to backbone 0.9999.
  - DEV-A: -11.2% (p < 1e-4), -2.8% (n.s.), -4.8% (p = 0.004), +0.8% (n.s.).
  - DEV-B: -17.6%, -12.1%, -9.8%, -8.1% (all p < 1e-4).

## VAL pre-registration (cell27, fixed before any VAL origin is scored)
- VAL origins: 29 (2024-11-16 to 2025-05-31), all with real-time vintages. TEST origins are not built (code guard).
- HALO state is carried through DEV into VAL (continuous rolling refits every 4 origins); only VAL rows are scored.
- **V1 (primary confirmation).** HALO[v1+pair-cliplong]@Chronos2-group vs Chronos2-group, WIS, h = 1..4, Wilcoxon across 51 locations, Holm over the four horizons. Confirmed if at least 2 horizons improve significantly and none is significantly worse.
- **V2 (external benchmark, secondary).** The same HALO vs real-time CDC Hub submissions (CovidHub-ensemble, CovidHub-baseline, UMass-gbqr, UM-DeepOutbreak) on the forecasts both issued. Caveat recorded in advance: Hub teams forecast at the reference date with the data then available, which matches our vintage rule, but their inputs and revisions differ. V2 is reported descriptively plus Wilcoxon, with no pass/fail rule.
- **Selection for TEST.** A = HALO[v1+pair-cliplong]@Chronos2-group; B = the same on Chronos2-group-EDcov. B is chosen only if its mean log relative WIS on VAL is lower and it is never more than 2% worse than A at any horizon.
- **Pivots if V1 fails** (decided now, so a failure is not followed by search):
  - P1. If V1 fails but the h = 1..2 gains hold, the claim narrows to short horizons, and the long-horizon gate becomes the open problem for the method section.
  - P2. If V1 fails overall, check whether the backbone itself changed regime (VAL is post-voluntary reporting). The fallback contribution becomes the diagnostic study (E1, E6/E10, E12: where lag information lives and which interface uses it), with HALO as one probe among several backbones.
MDEOF
python -c "import covidfc.trackp.hub as H; F = H.load(); print('Hub forecasts:', F.groupby('model').size().to_dict())"
git tag -l trackp-lock | grep -q . && { echo "trackp-lock exists: this cell is pre-lock only"; exit 1; } || true
echo "== 2-4. VAL run"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.val27
echo "== 5. Commit"
git add configs/trackp_decisions.md src/covidfc/trackp cells/cell27.sh results/trackp/val27_results.json && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell27: VAL confirmation of HALO[v1+pair-cliplong], Hub comparison, TEST selection" && git log --oneline -1
echo "[$(date +%H:%M)] cell27 done. Upload logs/cell27.log."
