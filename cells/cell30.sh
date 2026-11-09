#!/usr/bin/env bash
# cell30: Track P LOCK + sealed TEST evaluation of HALO-RT.  *** This unseals TEST. It cannot be undone. ***
#   1. adds test30.py; writes the locked TEST protocol into the decision log; commits; creates the git tag
#      trackp-lock (if it already exists, checks that the Track P code and protocol are unchanged since the tag)
#   2. Chronos-2 backbones on the 60 TEST origins (raw and nowcast-completed vintages), HALO-RT on four backbones,
#      ED ablation, per-horizon RT, HALO v1 reversal check, leaderboard, periods, primary tests T1 and T2
#   One GPU, no internet. Expect about 20-40 minutes. Run in tmux. Rerunnable (cached; the lock is checked).
set -eo pipefail
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"; mkdir -p logs results/trackp; exec > >(tee -a logs/cell30.log) 2>&1
eval "$(conda shell.bash hook)"; conda activate covidfc
export PYTHONPATH="$PROJ/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source cells/lib.sh
echo "[$(date +%H:%M)] cell30 start"
if git tag -l trackp-lock | grep -q .; then
  echo "trackp-lock exists at $(git rev-list -n 1 trackp-lock); checking that locked files are unchanged"
  git diff --quiet trackp-lock -- src/covidfc/trackp configs/trackp_decisions.md || { echo "Track P files changed since the lock: stop"; exit 1; }
else
  echo "== 1. Write the locked protocol and code, commit, tag"
cat > src/covidfc/trackp/test30.py <<'PYEOF'
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

## Results of cell27 (server, VAL, 22 Oct 2026): pre-registered confirmation FAILED
- **E16. V1 not confirmed. HALO[v1+pair-cliplong] is worse than Chronos2-group at every horizon on VAL:** +4.6%, +15.9%, +11.3%, +17.7% WIS (Holm p < 1e-4 each). Relative WIS vs persistence: backbone 0.975, 0.871, 0.787, 0.699; HALO 1.020, 1.017, 0.895, 0.822. The EDcov variant also loses. Point errors are hurt (relative AE 1.11 to 1.15), so the median correction is wrong, not just the width.
- **E17. The real-time CDC Hub ensemble is far stronger than our backbone on VAL:** relative WIS 0.76, 0.74, 0.71, 0.65 vs 0.97, 0.87, 0.79, 0.71 for Chronos2-group on shared rows. Our HALO beats CovidHub-baseline only at h = 3..4 (n.s.) and UM-DeepOutbreak at h = 3..4.
- Pre-registered P2 applies: the DEV gains do not carry to real time.

## Local diagnosis after cell27 (persistence and Hub-ensemble proxies; Chronos not available locally)
- **E18. Reporting revisions dominate real-time error at short horizons.** On VAL the first release of the last week is low by a median of 8.5% (mean |log error| 0.164; 24% of state-weeks low by more than 20%; 10% missing); lag 1: -2.8%, lag 2: -1.4%. Revisions are state-specific (split-half correlation of state means 0.53) but only weakly persistent week to week (r = 0.28). Persistence on finalized inputs is 25% better at h = 1 than on real-time inputs.
- **E19. Two causes of the HALO failure, separated with an oracle-input counterfactual** (HALO on persistence): DEV-B -17% to -20%; VAL with finalized inputs -7.7%, -0.8%, -0.1%, -2.5%; VAL real time +8%, +18%, 0%, -7%. (a) Train/serve skew: the corrector learned on finalized history meets revised real-time inputs. (b) Regime change: even with finalized inputs, the lead-information gain shrinks sharply in 2024-25. Settled-target training and a per-state completion factor did not repair it.
- **E20. A corrector trained only on real-time residuals (RT) helps at h = 1 and hurts at h = 3..4.** On the Hub ensemble after a 10-origin warm-up: -9.3% at h = 1 (p < 1e-4), -1% at h = 2, +5% to +9% at h = 3..4. Feature ablation at h = 1: backbone change only -3.4%; own -6.6%; own + collective -4.9%; own + ED -9.9%; full -9.3%. The ED leading signal survives real time; collective lag information does not.
- **Literature check (perplexity_search, 22 Oct 2026).** Revision-aware refinement exists: Back2Future (Kamarthi et al., ICLR 2022); Delphi-RF (PLOS Comput Biol 2025); STS-Residual (2026) corrects the FluSight ensemble with residual learning, -11.6% at h0 down to -2.7% at h3, robust to backfill because it is trained on archived real-time forecasts; MACROCAST (2026) names revision bias in time-series foundation models for macroeconomics. Claims must be positioned against these.

## Amendment A1 (22 Oct 2026): VAL becomes real-time development
- VAL was used for confirmation and failed. It is now real-time development data. TEST (60 origins) is the only confirmation set and stays sealed until `trackp-lock`. The TEST hypothesis, method, variant and analysis will be fixed in this file before the lock.
- **New primary hypothesis H-RT.** In real-time operation, lead information helps only where it is fresh: (i) the exogenous ED signal through a corrector trained on settled real-time residuals, at short horizons; (ii) revision-aware input completion for the backbone. Finalized-history lead correction (HALO v1) is reported as a negative result with the oracle decomposition.
- **cell28 (VAL, exploratory):** oracle-input decomposition for Chronos2-group (D1), revision diagnostics (D2), Chronos2-group on nowcast-completed inputs (M1), the RT corrector on Chronos2-group, Chronos2-group-nowcast, the Hub ensemble and UMass-gbqr with feature ablations (M2), and an export of all forecasts for local analysis.
- **Paper framing if H-RT holds on TEST.** (1) Where lag information lives: pairwise leads carry nothing (E1), collective information helps only on finalized data, ED survives real time. (2) Finalized-data development reverses sign in real time, decomposed into revision and regime parts. (3) A horizon-aware real-time corrector that is backbone-agnostic (foundation model and the operational Hub ensemble). If H-RT fails on TEST, the paper is the diagnostic study (1) + (2).

## Results of cell28 (server, VAL as real-time development)
- **E21. Revisions bias Chronos-2 down at every horizon.** In real time Chronos2-group's median is low by 0.13 log units at h = 1..4; with finalized inputs the bias is about 0.01 to 0.03. Finalized inputs would improve its WIS by 30%, 20%, 15%, 11% (h = 1..4).
- **E22. For Chronos2-group the HALO v1 failure is mainly regime change, not revisions.** HALO vs backbone with finalized inputs on VAL: 0.976, 1.081, 1.164, 1.227 (real time: 1.046, 1.168, 1.137, 1.176). Lead corrections learned on 2020-24 do not transfer to 2024-25 even with clean data.
- **E23. Learned last-week completion of the inputs (nowcast) removes most of Chronos's bias** (-0.03 at h = 1..2) and lowers WIS by 8.6%, 3.1%, 0.5% (h = 1..3), +1.3% at h = 4.
- **E24. The real-time corrector (RT, own + ED + revision features) helps at short horizons on all four backbones and hurts at long horizons** (after warm-up, vs the same backbone):
  - Chronos2-group: -18.2%, -13.5%, -7.1%, +1.1% (n.s.).
  - Chronos2-group with nowcast inputs: -10.0%, -8.4%, +1.0%, +9.4%.
  - CovidHub-ensemble: -7.6%, -3.6%, +4.4%, +8.0%.
  - UMass-gbqr: -12.6%, -5.1%, +4.8%, +13.6%.
  At h = 1 the corrected Hub ensemble is the best real-time forecast (relative WIS 0.758 vs 0.835 for the ensemble). The pattern fits label staleness: a horizon-h label is at least h + 2 weeks old when it can be used.

## Local check after cell28 (Hub ensemble and persistence)
- **Horizon anchoring.** Learn only the h = 1 correction and apply rho^(h-1) times it at later horizons. On the Hub ensemble (rho = 0.8) this gives -9.4%, -3.6%, -1.2%, +0.2%, against +5.1% and +7.8% at h = 3..4 for per-horizon learning. A rule using only h <= 2 gives -9.4%, -2.5%, 0, 0.

## cell29 pre-registration (written before the run): HALO-RT design selection on VAL
- HALO-RT = (1) a real-time corrector learned only at h = 1 (own + ED + revision-state features, settled labels, honest hold-out gate); (2) horizon anchoring to h = 2..4; (3) optional online two-scale width recalibration.
- Variants: per-horizon RT (reference); fixed rho in {0.6, 0.8, 1}; online-learned rho_h; learned rho + width; rho = 0.8 + width; width only.
- Backbones: Chronos2-group, Chronos2-group-nowcast, CovidHub-ensemble, UMass-gbqr.
- **Rule.** Lowest mean log relative WIS over 4 backbones x 4 horizons (VAL after the 10-origin warm-up), among variants never worse than the backbone by more than 2% in any cell. The selected variant, with no further tuning, becomes the TEST method. The TEST protocol is written into this file before `trackp-lock`.

## Results of cell29 (server, VAL after warm-up): HALO-RT design selected
- **E25. Horizon anchoring removes the long-horizon harm on every backbone.** Selected by the pre-registered rule: **HA[rho = 0.8]** (mean log relative WIS -0.0599; worst cell 1.013, eligible). rho = 1 had a slightly better mean but a worst cell of 1.046 (ineligible). Width recalibration helps Chronos but hurts the already calibrated Hub models, so it is not selected.
- **HALO-RT vs its backbone on VAL** (Wilcoxon across 51 locations):
  - Chronos2-group: -18.2%, -11.3%, -5.8%, -3.2% (all p < 0.001).
  - Chronos2-group-nowcast: -10.0%, -7.0%, -3.7%, -2.3% (all p < 1e-4).
  - CovidHub-ensemble: -7.6%, -3.5% (p < 0.001), -0.7%, -0.5% (n.s.).
  - UMass-gbqr: -12.6%, -5.7% (p < 0.001), -1.0%, +0.4% (n.s.).
- **Real-time leaderboard on VAL** (relative WIS vs persistence, h = 1..4):
  - HALO-RT@Hub ensemble: 0.758, 0.678, 0.604, 0.520.
  - Hub ensemble: 0.835, 0.703, 0.610, 0.518.
  - HALO-RT@Chronos2-group-nowcast: 0.820, 0.742, 0.668, 0.595.
  - Chronos2-group: 1.054, 0.872, 0.719, 0.603.

## TEST protocol (locked by the git tag `trackp-lock`, written before any TEST origin is forecast or scored)
- **Data.**
  - Origins: 60 TEST origins (2025-06-28 to 2026-09-19); the five federal-shutdown origins are excluded.
  - Inputs: real-time NHSN vintages; ED share from its own real-time vintages (available for every TEST origin).
  - Truth: the latest vintage in the Hub clone used by cell24.
- **Method (frozen, no TEST tuning): HALO-RT** =
  - at h = 1, a pooled ridge (lambda 5) of the backbone's log-median error on: the backbone's predicted change, own momentum (growth lags 0..3, drawdown), ED (growth lags 0..2, phase gap, availability), and revision-state features (the location's median past first-release revision from vintages released by the origin, a missing-last-week flag, and their product with last growth);
  - labels: real-time residuals at least 2 weeks settled;
  - gate: honest hold-out (the last 8 settled origins);
  - clip: 0.5;
  - correction at horizon h = 0.8^(h-1) times the issued h = 1 correction.
  - It runs continuously from the first VAL origin; only TEST origins are scored.
- **Primary hypotheses** (WIS; Wilcoxon across 51 locations on log mean-WIS ratios; Holm over all 8 tests; 95% moving-block bootstrap intervals over origins, block 4):
  - T1: HALO-RT@Chronos2-group beats Chronos2-group.
  - T2: HALO-RT@CovidHub-ensemble beats CovidHub-ensemble.
  - Each is confirmed if at least 2 horizons are significantly better and none is significantly worse.
- **Pre-registered sensitivity:** settled targets only (target weeks at least 3 weeks before the truth vintage).
- **Secondary (reported regardless of outcome):**
  - HALO-RT on Chronos2-group-nowcast and on UMass-gbqr;
  - ED ablation (HALO-RT without ED) on Chronos2-group and the Hub ensemble;
  - per-horizon RT (no anchoring);
  - the reversal check: HALO v1 (the finalized-history corrector) on TEST;
  - the full leaderboard against the Hub models on shared rows;
  - 50% and 95% coverage;
  - three pre-registered periods: summer-fall 2025 (2025-06-28 to 2025-11-29), winter 2025-26 (2025-12-06 to 2026-04-25), spring-summer 2026 (2026-05-02 to 2026-09-19).
- **Baselines added after the lock.** TNNLS (origin-safe), CellAtt-LSTM, other foundation models (TimesFM-2.5, Moirai-2, TiRex, TabPFN-TS), each with configurations fixed in this file before its TEST run. They never feed back into HALO-RT.
- **Outcome rules.**
  - If T1 and T2 are confirmed, the method claim stands: horizon-anchored real-time correction improves both a foundation model and the operational Hub ensemble.
  - If only T1 is confirmed, the claim is limited to non-operational backbones.
  - If neither is confirmed, the paper is the diagnostic study.
  - Any TEST rerun after a code fix is reported as such, with both results.
MDEOF
  python -m py_compile src/covidfc/trackp/test30.py && echo "test30.py compiles"
  git add configs/trackp_decisions.md src/covidfc/trackp cells/cell30.sh && \
    git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "Track P lock: HALO-RT and TEST protocol frozen before TEST" || true
  git -c user.name=becky1 -c user.email=becky1@andlab tag -a trackp-lock -m "Track P lock: HALO-RT (rho 0.8), TEST protocol T1/T2" 
  echo "locked at $(git rev-list -n 1 trackp-lock)"
fi
echo "== 2. TEST evaluation"
pick_gpus 1; export CUDA_VISIBLE_DEVICES=${FREE[0]}; echo "GPU ${FREE[0]}"
python -u -m covidfc.trackp.test30
echo "== 3. Commit results summary"
git add results/trackp/test30_results.json && \
  git -c user.name=becky1 -c user.email=becky1@andlab commit -q -m "cell30: TEST results of HALO-RT (post-lock)" && git log --oneline -2
echo "[$(date +%H:%M)] cell30 done. Upload logs/cell30.log."
