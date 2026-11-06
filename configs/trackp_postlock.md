# Track P post-lock analyses (amendment A2)

Written on 6 Nov 2026, before cell31 runs. HALO-RT and its TEST results (lock tag `trackp-lock`, commit a01103d, cell30 on 22 Oct 2026) are not changed. The locked HALO-RT forecasts are read from cell30's output files. This file is separate from `configs/trackp_decisions.md` so the locked files stay untouched.

## A2.1 Simple online residual correction (SORC), requested by Prof. Jang (review of 5 Nov 2026)
- **Definition.** At origin s, for location i and horizon h, the correction is the mean log residual of the backbone over the last W real-time origins whose label has settled (target week s' + h <= s - 2 weeks). Labels come from the latest release available at s, the same source and the same settlement delay as HALO-RT. The correction is clipped to +-0.5 (HALO-RT's clip) and shifts all quantiles in log(1 + y) space. It uses no features, no learning and no gate.
- **Variants:**
  - SORC-4 (W = 4);
  - SORC-8 (W = 8, equal to HALO-RT's gate window), the primary baseline;
  - SORC-8-anchored (W = 8, only the h = 1 bias, applied with HALO-RT's anchoring rho = 0.8).
- **Same-machinery ablation:** HALO-RT[bias-only] = HALO-RT's ridge learner, settled labels, hold-out gate, clip and anchoring, with only the backbone's predicted change as a feature (no momentum, ED or revision features).
- **Backbones:** Chronos2-group, Chronos2-group-nowcast, CovidHub-ensemble, UMass-gbqr.
- **Periods:** all methods run continuously from the first VAL origin. Scored on VAL after the 10-origin warm-up, and on TEST.
- **Primary comparison:** HALO-RT vs SORC-8 on TEST, for Chronos2-group and CovidHub-ensemble.
  - Wilcoxon across 51 jurisdictions on log mean-WIS ratios, Holm over 8 tests (2 backbones x 4 horizons).
  - 95% moving-block bootstrap over origins (block 4, 1000 replicates).
- **Secondary:** HALO-RT vs SORC-4, SORC-8-anchored and HALO-RT[bias-only]; each SORC variant vs its backbone; the other two backbones. All are reported regardless of outcome.
- **Interpretation (fixed now):**
  - If HALO-RT beats SORC-8 at h = 1 on both primary backbones, the paper states that the gain is not explained by bias correction.
  - If SORC matches or beats HALO-RT on a backbone, this is reported as such, and the claim is restricted accordingly.

## A2.2 Post-hoc sensitivity (labelled post hoc in the paper)
- **Grid:** settling delay delta in {1, 2, 3} weeks at rho = 0.8, and rho in {0.6, 0.8, 1.0} at delta = 2, on TEST, for Chronos2-group and CovidHub-ensemble.
- **Reproduction check:** delta = 2, rho = 0.8 must reproduce the locked HALO-RT forecasts exactly.
- **Use:** not used to change HALO-RT.

## A2.3 Table III diagnostics
- Number of origins and forecasts in the shared subset of the leaderboard (all models), and the same without UM-DeepOutbreak (25 of 60 TEST origins submitted).
- HALO-RT@ensemble vs ensemble on all ensemble forecasts and on the Table III subset (geometric-mean relative WIS and median per-jurisdiction change), to explain the h = 4 difference between Tables II and III.
