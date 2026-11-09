# HALO-RT

Code for **"HALO-RT: Horizon-Anchored Real-Time Correction of COVID-19 Hospitalization Forecasts With Emergency-Department Leading Indicators"** (B. F. Gudeta, J. W. Kim, B. Jang; submitted to IEEE Journal of Biomedical and Health Informatics).

HALO-RT is a post hoc corrector for probabilistic forecasts of weekly COVID-19 hospital admissions. It learns only the one-week-ahead log error of a frozen backbone from settled real-time residuals, admits the correction through a leakage-safe validation gate, and applies it to horizon h after geometric anchoring by rho^(h-1).

## Repository layout

```
src/covidfc/trackp/        locked Track P code (HALO-RT, backbones, data, metrics)
src/covidfc/trackp_post/   post-lock analyses (SORC baseline, bias-only ablation, sensitivity)
cells/                     experiment scripts, run in numerical order
configs/trackp_decisions.md   decision log: conventions, VAL selection rule, locked TEST protocol, amendment A1
configs/trackp_postlock.md    amendment A2 (post-lock analyses), written before cell31 ran
paper/make_figures.py      figures 2, 3, 5 and 6 of the manuscript
```

## Setup

```bash
conda create -n covidfc python=3.11 -y && conda activate covidfc
pip install -r requirements.txt
export PROJ=$(pwd) PYTHONPATH=$(pwd)/src
```

Data: clone the CDC COVID-19 Forecast Hub into `data/trackp/hub` (or set `TP_HUB`):

```bash
git clone https://github.com/CDCgov/covid19-forecast-hub data/trackp/hub
```

All inputs are rebuilt from the archived releases in `target-data/time-series.parquet` (column `as_of`). The paper scores against the release of 21 October 2026.

Backbone: download the Chronos-2 weights (`amazon/chronos-2` on Hugging Face) into `models/chronos-2`. A CUDA GPU is needed for the Chronos-2 runs; HALO-RT itself runs on a CPU in under a minute.

## Reproducing the paper

| Script | Period | Paper |
|---|---|---|
| `cells/cell24.sh` to `cell26.sh` | DEV | Section IV (leading-indicator analyses) |
| `cells/cell27.sh`, `cell28.sh` | VAL | VAL confirmation of HALO-F; real-time development (amendment A1) |
| `cells/cell29.sh` | VAL | Table I, Fig. 5 (design selection) |
| `cells/cell30.sh` | TEST | Table II, Fig. 6, primary hypotheses T1 and T2 (run once after the lock) |
| `cells/cell31.sh` | VAL, TEST | Table III, Table IV, sensitivity analysis (post hoc, amendment A2) |

Results are written to `results/trackp/`, logs to `logs/`.

## Protocol record

The TEST protocol was frozen by the tag `trackp-lock` (commit `a01103d`, 22 October 2026) in the lab's internal project repository, before any TEST origin was forecast or scored. Amendment A2 was committed as `68d677a` (6 November 2026) before the post-lock analyses ran. This public repository contains the same code and decision log as a single release commit; the internal history is available to editors and reviewers on request. The lock check at the start of `cells/cell31.sh` needs that internal history and prints a warning otherwise.

## Citation

If you use this code, please cite the paper (citation will be added on publication).
