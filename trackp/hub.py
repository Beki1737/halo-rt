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
