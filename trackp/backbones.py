"""Base forecasters for Track P. Every function takes the data available at origin s only.
Rows returned: model, loc, s, h, q0.025..q0.975 (counts)."""
import os, numpy as np, pandas as pd
from covidfc.trackp.metrics import QS, QC
from covidfc.trackp.data import H

W7 = pd.Timedelta(days=7)


def _rows(model, s, loc, Q):
    out = []
    for i, l in enumerate(loc):
        for h in range(1, H + 1):
            out.append([model, l, s, h, *np.sort(Q[i, h - 1])])
    return out


def persistence(s, V):
    """Last observed value; quantiles from the location's trailing 52-week h-step log changes, centred at 0."""
    Y = np.log1p(V.clip(lower=0).ffill())
    last = Y.iloc[-1].to_numpy()
    Q = np.zeros((len(V.columns), H, len(QS)))
    for h in range(1, H + 1):
        d = (Y - Y.shift(h)).iloc[-52:]
        d = d - d.median()
        qd = np.nanquantile(d.to_numpy(), QS, axis=0).T            # loc x 7
        qd = np.nan_to_num(qd, nan=0.0)
        Q[:, h - 1] = np.expm1(last[:, None] + qd)
    return _rows("Persistence", s, list(V.columns), np.clip(Q, 0, None))


_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        d = os.path.join(os.environ["PROJ"], "models", "chronos-2")
        _PIPE = Chronos2Pipeline.from_pretrained(d, device_map="cuda")
    return _PIPE


def chronos(s, V, mode="uni", Ved=None, ctx_weeks=260):
    """Chronos-2 zero-shot on raw weekly counts. mode: 'uni' (series independent), 'group' (cross-learning over the
    51 locations of this one origin), 'edcov' (own ED share as a past-only covariate)."""
    V = V.iloc[-ctx_weeks:]
    loc = list(V.columns)
    frames = []
    for l in loc:
        f = pd.DataFrame({"id": l, "timestamp": V.index, "target": V[l].to_numpy(float)})
        if mode in ("edcov", "group-edcov"):
            e = Ved.reindex(V.index)[l].to_numpy(float) if (Ved is not None and l in Ved.columns) else np.full(len(V), np.nan)
            f["ed"] = e
        frames.append(f)
    C = pd.concat(frames, ignore_index=True)
    if mode in ("edcov", "group-edcov") and not np.isfinite(C["ed"].to_numpy(float)).any():
        C = C.drop(columns="ed")          # no ED data yet at this origin: identical to the univariate call
    R = _pipe().predict_df(C, id_column="id", timestamp_column="timestamp", target="target", prediction_length=H,
                           quantile_levels=list(QS), batch_size=64, cross_learning=mode.startswith("group"))
    R = R.sort_values(["id", "timestamp"])
    Q = np.zeros((len(loc), H, len(QS)))
    for i, l in enumerate(loc):
        r = R[R["id"] == l]
        Q[i] = r[[str(q) for q in QS]].to_numpy(float)[:H]
    return _rows({"uni": "Chronos2", "group": "Chronos2-group", "edcov": "Chronos2-EDcov",
                  "group-edcov": "Chronos2-group-EDcov"}[mode], s, loc,
                 np.clip(Q, 0, None))


def damped(s, V, phi=0.8):
    """Offline stand-in backbone for local testing (damped log-trend), same quantile recipe as persistence."""
    Y = np.log1p(V.clip(lower=0).ffill())
    g = (Y.iloc[-1] - Y.iloc[-2]).to_numpy()
    Q = np.zeros((len(V.columns), H, len(QS)))
    for h in range(1, H + 1):
        mu = Y.iloc[-1].to_numpy() + g * sum(phi ** k for k in range(1, h + 1))
        d = (Y - Y.shift(h)).iloc[-52:]; d = d - d.median()
        qd = np.nan_to_num(np.nanquantile(d.to_numpy(), QS, axis=0).T)
        Q[:, h - 1] = np.expm1(mu[:, None] + qd)
    return _rows("Damped", s, list(V.columns), np.clip(Q, 0, None))
