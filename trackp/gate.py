"""Evidence gate for pairwise fixed lags (the TNNLS / CellAtt-LSTM premise), development data only.
For target location r, horizon h, leader j and lead l >= h (leader change observed at the origin):
    rho = corr( target change y_r(s+h) - y_r(s) - base_h(s) ,  leader change y_j(s+h-l) - y_j(s-l) )
base 'persist' = 0 (raw change), base 'trend' = h * last weekly growth (target's own momentum).
The maximum over leaders and lags is compared with a circular-shift permutation null (leaders shifted by
26..T-26 weeks, target untouched), so the selection over ~400 candidates is accounted for."""
import numpy as np, pandas as pd

H, LMAX = (1, 2, 3, 4), 8


def _corrmat(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-12); B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return A.T @ B / len(A)          # R x R : target x leader


def best_rho(y, ylead, base):
    T, R = y.shape; s = np.arange(LMAX + 2, T - max(H))
    best = np.full((R, len(H)), -np.inf); arg = np.zeros((R, len(H)), int)
    for hi, h in enumerate(H):
        tgt = y[s + h] - y[s]
        if base == "trend": tgt = tgt - h * (y[s] - y[s - 1])
        for l in range(h, LMAX + 1):
            C = _corrmat(tgt, ylead[s + h - l] - ylead[s - l]); np.fill_diagonal(C, -np.inf)
            m = C.max(1); upd = m > best[:, hi]; best[upd, hi] = m[upd]; arg[upd, hi] = l
    return best, arg


def run(final, start="2020-08-08", end="2024-04-27", perms=200, seed=2026):
    A = final.loc[start:end]
    y = np.log1p(A.clip(lower=0)).interpolate(limit=2).bfill().ffill().to_numpy()
    T, R = y.shape; rng = np.random.default_rng(seed); out = []
    for base in ("persist", "trend"):
        best, arg = best_rho(y, y, base)
        null = np.stack([best_rho(y, np.stack([np.roll(y[:, j], k) for j, k in
                                               enumerate(rng.integers(26, T - 26, R))], 1), base)[0]
                         for _ in range(perms)])
        p = (null >= best[None]).mean(0)
        for hi, h in enumerate(H):
            out.append({"base": base, "h": h, "median_best_rho": np.median(best[:, hi]),
                        "median_null_rho": np.median(null[:, :, hi]), "share_p<0.05": (p[:, hi] < 0.05).mean(),
                        "modal_lag": int(np.bincount(arg[:, hi]).argmax())})
    return pd.DataFrame(out)
