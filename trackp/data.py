"""Track P data: weekly NHSN COVID-19 admissions and NSSP ED-visit share for 50 states + DC,
with real-time vintages from the CDC COVID-19 Forecast Hub target data (as_of, weekly from 2024-11-20).

Conventions (pre-registered in configs/trackp_decisions.md):
  * week index = epiweek ending Saturday (target_end_date).
  * origin s = the last week whose data are available at forecast time (week ending the Saturday before the
    Hub reference date R, i.e. s = R - 7 days). Horizon h = 1..4 targets week s + 7h  (Hub horizons 0..3).
  * inputs at origin s: the vintage published on the Wednesday after s (as_of = s + 4 days) when such a vintage
    exists, otherwise the finalized series truncated at s (dev period; stated limitation).
  * truth for scoring: the latest vintage (finalized values).
"""
import os, subprocess
import numpy as np, pandas as pd

PROJ = os.environ.get("PROJ", ".")
HUB = os.environ.get("TP_HUB", f"{PROJ}/data/trackp/hub")
OUT = os.environ.get("TP_OUT", f"{PROJ}/results/trackp")
HOSP, ED = "wk inc covid hosp", "wk inc covid prop ed visits"
EXCLUDE = ("US", "72")
SPLITS = {   # origin s (week-ending date) ranges; targets of every origin lie inside the same period
    "DEV":  ("2020-11-07", "2024-03-30"),   # targets <= 2024-04-27 (before the voluntary-reporting interval)
    "VAL":  ("2024-11-16", "2025-05-31"),   # mandatory reporting, partly versioned; targets <= 2025-06-28
    "TEST": ("2025-06-28", "2026-09-19"),   # sealed: fully versioned; targets <= 2026-10-17
}
VOLUNTARY = ("2024-05-01", "2024-10-31")
H = 4


def sync_hub():
    """Sparse clone (or update) of the CDC COVID-19 Forecast Hub: target data, locations, two reference models."""
    paths = ["target-data", "auxiliary-data/locations.csv", "model-output/CovidHub-ensemble",
             "model-output/CovidHub-baseline", "model-output/UMass-gbqr", "model-output/UM-DeepOutbreak"]
    if not os.path.isdir(f"{HUB}/.git"):
        os.makedirs(os.path.dirname(HUB), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/cdcgov/covid19-forecast-hub.git", HUB], check=True)
    else:
        subprocess.run(["git", "-C", HUB, "pull", "--ff-only"], check=True)
    subprocess.run(["git", "-C", HUB, "sparse-checkout", "set", "--no-cone", *paths], check=True)
    return subprocess.check_output(["git", "-C", HUB, "rev-parse", "HEAD"]).decode().strip()


def load_raw():
    t = pd.read_parquet(f"{HUB}/target-data/time-series.parquet")
    t["target_end_date"] = pd.to_datetime(t["target_end_date"]); t["as_of"] = pd.to_datetime(t["as_of"])
    t["location"] = t["location"].astype(str)
    t = t[~t["location"].isin(EXCLUDE)]
    return t


class Panel:
    """Week x location arrays for one target, finalized values plus vintage access."""

    def __init__(self, raw, target):
        d = raw[raw["target"] == target].sort_values("as_of")
        self.target = target
        fin = d.groupby(["location", "target_end_date"]).tail(1)
        P = fin.pivot(index="target_end_date", columns="location", values="observation").sort_index()
        full = pd.date_range(P.index.min(), raw["target_end_date"].max(), freq="7D")
        self.final = P.reindex(full)
        self.locs = list(self.final.columns)
        self.weeks = self.final.index
        self.vint = {a: g.pivot(index="target_end_date", columns="location", values="observation")
                     for a, g in d.groupby("as_of")}
        self.vdates = np.array(sorted(self.vint))

    def asof(self, s):
        """Values available at origin s (week-ending date): the Wednesday vintage, else finalized truncated at s."""
        s = pd.Timestamp(s); wed = s + pd.Timedelta(days=4)
        ok = self.vdates[(self.vdates <= wed) & (self.vdates > wed - pd.Timedelta(days=15))]   # latest vintage, at most 2 weeks old
        V = self.final.loc[:s].copy(); src = "final"
        if len(ok):   # the vintage overrides every week it covers; older weeks (rarely revised) come from final
            W = self.vint[ok[-1]].reindex(columns=self.locs)
            W = W[W.index <= s]
            V.loc[W.index] = W.where(W.notna(), V.loc[W.index])
            # weeks inside the vintage's span that it does not report yet are unknown at the origin
            V.loc[V.index > W.index.max()] = np.nan
            src = "vintage"
        assert len(V) == 0 or V.index.max() <= s, "provenance: a value dated after the origin"
        return V, src


def pop():
    loc = pd.read_csv(f"{HUB}/auxiliary-data/locations.csv", dtype={"location": str})
    return loc.set_index("location")["population"].astype(float)


def origins(split, panel=None):
    """Origins of a split. For VAL / TEST (versioned periods) an origin is kept only if a vintage at most two weeks
    old exists, which drops the Oct-Nov 2025 federal-shutdown weeks when no NHSN data were released and the Hub
    paused (reference dates 2025-10-04 to 2025-11-15)."""
    a, b = SPLITS[split]
    O = pd.date_range(a, b, freq="7D")
    if panel is not None and split != "DEV":
        O = pd.DatetimeIndex([s for s in O if panel.asof(s)[1] == "vintage"])
    return O


def audit(hosp, ed):
    """Coverage, reporting dropouts, vintage availability and revision size; returns a dict printed by the cell."""
    A = hosp.final
    rep = {}
    rep["weeks"] = (str(A.index.min().date()), str(A.index.max().date()), len(A))
    rep["locations"] = len(hosp.locs)
    vol = A.loc[VOLUNTARY[0]:VOLUNTARY[1]]
    pre = A.loc["2024-01-01":"2024-04-27"].median()
    drop = (vol.median() < 0.2 * pre).sum()
    rep["voluntary_interval_dropout_states"] = int(drop)
    rep["hosp_vintages"] = (str(hosp.vdates.min().date()), str(hosp.vdates.max().date()), len(hosp.vdates))
    rep["ed_weeks"] = (str(ed.final.dropna(how="all").index.min().date()), str(ed.final.index.max().date()))
    rep["ed_vintages"] = (str(ed.vdates.min().date()), str(ed.vdates.max().date()), len(ed.vdates))
    # first-release vs final relative revision of admissions (versioned period)
    revs = []
    for a in hosp.vdates[1:]:
        V = hosp.vint[a]
        last = V.index.max()
        if last not in A.index: continue
        f, v = A.loc[last, V.columns], V.loc[last]
        m = (f > 20)
        revs.append(((f[m] - v[m]) / f[m]).to_numpy())
    r = np.concatenate(revs) if revs else np.array([np.nan])
    rep["first_release_revision_median_pct"] = round(100 * float(np.nanmedian(r)), 1)
    rep["first_release_revision_p90_pct"] = round(100 * float(np.nanpercentile(r, 90)), 1)
    for sp in SPLITS:
        O0, O = origins(sp), origins(sp, hosp)
        rep[f"{sp}_origins"] = (len(O), f"{len(O0) - len(O)} dropped (no vintage within 2 weeks)")
    return rep
