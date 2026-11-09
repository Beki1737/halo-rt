"""Figures for the HALO-RT manuscript (IEEE JBHI submission).
Data panels (Figs. 2, 3) are computed from the CDC COVID-19 Forecast Hub target data (time-series.parquet with as_of
release dates, restricted to releases up to 21 October 2026, the truth release used in the evaluation); result
panels (Figs. 5, 6) use the scores logged by the locked evaluation runs (cell29 VAL, cell30 TEST).
usage: python make_figures.py --hub <path to covid19-forecast-hub>"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HUB = sys.argv[sys.argv.index("--hub") + 1] if "--hub" in sys.argv else "hub"
OUT = os.path.dirname(os.path.abspath(__file__))
TRUTH_ASOF = pd.Timestamp("2026-10-21")
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#8a8984", "grid": "#e4e3df", "band": "#f2f1ed"}
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6, "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "hatch.color": "#c9c8c3", "hatch.linewidth": 0.5})
W1, W2 = 3.5, 7.16
H = [1, 2, 3, 4]
x = np.arange(4)
def grid(ax, axis="y"):
    ax.grid(axis=axis, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
def tag(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")

# ------------------------------------------------------------------ data
t = pd.read_parquet(f"{HUB}/target-data/time-series.parquet")
t["target_end_date"] = pd.to_datetime(t["target_end_date"]); t["as_of"] = pd.to_datetime(t["as_of"])
t["location"] = t["location"].astype(str); t = t[~t["location"].isin(["US", "72"]) & (t["as_of"] <= TRUTH_ASOF)]
hosp = t[t["target"] == "wk inc covid hosp"]
fin = hosp.sort_values("as_of").groupby(["location", "target_end_date"]).tail(1)
A = fin.pivot(index="target_end_date", columns="location", values="observation").sort_index()

# Fig. 2 ------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.28})
nat = A.sum(1, min_count=40) / 1000
P = {"DEV": ("2020-11-07", "2024-03-30", C["band"]), "VAL": ("2024-11-16", "2025-05-31", "#e6eef9"),
     "TEST": ("2025-06-28", "2026-09-19", "#fde9e0")}
for nm, (a, b, col) in P.items():
    a1.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(weeks=4), color=col, lw=0, zorder=0)
a1.axvspan(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-11-09"), fc="white", hatch="//////", ec="#c9c8c3", lw=0, zorder=0.5)
a1.axvspan(pd.Timestamp("2025-10-04"), pd.Timestamp("2025-11-15"), fc="white", hatch="//////", ec="#c9c8c3", lw=0, zorder=0.5)
a1.plot(nat.index, nat.values, color=C["ink"], lw=1.0, zorder=3)
a1.set_ylabel("Weekly admissions (thousands)")
a1.set_xlim(pd.Timestamp("2020-08-01"), pd.Timestamp("2026-10-31")); a1.set_ylim(0, None); grid(a1)
ymax = a1.get_ylim()[1]
for nm, (a, b, col) in P.items():
    mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
    a1.text(mid, ymax * 0.97, nm, ha="center", va="top", fontsize=7, color=C["ink2"])
a1.text(pd.Timestamp("2024-08-01"), ymax * 0.80, "voluntary\nreporting", ha="center", fontsize=6.3, color=C["ink2"],
        bbox=dict(fc="white", ec="none", pad=0.6))
a1.text(pd.Timestamp("2025-10-25"), ymax * 0.62, "federal\nshutdown", ha="center", fontsize=6.3, color=C["ink2"],
        bbox=dict(fc="white", ec="none", pad=0.6))
tag(a1, "(a)")
rows = []
vint = {a: g.pivot(index="target_end_date", columns="location", values="observation") for a, g in hosp.groupby("as_of")}
for a in sorted(vint):
    V = vint[a]; last = V.index.max()
    if last > pd.Timestamp("2026-09-19"): continue
    for l in range(4):
        w = last - pd.Timedelta(weeks=l)
        if w not in V.index or w not in A.index: continue
        f, v = A.loc[w, V.columns], V.loc[w]; m = (f > 20) & v.notna()
        rows += [(l, q) for q in (100 * (np.expm1(np.log1p(v[m]) - np.log1p(f[m])))).to_numpy()]
R = pd.DataFrame(rows, columns=["lag", "rev"])
data = [R.loc[R["lag"] == l, "rev"].clip(-60, 25).to_numpy() for l in range(4)]
bp = a2.boxplot(data, positions=range(4), widths=0.5, showfliers=False, patch_artist=True, whis=(5, 95),
                medianprops=dict(color=C["ink"], lw=1.0), whiskerprops=dict(color=C["ink2"], lw=0.6),
                capprops=dict(color=C["ink2"], lw=0.6), boxprops=dict(lw=0.6, color=C["ink2"]))
for k, b in enumerate(bp["boxes"]): b.set_facecolor(C["blue"] if k == 0 else "#bcd3f1")
a2.axhline(0, color=C["ink2"], lw=0.6, ls=(0, (2, 2)))
a2.set_xticks(range(4), ["0", "1", "2", "3"]); a2.set_xlabel("Weeks before release date (reporting lag)")
a2.set_ylabel("First release vs. final value (%)"); grid(a2); tag(a2, "(b)")
med = R.groupby("lag")["rev"].median(); a2.set_ylim(-42, 7)
for l in range(4): a2.text(l, 2.5, f"{med[l]:.1f}%", ha="center", fontsize=6.3, color=C["ink2"])
fig.savefig(f"{OUT}/fig2_data.pdf"); plt.close(fig)
print("revision stats (median %, share below -20%):")
print(R.groupby("lag")["rev"].agg(["median", lambda q: (q < -20).mean()]).round(3))
print("admission releases:", hosp["as_of"].nunique(), hosp["as_of"].min().date(), hosp["as_of"].max().date())
edv = t[t["target"] != "wk inc covid hosp"]
print("other targets:", edv["target"].unique(), edv["as_of"].nunique(), edv["as_of"].min().date())

# Fig. 3 ------------------------------------------------------------------------------------------
fig, (b1, b2) = plt.subplots(1, 2, figsize=(W2, 1.95), gridspec_kw={"wspace": 0.3})
raw = [96.1, 98.0, 86.3, 49.0]; w = 0.36
b1.bar(x - w / 2 - 0.01, raw, w, color=C["muted"], label="raw change of the target", zorder=3)
b1.bar(x + w / 2 + 0.01, [0.6] * 4, w, color=C["orange"], label="net of the target's own trend", zorder=3)
for i in range(4): b1.text(x[i] + w / 2 + 0.01, 3, "0", ha="center", fontsize=6.5, color=C["ink2"])
b1.set_xticks(x, [f"$h={h}$" for h in H]); b1.set_ylim(0, 110); b1.set_ylabel("States with a significant\nfixed-lag leader (%)")
b1.legend(loc="upper right", handlelength=1.0); grid(b1); tag(b1, "(a)")
own = np.array([0.1738, 0.2526, 0.3351, 0.4255]); nat_ = np.array([0.1560, 0.2215, 0.2930, 0.3732])
ed_ = np.array([0.1398, 0.1952, 0.2614, 0.3453]); both = np.array([0.1390, 0.1936, 0.2548, 0.3338])
for k, (v, nm, col) in enumerate(((nat_, "+ national wave", C["blue"]), (ed_, "+ ED-visit share", C["orange"]),
                                  (both, "+ national + ED", C["aqua"]))):
    b2.bar(x + (k - 1) * 0.26, 100 * (v / own - 1), 0.24, color=col, label=nm, zorder=3)
b2.set_xticks(x, [f"$h={h}$" for h in H]); b2.set_ylabel("Change in MAE vs. own\nmomentum only (%)")
b2.axhline(0, color=C["ink2"], lw=0.6); b2.set_ylim(-32, 0)
b2.legend(loc="lower center", ncol=3, handlelength=1.0, columnspacing=0.7, handletextpad=0.4, borderaxespad=0.2)
grid(b2); tag(b2, "(b)")
fig.savefig(f"{OUT}/fig3_leads.pdf"); plt.close(fig)

# Fig. 5: horizon anchoring vs per-horizon correction on VAL -------------------------------------------
col = {"Hub ensemble": C["blue"], "Chronos-2": C["orange"], "UMass-gbqr": C["aqua"]}
VALd = {"Chronos-2": ([0.811, 0.872, 0.948, 1.027], [0.811, 0.891, 0.945, 0.976]),
        "Hub ensemble": ([0.906, 0.975, 1.051, 1.078], [0.906, 0.964, 0.988, 1.002]),
        "UMass-gbqr": ([0.871, 0.952, 1.067, 1.109], [0.871, 0.948, 0.992, 1.013])}
fig, ax = plt.subplots(figsize=(W1, 2.05))
for bk, (ph, an) in VALd.items():
    ax.plot(H, 100 * (np.array(ph) - 1), color=col[bk], lw=1.1, ls=(0, (3, 2)))
    ax.plot(H, 100 * (np.array(an) - 1), color=col[bk], lw=2.0, marker="o", ms=4, mec="white", mew=0.7)
ax.axhline(0, color=C["ink2"], lw=0.6); ax.set_xticks(H, [f"$h={h}$" for h in H]); grid(ax)
ax.set_xlim(0.85, 4.15); ax.set_ylim(-21, 13)
ax.set_ylabel("Relative WIS vs. backbone (%)")
hl = [Line2D([], [], color=col[k], lw=2) for k in col] + [Line2D([], [], color=C["ink2"], lw=2, marker="o", ms=4),
      Line2D([], [], color=C["ink2"], lw=1.1, ls=(0, (3, 2)))]
ax.legend(hl, list(col) + ["anchored ($\\rho=0.8$)", "per-horizon"], loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=3, columnspacing=0.9, handlelength=1.6)
fig.savefig(f"{OUT}/fig5_anchoring.pdf"); plt.close(fig)

# Fig. 6: periods heatmap -----------------------------------------------------------------------------
PER = ["Summer-fall 2025", "Winter 2025-26", "Spring-summer 2026"]
D = {"Chronos-2": [[0.959, 0.983, 0.986, 0.992], [0.944, 0.967, 0.977, 0.987], [0.83, 0.859, 0.889, 0.915]],
     "Hub ensemble": [[0.919, 0.959, 0.969, 0.979], [0.933, 0.969, 0.995, 1.016], [0.939, 0.954, 0.966, 0.971]],
     "UMass-gbqr": [[1.10, 1.043, 1.01, 0.962], [0.921, 0.963, 0.984, 1.0], [0.897, 0.917, 0.944, 0.956]]}
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.45), gridspec_kw={"wspace": 0.08})
cm = LinearSegmentedColormap.from_list("div", ["#1f5fae", "#8fb7e8", "#f2f1ed", "#f2a98a", "#c43d1f"])
nm = TwoSlopeNorm(vcenter=0, vmin=-17, vmax=17)
for k, (ax, (bk, M)) in enumerate(zip(axs, D.items())):
    M = 100 * (np.array(M) - 1); ax.imshow(M, cmap=cm, norm=nm, aspect="auto")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(M[i, j]) > 11 else C["ink"])
    ax.set_xticks(range(4), [f"$h={h}$" for h in H]); ax.set_yticks(range(3), PER if k == 0 else [])
    ax.set_title(bk); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
fig.savefig(f"{OUT}/fig6_periods.pdf"); plt.close(fig)
print("figures written to", OUT)
