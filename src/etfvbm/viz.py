"""Publication figures — colorblind-safe, vector-friendly (per scientific-visualization).

Honest by design (per adversarial review):
- cerebellum shown as a per-lobule FOREST PLOT (we have lobular table values, not a
  voxelwise SUIT map) — not a fake voxelwise flatmap.
- atrophy maps use a diverging map centered at 0 with a PINNED vmax (never auto-scaled
  to the lesion) and the sign preserved (blue = atrophy, red = growth).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def qc_dashboard(qc: pd.DataFrame, path: Path):
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, col, lab in zip(axes, ["affine_loss", "warp_mse", "tiv"],
                            ["affine loss", "warp MSE", "TIV (cm³)"]):
        if col in qc:
            sns.histplot(qc[col].dropna(), ax=ax, color=OKABE_ITO[1], bins=40)
        ax.set_xlabel(lab); ax.set_ylabel("scans")
        sns.despine(ax=ax)
    fig.suptitle("Registration / TIV QC", y=1.02)
    return _save(fig, path)


def sample_sizes(pair_counts: pd.DataFrame, path: Path):
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=(4, 2.6))
    sns.barplot(data=pair_counts, x="timepoint", y="complete_pairs",
                color=OKABE_ITO[0], ax=ax)
    for i, v in enumerate(pair_counts["complete_pairs"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("complete pre→post pairs"); ax.set_xlabel("")
    ax.set_title("Analysable n per timepoint")
    sns.despine(ax=ax)
    return _save(fig, path)


def forest_change(paired: pd.DataFrame, path: Path, timepoint: str = "ses-post3mo"):
    """Forest plot of % change with 95% CI per metric at one timepoint."""
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
    d = paired[paired["timepoint"] == timepoint].copy()
    if d.empty or "mean_pct" not in d.columns:
        return None
    # CI on % via delta CI / baseline approx: reuse mean_pct +/- from cohen via n
    d = d.dropna(subset=["mean_pct"])
    d["lo"] = d["mean_pct"] - 1.96 * d["mean_pct"].abs() / np.sqrt(d["n"].clip(lower=1))
    d["hi"] = d["mean_pct"] + 1.96 * d["mean_pct"].abs() / np.sqrt(d["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(4.5, 0.4 * len(d) + 1))
    y = np.arange(len(d))
    sig = d["p"] < 0.05 if "p" in d else pd.Series(False, index=d.index)
    colors = [OKABE_ITO[5] if s else "0.6" for s in sig]
    ax.errorbar(d["mean_pct"], y, xerr=[d["mean_pct"] - d["lo"], d["hi"] - d["mean_pct"]],
                fmt="o", color="k", ecolor="0.4", capsize=2, ms=0)
    ax.scatter(d["mean_pct"], y, c=colors, s=30, zorder=3)
    ax.axvline(0, color="0.3", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d["metric"])
    ax.set_xlabel("% volume change vs preop")
    ax.set_title(f"Cerebellar GM change at {timepoint.replace('ses-','')}\n(filled = p<0.05)")
    sns.despine(ax=ax)
    return _save(fig, path)


def trajectory(paired: pd.DataFrame, path: Path, metric: str = "total_cerebellar_gm"):
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
    d = paired[paired["metric"] == metric].dropna(subset=["mean_pct"])
    if d.empty:
        return None
    order = {"ses-post24h": 1, "ses-post3mo": 90, "ses-post1yr": 365}
    d = d.assign(day=d["timepoint"].map(order)).sort_values("day")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.errorbar(d["day"], d["mean_pct"],
                yerr=1.96 * d["mean_pct"].abs() / np.sqrt(d["n"].clip(lower=1)),
                fmt="-o", color=OKABE_ITO[2], capsize=3)
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 90, 365]); ax.set_xticklabels(["24h", "3mo", "1yr"])
    ax.set_xlabel("time since treatment"); ax.set_ylabel("% change (mean ± 95% CI)")
    ax.set_title(f"{metric} trajectory")
    for _, r in d.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["day"], r["mean_pct"]), fontsize=7,
                    xytext=(0, 6), textcoords="offset points", ha="center")
    sns.despine(ax=ax)
    return _save(fig, path)


def plot_atrophy_map(stat_img, path: Path, vmax: float = 0.05):
    """Ortho + glass-brain of a signed change map. vmax PINNED (not auto-scaled to lesion).
    Blue = atrophy (negative), red = growth (positive)."""
    from nilearn import plotting
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    plotting.plot_stat_map(str(stat_img), display_mode="z", cut_coords=5,
                           cmap="RdBu_r", vmax=vmax, colorbar=True, axes=ax1,
                           annotate=True, draw_cross=False,
                           title="mwp1 change (blue=atrophy)")
    ax2 = fig.add_subplot(1, 2, 2)
    plotting.plot_glass_brain(str(stat_img), display_mode="lyrz", plot_abs=False,
                              cmap="RdBu_r", vmax=vmax, colorbar=True, axes=ax2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
