"""Primary analysis — within-subject cerebellar volume change (Path C).

Uses the pre-computed SUIT lobular volumes (no reprocessing). This is the metric the
adversarial review endorsed as the only trustworthy one available: scalar SUIT
lobular/hemisphere GM volume change, within subject, vs the preop baseline.

Outputs, per metric and post-timepoint:
- n complete pairs, mean & median % change, 95% CI, paired Cohen's d_z, paired-t p
- a QC flag for |%change| > `artifact_pct` (edema/segmentation artifact, esp. 24 h)
And a linear mixed-effects trajectory (random intercept per subject) across timepoints.
If treated side is known, the crossed-asymmetry test (contra vs ipsi over time) runs.

24 h is reported but flagged as oedema, not atrophy (ADR-0002). 3 mo is the primary
timepoint; 1 yr is descriptive (tiny n).
"""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd

from . import load_config
from .io import load_cerebellar, load_clinical
from .laterality import align_hemispheres

ARTIFACT_PCT = 30.0  # |%change| above this is almost certainly oedema/segmentation error

METRICS = ["total_cerebellar_gm", "CONTRA_hemi_GM", "IPSI_hemi_GM",
           "Left_hemi_GM", "Right_hemi_GM", "asymmetry_index"]


def _paired(df, metric, ref, post):
    """Paired within-subject change for one metric, ref->post."""
    piv = df.pivot_table(index="subject", columns="session", values=metric, aggfunc="first")
    if ref not in piv.columns or post not in piv.columns:
        return None
    pair = piv[[ref, post]].dropna()
    if len(pair) < 3:
        return {"metric": metric, "timepoint": post, "n": len(pair),
                "note": "too few pairs"}
    pre, pos = pair[ref].values, pair[post].values
    delta = pos - pre
    pct = 100.0 * delta / pre
    n = len(pair)
    sd = delta.std(ddof=1)
    dz = delta.mean() / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n)
    from scipy import stats
    tval, pval = stats.ttest_rel(pos, pre)
    ci = stats.t.interval(0.95, n - 1, loc=delta.mean(), scale=se) if se > 0 else (np.nan, np.nan)
    return {
        "metric": metric, "timepoint": post, "n": n,
        "mean_delta": delta.mean(), "mean_pct": pct.mean(), "median_pct": np.median(pct),
        "ci95_delta_lo": ci[0], "ci95_delta_hi": ci[1],
        "cohen_dz": dz, "t": tval, "p": pval,
        "n_artifact_flag": int((np.abs(pct) > ARTIFACT_PCT).sum()),
    }


def _lmm(df, metric, ref, posts):
    """Mixed-effects trajectory: metric ~ C(session) + (1|subject). Returns coef table."""
    import statsmodels.formula.api as smf
    order = [ref] + list(posts)
    long = df[df["session"].isin(order)][["subject", "session", metric]].dropna().copy()
    long = long.rename(columns={metric: "y"})
    if long["subject"].nunique() < 5 or long["session"].nunique() < 2:
        return pd.DataFrame()
    long["session"] = pd.Categorical(long["session"], categories=order, ordered=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm("y ~ C(session)", long, groups=long["subject"]).fit(method="lbfgs")
        except Exception as e:
            return pd.DataFrame([{"metric": metric, "error": str(e)}])
    out = []
    for name in res.params.index:
        if name.startswith("C(session)"):
            out.append({"metric": metric, "term": name, "beta": res.params[name],
                        "se": res.bse[name], "p": res.pvalues[name]})
    return pd.DataFrame(out)


def crossed_test(df, ref, posts):
    """Test whether contralesional cerebellar GM declines MORE than ipsilesional over
    time (the crossed-diaschisis prediction): y ~ C(session)*hemi + (1|subject) on a
    long ipsi/contra stack; the session:hemi interaction is the crossed effect."""
    import statsmodels.formula.api as smf
    sub = df.dropna(subset=["IPSI_hemi_GM", "CONTRA_hemi_GM"])
    if sub.empty:
        return pd.DataFrame()
    order = [ref] + list(posts)
    rows = []
    for _, r in sub.iterrows():
        if r["session"] not in order:
            continue
        rows.append({"subject": r["subject"], "session": r["session"],
                     "hemi": "ipsi", "y": r["IPSI_hemi_GM"]})
        rows.append({"subject": r["subject"], "session": r["session"],
                     "hemi": "contra", "y": r["CONTRA_hemi_GM"]})
    long = pd.DataFrame(rows)
    if long.empty or long["subject"].nunique() < 5:
        return pd.DataFrame()
    long["session"] = pd.Categorical(long["session"], categories=order, ordered=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm("y ~ C(session)*hemi", long, groups=long["subject"]).fit(method="lbfgs")
        except Exception as e:
            return pd.DataFrame([{"error": str(e)}])
    out = [{"term": n, "beta": res.params[n], "se": res.bse[n], "p": res.pvalues[n]}
           for n in res.params.index if ":" in n]
    return pd.DataFrame(out)


def run(cfg: dict) -> dict:
    cereb = load_cerebellar(cfg)
    if cereb.empty:
        raise SystemExit("No cerebellar volumes found (cfg['cerebellar_csv']).")
    clinical = load_clinical(cfg)
    df = align_hemispheres(cereb, clinical)

    ref = cfg["reference_session"]
    posts = cfg["post_sessions"]
    have_side = df["treated_side"].notna().any()

    paired_rows = [r for m in METRICS for r in [
        _paired(df, m, ref, p) for p in posts] if r is not None]
    paired_df = pd.DataFrame(paired_rows)

    lmm_df = pd.concat([_lmm(df, m, ref, posts) for m in
                        ["total_cerebellar_gm", "CONTRA_hemi_GM", "IPSI_hemi_GM"]],
                       ignore_index=True)
    crossed = crossed_test(df, ref, posts) if have_side else pd.DataFrame()

    out_dir = cfg["derivatives"] / "C_cerebellar_longitudinal"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(out_dir / "paired_change.csv", index=False)
    lmm_df.to_csv(out_dir / "lmm_trajectory.csv", index=False)
    if not crossed.empty:
        crossed.to_csv(out_dir / "crossed_asymmetry.csv", index=False)
    df.to_csv(out_dir / "cerebellar_aligned.csv", index=False)

    return {"paired": paired_df, "lmm": lmm_df, "crossed": crossed,
            "have_side": have_side, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description="Path C primary: cerebellar volume-change.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run(cfg)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"Treated side available: {res['have_side']}  "
          f"(fill config/clinical.csv to enable crossed ipsi/contra test)\n")
    print("=== Paired within-subject change (total + hemispheres) ===")
    cols = ["metric", "timepoint", "n", "mean_pct", "median_pct", "cohen_dz", "p", "n_artifact_flag"]
    show = res["paired"]
    print(show[[c for c in cols if c in show.columns]].to_string(index=False))
    print(f"\nResults -> {res['out_dir']}")
    if not res["crossed"].empty:
        print("\n=== Crossed asymmetry (session x hemi interaction) ===")
        print(res["crossed"].to_string(index=False))


if __name__ == "__main__":
    main()
