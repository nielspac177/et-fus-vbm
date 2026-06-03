"""Structure-symptom association (ROI-level) — does regional atrophy track outcomes?

Primary, feasible-without-reprocessing design (endorsed by the adversarial review):
ROI = SUIT cerebellar lobules + hemispheres (+ thalamus when available). Metric =
within-subject volume change (post - pre) of the scalar SUIT volume. Outcomes:
  - tremor improvement (continuous)  -> linear regression
  - imbalance / weakness (binary AE) -> Firth-penalized logistic (rare events)
Covariates (pre-specified): age, sex, TIV(/global), baseline severity, treated_side,
and LESION BURDEN. Multiple comparisons: Benjamini-Hochberg FDR across the ROI family,
per outcome. 3-month timepoint primary; 24 h excluded (oedema); 1 yr descriptive.

CRITICAL CONFOUND (do not run a causal claim without it): lesion volume causes BOTH
atrophy and outcome. Provide a lesion-burden column in clinical.csv (`lesion_volume`,
or a dose surrogate such as `n_sonications`/`max_temp`/`energy`). If absent, this module
runs but loudly flags every association as UNADJUSTED-FOR-LESION (confounded).
"""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd

from . import load_config
from .io import load_cerebellar, load_clinical
from .laterality import align_hemispheres, _PAIRED_STEMS

LESION_COLS = ["lesion_volume", "n_sonications", "max_temp", "energy"]


def _change_table(cfg, timepoint):
    """Per-subject within-subject % change for each ROI metric at one timepoint."""
    cereb = load_cerebellar(cfg)
    clin = load_clinical(cfg)
    df = align_hemispheres(cereb, clin)
    rois = (["total_cerebellar_gm", "CONTRA_hemi_GM", "IPSI_hemi_GM"]
            + [f"CONTRA_{s}" for s in _PAIRED_STEMS] + [f"IPSI_{s}" for s in _PAIRED_STEMS])
    rois = [r for r in rois if r in df.columns]
    ref = cfg["reference_session"]
    rows = {}
    for roi in rois:
        piv = df.pivot_table(index="subject", columns="session", values=roi, aggfunc="first")
        if ref in piv.columns and timepoint in piv.columns:
            pair = piv[[ref, timepoint]].dropna()
            rows[roi] = (100.0 * (pair[timepoint] - pair[ref]) / pair[ref])
    change = pd.DataFrame(rows)
    return change, clin


def _fit_one(change_col, outcome, covars, binary):
    """Return (beta, se, p, n) for the outcome term in: change ~ outcome + covars."""
    import statsmodels.api as sm
    data = pd.concat([change_col.rename("y"), outcome.rename("x"), covars], axis=1).dropna()
    if len(data) < 10:
        return dict(beta=np.nan, se=np.nan, p=np.nan, n=len(data), note="n<10")
    X = sm.add_constant(data[["x"] + list(covars.columns)], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if binary:
            try:  # Firth via penalized logit if available, else regular logit
                res = sm.Logit(data["y"], X).fit_regularized(disp=0)
            except Exception:
                res = sm.Logit(data["y"], X).fit(disp=0)
        else:
            res = sm.OLS(data["y"], X).fit()
    return dict(beta=res.params.get("x", np.nan), se=getattr(res, "bse", {}).get("x", np.nan),
                p=getattr(res, "pvalues", {}).get("x", np.nan), n=len(data))


def fdr_bh(p, q=0.05):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        val = pv[order[rank]] * m / (rank + 1)
        prev = min(prev, val)
        adj[order[rank]] = prev
    out[idx] = adj
    return out


def run(cfg, timepoint="ses-post3mo", outcome_col="tremor_improvement", binary=False):
    change, clin = _change_table(cfg, timepoint)
    if clin.empty:
        raise SystemExit("clinical.csv is empty — fill it (treated_side, tremor scores, "
                         "lesion burden) before running associations.")
    clin = clin.set_index("subject")
    change.index = change.index  # subject
    if outcome_col not in clin.columns:
        raise SystemExit(f"Outcome '{outcome_col}' not in clinical.csv columns {list(clin.columns)}")
    outcome = pd.to_numeric(clin[outcome_col], errors="coerce")
    covar_cols = [c for c in ["age", "sex", "tiv", "tremor_pre"] if c in clin.columns]
    lesion = [c for c in LESION_COLS if c in clin.columns]
    has_lesion = len(lesion) > 0
    covars = clin[covar_cols + lesion].copy()
    if "sex" in covars:  # 'M'/'F' (or 1/2) -> numeric so it survives to_numeric
        covars["sex"] = covars["sex"].map({"M": 0, "F": 1, "1": 0, "2": 1, 1: 0, 2: 1})
    covars = covars.apply(pd.to_numeric, errors="coerce")

    res = []
    for roi in change.columns:
        r = _fit_one(change[roi], outcome, covars, binary)
        r["roi"] = roi
        res.append(r)
    out = pd.DataFrame(res)
    out["q_fdr"] = fdr_bh(out["p"].values)
    out["lesion_adjusted"] = has_lesion
    out = out.sort_values("p")

    out_dir = cfg["derivatives"] / "association"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"assoc_{outcome_col}_{timepoint}.csv"
    out.to_csv(fname, index=False)
    if not has_lesion:
        print("WARNING: no lesion-burden column found -> associations are CONFOUNDED by "
              "lesion size (add lesion_volume / n_sonications / max_temp to clinical.csv).")
    print(f"Association {outcome_col} @ {timepoint}: {len(out)} ROIs -> {fname}")
    print(out[["roi", "beta", "p", "q_fdr", "n"]].head(10).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="ROI structure-symptom association.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--timepoint", default="ses-post3mo")
    ap.add_argument("--outcome", default="tremor_improvement")
    ap.add_argument("--binary", action="store_true", help="Outcome is a binary AE (Firth logistic).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.timepoint, args.outcome, args.binary)


if __name__ == "__main__":
    main()
