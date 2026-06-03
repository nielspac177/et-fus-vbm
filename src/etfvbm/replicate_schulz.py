"""Replicate Schulz et al. 2022 (Brain Communications, fcac203) for FUS-ET.

Design (adapted): acute-timepoint regional CEREBELLAR volumes predict later functional
OUTCOME via median-split ordinal logistic regression. For FUS we use the **24 h** scan as
the acute baseline T1 (the cerebellum is remote from the thalamic lesion, so 24 h cerebellar
volume is not edema-corrupted) and an outcome at T2 (3 mo / 1 yr).

Per cerebellar region (whole-lobule = L+R+vermis), the cohort is **median-split** into
larger vs smaller volume. Ordinal logistic regression (statsmodels OrderedModel, logit):

    outcome_T2  ~  volume_group(smaller vs larger=ref)
                 + log10(lesion_volume) + age_resid + ICV_resid
                 [+ baseline_severity]      # 'adjusted' model (NIHSS analog)

age and ICV are residualized against the regional volume first (collinearity), per the
paper. Odds ratios (ref = larger volume) for a WORSE outcome are reported with 95% CI and p;
OR<1 => larger volume protects. Leave-one-out (LOOA) re-fits to probe robustness of
significant regions. Significance p<0.05.

Requires: clinical.csv with the outcome column + age + baseline severity; lesion_burden.csv
(from etfvbm.lesion) for lesion_volume; TIV from the cohort manifest as ICV.
"""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd

from . import load_config
from .io import load_cerebellar, load_clinical, build_cohort
from .association import fdr_bh

# Whole-lobule SUIT regions available in our table (CERES 13-region analog).
LOBULES = ["I_IV", "V", "VI", "CrusI", "CrusII", "VIIb", "VIIIa", "VIIIb", "IX", "X"]


def _whole_lobule(cereb: pd.DataFrame) -> pd.DataFrame:
    out = cereb[["subject", "session"]].copy()
    for lob in LOBULES:
        cols = [c for c in (f"GM_Left_{lob}", f"GM_Right_{lob}", f"GM_Vermis_{lob}")
                if c in cereb.columns]
        if cols:
            out[lob] = cereb[cols].sum(axis=1, min_count=1)
    if "total_cerebellar_gm" in cereb.columns:
        out["total"] = cereb["total_cerebellar_gm"]
    return out


def _residualize(y, x):
    """Residual of y regressed on x (both 1-D, NaNs dropped pairwise) -> aligned series."""
    import statsmodels.api as sm
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(d) < 5:
        return y * np.nan
    res = sm.OLS(d["y"], sm.add_constant(d["x"])).fit()
    out = pd.Series(np.nan, index=y.index)
    out.loc[d.index] = res.resid
    return out


def _ordinal_or(df, region, outcome, covars):
    """Fit ordinal logit; return OR/CI/p for the (smaller-volume) group indicator."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    d = df.dropna(subset=[region, outcome] + covars).copy()
    if len(d) < 15 or d[outcome].nunique() < 2:
        return None
    med = d[region].median()
    d["smaller"] = (d[region] < med).astype(float)   # larger = reference (0)
    X = d[["smaller"] + covars].astype(float)
    y = pd.Categorical(d[outcome], ordered=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=0)
        except Exception as e:
            return {"region": region, "n": len(d), "error": str(e)}
    if "smaller" not in res.params:
        return None
    beta, se, p = res.params["smaller"], res.bse["smaller"], res.pvalues["smaller"]
    return {"region": region, "n": len(d), "OR": float(np.exp(beta)),
            "ci_lo": float(np.exp(beta - 1.96 * se)), "ci_hi": float(np.exp(beta + 1.96 * se)),
            "p": float(p)}


def _looa(df, region, outcome, covars):
    """Leave-one-out: fraction of refits that keep p<0.05 (robustness)."""
    sig = 0
    idx = df.dropna(subset=[region, outcome] + covars).index
    n = len(idx)
    if n < 16:
        return np.nan
    for drop in idx:
        r = _ordinal_or(df.drop(index=drop), region, outcome, covars)
        if r and r.get("p", 1) < 0.05:
            sig += 1
    return sig / n


def run(cfg, outcome="tremor_post3mo", predictor_session="ses-post24h", adjusted=False):
    cereb = load_cerebellar(cfg)
    wl = _whole_lobule(cereb)
    wl = wl[wl["session"] == predictor_session].set_index("subject")

    clin = load_clinical(cfg)
    if clin.empty:
        raise SystemExit("clinical.csv empty — need outcome + age (+ baseline severity).")
    clin = clin.set_index("subject")

    # ICV from TIV (preop) in the manifest
    cohort = build_cohort(cfg)
    tiv = (cohort[cohort["session"] == cfg["reference_session"]]
           .set_index("subject")["tiv"])

    lesion_csv = cfg["derivatives"] / "lesion_burden.csv"
    if not lesion_csv.exists():
        raise SystemExit("Run `python -m etfvbm.lesion` first (need lesion_volume covariate).")
    import pandas as pd
    lesvol = pd.read_csv(lesion_csv).set_index("subject")["lesion_volume_cm3"]

    df = wl.join(clin[[c for c in [outcome, "age", "tremor_pre"] if c in clin.columns]])
    df["ICV"] = tiv
    df["log10_lesion"] = np.log10(lesvol.reindex(df.index).clip(lower=1e-3))
    for c in [outcome, "age", "ICV", "tremor_pre"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    regions = [r for r in LOBULES + ["total"] if r in df.columns]
    rows = []
    for region in regions:
        d = df.copy()
        d["age_resid"] = _residualize(d["age"], d[region]) if "age" in d else np.nan
        d["ICV_resid"] = _residualize(d["ICV"], d[region]) if "ICV" in d else np.nan
        covars = ["log10_lesion", "age_resid", "ICV_resid"]
        if adjusted and "tremor_pre" in d.columns:
            covars.append("tremor_pre")
        covars = [c for c in covars if c in d.columns and d[c].notna().any()]
        r = _ordinal_or(d, region, outcome, covars)
        if r and "OR" in r:
            r["looa_frac_sig"] = _looa(d, region, outcome, covars) if r["p"] < 0.05 else np.nan
            r["adjusted"] = adjusted
        if r:
            rows.append(r)

    out = pd.DataFrame(rows)
    if "p" in out.columns:
        out["q_fdr"] = fdr_bh(out["p"].values)
        out = out.sort_values("p")
    out_dir = cfg["derivatives"] / "schulz_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"schulz_{outcome}_{predictor_session}_{'adj' if adjusted else 'unadj'}.csv"
    out.to_csv(fn, index=False)
    print(f"Schulz-style ordinal regression: {outcome} ~ regional vol @ {predictor_session} "
          f"({'adjusted' if adjusted else 'unadjusted'}) -> {fn}")
    if not out.empty:
        print(out.to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="Replicate Schulz 2022 fcac203 (FUS-ET adaptation).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outcome", default="tremor_post3mo")
    ap.add_argument("--predictor-session", default="ses-post24h",
                    help="Acute T1 timepoint (default ses-post24h, per the '24h as baseline' design).")
    ap.add_argument("--adjusted", action="store_true", help="Add baseline-severity adjustment.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.outcome, args.predictor_session, args.adjusted)


if __name__ == "__main__":
    main()
