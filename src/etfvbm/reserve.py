"""Deepened cerebellar-reserve -> imbalance analysis (extends the Schulz-style finding).

Per the adversarial design panel:
- PRIMARY = continuous dose-response: TIV-residualized, z-scored cerebellar volume ->
  imbalance (logistic), reported as OR per 1-SD DECREASE in volume (= less reserve).
- Pre-specified BALANCE-LOBULE set {I_IV, V, VI, VIIIa, VIIIb, vermis} (spinocerebellar
  leg/trunk representation) -> FDR over these 6 only (anatomically motivated, not post hoc);
  the all-region FDR is reported as exploratory.
- The make-or-break test (Model E): does reserve add over lesion size? Nested LR test +
  repeated cross-validated AUROC (clinical+lesion vs +reserve).
- Robustness: bootstrap CI on the OR, LOOA, and drop-side-discordant sensitivity.

Confounder note: lesion VOLUME is covaried (log10), but lesion LOCATION/inferior-extension
is the deeper confounder for ataxia and is NOT yet available — flagged, not solved.
Data blockers: no graded imbalance severity (outcome_3month is corrupted) -> binary only;
no dysarthria/dose columns.
"""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import load_config
from .io import load_clinical, build_cohort
from .association import fdr_bh

BALANCE = ["I_IV", "V", "VI", "VIIIa", "VIIIb", "vermis"]


def _whole(cereb, lob):
    cols = [c for c in (f"GM_Left_{lob}", f"GM_Right_{lob}", f"GM_Vermis_{lob}") if c in cereb]
    return cereb[cols].sum(axis=1, min_count=1) if cols else np.nan


def _build(cfg):
    al = pd.read_csv(cfg["derivatives"] / "C_cerebellar_longitudinal" / "cerebellar_aligned.csv")
    al = al[al["session"] == cfg["predictor_session"]].copy()
    df = al[["subject"]].copy()
    for lob in ["I_IV", "V", "VI", "VIIIa", "VIIIb"]:
        df[lob] = _whole(al, lob)
    verm = [c for c in al.columns if c.startswith("GM_Vermis_")]
    df["vermis"] = al[verm].sum(axis=1, min_count=1) if verm else np.nan
    df["total"] = al["total_cerebellar_gm"]

    clin = load_clinical(cfg).set_index("subject")
    cohort = build_cohort(cfg)
    tiv = cohort[cohort["session"] == cfg["reference_session"]].set_index("subject")["tiv"]
    df = df.set_index("subject")
    for c in ["imbalance", "weakness", "tremor_improvement", "age", "tremor_pre",
              "lesion_volume", "side_discordant"]:
        if c in clin.columns:
            df[c] = clin[c]
    df["sex"] = clin["sex"].map({"M": 0, "F": 1}) if "sex" in clin else np.nan
    df["TIV"] = tiv
    df["log10_lesion"] = np.log10(pd.to_numeric(df["lesion_volume"], errors="coerce").clip(lower=1e-3))
    for c in ["imbalance", "weakness", "tremor_improvement", "age", "tremor_pre", "TIV"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _zresid(vol, tiv):
    """TIV-residualized, z-scored volume (higher = more reserve)."""
    d = pd.concat([vol.rename("v"), tiv.rename("t")], axis=1).dropna()
    if len(d) < 10:
        return vol * np.nan
    res = sm.OLS(d["v"], sm.add_constant(d["t"])).fit().resid
    out = pd.Series(np.nan, index=vol.index)
    out.loc[d.index] = (res - res.mean()) / res.std()
    return out


def _logit_or(df, region, outcome, covars):
    """Logistic OR per 1-SD DECREASE of the reserve predictor (= exp(-beta_z))."""
    d = df.dropna(subset=[region, outcome] + covars).copy()
    if len(d) < 15 or d[outcome].nunique() < 2:
        return None
    X = sm.add_constant(d[[region] + covars].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(d[outcome].astype(float), X).fit(disp=0)
        except Exception as e:
            return {"region": region, "n": len(d), "error": str(e)}
    b, se = res.params[region], res.bse[region]
    return {"region": region, "n": int(len(d)), "n_events": int(d[outcome].sum()),
            "OR_per_SD_decrease": float(np.exp(-b)),
            "ci_lo": float(np.exp(-(b + 1.96 * se))), "ci_hi": float(np.exp(-(b - 1.96 * se))),
            "p": float(res.pvalues[region])}


def _bootstrap_or(df, region, outcome, covars, n=2000, seed=0):
    d = df.dropna(subset=[region, outcome] + covars)
    rng = np.random.default_rng(seed)
    ors = []
    for _ in range(n):
        s = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31))
        if s[outcome].nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sm.Logit(s[outcome].astype(float),
                             sm.add_constant(s[[region] + covars].astype(float))).fit(disp=0)
            ors.append(np.exp(-r.params[region]))
        except Exception:
            continue
    if len(ors) < 100:
        return (np.nan, np.nan)
    return tuple(np.percentile(ors, [2.5, 97.5]))


def _looa(df, region, outcome, covars):
    d = df.dropna(subset=[region, outcome] + covars)
    idx = d.index
    if len(idx) < 16:
        return np.nan
    sig = 0
    for drop in idx:
        r = _logit_or(d.drop(index=drop), region, outcome, covars)
        if r and r.get("p", 1) < 0.05:
            sig += 1
    return sig / len(idx)


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def _cv_auc(df, outcome, base, extra, k=10, reps=20, seed=1):
    d = df.dropna(subset=[outcome] + base + extra)
    y = d[outcome].astype(float).values
    rng = np.random.default_rng(seed)
    aucs = {"base": [], "full": []}
    for rep in range(reps):
        idx = rng.permutation(len(d))
        folds = np.array_split(idx, k)
        for cols, key in [(base, "base"), (base + extra, "full")]:
            preds = np.full(len(d), np.nan)
            for f in folds:
                tr = np.setdiff1d(idx, f)
                Xtr = sm.add_constant(d.iloc[tr][cols].astype(float), has_constant="add")
                Xte = sm.add_constant(d.iloc[f][cols].astype(float), has_constant="add")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = sm.Logit(y[tr], Xtr).fit(disp=0)
                    preds[f] = r.predict(Xte)
                except Exception:
                    pass
            ok = ~np.isnan(preds)
            if ok.sum() > 10:
                aucs[key].append(_auc(y[ok], preds[ok]))
    return (np.nanmean(aucs["base"]), np.nanmean(aucs["full"]),
            np.nanmean(aucs["full"]) - np.nanmean(aucs["base"]))


def run(cfg, outcome="imbalance"):
    df = _build(cfg)
    for r in ["total"] + BALANCE:
        df[f"z_{r}"] = _zresid(df[r], df["TIV"])
    covars = ["log10_lesion", "age", "sex", "tremor_pre"]
    rows = []
    # total (primary) + balance lobules
    for r in ["total"] + BALANCE:
        res = _logit_or(df, f"z_{r}", outcome, covars)
        if res and "OR_per_SD_decrease" in res:
            res["region"] = r
            if res["p"] < 0.05:
                res["looa"] = _looa(df, f"z_{r}", outcome, covars)
                lo, hi = _bootstrap_or(df, f"z_{r}", outcome, covars)
                res["boot_lo"], res["boot_hi"] = lo, hi
        if res:
            rows.append(res)
    out = pd.DataFrame(rows)
    # FDR over the 6 balance lobules only (pre-specified), total reported separately
    bal = out["region"].isin(BALANCE)
    out.loc[bal, "q_fdr_balance6"] = fdr_bh(out.loc[bal, "p"].values)

    # Model E: does reserve add over lesion size?  (nested LR + CV-AUROC)
    d = df.dropna(subset=[outcome, "z_total"] + covars)
    e = {}
    try:
        X0 = sm.add_constant(d[covars].astype(float))
        X1 = sm.add_constant(d[["z_total"] + covars].astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = sm.Logit(d[outcome].astype(float), X0).fit(disp=0)
            m1 = sm.Logit(d[outcome].astype(float), X1).fit(disp=0)
        lr = 2 * (m1.llf - m0.llf)
        from scipy.stats import chi2
        e["LR_chi2"], e["LR_p"] = float(lr), float(chi2.sf(lr, 1))
        a0, a1, da = _cv_auc(df, outcome, covars, ["z_total"])
        e["AUROC_clinical_lesion"], e["AUROC_plus_reserve"], e["dAUROC"] = a0, a1, da
    except Exception as ex:
        e["error"] = str(ex)

    # Sensitivity: drop side-discordant
    sens = None
    if "side_discordant" in df.columns:
        dd = df[df["side_discordant"] != True]
        sens = _logit_or(dd, "z_total", outcome, covars)

    out_dir = cfg["derivatives"] / "reserve"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"reserve_{outcome}_dose_response.csv", index=False)
    pd.DataFrame([e]).to_csv(out_dir / f"reserve_{outcome}_nested_model.csv", index=False)
    print(f"=== Cerebellar reserve -> {outcome} (continuous, OR per 1-SD DECREASE) ===")
    cols = ["region", "n", "n_events", "OR_per_SD_decrease", "ci_lo", "ci_hi", "p",
            "q_fdr_balance6", "looa", "boot_lo", "boot_hi"]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    print(f"\n=== Model E (does reserve add over lesion size?) ===\n{e}")
    if sens:
        print(f"\nSensitivity (drop side-discordant): total OR={sens['OR_per_SD_decrease']:.2f} "
              f"[{sens['ci_lo']:.2f},{sens['ci_hi']:.2f}] p={sens['p']:.3f}")
    print(f"\n-> {out_dir}")
    return out, e


def main():
    ap = argparse.ArgumentParser(description="Deepened cerebellar-reserve analysis.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outcome", default="imbalance")
    args = ap.parse_args()
    run(load_config(args.config), args.outcome)


if __name__ == "__main__":
    main()
