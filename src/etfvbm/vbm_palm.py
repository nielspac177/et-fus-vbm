"""Driver: per-subject change maps -> voxelwise PALM (structure-symptom map).

Generates per-subject 3-month GM change maps (post mwp1 - pre mwp1), flipped to the
common treated hemisphere, 6 mm smoothed, with cerebellar-outlier QC exclusion; builds a
GM analysis mask (group mean preop mwp1 > 0.1) with the lesion territory excluded; then
runs `etfvbm.palm.voxelwise_regression` of change ~ outcome + covariates with Freedman-Lane
max-stat FWE (+ optional TFCE) and BH-FDR.

Honest scope: cross-sectional mwp1 difference (NOT TBM). 3 mo primary (24 h = oedema).
The change map is the DV; the clinical outcome is the regressor of interest. Lesion volume
is covaried (log10); lesion LOCATION is not yet measured (deeper confound).
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import smooth_img, resample_to_img

from . import load_config
from .io import build_cohort, load_clinical, _norm_subject
from .laterality import flip_to_treated
from .atrophy_maps import _pairs, _cereb_pct, _lesion_consensus, FWHM
from . import palm


def prepare(cfg, session, outcome, covars):
    cohort = build_cohort(cfg); cohort = cohort[cohort["mwp1_exists"]]
    pairs = _pairs(cohort, cfg["reference_session"], session)
    clin = load_clinical(cfg).set_index("subject")
    side = clin["treated_side"]
    cpct = _cereb_pct(cfg, session)

    # numeric clinical
    cl = clin.copy()
    if "sex" in cl:
        cl["sex"] = cl["sex"].map({"M": 0, "F": 1})
    cl["log10_lesion"] = np.log10(pd.to_numeric(cl.get("lesion_volume"), errors="coerce").clip(lower=1e-3))
    need = [outcome] + [c for c in covars if c != "log10_lesion"]

    ref_img = nib.load(str(pairs.iloc[0]["pre"])); shape = ref_img.shape
    out_dir = cfg["derivatives"] / "change_maps" / session
    out_dir.mkdir(parents=True, exist_ok=True)
    spre = np.zeros(shape, np.float64); n = 0
    rows = []
    for _, r in pairs.iterrows():
        sub = _norm_subject(r["subject"])
        ts = side.get(sub)
        if not isinstance(ts, str):
            continue
        if cpct.get(sub, 0) > 30:
            continue
        # complete clinical case?
        vals = {c: pd.to_numeric(cl[c], errors="coerce").get(sub) if c in cl else np.nan
                for c in [outcome] + covars}
        if any(pd.isna(v) for v in vals.values()):
            continue
        try:
            pre = np.asarray(nib.load(str(r["pre"])).dataobj, np.float32)
            pos = np.asarray(nib.load(str(r["post"])).dataobj, np.float32)
        except Exception:
            continue
        if pre.shape != shape:
            continue
        pre = np.asarray(flip_to_treated(nib.Nifti1Image(pre, ref_img.affine), ts).dataobj)
        pos = np.asarray(flip_to_treated(nib.Nifti1Image(pos, ref_img.affine), ts).dataobj)
        d = np.asarray(smooth_img(nib.Nifti1Image(pos - pre, ref_img.affine), FWHM).dataobj)
        p = out_dir / f"sub-{sub}_mwp1diff.nii.gz"
        nib.Nifti1Image(d.astype(np.float32), ref_img.affine, ref_img.header).to_filename(str(p))
        spre += pre; n += 1
        rows.append({"subject": sub, "map_path": str(p), **vals})

    if n < 10:
        raise SystemExit(f"Too few complete cases ({n}).")
    design = pd.DataFrame(rows).reset_index(drop=True)

    # GM analysis mask: group mean preop mwp1 > 0.1, brain, minus lesion territory
    mean_pre = spre / n
    from pathlib import Path as _P
    if cfg.get("mni_mask") and _P(str(cfg["mni_mask"])).exists():
        brain_img = nib.load(str(cfg["mni_mask"]))
    else:
        from nilearn.datasets import load_mni152_brain_mask
        brain_img = load_mni152_brain_mask()
    brain = np.asarray(resample_to_img(brain_img, ref_img, interpolation="nearest",
                                       force_resample=True, copy_header=True).dataobj) > 0
    exclude = _lesion_consensus(cfg, side, ref_img, n)
    gm = (mean_pre > 0.1) & brain & ~exclude
    mask_path = out_dir / "gm_analysis_mask.nii.gz"
    nib.Nifti1Image(gm.astype(np.float32), ref_img.affine, ref_img.header).to_filename(str(mask_path))
    print(f"prepared {n} change maps; GM mask voxels={int(gm.sum())} (lesion territory excluded)")
    return design, str(mask_path)


def main():
    ap = argparse.ArgumentParser(description="Voxelwise PALM structure-symptom map.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", default="ses-post3mo")
    ap.add_argument("--outcome", default="tremor_improvement")
    ap.add_argument("--covars", nargs="*", default=["age", "sex", "log10_lesion"])
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--tfce", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    design, mask_path = prepare(cfg, args.session, args.outcome, args.covars)
    out_dir = cfg["derivatives"] / "vbm_palm" / f"{args.outcome}_{args.session}"
    paths = palm.voxelwise_regression(
        map_paths=design["map_path"].tolist(), design=design, mask_path=mask_path,
        out_dir=out_dir, outcome_col=args.outcome, covar_cols=args.covars,
        n_permutations=args.permutations, tfce=args.tfce, two_sided=True,
        prefix=args.outcome)
    # quick significance summary
    fwe = nib.load(paths["fwe"]).get_fdata(); fdr = nib.load(paths["fdr"]).get_fdata()
    print(f"FWE p<0.05 voxels (1-p>0.95): {int((fwe > 0.95).sum())}; "
          f"FDR q<0.05 voxels: {int((fdr > 0.95).sum())}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
