"""Group atrophy / volume-change NIfTI maps (clean, honest version).

HONEST SCOPE: cross-sectional difference of independently modulated GM (post mwp1 - pre
mwp1), NOT TBM/log-Jacobian. Filenames say `mwp1diff`. 3 mo is primary; 24 h is oedema.

Pipeline (per the adversarial viz spec):
  1. flip each pair to the common treated hemisphere (target L) FIRST
  2. QC-exclude subjects with |whole-cerebellum %change| > 30 (oedema/segmentation artifact)
  3. d = post - pre ; smooth the difference at 6 mm FWHM
  4. accumulate mean, variance, and mean preop (for the GM mask)
  GM mask = (mean_pre > 0.2) & brain ; lesion-territory exclusion = flipped lesion consensus
  (>=10% of subjects) dilated ~6 mm. Save GM-masked AND lesion-excluded mean + one-sample t.
"""
from __future__ import annotations
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.image import smooth_img, resample_to_img
from scipy import ndimage

from . import load_config
from .io import build_cohort, load_clinical, load_cerebellar, _norm_subject
from .laterality import flip_to_treated
from .cerebellum import ARTIFACT_PCT

FWHM = 6.0
GM_THRESH = 0.2
LESION_CONSENSUS = 0.10
DILATE_ITERS = 4
VMAX_MEAN = 0.05


def _pairs(cohort, ref, post):
    piv = cohort.pivot_table(index="subject", columns="session", values="mwp1", aggfunc="first")
    if ref not in piv.columns or post not in piv.columns:
        return pd.DataFrame()
    pair = piv[[ref, post]].dropna(); pair.columns = ["pre", "post"]
    return pair.reset_index()


def _cereb_pct(cfg, post):
    """{subject: |total cerebellum %change|} for QC exclusion."""
    cereb = load_cerebellar(cfg)
    if cereb.empty:
        return {}
    piv = cereb.pivot_table(index="subject", columns="session",
                            values="total_cerebellar_gm", aggfunc="first")
    if cfg["reference_session"] not in piv.columns or post not in piv.columns:
        return {}
    p = piv[[cfg["reference_session"], post]].dropna()
    pct = 100.0 * (p[post] - p[cfg["reference_session"]]) / p[cfg["reference_session"]]
    return pct.abs().to_dict()


def _lesion_consensus(cfg, side, ref_img, n_expected):
    """Flipped lesion-frequency consensus (>=10%) dilated ~6 mm, on the map grid."""
    masks_dir = cfg.get("lesion_masks_dir")
    if not masks_dir or side is None:
        return np.zeros(ref_img.shape, bool)
    from glob import glob
    import re
    acc = np.zeros(ref_img.shape, np.float32); n = 0
    for f in sorted(glob(str(masks_dir / cfg.get("lesion_glob", "*ET*.nii*")))):
        if "/._" in f:
            continue
        m = re.search(r"(ET[0-9]+)", f)
        ts = side.get(_norm_subject(m.group(1))) if m else None
        if not ts:
            continue
        img = nib.load(f)
        fl = flip_to_treated(img, ts, "L")
        rs = resample_to_img(fl, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
        acc += (np.asarray(rs.dataobj) > 0).astype(np.float32); n += 1
    if n == 0:
        return np.zeros(ref_img.shape, bool)
    consensus = acc >= (LESION_CONSENSUS * n)
    return ndimage.binary_dilation(consensus, iterations=DILATE_ITERS)


def run(cfg, session, flip=True):
    cohort = build_cohort(cfg); cohort = cohort[cohort["mwp1_exists"]]
    pairs = _pairs(cohort, cfg["reference_session"], session)
    if pairs.empty:
        raise SystemExit(f"No {cfg['reference_session']}->{session} mwp1 pairs.")
    clin = load_clinical(cfg)
    side = (clin.set_index("subject")["treated_side"]
            if (flip and not clin.empty and "treated_side" in clin.columns) else None)
    cpct = _cereb_pct(cfg, session)

    ref_img = nib.load(str(pairs.iloc[0]["pre"])); shape = ref_img.shape
    s = np.zeros(shape, np.float64); ss = np.zeros(shape, np.float64); spre = np.zeros(shape, np.float64)
    n = 0; skipped = {"shape": 0, "cereb_outlier": 0, "no_side": 0, "read": 0}
    for _, r in pairs.iterrows():
        ts = side.get(_norm_subject(r["subject"])) if side is not None else None
        if side is not None and not ts:
            skipped["no_side"] += 1; continue
        if cpct.get(_norm_subject(r["subject"]), 0) > ARTIFACT_PCT:
            skipped["cereb_outlier"] += 1; continue
        try:
            pre = np.asarray(nib.load(str(r["pre"])).dataobj, np.float32)
            pos = np.asarray(nib.load(str(r["post"])).dataobj, np.float32)
        except Exception:
            skipped["read"] += 1; continue
        if pre.shape != shape or pos.shape != shape:
            skipped["shape"] += 1; continue
        if ts:
            pre = np.asarray(flip_to_treated(nib.Nifti1Image(pre, ref_img.affine), ts).dataobj)
            pos = np.asarray(flip_to_treated(nib.Nifti1Image(pos, ref_img.affine), ts).dataobj)
        d = np.asarray(smooth_img(nib.Nifti1Image(pos - pre, ref_img.affine), FWHM).dataobj)
        s += d; ss += d * d; spre += pre; n += 1
    if n < 3:
        raise SystemExit(f"Too few usable pairs ({n}).")

    mean = s / n; mean_pre = spre / n
    var = (ss - n * mean ** 2) / (n - 1); se = np.sqrt(np.maximum(var, 0) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        tmap = np.where(se > 0, mean / se, 0.0)

    from pathlib import Path as _P
    if cfg.get("mni_mask") and _P(str(cfg["mni_mask"])).exists():
        brain_img = nib.load(str(cfg["mni_mask"]))
    else:  # fall back to nilearn's MNI brain mask (no upstream clone needed)
        from nilearn.datasets import load_mni152_brain_mask
        brain_img = load_mni152_brain_mask()
    brain = resample_to_img(brain_img, ref_img,
                            interpolation="nearest", force_resample=True, copy_header=True)
    gm = (mean_pre > GM_THRESH) & (np.asarray(brain.dataobj) > 0)
    exclude = _lesion_consensus(cfg, side, ref_img, n)
    analysis = gm & ~exclude

    out_dir = cfg["derivatives"] / "atrophy_maps"; out_dir.mkdir(parents=True, exist_ok=True)
    edema = "_EDEMA" if session == "ses-post24h" else ""
    base = f"group_mwp1diff_{session}{edema}{'' if side is not None else '_unflipped'}"

    def save(arr, mask, name):
        nib.Nifti1Image((arr * mask).astype(np.float32), ref_img.affine, ref_img.header
                        ).to_filename(str(out_dir / f"{base}_{name}.nii.gz"))
    save(mean, gm, "mean"); save(tmap, gm, "t")
    save(mean, analysis, "mean_lesionExcl"); save(tmap, analysis, "t_lesionExcl")
    nib.Nifti1Image(exclude.astype(np.float32), ref_img.affine, ref_img.header
                    ).to_filename(str(out_dir / f"{base}_lesionTerritory.nii.gz"))
    (out_dir / f"{base}_provenance.txt").write_text(
        f"n={n}\nflipped_to_treated={side is not None}\nfwhm={FWHM}\n"
        f"metric=cross-sectional mwp1 diff (post-pre), NOT TBM\nskipped={skipped}\n"
        f"gm_thresh={GM_THRESH} lesion_consensus={LESION_CONSENSUS} dilate={DILATE_ITERS}\n")
    print(f"{base}: n={n}, skipped={skipped} -> {out_dir}")
    return {"mean_path": out_dir / f"{base}_mean_lesionExcl.nii.gz",
            "t_path": out_dir / f"{base}_t_lesionExcl.nii.gz", "base": base, "n": n, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description="Clean group atrophy (mwp1-diff) maps.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", default="ses-post3mo")
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run(cfg, args.session, flip=not args.no_flip)
    try:
        from .viz import plot_atrophy_map
        fig = plot_atrophy_map(res["mean_path"], cfg["derivatives"] / "figures" / f"{res['base']}.png",
                               vmax=VMAX_MEAN)
        print(f"figure -> {fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
