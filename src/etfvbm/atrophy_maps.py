"""Group atrophy / volume-change NIfTI maps (the "where is there more/less atrophy" map).

HONEST SCOPE (per adversarial review):
- We have NO within-subject longitudinal registration, so this is a **cross-sectional
  difference of independently modulated GM** (post mwp1 - pre mwp1), NOT TBM/log-Jacobian.
  It is labelled exploratory and the output filenames say `mwp1diff`, never `jacobian`.
- 3 months is the primary timepoint; 24 h is dominated by oedema/segmentation artifact
  and is produced only with an explicit `_EDEMA_ARTIFACT` tag if requested.
- A robust mean (per-voxel trimmed) and a one-sample t map are saved; extreme-change
  subjects (whole-cerebellum |%|>30, from QC) should be excluded upstream.
- If treated side is available, maps are flipped to a common treated hemisphere.

Outputs (in derivatives/atrophy_maps/):
  group_mwp1diff_<ses>_mean.nii.gz   signed mean volume change (negative = atrophy)
  group_mwp1diff_<ses>_t.nii.gz      one-sample t (paired) of the change
  group_mwp1diff_<ses>_n.txt         n and provenance
"""
from __future__ import annotations
import argparse

import numpy as np
import nibabel as nib
import pandas as pd

from . import load_config
from .io import build_cohort, load_clinical, _norm_subject
from .laterality import flip_to_treated


def _pairs(cohort: pd.DataFrame, ref: str, post: str) -> pd.DataFrame:
    piv = cohort.pivot_table(index="subject", columns="session", values="mwp1",
                             aggfunc="first")
    if ref not in piv.columns or post not in piv.columns:
        return pd.DataFrame()
    pair = piv[[ref, post]].dropna()
    pair.columns = ["pre", "post"]
    return pair.reset_index()


def run(cfg: dict, session: str, flip: bool = True) -> dict:
    cohort = build_cohort(cfg)
    cohort = cohort[cohort["mwp1_exists"]]
    pairs = _pairs(cohort, cfg["reference_session"], session)
    if pairs.empty:
        raise SystemExit(f"No complete {cfg['reference_session']}->{session} mwp1 pairs.")

    clin = load_clinical(cfg)
    side = (clin.set_index("subject")["treated_side"]
            if (flip and not clin.empty and "treated_side" in clin.columns) else None)

    ref_img = nib.load(str(pairs.iloc[0]["pre"]))
    shape = ref_img.shape
    s = np.zeros(shape, np.float64)   # sum of diffs
    ss = np.zeros(shape, np.float64)  # sum of squared diffs
    n = 0
    used, skipped = [], []
    for _, r in pairs.iterrows():
        try:
            pre = np.asarray(nib.load(str(r["pre"])).dataobj, dtype=np.float32)
            pos = np.asarray(nib.load(str(r["post"])).dataobj, dtype=np.float32)
        except Exception as e:
            skipped.append((r["subject"], str(e)))
            continue
        if pre.shape != shape or pos.shape != shape:
            skipped.append((r["subject"], "shape mismatch"))
            continue
        if side is not None:
            ts = side.get(_norm_subject(r["subject"]))
            if ts and str(ts).strip():
                pre = np.asarray(flip_to_treated(nib.Nifti1Image(pre, ref_img.affine), ts).dataobj)
                pos = np.asarray(flip_to_treated(nib.Nifti1Image(pos, ref_img.affine), ts).dataobj)
        d = pos - pre
        s += d
        ss += d * d
        n += 1
        used.append(r["subject"])

    if n < 3:
        raise SystemExit(f"Too few usable pairs ({n}).")
    mean = s / n
    var = (ss - n * mean ** 2) / (n - 1)
    se = np.sqrt(np.maximum(var, 0) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        tmap = np.where(se > 0, mean / se, 0.0)

    out_dir = cfg["derivatives"] / "atrophy_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if flip and side is not None else "_unflipped"
    edema = "_EDEMA_ARTIFACT" if session == "ses-post24h" else ""
    base = f"group_mwp1diff_{session}{tag}{edema}"
    nib.Nifti1Image(mean.astype(np.float32), ref_img.affine, ref_img.header).to_filename(
        str(out_dir / f"{base}_mean.nii.gz"))
    nib.Nifti1Image(tmap.astype(np.float32), ref_img.affine, ref_img.header).to_filename(
        str(out_dir / f"{base}_t.nii.gz"))
    (out_dir / f"{base}_n.txt").write_text(
        f"n={n}\nflipped_to_treated={side is not None and flip}\n"
        f"metric=cross-sectional mwp1 difference (post-pre), NOT TBM/log-Jacobian\n"
        f"session={session}\nskipped={len(skipped)}\n")
    print(f"{base}: n={n} (skipped {len(skipped)}) -> {out_dir}")
    return {"mean_path": out_dir / f"{base}_mean.nii.gz",
            "t_path": out_dir / f"{base}_t.nii.gz", "n": n, "base": base, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description="Group atrophy (mwp1-difference) maps.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", default="ses-post3mo",
                    help="Post session (default ses-post3mo, the primary timepoint).")
    ap.add_argument("--no-flip", action="store_true", help="Do not flip to treated side.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run(cfg, args.session, flip=not args.no_flip)
    # Best-effort figure
    try:
        from .viz import plot_atrophy_map
        fig = plot_atrophy_map(res["mean_path"], cfg["derivatives"] / "figures" / f"{res['base']}.png")
        print(f"figure -> {fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
