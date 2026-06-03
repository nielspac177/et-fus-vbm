"""Path A - single-subject normative atrophy Z-maps.

For each scan, resample GM (mwp1) and WM (mwp2) to 2 mm MNI and compute a voxelwise
Z relative to the bundled normative control distribution (Calvinwhow/vbm ctrl_dist):

    z = (subject - control_mean) / control_std        (within the MNI brain mask)

Atrophy is the negative tail (less tissue than controls); we also save a thresholded
map keeping |z| > z_threshold. NOTE: the control distribution is an external cohort
whose age/scanner may differ from this ET sample — interpret absolute Z magnitudes
with that caveat (see docs/PLAN.md).
"""
from __future__ import annotations
import argparse

import numpy as np
import nibabel as nib
import pandas as pd

from . import load_config
from .io import build_cohort
from .resample import to_2mm

_TISSUE = {"mwp1": "grey_matter", "mwp2": "white_matter"}


def _load_stats(ctrl_dist, tissue):
    """Load mean/std NIfTIs (handles .nii or .nii.gz)."""
    def _find(name):
        for ext in (".nii.gz", ".nii"):
            p = ctrl_dist / f"{tissue}_{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing {tissue}_{name}.nii[.gz] in {ctrl_dist}")
    return nib.load(str(_find("mean"))), nib.load(str(_find("std")))


def z_map(src_path, mean_img, std_img, mask, eps=1e-6):
    x = to_2mm(src_path, _mask_path_global).get_fdata(dtype=np.float32)
    z = (x - mean_img.get_fdata(dtype=np.float32)) / (std_img.get_fdata(dtype=np.float32) + eps)
    z *= mask
    return z


def run(cfg: dict, sessions=None) -> pd.DataFrame:
    global _mask_path_global
    _mask_path_global = cfg["mni_mask"]
    mask_img = nib.load(str(cfg["mni_mask"]))
    mask = (mask_img.get_fdata() > 0).astype(np.float32)
    ctrl = cfg["ctrl_dist"]
    zthr = cfg["z_threshold"]

    stats = {col: _load_stats(ctrl, t) for col, t in _TISSUE.items()}
    cohort = build_cohort(cfg)
    if sessions:
        cohort = cohort[cohort["session"].isin(sessions)]
    cohort = cohort[cohort["mwp1_exists"]]

    out_root = cfg["derivatives"] / "A_normative"
    rows = []
    for _, r in cohort.iterrows():
        subj, ses = r["subject"], r["session"]
        out_dir = out_root / f"sub-{subj}" / ses
        out_dir.mkdir(parents=True, exist_ok=True)
        for col, tissue in _TISSUE.items():
            src = r[col]
            if src is None or not src.exists():
                continue
            mean_img, std_img = stats[col]
            z = z_map(src, mean_img, std_img, mask)
            nib.Nifti1Image(z, mask_img.affine, mask_img.header).to_filename(
                str(out_dir / f"sub-{subj}_{ses}_{tissue}_z.nii.gz"))
            zt = z.copy()
            zt[np.abs(zt) < zthr] = 0
            nib.Nifti1Image(zt, mask_img.affine, mask_img.header).to_filename(
                str(out_dir / f"sub-{subj}_{ses}_{tissue}_z_thr.nii.gz"))
            atrophy = z[(z < 0) & (mask > 0)]
            rows.append({"subject": subj, "session": ses, "tissue": tissue,
                         "mean_atrophy_z": float(atrophy.mean()) if atrophy.size else np.nan,
                         "frac_voxels_z_lt_-2": float((z < -zthr).sum() / mask.sum())})
        print(f"  done sub-{subj} {ses}")

    summary = pd.DataFrame(rows)
    out = out_root / "normative_summary.csv"
    summary.to_csv(out, index=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Path A: normative atrophy Z-maps.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Restrict to these sessions (default: all).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    summary = run(cfg, sessions=args.sessions)
    print(f"\nPath A complete. {len(summary)} tissue-maps summarised -> "
          f"{cfg['derivatives'] / 'A_normative' / 'normative_summary.csv'}")


if __name__ == "__main__":
    main()
