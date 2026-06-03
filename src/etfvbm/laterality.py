"""Laterality handling — the #1 peer-review fix (ADR-0001).

Thalamotomy is unilateral, so the lesion and its crossed downstream effects live on
a subject-specific side. Pooling without aligning sides cancels the main effect.

This module provides:
- `align_hemispheres`: given a per-scan lobular cerebellar table and each subject's
  treated side, relabel Left/Right columns into IPSILESIONAL / CONTRALESIONAL, so
  the crossed-degeneration hypothesis (contra > ipsi) is testable across subjects.
- `flip_to_treated`: left<->right flip of a NIfTI (for image-level group analyses),
  so every subject's treated hemisphere lands on the same side.

Treated side MUST come from the operative record (clinical.csv `treated_side`), never
from the image.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd


def flip_to_treated(img: nib.Nifti1Image, treated_side: str,
                    target_side: str = "L") -> nib.Nifti1Image:
    """Flip a brain map left<->right so the treated hemisphere is on `target_side`.

    No-op if the treated side already matches the target. Assumes the image is in a
    standard MNI orientation where the first axis is L->R (RAS/LAS handled via affine
    sign); we flip along the x voxel axis, which is the convention for MNI-space VBM
    maps produced here.
    """
    treated_side = str(treated_side).strip().upper()[:1]
    if treated_side not in ("L", "R"):
        raise ValueError(f"treated_side must be 'L' or 'R', got {treated_side!r}")
    if treated_side == target_side:
        return img
    data = np.asarray(img.dataobj)
    return nib.Nifti1Image(np.ascontiguousarray(data[::-1]), img.affine, img.header)


# Cerebellar lobular column stems that have Left/Right counterparts in the SUIT table.
_PAIRED_STEMS = ["I_IV", "V", "VI", "CrusI", "CrusII", "VIIb", "VIIIa", "VIIIb", "IX", "X"]


def align_hemispheres(cereb: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Add ipsilesional/contralesional cerebellar GM columns using treated side.

    For each paired SUIT lobule, IPSI = the hemisphere on the treated side and CONTRA
    = the opposite. Also adds whole-hemisphere sums and an asymmetry index
    AI = (contra - ipsi)/(contra + ipsi)  (positive AI => contra smaller? no:
    positive => contra larger; crossed atrophy predicts contra LOSS => AI decreases
    over time). Rows without a known treated side are returned with NaN ipsi/contra.
    """
    df = cereb.copy()
    side = clinical.set_index("subject")["treated_side"].map(
        lambda s: str(s).strip().upper()[:1] if pd.notna(s) and str(s).strip() else None
    ) if not clinical.empty and "treated_side" in clinical.columns else pd.Series(dtype=object)
    df["treated_side"] = df["subject"].map(side)

    def _col(stem, hemi):
        c = f"GM_{hemi}_{stem}"
        return df[c] if c in df.columns else np.nan

    for stem in _PAIRED_STEMS:
        treated_is_L = df["treated_side"] == "L"
        left, right = _col(stem, "Left"), _col(stem, "Right")
        # ipsi = treated-side hemisphere; contra = other
        df[f"IPSI_{stem}"] = np.where(treated_is_L, left, right)
        df[f"CONTRA_{stem}"] = np.where(treated_is_L, right, left)
        # null out where side unknown
        unknown = df["treated_side"].isna()
        df.loc[unknown, [f"IPSI_{stem}", f"CONTRA_{stem}"]] = np.nan

    ipsi_cols = [f"IPSI_{s}" for s in _PAIRED_STEMS]
    contra_cols = [f"CONTRA_{s}" for s in _PAIRED_STEMS]
    df["IPSI_hemi_GM"] = df[ipsi_cols].sum(axis=1, min_count=1)
    df["CONTRA_hemi_GM"] = df[contra_cols].sum(axis=1, min_count=1)
    denom = df["CONTRA_hemi_GM"] + df["IPSI_hemi_GM"]
    df["asymmetry_index"] = (df["CONTRA_hemi_GM"] - df["IPSI_hemi_GM"]) / denom

    # Side-agnostic fallbacks (always available, even without treated_side)
    left_cols = [f"GM_Left_{s}" for s in _PAIRED_STEMS if f"GM_Left_{s}" in df.columns]
    right_cols = [f"GM_Right_{s}" for s in _PAIRED_STEMS if f"GM_Right_{s}" in df.columns]
    df["Left_hemi_GM"] = df[left_cols].sum(axis=1, min_count=1)
    df["Right_hemi_GM"] = df[right_cols].sum(axis=1, min_count=1)
    return df
