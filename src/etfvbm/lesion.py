"""Lesion burden + treated side from per-subject lesion masks.

Point `lesion_masks_dir` (config) at a directory of ET-cohort binary lesion masks in a
known space. We compute, per subject:
- lesion_volume (mm^3 and cm^3) = nonzero voxels * voxel volume
- treated_side ('L'/'R') from the lesion centroid's hemisphere (x relative to midline)

These feed: the laterality flip / crossed-asymmetry test (treated_side) and the
association/Schulz models (lesion_volume covariate, log10-transformed). NOTE: do NOT use
the Parkinson's `pd*` masks under `FUS FC v1/` — that is a different cohort (contamination).
"""
from __future__ import annotations
import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

from . import load_config
from .io import _norm_subject

_SUB = re.compile(r"(ET[0-9]+)", re.IGNORECASE)


def _side_from_mask(img) -> str | None:
    data = np.asarray(img.dataobj) > 0
    if data.sum() == 0:
        return None
    xs = np.where(data)[0]                       # voxel x-indices of lesion
    mid = data.shape[0] / 2.0
    cx = xs.mean()
    # In MNI voxel space with a standard affine, +x is typically R; check affine sign.
    x_is_R = img.affine[0, 0] > 0
    left = cx < mid
    if x_is_R:
        return "L" if left else "R"
    return "R" if left else "L"


def compute(cfg: dict) -> pd.DataFrame:
    masks_dir = cfg.get("lesion_masks_dir")
    pattern = cfg.get("lesion_glob", "*ET*.nii*")
    if not masks_dir or not Path(masks_dir).exists():
        raise SystemExit("Set cfg['lesion_masks_dir'] to a directory of ET lesion masks.")
    files = [f for f in sorted(glob(str(Path(masks_dir) / pattern)))
             if not Path(f).name.startswith("._")]
    rows = []
    for f in files:
        m = _SUB.search(Path(f).name)
        if not m:
            continue
        img = nib.load(f)
        data = np.asarray(img.dataobj) > 0
        vox_mm3 = float(np.abs(np.linalg.det(img.affine[:3, :3])))
        vol_mm3 = data.sum() * vox_mm3
        # lesion location in world (RAS) coords: centroid + inferior extension.
        # Inferior extension toward cerebellar outflow (DRTT / zona incerta) is the
        # mechanistic ataxia driver; lower world-z = more inferior.
        loc = {}
        if data.sum() > 0:
            ijk = np.array(np.where(data))               # (3, nvox) voxel coords
            world = nib.affines.apply_affine(img.affine, ijk.T)  # (nvox, 3) RAS mm
            cx, cy, cz = world.mean(axis=0)
            loc = {"centroid_x": float(cx), "centroid_y": float(cy), "centroid_z": float(cz),
                   "inferior_z_min": float(world[:, 2].min()),   # most inferior voxel (mm)
                   "frac_below_acpc": float((world[:, 2] < 0).mean())}  # fraction inferior to AC-PC
        rows.append({"subject": _norm_subject(m.group(1).upper()),
                     "lesion_volume_mm3": vol_mm3,
                     "lesion_volume_cm3": vol_mm3 / 1000.0,
                     "treated_side": _side_from_mask(img),
                     **loc, "mask": f})
    df = pd.DataFrame(rows).drop_duplicates("subject")
    out = cfg["derivatives"] / "lesion_burden.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Lesion volume + treated side from masks.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = compute(cfg)
    print(f"{len(df)} lesions -> {cfg['derivatives'] / 'lesion_burden.csv'}")
    if not df.empty:
        print(df[["subject", "lesion_volume_cm3", "treated_side"]].head(10).to_string(index=False))
        print("\nTreated side counts:\n" + df["treated_side"].value_counts().to_string())


if __name__ == "__main__":
    main()
