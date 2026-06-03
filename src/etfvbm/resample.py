"""Resample deepmriprep maps (1.5 mm MNI) to the 2 mm MNI grid.

The normative control distribution and ROI atlases from Calvinwhow/vbm live on a
2 mm MNI grid (91x109x91). deepmriprep writes 1.5 mm (113x137x113), so every map
must be resampled onto the 2 mm reference before Z-scoring or ROI sampling.

Modulated maps encode *volume*: resampling changes voxel size, so we use linear
interpolation and the maps remain comparable because deepmriprep modulation already
accounts for the warp Jacobian. (Total tissue volume is preserved up to interpolation
error; for group/longitudinal contrasts this is the standard CAT12-style approach.)
"""
from __future__ import annotations
from pathlib import Path
import functools

import nibabel as nib
from nilearn.image import resample_to_img


@functools.lru_cache(maxsize=4)
def _ref(mask_path: str):
    return nib.load(mask_path)


def to_2mm(src_path: str | Path, mask_path: str | Path,
           interpolation: str = "linear") -> nib.Nifti1Image:
    """Resample a single NIfTI onto the 2 mm MNI reference grid."""
    ref = _ref(str(mask_path))
    img = nib.load(str(src_path))
    return resample_to_img(img, ref, interpolation=interpolation, copy_header=True,
                           force_resample=True)


def resample_cached(src_path: str | Path, mask_path: str | Path, out_dir: Path,
                    tag: str, interpolation: str = "linear") -> Path:
    """Resample to 2 mm and cache to ``out_dir`` (skip if already present)."""
    src_path = Path(src_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src_path.name.replace('.nii.gz', '').replace('.nii', '')}_{tag}_2mm.nii.gz"
    if out.exists():
        return out
    to_2mm(src_path, mask_path, interpolation).to_filename(str(out))
    return out
