"""etfvbm.palm — voxelwise OLS regression with max-stat permutation FWE (+ optional TFCE).

Adapted (standalone) from CircuitPyPer's VoxelwiseRegression / TFCalculator for VBM/atrophy.
Deps: numpy / pandas / scipy / nibabel only.

Model per voxel v:  change_map[:, v] ~ b_interest * x_interest + B_nuis @ Nuis + b0
Tested statistic = t for b_interest. Inference = permutation max-stat FWE with the
Freedman-Lane scheme (nuisance partialled out, interest regressor permuted), valid for
TFCE when tfce=True. BH-FDR on parametric p also returned.

VBM cautions (see integration notes): mask to GM; exclude/covary the lesion; use 3 mo
(not 24 h oedema) for atrophy inference; do NOT reuse the connectivity cohort's maps.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.stats import t as student_t


def _load_mask_on_grid(mask_path, ref_img):
    m = nib.load(str(mask_path))
    mdat = (np.asanyarray(m.dataobj) > 0).astype(np.float32)
    if m.shape == ref_img.shape and np.allclose(m.affine, ref_img.affine, atol=1e-3):
        return mdat > 0
    xfm = np.linalg.inv(m.affine) @ ref_img.affine
    res = ndimage.affine_transform(mdat, xfm[:3, :3], offset=xfm[:3, 3],
                                   output_shape=ref_img.shape, order=0)
    return res > 0.5


def _stack_maps(paths, mask):
    n = int(mask.sum())
    out = np.empty((len(paths), n), dtype=np.float64)
    for i, p in enumerate(paths):
        d = np.asanyarray(nib.load(str(p)).dataobj)
        if d.shape != mask.shape:
            raise ValueError(f"{p}: shape {d.shape} != mask {mask.shape}")
        out[i] = d[mask].astype(np.float64)
    return out


def _unmask(vec, mask):
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = vec.astype(np.float32)
    return vol


def _build_design(design, outcome_col, covar_cols):
    cols = [outcome_col] + list(covar_cols)
    sub = design[cols].apply(pd.to_numeric, errors="coerce")
    keep = sub.notna().all(axis=1).values
    x = sub[outcome_col].values[keep].astype(np.float64)
    if covar_cols:
        N = sub[list(covar_cols)].values[keep].astype(np.float64)
        N = N - N.mean(axis=0, keepdims=True)
    else:
        N = np.empty((int(keep.sum()), 0))
    N = np.column_stack([N, np.ones(N.shape[0])])
    return x, N, keep


def _tstat_interest(x, N, Y):
    X = np.column_stack([x, N])
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta
    dof = n - p
    mse = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(XtX_inv[0, 0] * mse)
    se = np.where(se <= 0, np.inf, se)
    tmap = beta[0] / se
    pmap = 2.0 * student_t.sf(np.abs(tmap), dof)
    return tmap.astype(np.float64), pmap, dof


def _residualize(Y, N):
    return Y - N @ (np.linalg.pinv(N) @ Y)


def _tfce_signed(vol, mask, H=2.0, E=0.5, dh=0.1):
    out = np.zeros_like(vol, dtype=np.float64)
    for sign in (1.0, -1.0):
        s = np.where(mask, vol * sign, 0.0)
        smax = s.max()
        if smax <= 0:
            continue
        for h in np.arange(dh, smax + dh, dh):
            lab, k = ndimage.label(s >= h)
            if k == 0:
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, k + 1))
            ext = np.zeros(k + 1); ext[1:] = sizes
            supra = lab > 0
            out[supra] += sign * (ext[lab[supra]] ** E) * (h ** H) * dh
    return out


def voxelwise_regression(map_paths, design, mask_path, out_dir, outcome_col,
                         covar_cols=(), n_permutations=5000, tfce=False,
                         two_sided=True, tfce_H=2.0, tfce_E=0.5, tfce_dh=0.1,
                         prefix="vbm", seed=42):
    if len(map_paths) != len(design):
        raise ValueError(f"{len(map_paths)} maps vs {len(design)} design rows")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = nib.load(str(map_paths[0]))
    mask = _load_mask_on_grid(mask_path, ref)
    Y_all = _stack_maps(map_paths, mask)
    x, N, keep = _build_design(design, outcome_col, covar_cols)
    Y = Y_all[keep]
    n_vox = Y.shape[1]
    print(f"[palm] n={Y.shape[0]} subjects, {n_vox} voxels, {N.shape[1]-1} nuisance, "
          f"perms={n_permutations}, tfce={tfce}")
    tmap, pmap, dof = _tstat_interest(x, N, Y)
    stat = tmap.copy()
    if tfce:
        stat = _tfce_signed(_unmask(tmap, mask), mask, tfce_H, tfce_E, tfce_dh)[mask]
    Yr = _residualize(Y, N)
    xr = x - N @ (np.linalg.pinv(N) @ x)
    fwe_count = np.zeros(n_vox, dtype=np.int64)
    obs = np.abs(stat) if two_sided else stat
    for _ in range(n_permutations):
        xp = xr[rng.permutation(xr.shape[0])]
        tp, _, _ = _tstat_interest(xp, N, Yr)
        sp = _tfce_signed(_unmask(tp, mask), mask, tfce_H, tfce_E, tfce_dh)[mask] if tfce else tp
        sp = np.abs(sp) if two_sided else sp
        fwe_count += (np.nanmax(sp) >= obs)
    fwe_p = (fwe_count + 1) / (n_permutations + 1)
    order = np.argsort(pmap); ranked = pmap[order]; m = n_vox
    q = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    fdr_q = np.empty_like(q); fdr_q[order] = np.clip(q, 0, 1)

    def _save(vec, name):
        path = out_dir / f"{prefix}_{name}.nii.gz"
        nib.Nifti1Image(_unmask(vec, mask), ref.affine, ref.header).to_filename(str(path))
        return str(path)

    paths = {"t": _save(tmap, "t"), "fwe": _save(1.0 - fwe_p, "fwe_1minusp"),
             "fdr": _save(1.0 - fdr_q, "fdr_1minusq")}
    if tfce:
        paths["tfce"] = _save(stat, "tfce")
    print(f"[palm] wrote: {', '.join(paths.values())}")
    return paths
