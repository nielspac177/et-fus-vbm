#!/usr/bin/env python3
"""SUITPy cerebellar flatmap painted with per-lobule OR for imbalance.

HONEST: this is an ROI choropleth (each SUIT lobule filled with its scalar OR), NOT a
voxelwise statistical flatmap — our cerebellar data are lobular volumes, not voxelwise.
Run with the SUITPy venv:  .venv-suit/bin/python scripts/suit_flatmap.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-suit")
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap

ROOT = Path(__file__).resolve().parent.parent
ATL = ROOT/"external/suit_atlas/Diedrichsen_2009/atl-Anatom_space-SUIT_dseg.nii"
LUT = ROOT/"external/suit_atlas/Diedrichsen_2009/atl-Anatom.lut"
lob_or = pd.read_csv(ROOT/"derivatives/figures/lobule_or.csv").set_index("lobule")["OR"].to_dict()

lut = {}
for line in Path(LUT).read_text().splitlines():
    p = line.split()
    if len(p) >= 5 and p[0].isdigit():
        lut[int(p[0])] = p[4]

# Build a SUIT-space volume where each lobule voxel = log2(OR) (centered at 0; +=more imbalance)
amap = nib.load(str(ATL)); adata = np.asarray(amap.dataobj)
paint = np.full(adata.shape, np.nan, np.float32)
def lobule_of(name):  # Left_VIIIa -> VIIIa ; map names to our whole-lobule keys
    for k in lob_or:
        if name.endswith(k) or name == f"Left_{k}" or name == f"Right_{k}" or name == f"Vermis_{k}":
            return k
    return None
for idx, name in lut.items():
    k = lobule_of(name)
    if k is not None:
        paint[adata == idx] = np.log2(lob_or[k])
vol = nib.Nifti1Image(np.nan_to_num(paint, nan=0.0), amap.affine, amap.header)

surf = flatmap.vol_to_surf(vol, space="SUIT", ignore_zeros=True)
fig = plt.figure(figsize=(6, 5))
vmax = float(np.nanmax(np.abs([np.log2(v) for v in lob_or.values()])))
flatmap.plot(surf, cmap="RdBu_r", cscale=[-vmax, vmax], colorbar=True, render="matplotlib",
             new_figure=False)
plt.title("Cerebellar lobular effect on 3-month imbalance\nlog2(OR per 1-SD lower volume); red = higher risk (ROI choropleth)", fontsize=10)
out = ROOT/"derivatives/figures/fig_suit_flatmap_imbalance.png"
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("wrote", out)
print("lobule ORs painted:", {k: round(v, 2) for k, v in lob_or.items()})
