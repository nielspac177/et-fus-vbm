#!/usr/bin/env python3
"""Cerebellar small-volume-corrected PALM: baseline GM in lobules I-IV/V & VIII vs imbalance.

Voxelwise permutation FWE (etfvbm.palm) on PREOP mwp1 (baseline reserve; n~140), 4mm-smoothed,
restricted to anterior (lobules I-IV/V) and VIII cerebellar masks (AAL, resampled to grid).
Outcome = imbalance at 3mo and 1yr; covars age/sex/TIV/log10_lesion. SVC within each lobule
mask (legitimate focal test, vs the null whole-brain PALM). Honest: MNI-space cerebellum is
alignment-limited (SUIT not available without reprocessing).
"""
import sys, os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  # fix SSL cert for atlas download
os.environ["SSL_CERT_FILE"] = certifi.where()
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, warnings
from nilearn.image import smooth_img, resample_to_img
from nilearn.datasets import fetch_atlas_aal
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from etfvbm import load_config
from etfvbm.io import build_cohort
from etfvbm import palm

cfg = load_config(str(ROOT / "config/cohort.yaml"))
cohort = build_cohort(cfg)
pre = cohort[(cohort.session == "ses-preop") & (cohort.mwp1_exists)].set_index("subject")
les = pd.read_csv(cfg["derivatives"]/"lesion_burden.csv").set_index("subject")
cw = pd.read_csv(ROOT/".."/"BIDS"/"phenotype"/"clinical_wide_v3.tsv", sep="\t", dtype=str)
cw["subject"] = cw["participant_id"].str.replace("sub-", "", regex=False); cw = cw.set_index("subject")
num = lambda c: pd.to_numeric(cw[c], errors="coerce")

ref = nib.load(str(pre.iloc[0]["mwp1"]))

# Diedrichsen 2009 lobular atlas in MNI space (fetched via SUITPy; same naming as our volumes)
ATL = ROOT/"external/suit_atlas/Diedrichsen_2009/atl-Anatom_space-MNI_dseg.nii"
LUT = ROOT/"external/suit_atlas/Diedrichsen_2009/atl-Anatom.lut"
amap = nib.load(str(ATL)); adata = np.asarray(amap.dataobj)
lut = {}
for line in Path(LUT).read_text().splitlines():
    p = line.split()
    if len(p) >= 5 and p[0].isdigit():
        lut[int(p[0])] = p[4]
LOBULE_NAMES = {n for n in lut.values() if any(k in n for k in
                ["I_IV","_V","VI","Crus","VIIb","VIII","_IX","_X"]) and "Dentate" not in n}
def region_mask(name_filter):
    idx = [i for i, n in lut.items() if name_filter(n)]
    m = np.isin(adata, idx).astype(np.float32)
    rs = resample_to_img(nib.Nifti1Image(m, amap.affine), ref, interpolation="nearest",
                         force_resample=True, copy_header=True)
    return (np.asarray(rs.dataobj) > 0.5)
MASKS = {"anterior_I_IV_V": region_mask(lambda n: n in {"Left_I_IV","Right_I_IV","Left_V","Right_V"}),
         "VIIIa": region_mask(lambda n: "VIIIa" in n),
         "full_cerebellum": region_mask(lambda n: n in LOBULE_NAMES)}
for k, m in MASKS.items():
    print(f"mask {k}: {int(m.sum())} voxels")

# Smooth preop mwp1 once (4mm) -> baseline_maps/
sm_dir = cfg["derivatives"]/"baseline_maps"; sm_dir.mkdir(parents=True, exist_ok=True)
def smoothed_path(sub):
    p = sm_dir/f"sub-{sub}_preop_mwp1_s4.nii.gz"
    if not p.exists():
        smooth_img(str(pre.loc[sub, "mwp1"]), 4.0).to_filename(str(p))
    return str(p)

design_base = pd.DataFrame({
    "subject": pre.index, "age": num("current_age").reindex(pre.index),
    "sex": num("sex_2").reindex(pre.index).map({1: 0, 2: 1}),
    "TIV": pre["tiv"], "log10_lesion": np.log10(pd.to_numeric(les["lesion_volume_cm3"], errors="coerce").reindex(pre.index).clip(lower=1e-3)),
}).set_index("subject")

for outcome in ["imbalance_3month", "imbalance_1year"]:
    d = design_base.copy(); d[outcome] = num(outcome).reindex(d.index)
    d = d.dropna()
    print(f"\n===== {outcome}: n={len(d)} complete cases, events={int(d[outcome].sum())} =====")
    if len(d) < 20 or d[outcome].sum() < 8:
        print("  too few -> skip voxelwise"); continue
    # write a temp mask file per region and run palm within it
    for region, m in MASKS.items():
        mask_path = sm_dir/f"mask_{region}.nii.gz"
        nib.Nifti1Image(m.astype(np.float32), ref.affine, ref.header).to_filename(str(mask_path))
        dd = d.reset_index()
        dd["map_path"] = [smoothed_path(s) for s in dd["subject"]]
        out_dir = cfg["derivatives"]/"cerebellar_palm"/f"{outcome}_{region}"
        paths = palm.voxelwise_regression(
            map_paths=dd["map_path"].tolist(), design=dd, mask_path=str(mask_path),
            out_dir=out_dir, outcome_col=outcome,
            covar_cols=["age", "sex", "TIV", "log10_lesion"],
            n_permutations=2000, tfce=True, two_sided=True, prefix=region)
        fwe = nib.load(paths["fwe"]).get_fdata(); t = nib.load(paths["t"]).get_fdata()
        nsig = int((fwe > 0.95).sum())
        # peak: most negative t (lower volume -> more imbalance) within mask
        tm = np.where(m, t, np.nan)
        print(f"  {region:16s}: TFCE-FWE p<0.05 voxels={nsig}; peak t={np.nanmin(tm):.2f}/{np.nanmax(tm):.2f}; "
              f"best FWE p={1-np.nanmax(np.where(m,fwe,np.nan)):.3f}")
