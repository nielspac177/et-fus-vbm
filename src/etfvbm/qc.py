"""Quality control from deepmriprep registration metrics + TIV.

Bad registrations and segmentation failures masquerade as 'atrophy', so QC must
run before any statistics. We use deepmriprep's per-scan ``affine_loss`` and
``warp_mse`` (registration goodness) plus TIV outlier detection. Scans are *flagged*,
not silently dropped — you decide what to exclude.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from . import load_config
from .io import build_cohort


def flag_scans(cohort: pd.DataFrame, qc_cfg: dict) -> pd.DataFrame:
    df = cohort.copy()
    tiv = df["tiv"]
    tiv_z = (tiv - tiv.mean()) / tiv.std(ddof=0)

    df["flag_affine"] = df["affine_loss"].abs() > qc_cfg["affine_loss_abs_max"]
    df["flag_warp"] = df["warp_mse"] > qc_cfg["warp_mse_max"]
    df["flag_tiv"] = tiv_z.abs() > qc_cfg["tiv_sd_outlier"]
    df["flag_missing"] = ~df["mwp1_exists"]
    df["qc_fail"] = df[["flag_affine", "flag_warp", "flag_tiv", "flag_missing"]].any(axis=1)
    df["tiv_z"] = tiv_z
    return df


def main():
    ap = argparse.ArgumentParser(description="QC report for the cohort.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cohort = build_cohort(cfg)
    qc = flag_scans(cohort, cfg["qc"])

    out = cfg["derivatives"] / "qc_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(out, index=False)

    n = len(qc)
    print(f"QC over {n} scans -> {out}")
    for col in ["flag_affine", "flag_warp", "flag_tiv", "flag_missing", "qc_fail"]:
        print(f"  {col:14s}: {int(qc[col].sum()):5d} ({100*qc[col].mean():.1f}%)")
    print("\nRegistration metric ranges (tune thresholds in cohort.yaml):")
    for m in ["affine_loss", "warp_mse", "tiv"]:
        s = qc[m]
        print(f"  {m:12s} min={s.min():.4g}  p50={s.median():.4g}  "
              f"p99={np.nanpercentile(s, 99):.4g}  max={s.max():.4g}")

    # Optional figure (best-effort; QC table is the source of truth)
    try:
        from .viz import qc_dashboard
        fig_path = cfg["derivatives"] / "figures" / "qc_dashboard.png"
        qc_dashboard(qc, fig_path)
        print(f"\nQC dashboard figure -> {fig_path}")
    except Exception as exc:  # pragma: no cover
        print(f"(QC figure skipped: {exc})")


if __name__ == "__main__":
    main()
