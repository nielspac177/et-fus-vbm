#!/usr/bin/env python3
"""Map the BIDS REDCAP phenotype (clinical_wide_v3.tsv) -> et-fus-vbm clinical.csv.

Writes config/clinical.csv (GITIGNORED — contains clinical data). Column codings
(from clinical_wide_v3.json): fus_laterality 1=Left,2=Right; imbalance/weakness 0/1.
Lesion volume is added separately by `python -m etfvbm.lesion`.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TSV = ROOT / "data" / ".." / ".." / "BIDS" / "phenotype" / "clinical_wide_v3.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str((ROOT / ".." / "BIDS" / "phenotype" / "clinical_wide_v3.tsv").resolve()))
    ap.add_argument("--out", default=str(ROOT / "config" / "clinical.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype=str)
    g = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan

    out = pd.DataFrame({
        "subject": df["participant_id"].str.replace("sub-", "", regex=False),
        "treated_side": g("fus_laterality").map({1: "L", 2: "R"}),
        "age": g("current_age"),
        "sex": g("sex_2").map({1: "M", 2: "F"}),
        "tremor_pre": g("fts_total"),                 # FTM preop total (0-18)
        "tremor_post3mo": g("fts_3month_total"),
        "tremor_post1yr": g("fts_1year_total"),
        "tremor_improvement": g("fts_3month_percent"),  # % improvement at 3 mo
        "imbalance": g("imbalance_3month"),           # 0/1 AE at 3 mo
        "weakness": g("weakness_3month"),             # 0/1 AE at 3 mo (rare -> Firth)
        "outcome_3month": g("outcome_3month"),
    }).dropna(subset=["subject"]).drop_duplicates("subject")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} subjects -> {args.out}")
    print("treated_side:", out['treated_side'].value_counts(dropna=False).to_dict())
    print("imbalance:", out['imbalance'].value_counts(dropna=False).to_dict())
    print("weakness:", out['weakness'].value_counts(dropna=False).to_dict())
    print("tremor_improvement median:", out['tremor_improvement'].median())


if __name__ == "__main__":
    main()
