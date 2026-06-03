"""Parse deepmriprep outputs into a tidy cohort table.

deepmriprep wrote one folder per scan named like ``sub-ET001_ses-preop_T1w`` with
``mwp1`` (GM), ``mwp2`` (WM), ``s6mwp1``/``s8mwp1`` (smoothed GM), ``tiv*.csv`` and
registration QC CSVs. The manifest ``deepmriprep_outputs.csv`` lists absolute paths;
we re-base them onto ``data_root`` (the symlink) so the table is portable.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd

from . import load_config

_SUB = re.compile(r"sub-([A-Za-z0-9]+)")
_SES = re.compile(r"(ses-[A-Za-z0-9]+)")


def _rebase(path_str: str, data_root: Path) -> Path | None:
    """Re-base an absolute deepmriprep path onto the local data_root.

    Uses the final ``<scan_folder>/<file>`` components, which is how deepmriprep
    laid the output out, so the table works regardless of where the volume mounts.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    parts = Path(path_str).parts
    rebased = data_root / parts[-2] / parts[-1]
    return rebased


def _read_value_csv(path: Path | None):
    try:
        return float(pd.read_csv(path).iloc[0, 0])
    except Exception:
        return float("nan")


def build_cohort(cfg: dict) -> pd.DataFrame:
    data_root = cfg["data_root"]
    manifest = cfg["manifest_csv"]
    df = pd.read_csv(manifest)

    rows = []
    for _, r in df.iterrows():
        t1 = str(r.get("t1", ""))
        sub = _SUB.search(t1)
        ses = _SES.search(t1)
        rec = {
            "subject": sub.group(1) if sub else None,
            "session": ses.group(1) if ses else None,
            "mwp1": _rebase(r.get("mwp1"), data_root),   # GM, modulated, MNI 1.5 mm
            "mwp2": _rebase(r.get("mwp2"), data_root),   # WM
            "s8mwp1": _rebase(r.get("s8mwp1"), data_root),  # GM smoothed 8 mm (Path B input)
            "tiv": float(r.get("tiv_value", float("nan"))),
            "affine_loss": float(r.get("affine_loss_value", float("nan"))),
            "warp_mse": float(r.get("warp_mse_value", float("nan"))),
        }
        rec["mwp1_exists"] = rec["mwp1"] is not None and rec["mwp1"].exists()
        rows.append(rec)

    cohort = pd.DataFrame(rows)
    cohort = cohort.dropna(subset=["subject", "session"]).reset_index(drop=True)
    cohort["timepoint"] = cohort["session"].str.replace("ses-", "", regex=False)
    return cohort.sort_values(["subject", "session"]).reset_index(drop=True)


def save_cohort(cfg: dict, cohort: pd.DataFrame) -> Path:
    out = cfg["derivatives"] / "cohort.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False)
    return out


def _norm_subject(s):
    """Normalize subject keys: 'sub-ET001' and 'ET001' both -> 'ET001'."""
    s = str(s)
    return s[4:] if s.startswith("sub-") else s


def load_cerebellar(cfg: dict) -> pd.DataFrame:
    """Load pre-computed SUIT cerebellar lobular volumes, normalising keys."""
    path = cfg.get("cerebellar_csv")
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # The two known exports use either (subject, session) or (participant_id, session).
    sub_col = "subject" if "subject" in df.columns else "participant_id"
    df = df.rename(columns={sub_col: "subject"})
    df["subject"] = df["subject"].map(_norm_subject)
    df["session"] = df["session"].astype(str)
    return df


def load_clinical(cfg: dict) -> pd.DataFrame:
    """Load clinical metadata (treated_side, tremor scores, adverse effects)."""
    path = cfg.get("clinical_csv")
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["subject"] = df["subject"].map(_norm_subject)
    return df


def write_clinical_template(cfg: dict) -> Path:
    """Emit a clinical.csv template (one row per subject) for the user to fill."""
    cohort = build_cohort(cfg)
    subs = sorted(cohort["subject"].unique())
    tmpl = pd.DataFrame({
        "subject": subs,
        "treated_side": "",          # 'L' or 'R' (hemisphere of the VIM lesion)
        "age": "",
        "sex": "",                   # 'M'/'F'
        "tremor_pre": "",            # baseline TETRAS/CRST
        "tremor_post24h": "",
        "tremor_post3mo": "",
        "tremor_post1yr": "",
        "adverse_effects": "",       # free text / coded
    })
    out = cfg["clinical_csv"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if Path(out).exists():
        out = Path(out).with_name("clinical_template.csv")  # don't overwrite real data
    tmpl.to_csv(out, index=False)
    return Path(out)


def pair_counts(cohort: pd.DataFrame, reference: str, posts) -> pd.DataFrame:
    """Count complete reference->post pairs per timepoint (the real analysis n)."""
    piv = cohort.pivot_table(index="subject", columns="session",
                             values="mwp1_exists", aggfunc="first")
    piv = piv.fillna(False).astype(bool)
    rows = []
    for ses in posts:
        if reference in piv.columns and ses in piv.columns:
            n = int((piv[reference] & piv[ses]).sum())
        else:
            n = 0
        rows.append({"timepoint": ses, "complete_pairs": n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build tidy cohort table from deepmriprep outputs.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--write-clinical-template", action="store_true",
                    help="Write a clinical.csv template (one row per subject) and exit.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.write_clinical_template:
        out = write_clinical_template(cfg)
        print(f"Clinical template written -> {out}\nFill treated_side (L/R), tremor scores, etc.")
        return

    cohort = build_cohort(cfg)
    out = save_cohort(cfg, cohort)

    n_sub = cohort["subject"].nunique()
    print(f"Parsed {len(cohort)} scans from {n_sub} subjects -> {out}")
    print("\nScans per session:")
    print(cohort["session"].value_counts().to_string())
    print("\nComplete pre->post pairs (the real analysis n):")
    pc = pair_counts(cohort, cfg["reference_session"], cfg["post_sessions"])
    print(pc.to_string(index=False))
    missing = (~cohort["mwp1_exists"]).sum()
    if missing:
        print(f"\nWARNING: {missing} scans have a missing/unreadable mwp1 on disk.")


if __name__ == "__main__":
    main()
