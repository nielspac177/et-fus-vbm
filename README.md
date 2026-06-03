# et-fus-vbm

Voxel-based morphometry (VBM) of MR-guided focused ultrasound (MRgFUS) thalamotomy
for essential tremor (ET), driven by [`deepmriprep`](https://github.com/wwu-mmll/deepmriprep)
tissue segmentations.

This project takes `deepmriprep` VBM outputs (modulated, MNI-warped grey- and
white-matter maps) and runs **three complementary analyses**:

| Path | Module | Question it answers |
|------|--------|---------------------|
| **A — Normative atrophy** | `etfvbm.normative` | Per-scan: where is this individual abnormal vs a normative control distribution? (single-subject Z-maps + ROI tables) |
| **B — Group VBM-GLM** | `etfvbm.group_glm` | Population: is there a group-level GM difference between timepoints? (permutation/TFCE, FWE-corrected) |
| **C — Longitudinal change** | `etfvbm.longitudinal` | Within-subject: what changes from pre-op to each post-op timepoint? (paired, the most sensitive design for a focal FUS lesion) |

Paths A/B/C share one preprocessing core: parse → QC → resample (1.5 mm → 2 mm MNI).

## Why these design choices

- **Within-subject (C) is the primary analysis.** The cohort is longitudinal
  (`ses-preop / post24h / post3mo / post1yr`); pairing each post-op scan to the
  patient's own pre-op scan removes between-subject anatomical variance and is far
  more sensitive to a small focal thalamic lesion than normative Z-scoring.
- **GM + WM, no CSF.** `deepmriprep` (`outputs='vbm'`) emitted `mwp1` (GM) and
  `mwp2` (WM). GM captures the target lesion; WM captures dentato-rubro-thalamic
  tract (DRTT) degeneration. CSF (`mwp3`) was not produced and adds little for a
  deep focal lesion, so the CSF-based composite "H-score" is intentionally omitted.
- **Interpret `post24h` as lesion + oedema, not atrophy.** True secondary atrophy
  emerges at 3 mo / 1 yr.
- **Sample sizes are imbalanced** (post24h ≫ post1yr); see the cohort report before
  making longitudinal claims.

## Upstream

Z-scoring logic and the normative control distributions (`ctrl_dist/`, 2 mm MNI mean/std
for GM/WM) are reused from [`Calvinwhow/vbm`](https://github.com/Calvinwhow/vbm).
Run `scripts/fetch_upstream.sh` to clone it into `external/` (gitignored) — that
provides the MNI 2 mm mask, ROI atlases, and control distributions.

## Quick start

```bash
conda env create -f environment.yml && conda activate etfvbm
pip install -e .
bash scripts/fetch_upstream.sh          # gets mask + ctrl_dist + ROIs into external/
python -m etfvbm.io     --config config/cohort.yaml   # build tidy cohort table
python -m etfvbm.qc     --config config/cohort.yaml   # QC report + scan flags
python -m etfvbm.normative   --config config/cohort.yaml   # Path A
python -m etfvbm.longitudinal --config config/cohort.yaml  # Path C
python -m etfvbm.group_glm    --config config/cohort.yaml  # Path B
```

Data never leaves your disk: `data/` is a symlink to `deepmriprep_output`, and
`data/`, `derivatives/`, `external/` and all NIfTIs are gitignored.

### Structure–symptom association & atrophy maps

`etfvbm.association` regresses regional volume change against clinical outcomes (tremor
improvement, imbalance, weakness) at 3 months — ROI-level, FDR-corrected, with a
**lesion-burden covariate that is mandatory** (lesion size causes both atrophy and
outcome; unmeasured it confounds everything — see ADR-0011). `etfvbm.atrophy_maps`
writes group `mwp1`-difference NIfTIs (signed: blue = atrophy). These are **cross-sectional
differences, not TBM/Jacobian** (no within-subject registration was done) and are
exploratory.

### Status / first result

Primary cerebellar analysis runs on the existing SUIT volumes: **total cerebellar GM at 3
months = −1.6 % median / −3.4 % mean, Cohen's d_z = −0.25, p = 0.045 (n = 68)** — a modest
early decline. 24 h is oedema-dominated (artifact-flagged) and 1 yr is descriptive (n≈6–7).
The crossed contralateral-vs-ipsilateral test activates once `config/clinical.csv`
(treated side) is filled.

See [`docs/PLAN.md`](docs/PLAN.md) for the full analysis plan, [`docs/adr/`](docs/adr/) for
decisions, and [`docs/preregistration.md`](docs/preregistration.md) before running
confirmatory analyses.
