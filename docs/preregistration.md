# Pre-registration — et-fus-vbm (draft)

> Timestamp and freeze (e.g. OSF) BEFORE running confirmatory analyses. Everything not
> listed as confirmatory is exploratory / hypothesis-generating.

## Background & rationale
Longitudinal structural MRI after unilateral MRgFUS VIM thalamotomy for essential tremor.
We test whether the ablation produces lateralized secondary degeneration along the
cerebello-thalamic pathway and whether regional change tracks clinical outcome.

## Primary confirmatory hypothesis (H1)
At **3 months**, cerebellar GM volume declines, and the **contralesional** cerebellar
hemisphere declines **more than the ipsilesional** (crossed dentato-thalamic degeneration),
after flipping to a common treated hemisphere.
- **Endpoint:** session×hemisphere interaction (ipsi vs contra) in a linear mixed-effects
  model on SUIT hemispheric GM volume (random intercept per subject).
- **Effect reporting:** mean % change + 95% CI + Cohen's d_z per timepoint.

## Secondary confirmatory hypothesis (H2)
Greater 3-month cerebellar/thalamic volume loss in pre-specified ROIs associates with
clinical outcome (tremor improvement; imbalance; weakness).
- **Model:** ROI %change ~ outcome + age + sex + TIV + baseline severity + treated_side +
  **lesion burden** (linear for tremor; Firth logistic for binary AEs).
- **Multiplicity:** Benjamini-Hochberg FDR q<0.05 across the ROI set, **per outcome**.

## Timepoint hierarchy
- **Primary: 3 months** (n≈86 complete pairs).
- **24 h: excluded** from atrophy inference (oedema; descriptive/sensitivity only).
- **1 year: descriptive only** (n≈7; no voxelwise inference).

## ROIs (pre-specified, anatomical — independent of the data)
SUIT lobules I–IV, V, VI, Crus I/II, VIIb, VIIIa/b, IX, X, vermis; hemispheric sums;
thalamus (treated/contralateral) when available. Confirmatory core: treated-side V/VI,
dentate region, VIM-thalamus + contralesional cerebellar hemisphere.

## Covariates (fixed)
age, sex, TIV (from preop), baseline tremor severity, treated_side, **lesion burden**.

## Analysis decisions (locked)
- Flip to common treated hemisphere (ADR-0001).
- Metric = scalar SUIT/thalamic volume change; NOT TBM (ADR-0006).
- TIV once, from preop (ADR-0009).
- QC exclusion: scans with |whole-cerebellum %change|>30, or failing
  affine_loss/warp_mse/TIV thresholds in `config/cohort.yaml`.
- Exploratory voxelwise `mwp1`-difference maps: smoothed (s8), lesion-masked, 3 mo only,
  PALM/TFCE FWE within an a priori cerebellar+thalamic small-volume mask.

## Known limitations / blockers
- **Lesion burden must be measured** before H2 is interpretable (ADR-0011); otherwise
  associations are reported as confounded/exploratory.
- No within-subject longitudinal registration in v1 → no Jacobian/TBM.
- Cerebellar VBM is pipeline-unstable; cross-check with CerebNet; crossed-asymmetry is the
  protected inference.

## Negative controls
Contralesional hemisphere (should be the *less*-affected side); 24 h (should reflect
oedema, opposite sign to atrophy).
