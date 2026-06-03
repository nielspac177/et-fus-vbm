# Pre-registration — H3: Anterior-cerebellar reserve predicts post-thalamotomy imbalance

> Status: DRAFT to be timestamped/frozen (OSF) BEFORE the confirmatory test is run on
> independent data. The effect below was found in an exploratory/adversarial analysis
> (`RESULTS_cerebellar_reserve.md`); H3 exists to confirm it without HARKing. The composite,
> region, sign, model, and one-sided direction are fixed HERE, in advance of the confirmatory
> sample.

## Background & rationale
Post-MRgFUS VIM thalamotomy imbalance/ataxia is the commonest adverse event. The anterior
cerebellum (lobules I–V + anterior vermis) is the spinocerebellar leg/trunk/stance-gait
substrate (Schmahmann functional topography; alcoholic cerebellar degeneration as a lesion
model; Mitoma–Manto cerebellar-reserve framework). We hypothesise that lower **baseline**
anterior-cerebellar volume = less structural reserve to buffer the thalamotomy perturbation =
higher imbalance risk.

## Confirmatory hypothesis (H3)
Lower preoperative anterior-cerebellar GM volume predicts higher odds of 3-month imbalance,
independent of lesion volume, lesion inferior-extension, global GM, age, sex, and TIV.

## Predictor (fixed, 1 df)
`AnteriorCb` = SUIT GM volume, lobules **I–IV + V** (bilateral, hemispheric sum), preop;
TIV-residualised; z-scored. Sign: **lower volume → higher risk** (directional).

## Outcome (primary)
3-month imbalance, binary (`imbalance`). Prevalence ≈ 0.36.

## Model
Logistic (Firth if events sparse):
`imbalance ~ AnteriorCb_z + age + sex + TIV + log10(lesion_volume) + lesion_inferior_extension + global_GM`
- `lesion_inferior_extension` = `centroid_z` and `frac_below_acpc` (from `etfvbm.lesion`).
- `global_GM` = total GM from preop mwp1.

## Test & threshold
Coefficient on `AnteriorCb_z`. **One-sided α = 0.05** (direction pre-registered here). One test;
no multiplicity correction.

## Negative control (predicted null)
`PosteriorCb` = Crus I/II + VIIb + IX + X (same model) → expected OR ≈ 1, non-significant.

## Robustness (all pre-specified)
Permutation null (5000); timepoint replication (preop AND 24h, same sign/magnitude);
dose-response monotonicity across volume quartiles; bootstrap BCa CI on the OR.

## Decision rule
- One-sided p<0.05 AND CI excludes 1 AND posterior control null AND monotone dose-response →
  **confirmed** anterior-cerebellar reserve effect.
- Non-significant with CI bracketing 1.0 at adequate n → the exploratory trend was an
  underpowered/HARKing artifact; report null.

## Power / sample size
Exploratory effect OR≈1.5–1.6 per SD, event rate 0.36, ~5–6 covariates → **n ≈ 226 for 80%
power (one-sided)**. Current exploratory n=139 (~55% power). Confirmatory test requires either
additional SUIT-processed subjects (clinical outcomes exist for 663; SUIT coverage is the
binding constraint) or an independent cohort.

## Exploratory result motivating H3 (for transparency — NOT the confirmatory test)
Anterior composite, n=139, fully adjusted (global GM + lesion location): OR=1.61 [1.05, 2.49],
two-sided p=0.030; posterior control null (OR=1.24, p=0.28); replicates preop & 24h. Inferior
lesion extension independently predicted imbalance (p=0.035).
