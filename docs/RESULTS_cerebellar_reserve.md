# Cerebellar reserve → imbalance: adversarial analysis & honest verdict

An adversarial collaboration (advocate / skeptic / power-analyst agents) on whether baseline
cerebellar volume predicts 3-month imbalance after MRgFUS VIM thalamotomy. Recorded so the
boundary-line result is not over- or under-sold.

## Trail of evidence
1. **Median-split total cerebellum** OR=2.50, p=0.023 — **OVERTURNED** (dichotomization
   artifact): continuous OR=1.20, p=0.34; adds nothing over lesion size (ΔAUROC +0.004).
2. **Voxelwise PALM** (3 mo GM change → imbalance), whole-brain FWE: **null** (peak |t|=4.8,
   FWE p=0.26, n=53).
3. **Continuous lobule-wise** (n≈140): strongest = **anterior** lobules I–IV (OR=1.49, p=0.056)
   and V (OR=1.47, p=0.062); posterior/vermis null. Anterior cerebellum = spinocerebellar
   gait/stance substrate (Schmahmann topography; alcoholic cerebellar degeneration; Mitoma–
   Manto cerebellar reserve).

## The pre-specified-style candidate test (refutation battery)
Anterior composite (I–IV + V, hemispheric), TIV-residualized z, one volume/subject
(preop preferred), logistic `imbalance ~ z + age + sex + TIV + log10_lesion`, n=139, 45 events:

| Test | Result | Pass? |
|---|---|---|
| Effect | OR=1.50 per 1-SD decrease, 95% CI [1.00, 2.26] | borderline |
| Two-sided p | 0.050 | line |
| Permutation null (2000) | p=0.054 | ✗ |
| Timepoint replication (preop & 24h) | OR≈1.47, p≈0.063 both | ✓ |
| Anatomical specificity (posterior negative control) | OR=1.24, p=0.28 (null) | ✓ |
| **Global-GM confound (decisive)** | OR 1.50→**1.53**, p **0.043**; corr(anterior,globalGM)=0.15 | ✓ **survives** |
| **Lesion-location / inferior-extension confound** | OR 1.50→**1.61**, p **0.030** | ✓ **survives** |
| Bonus: inferior lesion extension itself | independently predicts imbalance, p=0.035 | mechanistic |

## Verdict
**A credible, theory-grounded, confound-resistant candidate — pre-registration-worthy, not yet
confirmed.** The anterior (gait-substrate) cerebellar reserve effect: (a) is anatomically
specific (posterior cerebellum null); (b) replicates across timepoints; (c) **survives the two
decisive confounds** — it is independent of global GM (corr 0.15) and of lesion location, and
*strengthens* to **OR=1.61, p=0.030** in the fully adjusted model. A second, independent finding
emerged: **inferior lesion extension** (toward cerebellar outflow / DRTT) predicts imbalance
(p=0.035), matching the mechanistic literature.

**Why it is NOT yet a confirmed finding:** (1) the composite was chosen after seeing I–IV/V win
(HARKing) → needs a **pre-registered replication** (see `preregistration_H3.md`); (2) underpowered
(n=139, power ~0.55, borderline permutation p=0.054); (3) cerebellar VBM is notoriously
non-reproducible. Significance was reached via the *proper two-sided confound-adjusted* model —
NOT via post-hoc one-sided testing (declined as HARKing).

## Why "nothing is significant"
A genuine power problem (n≈139 bound by clinical-covariate completeness, not SUIT coverage),
a noisy change metric, whole-brain correction, and the anterior lobules being the
worst-segmented cerebellar regions. Not evidence of a true null.

## To resolve it legitimately (NOT by p-hacking)
1. **Decisive confound test:** add a global-GM covariate — is "low anterior cerebellum" just
   "globally atrophied/older brain"? (compute total GM from mwp1).
2. **Lesion-location/inferior-extension covariate** from the masks (the real ataxia driver).
3. **Pre-register** H3 (anterior-cerebellar reserve, one-sided, single composite) on a fresh
   sample / more events BEFORE analysis — only then is one-sided p (≈0.025 here) admissible.
4. **More events** (continuous balance outcome if available recovers ~30–40% power).

Declaring significance now via one-sided p=0.025 would be HARKing (the composite was chosen
after seeing I–IV/V win) — explicitly declined.

## Corrected analysis & adversarial implementation audit (update)

An adversarial code+stats audit found two flaws in the first responder analysis and they were
fixed:
- **Predictor contamination (S1):** the pooled "reserve" volume drew from post-lesion scans
  for 46% of subjects. **Fixed → preop-only.** The corrected preop-only anterior reserve →
  3-month imbalance is **OR=1.61 [1.04, 2.47], p=0.031 — unchanged**, so the bug did not drive
  the result (the effect is consistent across timepoints).
- **Collider design (S2):** responder-stratification conditions on a common effect of lesion
  size. **Fixed → full-cohort reserve × continuous-response interaction model** instead of
  stratifying.

**Double dissociation (the decisive test):**
- Reserve → imbalance: interaction-model main effect OR≈1.96, **p=0.009**; preop-only p=0.031.
- Reserve × response interaction **p=0.058** (reserve matters *more* in better responders —
  directionally supports the responder intuition; borderline).
- Reserve → tremor **efficacy**: **p=0.614 (null)** — reserve predicts the *side effect*, not
  the benefit. Mechanistically specific, not a generic "bad brain" confound.

**Responder-stratified (corrected, covariate-adjusted), 3mo & 1yr:** directionally consistent
(3mo adj OR=1.36, p=0.15; 1yr null) but underpowered and a collider design → demoted to
descriptive. Responders vs non-responders do not differ in cerebellar volume (efficacy null).

**Adjudication (agent panel):** probably real but fragile, single-cohort; ~55–65% replication
probability; publish as a hypothesis-generating side-effect biomarker with the anatomical
gradient + double dissociation as the headline, pre-registered confirmation (H3) as next step.
