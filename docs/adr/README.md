# Architecture / Analysis Decision Records

Decisions driving `et-fus-vbm`, with the context that forced each. Status: Accepted
2026-06-03 unless noted. Several were forced by a 5-agent research + peer-review panel
(stats methodologist, FUS/ET clinician) and a Round-2 adversarial critique.

| ADR | Decision | Status |
|-----|----------|--------|
| 0001 | Flip all brains to a common treated hemisphere | Accepted |
| 0002 | 24 h = oedema, not atrophy; analyse descriptively only | Accepted |
| 0003 | Within-subject (paired/LMM), never pooled cross-sectional GLM | Accepted |
| 0004 | Internal preop reference, not external normative | Accepted |
| 0005 | Lesion-aware normalization (cost-function masking) | Proposed (needs lesion masks) |
| 0006 | Longitudinal metric: scalar ROI volume change (NOT TBM) | Accepted (revised by adversary) |
| 0007 | Cerebellum via SUIT tables + forest plots, cross-check CerebNet | Accepted |
| 0008 | 1 yr (n≈7) descriptive only; 3 mo (n≈86) is primary | Accepted |
| 0009 | TIV covariate once, from preop | Accepted |
| 0010 | One pre-registered confirmatory contrast; rest exploratory | Accepted |
| 0011 | Measure lesion burden before any association claim | Proposed (blocking) |
| 0012 | No reprocessing for v1 (use existing GM/WM + SUIT volumes) | Accepted |

---

**ADR-0001 — Flip to treated hemisphere.** *Context:* thalamotomy is unilateral;
pooling unflipped images cancels the lateralized lesion + crossed-cerebellar effect.
*Decision:* flip every subject L↔R so the treated side is common; keep `treated_side` as
nuisance. Treated side comes from the operative record, never the image.

**ADR-0002 — 24 h is oedema.** *Context:* at 24 h the lesion+oedema is space-occupying
(~3.7× the necrotic core; literature). *Decision:* never call 24 h change "atrophy";
analyse it descriptively (lesion/oedema), ideally from T2/FLAIR. Confirmed empirically: 20
of 139 24 h scans show |ΔGM|>30 % (segmentation/oedema artifact).

**ADR-0003 — Within-subject, not pooled GLM.** *Context:* repeated measures violate
independence; cross-sectional pooling confounds timepoint with who returned. *Decision:*
paired change scores / linear mixed-effects with random intercept per subject.

**ADR-0004 — Internal reference.** *Context:* the external normative (Calvinwhow/vbm
`ctrl_dist`) is age/scanner/pipeline-mismatched. *Decision:* each subject's own preop is
the reference; external-normative kept only as QC visualisation.

**ADR-0005 — Lesion-aware normalization.** *Context:* the lesion biases the warp into
peri-lesional "change." *Decision:* cost-function masking / enantiomorphic normalization
once lesion masks exist (see ADR-0011).

**ADR-0006 — Scalar ROI volume change, NOT TBM.** *Context:* the adversarial review showed
we have NO within-subject longitudinal registration, so log-Jacobian/TBM **does not exist**
in the data. *Decision:* primary metric = scalar SUIT lobular/hemisphere (+thalamus)
volume change vs preop. Voxelwise `mwp1` differences are exploratory and never labelled
"Jacobian/TBM". *Consequence:* the pitched voxelwise-TBM story requires reprocessing
(CAT12/ANTs longitudinal); deferred.

**ADR-0007 — Cerebellum honestly.** *Context:* we have a 28-lobule SUIT *table*, not a
voxelwise cerebellar map; standard MNI VBM mis-registers the cerebellum; cerebellar VBM is
pipeline-unstable (Wang 2023). *Decision:* report lobular results as **forest plots** (not
a fake voxelwise flatmap); cross-check with CerebNet; lean on the crossed-asymmetry
contrast; report effect sizes + ICC.

**ADR-0008 — Power-honest timepoints.** *Context:* complete pre→post pairs = 400 (24 h),
86 (3 mo), 7 (1 yr). *Decision:* 3 mo primary; 1 yr descriptive (no voxelwise inference);
use an LMM trajectory to borrow strength.

**ADR-0009 — TIV once.** Modulated maps already encode volume → TIV is a single covariate,
from preop; no proportional scaling (no double-correction).

**ADR-0010 — Pre-register one confirmatory contrast.** Family spans outcomes × timepoints ×
ROI/voxelwise streams; declare one primary (crossed contralateral-cerebellar + ipsilesional
GM loss at 3 mo / tremor-change ROI), FDR within the ROI set; all else exploratory.

**ADR-0011 — Measure lesion burden (BLOCKING for association).** *Context:* lesion volume
causes BOTH atrophy and outcome; unmeasured, every association is confounded. We have no
lesion masks for this cohort. *Decision:* before any association *claim*, obtain a lesion
burden measure — segment cavity+ring in-cohort (manual/semi-auto, inter-rater on a subset)
or use treatment-dose surrogates (n sonications, energy, max temperature) as covariate.

**ADR-0012 — No reprocessing for v1.** Use existing deepmriprep GM/WM maps + SUIT volumes.
SUIT-VBM, dentate (T2starw/QSM), and CAT12 true-CSF are deferred (~2 days CPU each).
