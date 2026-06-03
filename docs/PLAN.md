# et-fus-vbm — Analysis Plan (peer-reviewed draft)

**Study:** Longitudinal structural VBM of MR-guided focused ultrasound (MRgFUS) unilateral VIM
thalamotomy for essential tremor (ET).
**Data:** ~1115 deepmriprep scans (sub-ET###; ses-preop/post24h/post3mo/post1yr); modulated GM
(`mwp1`), WM (`mwp2`), `s8mwp1`, TIV at 1.5 mm MNI. Native T1w + T2w + T2starw(FIESTA) retained in
`BIDS/`. Existing SUIT cerebellar volumes for ~525 scans. Local CerebNet/FastSurfer.

This plan was reviewed by a statistics methodologist and an FUS/ET neuroimaging clinician (agent
panel, 2026-06-03). Their major points are folded in below and tracked as ADRs in `docs/adr/`.

---

## 1. Primary hypothesis (confirmatory)

> Unilateral VIM ablation produces, at **1 year**, **lateralized** secondary degeneration along the
> dentato-rubro-thalamic tract: **ipsilesional thalamic / peri-lesional GM volume loss** and—because
> the DRTT decussates—**contralateral cerebellar GM loss greater than ipsilateral**, distinguishable
> from symmetric disease-/age-related change by its **crossed asymmetry**.

**Primary endpoint:** within-subject change (1 yr − preop) in two *a priori* ROIs after flipping all
brains to a common treated hemisphere: (i) ipsilesional thalamus/peri-lesional region; (ii)
contralateral vs ipsilateral cerebellar GM (the crossed-asymmetry contrast). Report effect sizes
(Cohen's d / mean % volume change) with 95% CIs; ROI-level inference, not whole-brain voxelwise.

Everything else (24h/3mo maps, whole-brain voxelwise, Path A normative maps, WM beyond DRTT,
dentate) is **exploratory / hypothesis-generating** and reported without confirmatory significance
claims. Pre-register the primary hypothesis, ROIs, model, QC and exclusion rules before analysis
(OSF, timestamped).

---

## 2. What the data can and cannot support (key facts)

| Question | Answer |
|---|---|
| CSF voxelwise (mwp3)? | deepmriprep segments CSF but **cannot output modulated-warped CSF**; only native/affine `p3` + scalar `csfv`. A re-run won't add `mwp3`. True CSF VBM needs CAT12/SPM. **Omit voxelwise CSF**; ex-vacuo signal, if wanted, via ventricular/`csfv` volumes (CAT12 or atlas), not deepmriprep VBM. |
| Reprocess time | ~150 s/scan CPU (~2 days for cohort); ~10× on CUDA GPU; **Apple MPS unsupported**. Re-runs recompute fully (no incremental output). |
| Cerebellum — whole | **Done** (`total_cerebellar_gm`, 525 scans). Trust **moderate–high** (ICC > 0.95). |
| Cerebellum — voxelwise | **Not done.** Quick masked-`s8mwp1` map = exploratory only (whole-brain MNI mis-aligns cerebellar fissures). Proper = **SUIT VBM** from native T1. Trust **low–moderate**. |
| Cerebellum — lobular | Cortex **done** (28 SUIT lobules). Trust **low–moderate** (large lobules ok; thin VIIb/VIIIa/b poor). Dentate/deep nuclei **not done and not reliable on T1** — needs **QSM/SWI (you have T2starw FIESTA)**. |
| Biggest cerebellar caution | Cerebellar VBM **does not reproduce across pipelines** (Wang 2023, N=211; ET controversy Daniels 2006). → cross-check with **CerebNet** (2nd segmenter), prefer **ROI/lobar over voxel** for inference, lean on the **crossed-asymmetry** contrast, report ICC + effect sizes. |

---

## 3. Methodological decisions forced by peer review (→ ADRs)

1. **Laterality / image flipping (ADR-0001).** Thalamotomy is unilateral; pooling unflipped images
   cancels the main effect. **Flip every subject to a common "treated" hemisphere** (treatment side
   read from operative records, not the image). Highest-value single fix.
2. **24h = oedema, not atrophy (ADR-0002).** Reframe timepoints by substrate (24h = acute
   lesion+oedema, space-occupying; 3mo = cavitation/oedema resolution; 1yr = scar + secondary
   degeneration). VBM "atrophy" claims only at 3mo/1yr. Analyse 24h descriptively (lesion/oedema),
   ideally from T2/FLAIR, not GM-VBM.
3. **Within-subject is primary; drop the pooled cross-sectional GLM (ADR-0003).** A GLM over all
   scans violates independence (repeated measures) and confounds timepoint with which subjects
   returned. Use **linear mixed-effects** (random intercept/slope per subject, time continuous) or a
   **paired difference** design. Permutation must use within-subject exchangeability blocks.
4. **Internal reference beats external normative (ADR-0004).** Path A z-scoring against an
   age/scanner-mismatched external cohort is invalid for inference → use **each subject's own preop**
   as reference. External-normative demoted to exploratory/QC visualisation only.
5. **Lesion-aware normalization (ADR-0005).** Delineate lesion (±oedema) per post session (from
   T2/FLAIR) and apply **cost-function masking** or **enantiomorphic** normalization so the lesion
   doesn't bias the warp into peri-lesional "change."
6. **Proper longitudinal metric (ADR-0006).** Replace difference-of-independently-modulated maps with
   a **within-subject unbiased template + Jacobian (tensor-based morphometry)** (ANTs/CAT12
   longitudinal).
7. **Cerebellum via SUIT + cross-check (ADR-0007).** Dedicated **SUIT** normalization from native T1
   for voxelwise/lobular cerebellar work; **≤4–6 mm** smoothing (not 8 mm) for focal structures;
   cross-validate lobular volumes with **CerebNet**; dentate only from **T2starw/QSM**.
8. **Power-honest late timepoint (ADR-0008).** 1 yr (n≈19, fewer complete pairs) cannot support
   whole-brain voxelwise inference → restrict to a priori ROIs, report effect sizes + CIs, and use the
   LMM trajectory across all post timepoints to borrow strength. State the minimum detectable effect.
9. **TIV once (ADR-0009).** Modulated maps already encode volume → TIV is a **covariate, applied
   once**, taken from the **preop** scan; it largely cancels within-subject. No proportional scaling.
10. **Multiplicity / pre-registration (ADR-0010).** One confirmatory contrast; all else exploratory.
    TFCE/cluster-FWE within each map; small-volume correction within a priori ROIs. No selecting the
    "best" timepoint/tissue/atlas post hoc.

---

## 4. Pipeline (phased)

**Phase 0 — Cohort & QC (no reprocessing).** Parse `deepmriprep_outputs.csv` → tidy table (subject,
session, paths, TIV, affine_loss, warp_mse). Pre-specify QC thresholds; flag bad registrations and
TIV outliers; QC dashboard. Tabulate **complete pre→post pairs per timepoint** (the real n). Ingest
existing `cerebellar_volumes.csv`. **Obtain treatment side per subject** (blocking input for flipping).

**Phase 1 — Core preprocessing.** Resample `mwp1/mwp2` to 2 mm MNI (for normative reference grid);
build flip-to-treated-hemisphere transform; assemble per-subject preop/post pairs.

**Phase 2 — Primary (Path C, within-subject longitudinal).**
- ROI endpoints from existing SUIT cerebellar volumes: contralateral vs ipsilateral cerebellar GM
  change at 1 yr (and 3 mo), LMM across post timepoints; thalamic/peri-lesional ROI from flipped
  modulated GM. Effect sizes + CIs; cross-check cerebellum with CerebNet.
- Optional voxelwise TBM (Jacobian) within DRTT/thalamus ROI, lesion-masked, sign-flip permutation +
  TFCE, SVC.

**Phase 3 — Exploratory.**
- Path B as **LMM** (not pooled GLM) on flipped `s8mwp1`, whole-brain, exploratory maps.
- Path A internal-preop z-maps for per-patient visualisation; external-normative only as QC.
- 24h lesion/oedema descriptive characterisation (T2/FLAIR-based).
- SUIT VBM (native T1) for voxelwise cerebellum; dentate via T2starw if pursued.

**Phase 4 — Reporting & viz.** Publication figures (glass-brain/ortho overlays, ROI forest/bar plots
with CIs, crossed-asymmetry plot, sample-size-per-timepoint), colorblind-safe, vector. Clinical
correlation with tremor scores (TETRAS/CRST) if available — framed around DRTT coverage, not scar
volume.

**Optional reprocessing (only if approved):** CAT12 for true CSF `mwp3`; deepmriprep `cobra_volumes`
or SUIT dentate for deep nuclei; SUIT-VBM stream. Each ~2 days CPU.

---

## 5. Repo / engineering

Modular `src/etfvbm` (io, qc, resample, normative[A], group_glm[B], longitudinal[C], cerebellum, viz);
config-driven; data/derivatives/external gitignored; ADRs in `docs/adr/`; tests for io/resample/stats;
pre-registration doc in `docs/`. Public GitHub repo `et-fus-vbm`, sole author Niels Pacheco-Barrios.

## 6. Biggest threats to validity (watch list)

- **Statistical:** non-independence if any pooled cross-sectional model is used (→ LMM/paired).
- **Overall:** lesion/oedema-driven registration & segmentation failure mistaken for morphometry
  (→ lesion masking + per-scan registration QC; treat 24h descriptively).
- **Biological:** ET's own + ageing cerebellar change over 1 yr mimicking diaschisis (→ crossed
  asymmetry, not absolute volume, is the protected inference).
- **Cerebellar:** pipeline-dependent non-reproducibility (→ second segmenter, ROI not voxel).
