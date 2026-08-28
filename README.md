# Vocal Repertoire Geometry: MeerKAT Case Study

This project uses public MeerKAT as the first case study for a reusable vocal repertoire geometry framework.

The goal is to separate four questions that are often mixed together:

- Do human labels correspond to natural acoustic clusters?
- Are human labels categorically separable in an acoustic representation?
- Does a single label contain multiple acoustic modes?
- Are calls arranged along continuous or graded acoustic geometry?

## First Case Study Question

The first analysis asks a deliberately narrow question:

> Do public MeerKAT human call labels correspond to natural structure in unsupervised acoustic space?

This step does not claim to infer cognition, urgency, predator context, or individual identity. It tests whether a public labelled vocal repertoire contains label-aligned acoustic geometry that can be quantified and compared across representations.

## Current Structure

```text
repertoire_geometry/
  data_manifest/
  features/
  figures/
  reports/
  scripts/
```

## Current Data Products

- `data_manifest/call_manifest.csv`: call-level manifest generated from public MeerKAT HDF5 labels.
- `data_manifest/label_summary.csv`: counts by label, label category, and focal status.
- `data_manifest/duration_summary.csv`: duration summary by label and focal status.
- `data_manifest/pilot_manifest_balanced_600.csv`: balanced focal pilot subset with 600 calls per target label.
- `data_manifest/pilot_manifest_capped_3000.csv`: larger focal pilot subset with up to 3000 calls per target label.

Generated CSV files and all audio/model artifacts stay out of Git.

## Scripts

- `scripts/build_meerkat_call_manifest.py`: reads HDF5 label files and creates call-level manifest tables.
- `scripts/make_pilot_manifest.py`: creates class-balanced or capped pilot manifests from the full call manifest.
- `scripts/extract_acoustic_features.py`: extracts raw and RMS-normalized acoustic features plus QC.
- `scripts/extract_egemaps_features.py`: extracts openSMILE eGeMAPSv02 Functionals with raw and RMS-normalized variants plus validity QC.
- `scripts/analyze_geometry_pilot.py`: runs first-pass PCA, UMAP, HDBSCAN, linear probe, kNN purity, and GMM+BIC.
- `scripts/extract_spectrogram_representation.py`: extracts fixed-size raw and RMS-normalized log-mel spectrogram vectors.
- `scripts/extract_animal2vec_embeddings.py`: extracts context-window animal2vec embeddings with focal-frame pooling when a compatible animal2vec environment and checkpoint are available.
- `scripts/analyze_embedding_pilot.py`: runs the same geometry analysis on dense vector representations such as spectrogram embeddings.

## Usage

Create the full call-level manifest:

```powershell
python scripts/build_meerkat_call_manifest.py --skip-empty-size 3208
```

Create the strict balanced geometry pilot:

```powershell
python scripts/make_pilot_manifest.py --max-per-label 600 --name balanced_600
```

Create the capped stratified pilot:

```powershell
python scripts/make_pilot_manifest.py --max-per-label 3000 --name capped_3000
```

## Target Public Labels

```text
sn, cc, ld, mo, al, soc, agg, oth
```

## First Analysis Plan

Start with `pilot_manifest_balanced_600.csv`, extract acoustic features, and run:

- label-cluster alignment
- categorical separability
- within-label multimodality
- continuous acoustic geometry
- cross-representation robustness

Use two pilot sets:

- `balanced_600`: strict class-balanced geometry pilot for PCA, UMAP, HDBSCAN, GMM, Fuzzy C-Means, silhouette, kNN purity, AMI, and ARI.
- `capped_3000`: capped stratified pilot for checking whether the same patterns remain visible under a more natural, long-tailed label prevalence.

For short calls such as `sn`, use raw call boundaries for hand-crafted acoustic features and eGeMAPS. For neural embeddings, use real acoustic context windows, then pool only the focal-call frames.

## Current Pilot Result

The first `balanced_600` acoustic-feature pilot is summarized in:

- `reports/balanced_600_acoustic_pilot.md`

The first result suggests that manual labels are moderately decodable from hand-crafted acoustic features, but do not align with eight clean unsupervised clusters under this representation. This is an initial pilot result only; eGeMAPS, animal2vec, and robustness analyses are still needed.

The HDBSCAN parameter sensitivity diagnostic is summarized in:

- `reports/hdbscan_sensitivity_balanced_600.md`

Across 200 HDBSCAN fits over raw/normalized features, full/PCA spaces, and multiple `min_cluster_size`/`min_samples` values, label-cluster alignment remained weak. The highest observed AMI was about 0.14 in PCA-10, with a high noise rate around 0.64.

The first log-mel spectrogram representation baseline is summarized in:

- `reports/logmel_spectrogram_balanced_600.md`
- `reports/paper_style_spectrogram_balanced_600.md`

This diagnostic checks whether weak clustering was caused by using overly coarse hand-crafted summary features. A simple 64x64 call-boundary log-mel representation did not improve label-aligned natural clustering or linear-probe separability. This suggests that the weak cluster result is not only a librosa-summary-feature artifact, although stronger context-aware embeddings still need to be tested.

The paper-style version uses 40 mel bins, approximately 30 ms frames, 3.75 ms hops, per-spectrogram z-transform, original call duration, and right-padding to a 500 ms grid. It produced a more continuous UMAP trajectory, but default HDBSCAN assigned all points to noise and the linear probe remained around Macro-F1 0.53.

The HDBSCAN sensitivity diagnostic for the paper-style spectrogram baseline is summarized in:

- `reports/hdbscan_sensitivity_logmel_paper_pad500_balanced_600.md`

Across 250 HDBSCAN fits over raw/normalized variants, PCA spaces, and HDBSCAN parameters, the highest observed AMI was about 0.184. This shows that the all-noise default result was partly parameter-dependent, but the recovered clusters were fragmented, noisy, and only weakly aligned with public labels. When screened first using unsupervised criteria such as noise rate, cluster count, cluster size, and persistence, the remaining settings still had weak AMI/ARI.

The first eGeMAPS baseline is summarized in:

- `reports/egemaps_balanced_600.md`

eGeMAPS produced somewhat higher default-HDBSCAN AMI than the earlier baselines, but feature validity was highly label-dependent. In particular, most `sn` calls were too short for valid eGeMAPS extraction under original call boundaries. A valid-only sensitivity analysis reduced the apparent separability, so eGeMAPS does not overturn the current conclusion that public labels are only weakly aligned with natural acoustic clusters.

The animal2vec extraction plan and script are summarized in:

- `reports/animal2vec_plan.md`

animal2vec should be run in a separate Python 3.9-compatible environment because the official repository depends on fairseq and warns that Python 3.10+ is not compatible. The extraction script uses real acoustic context as model input, but pools only focal-call frames for the final embedding.

The optional dependency list for that separate environment is:

- `requirements-animal2vec.txt`

The current script I/O, preprocessing, label use, dimensionality reduction, clustering parameters, metric definitions, leakage risks, NaN/imputation handling, and result-code consistency audit is:

- `reports/script_io_and_method_audit.md`

Current animal2vec local environment status:

- Python 3.9 environment: created locally as a project-specific environment; keep the exact machine path out of public Git.
- Installed and smoke-tested: `torch==1.13.1+cpu`, `torchaudio==0.13.1+cpu`, `librosa==0.10.1`
- Not yet installed: `fairseq`
- Windows blockers observed: PyPI `fairseq==0.12.2` requires Microsoft Visual C++ Build Tools; the official pinned GitHub fairseq commit fails without symlink privileges/developer mode.
- Preferred checkpoint for unsupervised representation geometry: `animal2vec_large_pretrained_MeerKAT_240507.pt`, not the finetuned checkpoint, because the finetuned version was trained with MeerKAT labels.
