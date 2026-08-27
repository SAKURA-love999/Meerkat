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
