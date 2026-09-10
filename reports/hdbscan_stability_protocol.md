# HDBSCAN Stability Protocol

This diagnostic addresses a stricter question than a single clustering run:

> Do the same acoustic structures reappear when HDBSCAN parameters or sampled
> calls are slightly changed?

The result should be described conservatively. A single run can only support:

> No clear label-aligned clusters were recovered under the tested setting.

To argue against stable clusters, we need evidence that cluster solutions are
not stable across parameter perturbations and sample perturbations.

## Inputs

- Feature matrix: one `.npz` file with an `X` array.
- Metadata: one `.csv` file with a `label` column.
- Labels are used only for post-hoc evaluation, not for fitting PCA or HDBSCAN.

## Preprocessing

1. Standardize features with `StandardScaler`.
2. Reduce to `--pca-dim`, default 50.
3. Run HDBSCAN over a parameter grid.

This keeps the protocol aligned with the previous PCA/UMAP/HDBSCAN/classifier
analyses.

## Parameter Stability

The grid varies:

- `min_cluster_size`
- `min_samples`
- `cluster_selection_epsilon`
- `cluster_selection_method`

For each setting, the script records:

- number of clusters
- noise rate
- largest cluster fraction
- observed cluster-size range
- HDBSCAN cluster persistence
- post-hoc AMI/ARI against manual labels

It also computes pairwise AMI/ARI between every pair of HDBSCAN assignments.
This is the direct cluster-stability test.

Important caveat:

- All-noise or single-cluster solutions can look artificially stable.
- Therefore the summary separately reports `nontrivial_*` stability, keeping
  only pairs where both solutions have at least two clusters and noise rate
  below 0.95.

## Bootstrap Stability

Bootstrap stability asks whether the same cluster structure returns after
resampling calls.

The script uses stratified subsampling by label only to keep the pilot balanced
during perturbation. Labels are not passed to HDBSCAN.

For each bootstrap:

- sample a fixed fraction of calls per label
- fit HDBSCAN with one chosen unsupervised parameter setting
- compare overlapping samples between bootstrap solutions using AMI/ARI

## What Supports No Stable Clusters

Evidence against stable clusters would look like:

- cluster counts vary strongly across nearby HDBSCAN settings
- noise rate varies strongly across settings
- pairwise solution AMI/ARI is low among nontrivial settings
- bootstrap solution AMI/ARI is low
- high observed label AMI occurs only in fragmented, low-persistence, high-noise,
  or tiny-cluster settings
- the same pattern appears across multiple representations

Then the correct claim is:

> Across tested representations, HDBSCAN did not recover stable natural
> clusters that consistently aligned with public MeerKAT call labels.

This is still not proof that the true repertoire has no clusters at all. It is
evidence that stable label-aligned clusters were not recovered by this protocol.

## What It Cannot Prove

This protocol cannot prove:

- the biological repertoire is continuous in an absolute sense
- no clustering exists under any future representation
- manual labels are wrong
- UMAP islands are meaningful without artifact checks

It can only compare stability under the tested representations, preprocessing,
sample design, and HDBSCAN parameter ranges.

