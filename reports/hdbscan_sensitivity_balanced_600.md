# HDBSCAN Sensitivity: Balanced 600 Acoustic Pilot

Date: 2026-08-27

## Goal

This diagnostic analysis tests whether the weak label-cluster alignment in the first pilot is caused by a single HDBSCAN parameter choice.

Question:

> Do public MeerKAT manual labels become clearly aligned with unsupervised acoustic clusters under different HDBSCAN settings?

## Input

- Feature file: `features/acoustic_features_balanced_600.csv`
- Pilot: `balanced_600`
- Labels: `agg`, `al`, `cc`, `ld`, `mo`, `oth`, `sn`, `soc`
- Amplitude variants: `raw`, `rms_normalized`
- Spaces:
  - full standardized acoustic feature space
  - PCA-10
  - PCA-20
  - PCA-50

## Parameter Grid

```text
min_cluster_size: 10, 25, 50, 100, 200
min_samples:      1, 5, 10, 25, 50
```

This gives 200 HDBSCAN fits:

```text
2 amplitude variants x 4 spaces x 25 parameter settings
```

## Metrics

- `n_clusters`: number of non-noise clusters found by HDBSCAN.
- `noise_rate`: fraction of points labelled as noise.
- `AMI`: adjusted mutual information between HDBSCAN clusters and manual labels.
- `ARI`: adjusted Rand index between HDBSCAN clusters and manual labels.
- `cluster_silhouette_no_noise`: cluster separation among non-noise points only.
- `cluster_majority_purity_no_noise`: for each cluster, take the majority manual label; report the weighted average majority fraction.

AMI/ARI are the main label-cluster alignment metrics. Cluster count alone is not enough: HDBSCAN can find eight clusters that do not correspond to the eight manual labels.

## Highest Observed AMI Results

| amplitude | space | min_cluster_size | min_samples | clusters | noise rate | AMI | ARI | majority purity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | PCA-10 | 50 | 1 | 7 | 0.636 | 0.140 | 0.037 | 0.386 |
| RMS-normalized | PCA-10 | 25 | 1 | 9 | 0.643 | 0.136 | 0.035 | 0.381 |
| RMS-normalized | PCA-20 | 25 | 1 | 7 | 0.756 | 0.129 | 0.018 | 0.433 |
| RMS-normalized | PCA-10 | 50 | 1 | 4 | 0.669 | 0.127 | 0.037 | 0.365 |
| RMS-normalized | PCA-20 | 50 | 1 | 4 | 0.777 | 0.123 | 0.018 | 0.421 |

## Interpretation

Changing HDBSCAN parameters improves label-cluster alignment slightly, especially in PCA-10 with `min_samples=1`, but it does not reveal a strong eight-label cluster structure.

Main observations:

1. The highest observed AMI is about 0.14, which is still weak.
2. The corresponding ARI is about 0.04, also weak.
3. Some settings find 7-9 clusters, but these clusters do not align cleanly with manual labels.
4. Noise rates remain high in the highest-AMI settings, around 0.64.
5. Full feature space performs worse than PCA-10/PCA-20, suggesting that high-dimensional noise affects density clustering.
6. Raw and RMS-normalized results are similar, so this diagnostic is not mainly driven by amplitude scale.

## Conclusion

The first weak HDBSCAN result is not just an artifact of the single original parameter setting (`min_cluster_size=50`, `min_samples=10`).

More precise statement:

> Under hand-crafted acoustic features, public MeerKAT manual labels are not strongly aligned with HDBSCAN density clusters across a broad parameter grid. Dimensionality reduction to PCA-10 improves alignment slightly, but the highest observed AMI settings remain weak and noisy.

## What This Does Not Prove

- It does not prove that the data have no cluster structure.
- It does not prove that meerkat calls are continuous.
- It does not rule out stronger clusters in eGeMAPS, spectrogram, or animal2vec representations.
- It does not rule out non-density-based cluster structure.

## Next Diagnostics

1. Run forced-cluster baselines: KMeans k=8 and GMM k=8.
2. Add Fuzzy C-Means to estimate soft membership and ambiguous calls.
3. Add duration/energy-controlled feature sets.
4. Repeat with eGeMAPS and animal2vec.
5. Run label-shuffle controls for kNN purity, silhouette, AMI/ARI, and linear probe.
