# Sanity-Check Replication Summary

## Protocol

Goal:

> Verify that the data, labels, spectrogram preprocessing, UMAP, and HDBSCAN
> pipeline do not show obvious errors.

This is not a parameter search and is not intended to tune results to match
previous papers.

Representation:

- raw focal call audio
- log-mel spectrogram
- sample rate: 8000 Hz
- frame: 30 ms
- hop: 3.75 ms
- mel bins: 40
- frequency range: 0-4000 Hz
- per-spectrogram z-score
- right padding/truncation to 500 ms
- flattened spectrogram vector

Dimensionality reduction and clustering:

- UMAP: 2D, `n_neighbors=30`, `min_dist=0.1`
- HDBSCAN: `min_cluster_size=30`, `min_samples=10`, `eom`

Datasets:

- original-distribution pilot: `data_manifest/pilot_manifest_capped_3000.csv`, 20,620 calls
- balanced pilot: `data_manifest/pilot_manifest_balanced_600.csv`, 4,800 calls

The full `call_manifest.csv` has 254,069 calls, so the capped original-distribution
pilot was used for this minimal sanity-check replication.

## Metrics

| Dataset | Evaluation | n calls | n clusters | noise rate | AMI | ARI | label silhouette, feature | label silhouette, UMAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original distribution | include noise | 20,620 | 5 | 0.0018 | 0.0627 | 0.0035 | -0.1460 | -0.0804 |
| original distribution | exclude noise | 20,582 | 5 | 0.0000 | 0.0626 | 0.0034 | -0.1455 | -0.0803 |
| balanced_600 | include noise | 4,800 | 3 | 0.0198 | 0.0429 | 0.0014 | -0.1578 | -0.0923 |
| balanced_600 | exclude noise | 4,705 | 3 | 0.0000 | 0.0407 | 0.0013 | -0.1559 | -0.0911 |

## Interpretation

The pipeline appears functional:

- spectrogram extraction completed for both pilots
- UMAP produced structured 2D manifolds rather than random scatter
- HDBSCAN found nontrivial density partitions
- HDBSCAN noise rate was low in this protocol

However, the recovered HDBSCAN clusters are weakly aligned with the manual labels:

- AMI is low in both original-distribution and balanced pilots
- ARI is close to zero
- label silhouette is negative in both feature space and UMAP space
- excluding HDBSCAN noise does not improve label correspondence

This supports the conservative sanity-check conclusion:

> The spectrogram -> UMAP -> HDBSCAN pipeline runs correctly and recovers
> density structure, but the recovered density structure is not strongly aligned
> with the eight public manual call labels.

## Sampling Effect

Original-distribution and balanced pilots both show weak label alignment.

The original-distribution pilot produced more HDBSCAN clusters:

- original distribution: 5 clusters
- balanced_600: 3 clusters

This suggests that density structure is affected by sampling distribution, as
expected for HDBSCAN. But the central result is consistent:

> Neither sampling scheme recovers clear label-aligned clusters under this
> replication protocol.

## Output Files

Metrics:

```text
reports/replication_spectrogram_umap_hdbscan_metrics.csv
```

Figures:

```text
figures/replication_original_distribution_umap_manual_label.png
figures/replication_original_distribution_umap_hdbscan_cluster.png
figures/replication_balanced_600_umap_manual_label.png
figures/replication_balanced_600_umap_hdbscan_cluster.png
```

Assignments:

```text
reports/replication_original_distribution_umap_hdbscan_assignments.csv
reports/replication_balanced_600_umap_hdbscan_assignments.csv
```

Composition tables:

```text
reports/replication_original_distribution_include_noise_composition_counts.csv
reports/replication_original_distribution_include_noise_composition_fraction.csv
reports/replication_original_distribution_exclude_noise_composition_counts.csv
reports/replication_original_distribution_exclude_noise_composition_fraction.csv
reports/replication_balanced_600_include_noise_composition_counts.csv
reports/replication_balanced_600_include_noise_composition_fraction.csv
reports/replication_balanced_600_exclude_noise_composition_counts.csv
reports/replication_balanced_600_exclude_noise_composition_fraction.csv
```

