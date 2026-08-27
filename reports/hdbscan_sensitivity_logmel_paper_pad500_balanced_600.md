# HDBSCAN Sensitivity: logmel_paper_pad500 balanced_600

## Question

The default HDBSCAN setting assigned all `logmel_paper_pad500` calls to noise. This diagnostic tests whether that result was caused by one overly strict parameter choice.

The sweep asks:

> Under a range of HDBSCAN parameters and PCA analysis spaces, do paper-style spectrogram representations reveal stable natural clusters aligned with public MeerKAT call labels?

## Input

- Representation: `logmel_paper_pad500`
- Pilot: `balanced_600`
- Calls per amplitude variant: 4,800
- Labels: `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, `oth`
- Variants:
  - `raw`
  - `rms_normalized`

Because every spectrogram is independently z-transformed, raw and RMS-normalized variants are effectively identical for this representation.

## Parameter Grid

PCA analysis spaces:

- PCA-2
- PCA-5
- PCA-10
- PCA-20
- PCA-50

HDBSCAN grid:

- `min_cluster_size`: 10, 25, 50, 100, 200
- `min_samples`: 1, 5, 10, 25, 50

This gives 250 HDBSCAN fits across the two amplitude variants.

## Best Results

Top settings by AMI:

| amplitude | space | min_cluster_size | min_samples | clusters | noise | AMI | ARI | majority purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | PCA-5 | 10 | 1 | 53 | 0.529 | 0.184 | 0.088 | 0.466 |
| rms normalized | PCA-5 | 10 | 1 | 53 | 0.529 | 0.184 | 0.088 | 0.466 |
| raw | PCA-10 | 50 | 1 | 2 | 0.681 | 0.171 | 0.078 | 0.409 |
| rms normalized | PCA-10 | 50 | 1 | 2 | 0.681 | 0.171 | 0.078 | 0.409 |
| raw | PCA-2 | 100 | 1 | 2 | 0.731 | 0.154 | 0.066 | 0.474 |

The strongest AMI appears only under permissive settings, especially low `min_samples`. These settings either produce many small fragments or high noise.

## Best Setting Cluster Composition

The best AMI setting was:

- space: PCA-5
- `min_cluster_size`: 10
- `min_samples`: 1
- clusters: 53
- noise rate: 0.529

Cluster size summary:

| statistic | cluster size |
|---|---:|
| mean | 42.64 |
| median | 14 |
| 75th percentile | 20 |
| 90th percentile | 26.4 |
| max | 1379 |

The largest cluster contained 1,379 points, but its majority label was only 41.7% `sn`:

```text
sn:575; agg:243; al:194; cc:157; oth:141; soc:48; mo:15; ld:6
```

This is not a clean label-aligned cluster. It is a large mixed region with `sn` enrichment.

## Interpretation

HDBSCAN parameter sensitivity changes the output, but it does not rescue a strong discrete-cluster interpretation.

There are three regimes:

- Permissive settings find many clusters, but the solution is fragmented and noisy.
- Intermediate settings sometimes find 2 clusters with AMI around 0.15-0.17, but noise remains high.
- Stricter settings often return all noise.

The best AMI is higher than the default setting, but it remains low in absolute terms. ARI is also low, which means point-level cluster assignments do not correspond well to the eight public labels.

## What This Supports

- Default all-noise HDBSCAN was partly a parameter effect.
- The paper-style spectrogram does contain local density structure.
- That local density structure is not stable evidence for eight public label-aligned natural clusters.
- The UMAP trajectory and kNN purity are more consistent with local label enrichment along a continuous geometry than with clean discrete categories.

## What This Cannot Support

- It cannot prove biological continuity by itself.
- It cannot prove the public labels are wrong.
- It cannot claim HDBSCAN found the true call repertoire.
- It cannot use GMM/BIC as direct evidence for discrete vs continuous structure.

## Current Working Conclusion

For the first case study question:

> Public MeerKAT call labels are partly recoverable from acoustic representations, but neither hand-crafted acoustic features nor duration-preserving paper-style spectrograms currently show robust, stable, label-aligned natural clusters.

The stronger next step is to quantify continuity directly, rather than only trying more cluster parameters. Candidate diagnostics:

- neighbour label mixing as a function of distance
- between-label trajectory / centroid ordering
- fuzzy membership entropy
- within-label multimodality
- context-window spectrogram or animal2vec embeddings
