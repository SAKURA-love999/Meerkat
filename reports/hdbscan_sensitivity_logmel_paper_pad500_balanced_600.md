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

## Highest Observed AMI Settings

The following settings are not "best" unsupervised HDBSCAN settings. They are the parameter settings yielding the highest observed AMI after comparing HDBSCAN output to the public labels.

| amplitude | space | min_cluster_size | min_samples | clusters | noise | median cluster size | weighted persistence | AMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | PCA-5 | 10 | 1 | 53 | 0.529 | 14 | 0.032 | 0.184 | 0.088 |
| rms normalized | PCA-5 | 10 | 1 | 53 | 0.529 | 14 | 0.032 | 0.184 | 0.088 |
| raw | PCA-10 | 50 | 1 | 2 | 0.681 | 765 | 0.287 | 0.171 | 0.078 |
| rms normalized | PCA-10 | 50 | 1 | 2 | 0.681 | 765 | 0.287 | 0.171 | 0.078 |
| raw | PCA-2 | 100 | 1 | 2 | 0.731 | 645.5 | 0.246 | 0.154 | 0.066 |

The strongest AMI appears only under permissive settings, especially low `min_samples`. These settings either produce many small fragments or high noise.

## Unsupervised Screening

As a separate step, settings were screened using unsupervised cluster-quality criteria before looking at AMI/ARI:

- cluster count between 2 and 30
- noise rate between 0.05 and 0.75
- largest cluster less than or equal to 50% of all points
- cluster persistence and membership probability used descriptively

Only four settings passed this screen, and they were duplicated across raw and RMS-normalized variants:

| amplitude | space | min_cluster_size | min_samples | clusters | noise | weighted persistence | mean membership probability | AMI | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | PCA-10 | 50 | 1 | 2 | 0.681 | 0.287 | 0.597 | 0.171 | 0.078 |
| rms normalized | PCA-10 | 50 | 1 | 2 | 0.681 | 0.287 | 0.597 | 0.171 | 0.078 |
| raw | PCA-2 | 100 | 1 | 2 | 0.731 | 0.246 | 0.672 | 0.154 | 0.066 |
| rms normalized | PCA-2 | 100 | 1 | 2 | 0.731 | 0.246 | 0.672 | 0.154 | 0.066 |

Under this unsupervised screen, AMI ranged from 0.154 to 0.171 and ARI ranged from 0.066 to 0.078. These values are still weak, so the unsupervised-plausible settings do not show strong alignment with public labels.

## Highest-AMI Setting Cluster Composition

The parameter setting yielding the highest observed AMI was:

- space: PCA-5
- `min_cluster_size`: 10
- `min_samples`: 1
- clusters: 53
- noise rate: 0.529

This setting is useful as a supervised diagnostic, but it is not a strong unsupervised cluster solution. It has many small clusters and low weighted cluster persistence.

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

## Label Inclusion Rate

The cluster composition line above describes only the largest non-noise cluster, not all 4,800 samples.

For the highest-observed-AMI setting:

- total samples: 4,800
- non-noise samples: 2,260
- noise samples: 2,540
- total clusters: 53

Cluster inclusion rate means:

> For a given public label, what fraction of its 600 calls were assigned to any non-noise HDBSCAN cluster?

| label | total count | non-noise count | noise count | inclusion rate | clusters touched | largest cluster count within label |
|---|---:|---:|---:|---:|---:|---:|
| sn | 600 | 578 | 22 | 0.963 | 4 | 575 |
| agg | 600 | 383 | 217 | 0.638 | 26 | 243 |
| cc | 600 | 320 | 280 | 0.533 | 26 | 157 |
| oth | 600 | 278 | 322 | 0.463 | 44 | 141 |
| al | 600 | 270 | 330 | 0.450 | 31 | 194 |
| soc | 600 | 196 | 404 | 0.327 | 32 | 48 |
| ld | 600 | 127 | 473 | 0.212 | 30 | 19 |
| mo | 600 | 108 | 492 | 0.180 | 21 | 15 |

This means `sn` is much more likely than other labels to fall inside HDBSCAN's local high-density structure. `mo` and `ld` are mostly treated as noise under this setting. However, `sn` being included does not mean the largest cluster is a pure `sn` cluster, because that cluster also contains many `agg`, `al`, `cc`, and `oth` calls.

## Interpretation

HDBSCAN parameter sensitivity changes the output, but it does not rescue a strong discrete-cluster interpretation.

There are three regimes:

- Permissive settings find many clusters, but the solution is fragmented and noisy.
- Intermediate settings sometimes find 2 clusters with AMI around 0.15-0.17, but noise remains high.
- Stricter settings often return all noise.

The highest observed AMI is higher than the default setting, but it remains low in absolute terms. ARI is also low, which means point-level cluster assignments do not correspond well to the eight public labels.

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
