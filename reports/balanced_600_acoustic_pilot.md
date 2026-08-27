# Balanced 600 Acoustic Pilot

Date: 2026-08-27

## Scope

This is the first pilot run for public MeerKAT vocal repertoire geometry.

Input:

- `data_manifest/pilot_manifest_balanced_600.csv`
- 8 focal labels: `agg`, `al`, `cc`, `ld`, `mo`, `oth`, `sn`, `soc`
- 600 calls per label, 4,800 calls total

Representation:

- librosa-style hand-crafted acoustic features
- two amplitude variants: raw amplitude and RMS-normalized amplitude

This pilot does not yet include eGeMAPS or animal2vec.

## QC Result

All 4,800 calls produced valid feature rows.

| label | n valid | mean duration (s) | median duration (s) | mean RMS dBFS | mean SNR proxy dB |
|---|---:|---:|---:|---:|---:|
| agg | 600 | 0.218 | 0.126 | -25.10 | 5.48 |
| al | 600 | 0.147 | 0.123 | -30.56 | 8.96 |
| cc | 600 | 0.125 | 0.121 | -26.86 | 7.17 |
| ld | 600 | 0.216 | 0.205 | -24.76 | 7.86 |
| mo | 600 | 0.211 | 0.208 | -28.39 | 7.09 |
| oth | 600 | 0.149 | 0.143 | -26.78 | 6.99 |
| sn | 600 | 0.043 | 0.041 | -27.18 | 4.27 |
| soc | 600 | 0.249 | 0.212 | -26.88 | 6.04 |

Interpretation:

- `sn` is much shorter than the other labels, so duration remains a major nuisance variable.
- `al` has the highest SNR proxy in this sample, even though its mean RMS is not the highest.
- No NaN/inf failures occurred in this feature run.

## Geometry Metrics

| amplitude variant | HDBSCAN clusters | HDBSCAN noise rate | AMI | ARI | label silhouette | kNN15 purity | linear Macro-F1 | linear balanced accuracy | GMM best BIC k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 2 | 0.874 | 0.072 | 0.008 | -0.022 | 0.440 | 0.591 | 0.602 | 10 |
| RMS-normalized | 2 | 0.903 | 0.060 | 0.005 | -0.027 | 0.434 | 0.581 | 0.593 | 12 |

## Preliminary Interpretation

This pilot supports separating label-cluster alignment from categorical separability.

1. Label-cluster alignment is weak in this feature space.
   HDBSCAN finds only two stable clusters and marks most points as noise. AMI/ARI between HDBSCAN clusters and manual labels are low.

2. Categorical separability is moderate, not absent.
   A linear probe reaches about 0.59-0.60 balanced accuracy across eight balanced labels. This means manual labels are partially decodable from acoustic features.

3. The manual labels do not behave like eight clean natural clusters under this first representation.
   Label silhouette is slightly negative and kNN label purity is only about 0.44, far above chance but far from a clean cluster structure.

4. `oth` behaves like the most mixed category.
   In the confusion matrix, `oth` is spread across multiple predicted labels rather than forming a strong diagonal category.

5. Raw and RMS-normalized results are similar.
   This suggests the first-pass geometry is not purely an amplitude artifact, though energy/loudness still needs to be treated as a nuisance factor.

## What This Does Not Prove

- It does not prove that meerkat calls are continuous.
- It does not prove that meerkats perceive calls as non-categorical.
- It does not infer urgency, emotion, predator context, individual identity, or sequence structure.
- It does not yet establish cross-representation robustness.

## Next Steps

1. Add a duration/energy-controlled feature set.
2. Run the same analysis on the capped 3000 pilot.
3. Add eGeMAPS as a second representation.
4. Add animal2vec with real-context windows for short calls.
5. Run HDBSCAN sensitivity over `min_cluster_size` and `min_samples`.
6. Add fuzzy C-means for soft membership and hybrid-call candidates.
