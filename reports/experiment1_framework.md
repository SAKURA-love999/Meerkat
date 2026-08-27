# Experiment 1: Vocal Repertoire Geometry in Public MeerKAT

Working formulation:

> Use public MeerKAT as the first case study to test whether a vocal repertoire geometry framework can separate categorical separability, multimodality, and continuous acoustic variation.

## Current Data Basis

- Source: public MeerKAT 10-second dataset.
- Audio: `wav/08000Hz/*.wav`.
- Labels: `lbl/08000Hz/*.h5`.
- HDF5 fields used: `start_time_lbl`, `end_time_lbl`, `start_frame_lbl`, `end_frame_lbl`, `lbl_cat`, `lbl`, `foc`.
- Current call-level manifest: `data_manifest/call_manifest.csv`.
- Total labelled call/event rows found: 254,069.
- Missing corresponding wav files: 0.
- Target focal labels for pilot: `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, `oth`.

## Module Map

| Module | Input | Question | Main metrics | Supports what conclusion | Cannot support what conclusion |
|---|---|---|---|---|---|
| 1A. Label-cluster alignment | Representation vectors plus human labels | Do human labels correspond to natural acoustic clusters? | AMI, ARI, cluster-based silhouette, label-based silhouette, HDBSCAN stability/noise rate | Whether unsupervised acoustic groupings align with manual labels | Whether labels are behaviorally meaningful; whether categories are cognitively perceived as discrete |
| 1B. Categorical separability | Representation vectors plus human labels | Can manual labels be decoded from the representation? | Macro-F1, balanced accuracy, confusion matrix, linear-probe accuracy | Whether label information is present and separable in the representation | Whether the labels are natural clusters; whether boundaries are biologically discrete |
| 1C. Within-label multimodality | Vectors from one label at a time | Does one manual label contain multiple acoustic modes/subtypes? | Per-label GMM BIC, within-label silhouette, dip test, cluster count stability | Whether a single call label may hide substructure | Whether modes form a continuous gradient; whether submodes have known behavioral meaning |
| 1D. Continuous acoustic geometry | High-dimensional vectors, local-neighborhood graph, optional low-dimensional embeddings | Are calls arranged along continuous acoustic axes or transition regions? | kNN label mixing, fuzzy membership entropy, principal/diffusion axis structure, centroid-distance pattern, ordinal-distance correlation when a prior order exists | Whether boundaries are graded, mixed, or geometrically ordered | A causal link to emotion, urgency, predator type, or individual identity without metadata |
| 1E. Cross-representation robustness | Results from librosa, eGeMAPS, spectrogram, and animal2vec separately | Are conclusions stable across representations and analysis choices? | Metric direction consistency, bootstrap confidence intervals, seed sensitivity, sampling sensitivity | Whether findings are robust to feature choices | That all representations encode the same biological information |

## Interpretation Rules

High classifier performance alone means labels are decodable, not that they are natural clusters.

High AMI/ARI between unsupervised clusters and labels supports label-cluster alignment, not necessarily continuous or discrete biological perception.

GMM+BIC estimates how many Gaussian components fit better. It is useful for multimodality, but cannot alone prove discreteness or continuity.

UMAP/PCA figures are diagnostic visualizations. The core evidence should come from high-dimensional metrics, cross-validation, bootstrap, and cross-representation stability.

`oth` is important because high fuzzy entropy, high kNN label mixing, or placement near multiple class boundaries can identify candidate intermediate/hybrid calls.

## Immediate Next Analysis

1. Use `pilot_manifest_balanced_600.csv` for first clean baseline.
2. Extract interpretable acoustic features with a simple package stack.
3. Run 1A/1B/1C/1D first on acoustic features.
4. Repeat with eGeMAPS and animal2vec separately.
5. Report a stability table across representations.
