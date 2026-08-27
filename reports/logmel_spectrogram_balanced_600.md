# Log-Mel Spectrogram Baseline: balanced_600

## Question

This diagnostic asks whether the weak cluster structure observed with hand-crafted acoustic summary features is caused by an overly coarse representation.

The test uses the same `balanced_600` focal-call pilot and the same downstream geometry analysis, but replaces the 80-dimensional librosa summary features with a fixed-size log-mel spectrogram vector.

## Input

- Manifest: `data_manifest/pilot_manifest_balanced_600.csv`
- Labels: `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, `oth`
- Calls: 4,800 focal calls, 600 per label
- Amplitude variants:
  - `raw`
  - `rms_normalized`

## Representation

Each call-level segment is converted into:

- 64 mel bands
- 64 time frames
- log-mel power scale
- flattened 4096-dimensional vector

The analysis script then standardizes vectors and uses PCA-50 as the analysis space for HDBSCAN, silhouette, kNN label purity, the linear classifier probe, and GMM+BIC. PCA and UMAP figures are generated for visualization.

Very short calls produced expected `n_fft=256 is too large for input signal` warnings during spectrogram extraction. This mostly reflects the short duration of some calls, especially `sn`, and is a limitation of call-boundary-only spectrogram representations.

## Results

| representation | amplitude | HDBSCAN clusters | noise rate | AMI | ARI | label silhouette | kNN15 purity | linear Macro-F1 | balanced accuracy | GMM best k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| log-mel call resize | raw | 2 | 0.889 | 0.065 | 0.005 | -0.041 | 0.425 | 0.533 | 0.545 | 4 |
| log-mel call resize | rms normalized | 2 | 0.949 | 0.035 | 0.002 | -0.039 | 0.432 | 0.529 | 0.542 | 5 |

For comparison, the hand-crafted acoustic summary baseline was:

| representation | amplitude | HDBSCAN clusters | noise rate | AMI | ARI | label silhouette | kNN15 purity | linear Macro-F1 | balanced accuracy | GMM best k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa summary | raw | 2 | 0.874 | 0.072 | 0.008 | -0.022 | 0.440 | 0.591 | 0.602 | 10 |
| librosa summary | rms normalized | 2 | 0.903 | 0.060 | 0.005 | -0.027 | 0.434 | 0.581 | 0.593 | 12 |

## Interpretation

The spectrogram baseline does not reveal stronger label-aligned natural clusters than the hand-crafted acoustic summary baseline. HDBSCAN still finds only two clusters under the default setting, most points are treated as noise, and AMI/ARI remain weak.

The linear probe remains above chance, so public labels are not acoustically arbitrary. However, the classifier result is weaker than the hand-crafted baseline, suggesting that this simple call-resized log-mel vector does not provide a better categorical representation.

The UMAP plots show local enrichment for some labels, but not eight isolated islands. This supports the current working interpretation:

> Public MeerKAT call labels are partially predictable from acoustic structure, but they do not currently appear as clean, label-aligned natural clusters in either hand-crafted summary features or this simple spectrogram representation.

## What This Cannot Conclude

This result does not prove that MeerKAT calls are continuous in a biological sense. It only weakens the explanation that the previous weak clustering was caused by using overly coarse summary features.

It also does not rule out stronger structure in:

- eGeMAPS
- animal2vec or other self-supervised audio embeddings
- context-window representations
- spectrogram models that preserve temporal context rather than resizing call boundaries
- per-label multimodality analysis
- sequence-aware representations

## Next Diagnostic

The next representation-level diagnostic should avoid forcing very short calls into isolated call-boundary spectrograms. A good next step is:

1. Use a fixed real-context window around each focal call, such as 500 ms.
2. Build either log-mel context representations or animal2vec embeddings from that window.
3. Pool or mask only the focal-call frames for the final call embedding.
4. Run 250/500/1000 ms context sensitivity analysis.

