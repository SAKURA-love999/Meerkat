# Paper-Style Spectrogram Baseline: balanced_600

## Why This Baseline Was Added

The first spectrogram baseline used variable-duration call segments and resized every spectrogram to 64 time bins. That made all calls comparable as vectors, but it also changed the temporal scale of each call.

This second baseline follows a more paper-like design:

1. Use the original call boundaries.
2. Compute log-mel spectrograms with approximately 30 ms frames and 3.75 ms hops.
3. Use 40 mel bins over 0-4 kHz.
4. Convert power to dB.
5. Z-transform each spectrogram independently.
6. Preserve the original number of time frames.
7. Right-pad each spectrogram to a fixed 500 ms grid.
8. Flatten the padded spectrogram into one vector.
9. Use the same PCA/UMAP/HDBSCAN/neighbour/classifier analysis.

## Difference From Resize-to-64

`logmel_call_resize` and `logmel_paper_pad500` answer slightly different representation questions.

`logmel_call_resize`:

- Forces every call into 64 time bins.
- Short calls are stretched and long calls are compressed.
- Duration information is weakened.
- Shape and spectral pattern are emphasized.
- Temporal rates can become distorted because a 40 ms call and a 250 ms call are both represented with 64 frames.

`logmel_paper_pad500`:

- Keeps the original frame count.
- Pads unused right-side frames with zero after per-spectrogram z-scoring.
- Duration and temporal extent remain encoded.
- Short calls retain their shortness.
- The vector includes both acoustic content and the amount/location of padding.

Because each spectrogram is z-scored independently, raw-amplitude and RMS-normalized versions become nearly identical. This baseline is therefore mainly about spectro-temporal shape plus duration, not loudness.

## Input

- Manifest: `data_manifest/pilot_manifest_balanced_600.csv`
- Calls: 4,800 focal calls
- Labels: `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, `oth`
- Variants: `raw`, `rms_normalized`

## Representation Details

At the public MeerKAT sample rate of 8 kHz:

- 30 ms frame = 240 samples
- 3.75 ms hop = 30 samples
- 500 ms maximum duration grid = 135 frames
- vector dimension = 40 mel bins x 135 frames = 5,400

Some very short calls are shorter than a 30 ms frame. Librosa therefore reports expected `n_fft=240 is too large for input signal` warnings for those calls.

## Duration/Padding QC

Mean original spectrogram frames by label:

| label | mean frames |
|---|---:|
| sn | 12.05 |
| cc | 33.88 |
| al | 39.83 |
| oth | 40.35 |
| mo | 56.79 |
| ld | 58.03 |
| agg | 58.65 |
| soc | 66.90 |

Only 256 of 9,600 representation rows had frames truncated beyond the 500 ms grid. This means 500 ms is a reasonable first cap for the balanced pilot, while still making truncation auditable.

## Results

| representation | amplitude | HDBSCAN clusters | noise rate | AMI | ARI | label silhouette | kNN15 purity | linear Macro-F1 | balanced accuracy | GMM best k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa summary | raw | 2 | 0.874 | 0.072 | 0.008 | -0.022 | 0.440 | 0.591 | 0.602 | 10 |
| log-mel resize64 | raw | 2 | 0.889 | 0.065 | 0.005 | -0.041 | 0.425 | 0.533 | 0.545 | 4 |
| log-mel paper pad500 | raw | 0 | 1.000 | 0.000 | 0.000 | -0.127 | 0.462 | 0.530 | 0.548 | 12 |
| log-mel paper pad500 | rms normalized | 0 | 1.000 | 0.000 | 0.000 | -0.127 | 0.462 | 0.529 | 0.547 | 12 |

## Interpretation

The paper-style representation does preserve a clearer temporal-duration structure. In UMAP, the calls form a broad continuous trajectory with visible local label enrichment: `sn` is strongly localized, while labels such as `cc`, `al`, `mo`, `ld`, `soc`, and `agg` occupy different but overlapping regions.

However, this did not produce label-aligned natural clusters under the default HDBSCAN setting. HDBSCAN assigned all points to noise, so AMI and ARI are zero for this particular clustering run.

The linear classifier probe stayed around Macro-F1 0.53, similar to resize64 and weaker than hand-crafted acoustic summary features. The one strong exception is `sn`, whose diagonal confusion-matrix value reached 0.94. That likely reflects its very short duration and compact temporal footprint.

The main conclusion is:

> A paper-style duration-preserving spectrogram changes the geometry from a mixed cloud into a more continuous trajectory, but it still does not support eight discrete, label-aligned natural clusters in the balanced MeerKAT pilot.

## What This Supports

- The weak cluster result is not only caused by using coarse librosa summary features.
- Duration-preserving spectrograms reveal more continuous geometric organization than resize-to-64 spectrograms.
- Public call labels are partly decodable, especially `sn`, but categorical separability remains different from natural cluster structure.

## What This Does Not Support

- It does not prove that the vocal system is biologically continuous.
- It does not prove that all public labels are invalid.
- It does not show that HDBSCAN can never find clusters under other parameters.
- It does not replace context-aware embeddings or sequence modelling.

## Next Step

The next diagnostic should use the same paper-style representation but run HDBSCAN parameter sensitivity, because default HDBSCAN was too strict and classified everything as noise. After that, the stronger representation step is context-window spectrogram or animal2vec embedding with focal-frame pooling.
