# eGeMAPS Baseline: balanced_600

## Question

This baseline adds eGeMAPSv02 Functionals as a third hand-crafted representation. It asks whether a standard openSMILE paralinguistic feature set reveals stronger label-aligned acoustic structure than the previous librosa summary and spectrogram baselines.

## Input

- Manifest: `data_manifest/pilot_manifest_balanced_600.csv`
- Calls: 4,800 focal calls, 600 per label
- Labels: `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, `oth`
- Feature set: openSMILE `eGeMAPSv02`
- Feature level: `Functionals`
- Feature dimension: 88
- Amplitude variants:
  - `raw`
  - `rms_normalized`

The extraction uses original call boundaries. It does not pad or loop short calls.

## Implementation Note

On this Windows setup, openSMILE could not initialize from a package path containing non-ASCII characters. The extraction script copies the openSMILE config directory into an ASCII cache directory and uses that as the config root. This only affects initialization; the project and outputs remain under `C:\学校\MeerKAT\repertoire_geometry`.

## Feature Validity QC

openSMILE reports very short segments as too short and fills their eGeMAPS values with NaN. This is especially severe for `sn`.

Counts below combine raw and RMS-normalized rows, so each label has 1,200 rows:

| label | total rows | valid rows | invalid rows | valid rate | mean NaNs | mean duration |
|---|---:|---:|---:|---:|---:|---:|
| agg | 1200 | 908 | 292 | 0.757 | 21.41 | 0.218 |
| al | 1200 | 1132 | 68 | 0.943 | 4.99 | 0.147 |
| cc | 1200 | 1158 | 42 | 0.965 | 3.08 | 0.125 |
| ld | 1200 | 1192 | 8 | 0.993 | 0.59 | 0.216 |
| mo | 1200 | 1188 | 12 | 0.990 | 0.88 | 0.211 |
| oth | 1200 | 1066 | 134 | 0.888 | 9.83 | 0.149 |
| sn | 1200 | 128 | 1072 | 0.107 | 78.61 | 0.043 |
| soc | 1200 | 1164 | 36 | 0.970 | 2.64 | 0.249 |

This means all-row eGeMAPS analyses must be interpreted with caution. Median imputation allows downstream analysis to run, but missingness is label-correlated and can distort geometry.

## Results: All Rows With Median Imputation

| amplitude | calls | HDBSCAN clusters | noise | AMI | ARI | label silhouette | kNN15 purity | Macro-F1 | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 4800 | 3 | 0.564 | 0.216 | 0.105 | -0.110 | 0.404 | 0.543 | 0.566 |
| rms normalized | 4800 | 2 | 0.104 | 0.168 | 0.084 | -0.102 | 0.398 | 0.535 | 0.560 |

The all-row eGeMAPS analysis gives the highest default-HDBSCAN AMI so far, especially in the raw-amplitude version. However, this is not sufficient evidence for clean natural clusters because the label silhouette remains negative, kNN purity is low, and feature validity is highly label-dependent.

## Results: Valid-Only Sensitivity Analysis

To check whether the all-row result is influenced by NaN/imputation, the same analysis was repeated using only rows with no NaN or Inf features.

| amplitude | calls | HDBSCAN clusters | noise | AMI | ARI | label silhouette | kNN15 purity | Macro-F1 | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 3968 | 2 | 0.175 | 0.158 | 0.097 | -0.071 | 0.403 | 0.519 | 0.564 |
| rms normalized | 3968 | 2 | 0.117 | 0.163 | 0.103 | -0.069 | 0.401 | 0.513 | 0.562 |

After removing invalid rows, AMI decreases and linear-probe Macro-F1 also decreases. The valid-only UMAP still shows broad mixed regions rather than eight label-specific islands.

## Comparison With Previous Representations

| representation | amplitude | HDBSCAN clusters | noise | AMI | ARI | label silhouette | kNN15 purity | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| librosa summary | raw | 2 | 0.874 | 0.072 | 0.008 | -0.022 | 0.440 | 0.591 |
| log-mel resize64 | raw | 2 | 0.889 | 0.065 | 0.005 | -0.041 | 0.425 | 0.533 |
| log-mel paper pad500 | raw | 0 | 1.000 | 0.000 | 0.000 | -0.127 | 0.462 | 0.530 |
| eGeMAPS all rows | raw | 3 | 0.564 | 0.216 | 0.105 | -0.110 | 0.404 | 0.543 |
| eGeMAPS valid only | raw | 2 | 0.175 | 0.158 | 0.097 | -0.071 | 0.403 | 0.519 |

## Interpretation

eGeMAPS does not overturn the current conclusion.

It provides somewhat higher HDBSCAN-label alignment than the first two baselines, but that improvement is fragile because short-call validity is highly label-dependent. `sn` is the clearest example: in the all-row imputed analysis, the linear probe classifies `sn` very strongly, but after valid-only filtering its diagonal value decreases substantially.

The current interpretation is:

> eGeMAPS captures some local acoustic structure, but under original call boundaries it is not a robust standalone representation for very short MeerKAT calls. It does not provide strong evidence that the eight public labels are clean natural acoustic clusters.

## What This Supports

- eGeMAPS is useful as a standardized baseline, but it is sensitive to very short focal calls.
- Missingness/feature validity must be treated as a first-class QC issue.
- Cross-representation robustness still points away from clean eight-way natural clustering.

## What This Cannot Support

- It cannot be used alone to claim that public labels are discrete natural categories.
- It cannot be used alone to reject the public labels.
- The all-row imputed result cannot be interpreted without the valid-only sensitivity check.

## Next Step

The next representation should handle short calls using real acoustic context rather than isolated call boundaries. A reasonable next step is:

1. Build a 500 ms real-context spectrogram around each focal call.
2. Extract either context-aware log-mel vectors or animal2vec embeddings.
3. Pool only focal-call frames.
4. Compare 250/500/1000 ms context windows.
