# Calibrated Proxy Validity + Behavioral Sufficiency Plan, v2

## Core Revision

Bat does not provide a response window for MeerKAT.

Instead:

> Bat quantifies what temporal adjacency loses under degraded observability.

The MeerKAT analysis should not claim that a later non-focal call is a response.
It should treat clip-local temporal adjacency as a conservative proxy for local
sequential activity, and it should explicitly report when proxy validity is
unresolved.

## Corrected Contribution

The central contribution is:

> A calibrated proxy-validity framework for animal vocal sequence analysis that
> separates local adjacency, annotated interaction membership, nuisance
> sensitivity, and residual acoustic information beyond manual call categories.

The paper should not be framed as:

> We prove that MeerKAT non-focal calls are responses.

It should be framed as:

> We test when degraded temporal observations can support conservative local
> sequence proxies, and whether manual categories exhaust the predictive
> information available under those proxies.

## Experiment 1: Bat Observability-Degradation Feasibility Audit

Before fitting any calibration model, audit whether Prat-style annotations allow
voiced segments to be mapped to annotated interaction intervals.

Ground truth:

```text
same annotated interaction interval
```

Not ground truth:

```text
second vocalization is a response to the first
```

Required mapping:

```text
voiced segment_i -> annotation / interaction interval_j
```

Audit outputs:

- number of voiced segments
- number of annotation intervals
- zero-match segment rate
- unique-match segment rate
- multi-match segment rate
- files with multiple interactions
- files where next voiced segment can be defined

If mapping quality is poor, proxy calibration must be redesigned before any
MeerKAT transfer.

Implemented script:

```text
scripts/audit_bat_interaction_mapping.py
```

Example command:

```powershell
Set-Location "C:\学校\MeerKAT\repertoire_geometry"

& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\audit_bat_interaction_mapping.py `
  --annotations "PATH\TO\Annotations.csv" `
  --fileinfo "PATH\TO\FileInfo.csv"
```

If column inference fails, pass explicit columns:

```powershell
  --annotation-file-col "File ID" `
  --segment-file-col "File ID" `
  --annotation-start-col "Start sample" `
  --annotation-end-col "End sample" `
  --segment-start-col "Voiced segment start sample" `
  --segment-end-col "Voiced segment end sample"
```

## Experiment 2: Bat Reliability Curves, Not a Magic Window

For every voiced segment `i`, find the next voiced segment `j` within the same
file and compute:

```text
delta_t = start_j - end_i
```

Then define:

```text
S_ij = 1
```

only when both segments uniquely map to the same annotated interaction interval.

Report:

```text
P(S = 1 | delta_t <= tau)
```

for:

```text
50, 100, 250, 500, 1000, 2000 ms
```

stratified by context.

Metrics:

- precision: probability that a temporal link is same-interaction
- false-link rate: 1 - precision
- coverage: fraction of consecutive edges captured by the window
- recall: fraction of same-interaction consecutive edges captured

The result should look like:

> Temporal adjacency reliability depends on lag and context.

Not:

> The best bat response window is 500 ms.

## Experiment 3: MeerKAT Clip-Local Adjacency Audit

The public MeerKAT data should be described as:

> 184 h ground-truth-complete subset with clip-local temporal completeness.

Do not call it:

> 184 h temporally continuous subset.

The public data are 10-second clips. Cross-clip order and individual identity
cannot be assumed recoverable from randomized file names. Therefore every
sequence endpoint must be computed within a single 10-second clip.

Primary MeerKAT windows should come from MeerKAT literature and sensitivity
analysis, not from bat:

```text
400 ms primary
800 ms sensitivity
1500 ms sensitivity
```

Clip-boundary censoring:

- for a 400 ms post window, events ending after 9.6 s are right-censored
- for an 800 ms post window, events ending after 9.2 s are right-censored
- for a 1500 ms post window, events ending after 8.5 s are right-censored
- analogous left-censoring applies to PRE windows

Implemented script:

```text
scripts/audit_meerkat_clip_local_adjacency.py
```

Command:

```powershell
Set-Location "C:\学校\MeerKAT\repertoire_geometry"

& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\audit_meerkat_clip_local_adjacency.py `
  --manifest data_manifest\call_manifest.csv
```

Outputs:

```text
reports\meerkat_proxy_audit\meerkat_clip_local_prepost_nf_by_focal_event.csv
reports\meerkat_proxy_audit\meerkat_clip_local_prepost_nf_rates_by_label.csv
reports\meerkat_proxy_audit\meerkat_clip_local_adjacency_audit_summary.json
```

This directly reports:

```text
N_y
P(NF_post,400)
P(NF_post,800)
P(NF_post,1500)
P(NF_pre,400)
P(NF_pre,800)
P(NF_pre,1500)
clip-boundary censoring
positive / negative counts
```

## Experiment 4: Behavioral Sufficiency as Nested Incremental Test

The main experiment should not be a loose comparison of:

- manual label only
- acoustic only
- label + acoustic
- nuisance only

Those can remain descriptive baselines, but the core estimand should be nested:

```text
M0: R_post ~ Y + C_pre + Q
M1: R_post ~ Y + C_pre + Q + A_focal
```

where:

- `R_post`: post-window non-focal local activity
- `Y`: manual call category
- `C_pre`: pre-window vocal activity / local call state
- `Q`: nuisance controls such as duration, SNR proxy, overlap, clipping,
  recording, clip position
- `A_focal`: strictly call-only acoustic representation

Core estimand:

```text
Delta = L_test(M0) - L_test(M1)
```

The key analysis is class-conditional:

```text
Delta_cc, Delta_sn, Delta_soc, ...
```

because the biological question is:

> Once we know that a call is `cc`, does within-cc acoustic variation still
> predict subsequent local non-focal activity?

## PRE Symmetry Negative Control

Run the same nested test for PRE activity:

```text
Delta_post_y vs Delta_pre_y
```

Interpretation:

- if PRE and POST gains are similar, the model may be detecting ongoing local
  interaction state
- if POST is stronger than PRE, there is better evidence for temporal
  specificity

This still does not prove response behavior.

## Certificate Outputs

Do not output overly strong binary labels such as:

```text
manual-label-sufficient
```

because failure to detect residual information is not proof of sufficiency.

Use conservative evidence states:

- `residual-acoustic-information detected`
- `no residual evidence detected`
- `proxy-validity unresolved`
- `sensitive to measured nuisance controls`

If bat calibration shows high false-link risk under comparable observable
conditions, the corresponding MeerKAT result should be marked:

```text
proxy-validity unresolved
```

and should not receive a strong behavioral interpretation.

## Current Status

Already completed:

- MeerKAT call manifest and pilot construction
- acoustic representation analyses
- clustering/separability diagnostics
- animal2vec analyses
- fuzzy c-means degeneration diagnosis
- GMM/HDBSCAN coarse-group comparison
- spectrogram UMAP/HDBSCAN sanity-check replication

New first steps:

1. Run bat interaction-mapping audit.
2. Run MeerKAT clip-local adjacency audit.
3. Only if both are interpretable, implement nested behavioral sufficiency
   models.

## Go / No-Go Logic

Go:

- bat voiced segments mostly map uniquely to annotation intervals
- bat reliability curves show interpretable lag/context dependence
- MeerKAT clip-local censoring leaves enough eligible focal events per label
- PRE/POST local activity rates are estimable for 400/800/1500 ms

No-go or redesign:

- bat segment-to-interval mapping is ambiguous
- reliability curves are dominated by file artifacts
- MeerKAT clip-boundary censoring removes too many samples
- post-window positives are too rare for class-conditional modeling
- nuisance controls explain all apparent acoustic gains

