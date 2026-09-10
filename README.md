# Repertoire Geometry and Calibrated Proxy Validity

This project studies animal vocal repertoires without assuming that manual call
labels are automatically equivalent to natural acoustic categories or verified
behavioral responses.

The current direction has two connected parts:

1. **Vocal repertoire geometry:** test whether manual call labels correspond to
   decodable, clustered, overlapping, coarse, graded, nuisance-confounded, or
   unresolved acoustic structure.
2. **Calibrated proxy validity:** quantify what temporal adjacency loses under
   degraded observability, then test whether manual call categories exhaust
   local sequence-predictive information under conservative, clip-local proxies.

## Core Principle

The project should not directly claim that a later non-focal MeerKAT call is a
response.

Instead:

> Temporal adjacency is treated as a proxy whose validity must be calibrated,
> stress-tested, and sometimes rejected as unresolved.

This framing is important because public bioacoustic datasets often lack the
spatial, identity, addressee, or behavioral ground truth needed to verify
directed vocal responses.

## Current MeerKAT Status

Completed analyses include:

- call manifest construction
- balanced and capped-natural pilot datasets
- librosa, log-mel, eGeMAPS, and animal2vec representations
- PCA / UMAP visualization
- HDBSCAN clustering
- classifier probes
- GMM coarse grouping
- fuzzy c-means degeneration diagnostics
- HDBSCAN parameter and bootstrap stability
- GMM-vs-HDBSCAN coarse-group comparison
- spectrogram-to-UMAP-to-HDBSCAN sanity-check replication

Main current conclusion:

> MeerKAT manual labels are acoustically decodable, but the tested
> representations do not recover stable natural clusters that cleanly align with
> the eight public manual call labels.

This is not a claim that the repertoire is proven continuous. It is a
conservative claim about what the tested protocols did and did not recover.

## Revised Research Direction

The next stage pivots from a MeerKAT-only clustering study to:

> A calibrated proxy-validity and behavioral-sufficiency framework for animal
> vocal sequence analysis.

The revised question is:

> Given imperfect public annotations, when can temporal adjacency be used as a
> conservative proxy for local interaction, and do manual call categories exhaust
> the information needed to predict that local interaction structure?

## Planned Experiments

### Experiment 1: Observability-Degradation Audit on High-Observability Data

Use Egyptian fruit bat data with richer annotations such as emitter, addressee,
and behavioral context.

Goal:

> Test whether voiced segments can be uniquely mapped to annotated interaction
> intervals, then estimate by behavioral context how often temporal adjacency
> links remain within the same annotated interval.

The bat data do not provide a MeerKAT response window. They provide a way to
measure how temporal-adjacency reliability changes with lag and context under
degraded observability.

### Experiment 2: Conservative MeerKAT Clip-Local Proxy

Apply MeerKAT-specific windows from meerkat literature and sensitivity analysis,
not from bat. The current primary design is:

- 400 ms primary
- 800 ms sensitivity
- 1500 ms sensitivity

Goal:

> Test local sequence-predictive information under a conservative adjacency
> proxy, not verified response behavior.

The public MeerKAT data should be described as:

> 184 h ground-truth-complete subset with clip-local temporal completeness.

Do not describe it as a temporally continuous 184 h sequence. Cross-clip order
and individual identity cannot be assumed recoverable from randomized clip
filenames.

### Experiment 3: Internal MeerKAT Proxy Checks

If the relevant metadata exists, check whether adjacent calls are supported by:

- individual identity
- spatial or group context
- co-occurring behavioral annotations
- call-rate and recording controls

If the metadata is absent, the analysis must explicitly report:

> MeerKAT cannot internally validate the temporal-adjacency proxy, so conclusions
> are conditional on the calibrated prior.

### Experiment 4: Behavioral Sufficiency

Use nested incremental tests as the main analysis:

```text
M0: R_post ~ Y + C_pre + Q
M1: R_post ~ Y + C_pre + Q + A_focal
```

where `Y` is manual category, `C_pre` is previous local vocal activity, `Q` is
nuisance structure, and `A_focal` is strictly call-only acoustics.

The core estimand is:

```text
Delta = L_test(M0) - L_test(M1)
```

reported overall and per manual category.

## Certificate Outputs

The framework should return a set-valued, calibrated conclusion, such as:

- `residual-acoustic-information detected`
- `no residual evidence detected`
- `proxy-validity unresolved`
- `sensitive to measured nuisance controls`

`unresolved` is an intended output, not a failure. It prevents over-interpreting
weak clustering or temporal-adjacency evidence.

## Key Reports

- `reports/replication_sanity_check_summary.md`
- `reports/animal2vec_diagnostic_runbook.md`
- `reports/hdbscan_stability_protocol.md`
- `reports/calibrated_proxy_validity_research_plan_v2.md`
- `reports/bat_proxy_audit_summary.md`

## First Audit Scripts

Bat observability-degradation audit:

```powershell
python scripts\audit_bat_interaction_mapping.py `
  --annotations "PATH\TO\Annotations.csv" `
  --fileinfo "PATH\TO\FileInfo.csv" `
  --skip-full-mapping-csv
```

The script outputs a segment-to-interaction mapping audit, next-segment edges,
and context-stratified temporal-adjacency reliability curves. The raw bat
metadata are not stored in this repository; use
`scripts\download_bat_metadata.py` to retrieve the public metadata locally.

MeerKAT clip-local adjacency audit:

```powershell
python scripts\audit_meerkat_clip_local_adjacency.py `
  --manifest data_manifest\call_manifest.csv
```
