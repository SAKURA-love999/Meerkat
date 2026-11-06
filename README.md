# MeerKAT Acoustic Gain Attribution

## Current Research Question

After manual call category and measured temporal history are known,where does
an apparent acoustic prediction increment come from,and what interpretation is
permitted? We study held-out prediction of nearby non-focal activity in anonymous
10-second clips,not a verified receiver response or continuous-time intensity.

The recent primary protocol fixes18345 calls/9756 clips,400ms complete PRE/POST
observation and five saved clip folds. It separates sample availability,received
amplitude/overlap controls,baseline expression,regularization,PCA and explicit
extraction-configuration inputs. Full and valid-only evaluations are not
different training populations unless a named experiment explicitly refits.

## Status And Bounded Findings (2026-11-06)

- Selection/random label-N controls,amplitude/overlap controls and OOF diagnostics
  are complete. Feature validity changes sample support,especially short sn.
- Known-truth synthetic controls,real linear/RMS-spline/HGB comparisons,nested C
  and separate PCA screens are complete. No general calibrated test is claimed.
- Overall controlled POST increments have not established stable positive gains.
  An exploratory al increment is baseline-expression-sensitive;no-information,
  manual-category sufficiency and causal conclusions are not established.
- The n_fft/hop_length/n_mels columns reconstruct from existing boundary inputs.
  Configuration-only inputs do not reproduce the linear al gain. The M78 reverse
  ablation completes the input-block comparison without re-extracting features.
- Independent dataset/caller/session validation and verified-response/causal
  experiments have not been completed. No pristine same-release confirmation
  set is claimed. Research novelty remains unresolved.

Audited aggregate tables and minimal native-run provenance are in
[`reports/research_snapshot_20261106`](reports/research_snapshot_20261106/).
Read its README before comparing protocols. This is an exploratory research
snapshot,not a submission-ready paper or a clean-machine replication package.

## Main Code

|Purpose|Scripts|
|---|---|
|Frozen cohort and controls|run_shared_cohort_selection_control.py,extend_shared_cohort_controls.py,run_control_block_attribution.py|
|Known-truth controls|calibration_dgp.py,calibration_dgp_asymmetric.py,run_baseline_capacity.py,run_baseline_capacity_b2.py|
|Real model capacity|run_baseline_capacity_b3.py,baseline_capacity_models.py|
|Training-only regularization/PCA|run_nested_logistic_regularization.py,run_pca_sensitivity.py|
|Configuration source/forward chain|audit_acoustic_config_fields.py,run_config_source_attribution.py|
|Reverse M78 comparison|run_reverse_config_ablation.py,audit_reverse_config_ablation.py|
|Public aggregate export|export_public_research_snapshot.py|

M0 contains category,measured side-appropriate history,duration/boundaries and
the named controls. Mconfig adds three configuration fields;M78 adds the other78;
M1 adds all81. Positive Delta is natural-logloss reduction. POST-minus-PRE is a
direction diagnostic,not a causal certificate. The predictive interaction is an
order-dependence contrast of fitted risks,not necessarily biological synergy.

## Environment And Tests

The current shallow-model environment uses Python3.9.13. Observed direct package
versions are pinned in requirements-controls-py39.txt;this is not a full lockfile.
The optional animal2vec stack has separate requirements and compatibility limits.

```powershell
python -m pip install -r requirements-controls-py39.txt
Set-Location scripts
python -m unittest test_reverse_config_ablation test_config_source_attribution test_baseline_capacity_b3 -v
```

Tests use temporary mock classifiers and small preprocessing fixtures,not the
scientific dataset. They do not validate a biological conclusion.

Python drivers expose explicit path arguments through --help. Existing PowerShell
wrappers use the author's local project/private workspace and external native
receipt helper;they are not portable launchers for a fresh clone. Raw data,
feature matrices,saved fold/source packets and row-level OOF are intentionally
not in Git. Reconstructing a run elsewhere requires these inputs and a fresh
local protocol;the full original absolute-path contracts are not published.
Never alter a completed frozen run's code/inputs or reuse its output directory
to silently introduce new parameters. Byte hashes may differ after Git line-ending
normalization;the public manifest distinguishes source and export hashes.

## Evidence Boundaries And Next Validation

Independent-data feasibility screening can start now. Before scoring a genuinely
untouched dataset,freeze the model/comparisons,primary endpoint,important-effect
margin and precision target. Confirm recording/caller/date provenance and raw
audio availability;another release of the same recordings is not independence.
Richer identity/spatial/behavioral records can validate a receiver-linked proxy,
but are not by themselves causal proof. Controlled playback/manipulation is a
separate collaboration/ethics-gated study,not an inference from this repository.

Private paper drafts,research-refinement reviews and run logs are excluded.
No new license or redistribution permission for the original datasets is granted.
Older geometry and bat audit reports remain historical exploratory material;
they are not automatically re-audited or promoted to current claims by this commit.

## Earlier Project Framing (Historical)

The following records the earlier geometry/proxy-calibration direction. In
particular,a calibrated certificate,behavioral sufficiency and stable natural
clusters are not established outcomes of the current prediction audit.

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
- `reports/bat_temporal_neighbor_categories_summary.md`
- `reports/meerkat_clip_local_proxy_audit_summary.md`
- `reports/m1_capped_history_runbook.md`
- `reports/m1_results_and_marked_process_framing.md`

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
