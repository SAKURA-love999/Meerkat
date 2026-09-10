# Calibrated Proxy Validity + Behavioral Sufficiency Plan

## Revised Framing

The project should no longer claim that a later non-focal MeerKAT call is a
vocal response. That claim is too strong without emitter/addressee, spatial, or
behavioral ground truth.

The revised contribution is:

> A calibrated proxy-validity framework for asking when simple temporal
> adjacency can be used as a conservative proxy for local vocal interaction, and
> whether manual call categories exhaust the locally predictive information in
> animal vocal sequences.

The key shift is from:

> We infer responses in MeerKAT.

to:

> We calibrate when temporal adjacency is likely to be an informative proxy, and
> we explicitly mark cases where the proxy is unresolved or likely invalid.

## What Changed

### Previous Direction

The earlier plan risked treating temporal adjacency as if it directly measured
vocal response:

- focal call followed by another call
- next call interpreted as response
- manual category tested as predictor of this response

This is vulnerable because MeerKAT lacks the ground truth needed to distinguish
directed response, broadcast calling, shared environmental triggering, and
opportunistic temporal proximity.

### Revised Direction

The new design separates three questions:

1. **Proxy validity:** Under what observable conditions does temporal adjacency
   approximate membership in the same local interaction?
2. **Behavioral sufficiency:** Given a conservative adjacency proxy, do manual
   call categories exhaust the information needed to predict local sequential
   structure?
3. **Unresolved evidence:** When spatial, identity, or behavioral ground truth
   is missing, which conclusions must be withheld?

This turns the data limitation into a method contribution: the framework is
designed to avoid over-interpreting temporal adjacency.

## Core Claims and Evidence Boundaries

### Supported Claim Target

The strongest defensible claim is:

> A calibrated temporal-adjacency proxy can be used as a conservative tool for
> testing whether manual call categories are behaviorally sufficient for local
> sequence prediction.

### Claims to Avoid

Do not claim:

- non-focal MeerKAT calls are responses
- temporal adjacency proves directed communication
- MeerKAT sequence structure alone identifies addressee or intent
- failure of manual labels implies graded communication

### Allowed Conservative Claim

If calibration and nuisance checks are incomplete, write:

> In MeerKAT, temporal adjacency is treated as a proxy for local sequential
> context rather than as a verified vocal response. Results are interpreted as
> evidence about behavioral sufficiency under this proxy, not as direct evidence
> of response behavior.

## Revised Experimental Design

### Experiment 1: Calibrate Temporal Adjacency on High-Observability Data

Use Egyptian fruit bat data from Prat et al. because it contains richer
interaction labels such as emitter, addressee, and behavioral context.

Question:

> In different behavioral contexts, how often does the next call within a time
> window belong to the same annotated interaction event?

Important limitation:

This calibration should not be framed as a response-vs-broadcast classifier,
because the dataset does not provide a clean broadcast control condition.

Outputs:

- context-stratified adjacency-validity curves
- time window vs same-interaction probability
- context-specific uncertainty intervals
- failure cases where adjacency is a poor proxy

Decision:

- choose conservative reference windows
- do not treat them as universally validated rules

### Experiment 2: Transfer the Calibrated Window to MeerKAT Conservatively

Apply the calibrated windows to the MeerKAT subset that has:

- complete annotation
- temporal continuity
- known inclusion in the 184h continuous subset

Question:

> Under a conservative adjacency window, do manual call categories explain local
> sequence prediction, or do acoustic representations add predictive information?

Interpretation:

- The target is local predictive information, not verified vocal response.
- The window is a prior/reference benchmark, not a validated MeerKAT response
  rule.

### Experiment 3: Internal MeerKAT Proxy Check

If identity or group-level behavioral metadata is available:

- test whether temporally adjacent calls are enriched for plausible interacting
  individuals
- test whether adjacency relates to co-occurring behavioral annotations
- test whether short-note-like broadcast calls violate the proxy more often

If identity or behavior metadata is unavailable:

State explicitly:

> This study cannot internally validate the temporal-adjacency proxy in MeerKAT.
> The MeerKAT analysis is therefore interpreted as a conservative sequence
> prediction analysis under a calibrated prior from a higher-observability
> dataset.

### Experiment 4: Behavioral Sufficiency Test

Compare predictive models for next-call/local-context structure:

1. manual call category only
2. acoustic representation only
3. manual category + acoustic representation
4. nuisance-only controls, such as duration, recording, individual, local call
   rate, and time since previous call

Main question:

> Do acoustic representations provide predictive information beyond manual
> categories under the conservative adjacency proxy?

Possible outputs:

- manual labels sufficient
- acoustic representations add information
- nuisance variables explain the apparent gain
- unresolved because proxy validity is weak

## Certificate-Style Decision Outputs

The method should return a set-valued conclusion rather than a single
overconfident label.

Possible outputs:

- `proxy-supported`: adjacency is empirically supported in a comparable context
- `proxy-context-dependent`: adjacency works only in some behaviors or windows
- `proxy-unresolved`: target dataset lacks validation metadata
- `manual-label-sufficient`: manual categories explain local predictive signal
- `manual-label-insufficient`: acoustic features add robust information
- `nuisance-confounded`: gains are explained by duration, recording, individual,
  call rate, or padding/context artifacts
- `unresolved`: evidence is unstable or contradictory

## Why This Is More Novel

The earlier MeerKAT-only analysis mainly showed:

> manual labels are decodable but not cleanly clustered.

That is useful but close to prior latent-space taxonomy work.

The revised contribution is broader:

> The project calibrates when weak behavioral proxies should be trusted, when
> they should be rejected, and when the correct scientific output is unresolved.

This is valuable because animal vocalization studies often rely on imperfect
observational proxies. A calibrated proxy-validity framework directly addresses
the risk of over-interpreting temporal adjacency as interaction.

## Current Progress

Completed:

- MeerKAT manifest construction
- balanced and capped-natural pilots
- multiple acoustic representations
- label separability and clustering diagnostics
- animal2vec analysis
- HDBSCAN stability
- GMM/HDBSCAN coarse-group comparison
- fuzzy c-means degeneration analysis
- spectrogram-to-UMAP-to-HDBSCAN sanity-check replication

Not yet completed:

- Egyptian fruit bat calibration
- temporal adjacency proxy curves
- MeerKAT 184h subset extraction
- internal MeerKAT identity/behavior validation check
- behavioral sufficiency prediction models
- certificate decision rule

## Immediate Next Steps

1. Identify and obtain the Egyptian fruit bat dataset and its annotation schema.
2. Define event membership, emitter, addressee, and behavioral-context fields.
3. Implement context-stratified temporal adjacency calibration.
4. Extract the MeerKAT 184h temporally continuous subset.
5. Check whether MeerKAT identity or behavioral metadata survived in the public
   release.
6. Build the behavioral sufficiency prediction task.
7. Freeze certificate outputs and decision language before interpreting MeerKAT
   results.

## Paper Positioning

Candidate title:

> Calibrated Proxy Validity for Behavioral Sufficiency Tests in Animal Vocal
> Repertoires

Alternative title:

> When Does Temporal Adjacency Mean Interaction? Calibrating Proxy Validity in
> Bioacoustic Sequence Analysis

Reviewer-facing message:

> The paper does not claim to infer responses from MeerKAT temporal adjacency.
> Instead, it calibrates the conditions under which temporal adjacency can serve
> as a conservative proxy, and uses that proxy to test whether manual call
> categories exhaust local interaction-predictive information.

