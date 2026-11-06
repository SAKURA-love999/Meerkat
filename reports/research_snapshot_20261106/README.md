# Audited Research Snapshot:2026-11-06

## Scope

These are byte-identical allowlisted aggregate exports of completed local
experiments. No raw audio,feature matrix,per-call OOF,private paper or reviewer
transcript is included. provenance.json records successful native run IDs,
actual invocation dates,protocol identities,limited protocol excerpts,source
code byte hashes and SHA256 of every exported result. The original full
protocols contain absolute local input paths and are not published here.

This permits checking reported aggregate numbers and comparing named settings.
It does not let a fresh clone reconstruct OOF or independently reproduce the
science without obtaining data and generating the missing source packets.
Native audit results are deterministic checks,not independent semantic reviews.

## Shared Real-Data Contract

18345 calls/9756 anonymous10-second clips;400ms both-side eligible PRE/POST;
original5clipfolds;14786 original-valid evaluation rows. Seven shared controls:
received RMS/peak/crest/clipping and focal-interval overlap fraction/nonfocal
fraction/count. Existing SNR was excluded because it used surrounding audio.
The capped pilot is not the entire natural population. Anonymous clip grouping
is not a caller/session holdout.

M0 uses manual category,measured history,duration/boundary terms and controls.
M1 adds81 cached raw librosa fields. Mconfig adds only n_fft/hop_length/n_mels;
M78 adds the remaining78. Those78 still contain duration/amplitude and
resolution-dependent estimates. Positive Delta means lower natural-logloss.
NF_post/NF_pre are activity proxies,not verified responses or causal effects.
Intervals are1000 paired clip draws,unadjusted,conditional on stored OOF and
excluding full-training uncertainty. No practical equivalence margin was fixed.

## Read The Folders

|Folder|Meaning|
|---|---|
|b3|Original fixedC1 linear/RMS-spline/HGB,81fields,PCA30|
|nested_c|Separate PCA30,training-only inner clip selection of C{.1,1,10};not joint PCA tuning|
|pca|Original fixed learners,PCA30/full/no-PCA;do not mix with tuned-C rows|
|config|No-PCA forward chain,30 new Mconfig fits;old M0/M1 reused|
|reverse_prepare|Historical zero-fit preparation record,not another scientific fit|
|reverse|30 new M78 fits,old three models reused;five gain contrasts and predictive interaction|
|*_audit|Zero-fit recomputation/integrity reports for the corresponding runs|

For every table filter view=all as primary and manual_label=overall or a named
label. original_valid_rows changes evaluation rows only in these blocks.
Never compare unmatched cohorts as a representation effect.

## Bounded Findings

- No stable positive overall POST increment is established across tested controls
  and model settings. A positive HGB no-PCA point remains with uncertainty.
- Linear al's PCA30 increment.003703 shrinks to.000657 under RMS-spline baseline
  expression;paired change-.003046[-.004211,-.001773]. Absolute M0 improves more
  than M1. Separate C/PCA screens retain attenuation,not universal direction
  symmetry or biological absence.
- Config-only fields do not reproduce linear al gain. In the reverse ablation,
  linear al M78 gain=.004122647[.000257271,.007642221];D=.003495403
  [-.001127470,.007614764]. Spline/HGB M78 al POST intervals span0.
- HGB M78 and81-field M1 predictions are exactly identical in all folds/targets.
  In logistic models configurations change some predictions and can worsen
  overall POST risk. These are learner-specific findings,not universal irrelevance.
- valid-sn contains only271 calls/11 POST positives. Selected-row positive
  config increments do not characterize excluded short sn calls.

No categorical-sufficiency,no-information,causal,source-purity or confirmed
novelty claim follows. Predictive interaction measures order dependence of
fitted losses,not necessarily feature/biological interaction.

## When To Begin The Next Validation

Independent-data screening can start now. Freeze the replication question and
comparisons before examining confirmation outcomes;use a separate development
batch for format/QC and never silently reuse it as untouched confirmation.
Verify raw audio,complete event timing,manual labels,recording/caller/session
provenance and non-overlap with existing sources. Another distribution of the
same recordings is not independence. A new species tests method scope,not an
exact replication of the meerkat al result;choose species-appropriate endpoints.

Receiver-linked validation additionally needs synchronized caller/receiver
identity and behavioral/spatial observations or independently annotated response
events. Temporal adjacency alone is insufficient. Demartsev et al.2024 used
identity-linked movement and vocal observations to distinguish exchange from
broadcast;their available figure data are not assumed to contain the raw audio
needed by this project without further audit.
[Primary paper](https://pubmed.ncbi.nlm.nih.gov/38768207/).

Causal acoustic testing is a separate controlled-manipulation/playback design,
with valid within-type stimuli,randomization,multiple callers/receivers and
blind behavioral outcomes,subject to collaboration and ethics review. Prior
meerkat work already uses playback;the intervention is not itself a new method.
[Rauber and Manser2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5358691/).
Such a test can support an acoustic intervention's behavioral effect,not
automatically a neural/cognitive mechanism. No animal intervention ran here.
