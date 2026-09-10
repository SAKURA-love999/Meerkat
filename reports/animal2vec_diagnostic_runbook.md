# Animal2vec Diagnostic Runbook

This runbook starts from the finished animal2vec balanced-600 embedding and
asks a stricter version of the cluster question:

> Is the animal2vec structure a stable biological/acoustic structure, or a
> nuisance artifact from recording, individual, duration, context, or quality?

Long-running commands are marked as **Run yourself**.

## Current Inputs

Expected project root:

```powershell
Set-Location "C:\学校\MeerKAT\repertoire_geometry"
```

Main embedding:

```text
features\animal2vec_context500ms_focal_mean_balanced_600.npz
features\animal2vec_context500ms_focal_mean_balanced_600_metadata.csv
```

The `.npz` contains:

```text
X: 4800 x 1024 float32 animal2vec vectors
```

The metadata should contain at least:

```text
label
```

Ideally it also contains:

```text
sample ID / call ID
individual
recording / file / session
duration
focal_pooled_frames
model_total_frames
RMS / SNR proxy / clipping / quality columns
```

If some nuisance columns are absent, the audit script skips them rather than
failing.

## Step 1: UMAP Island, kNN Geometry, and Fuzzy Membership

**Question**

Do visible UMAP islands, pairwise label neighborhoods, and fuzzy memberships
exist in animal2vec space?

Important:

- The fuzzy result must pass the degeneration check before interpretation.
- A table that is exactly `0.125 x 8` with entropy close to `ln(8)` is a failed
  fuzzy solution, not evidence that all calls are maximally ambiguous.

**Run yourself**

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\diagnose_animal2vec_umap_fuzzy.py
```

**Outputs**

```text
reports\animal2vec_left_umap_islands_samples.csv
reports\animal2vec_pairwise_label_centroid_distances.csv
reports\animal2vec_knn15_label_mixing_matrix.csv
reports\animal2vec_fuzzy_membership_by_sample.csv
reports\animal2vec_fuzzy_membership_by_label.csv
reports\animal2vec_fuzzy_diagnostics_by_c.csv
reports\animal2vec_fuzzy_selected_diagnostics.json
figures\umap_animal2vec_fuzzy_entropy.png
```

**How to read**

- `left_umap_islands_samples.csv`: diagnostic subset from the far-left UMAP
  region. This is not automatically a biological cluster.
- `pairwise_label_centroid_distances.csv`: which manual labels are close in
  PCA-reduced animal2vec space.
- `knn15_label_mixing_matrix.csv`: for each label, what fraction of local
  neighbors come from each label.
- `fuzzy_membership_by_sample.csv`: soft cluster membership for every call.
- `fuzzy_membership_by_label.csv`: average soft-membership profile per manual
  label.
- `fuzzy_diagnostics_by_c.csv`: fuzzy validity diagnostics for multiple values
  of `c`.
- `fuzzy_selected_diagnostics.json`: validity diagnostics for the selected
  `c`, default `c=8`.

**Fuzzy validity check**

Before using fuzzy membership for transition claims, check:

```text
reports\animal2vec_fuzzy_selected_diagnostics.json
```

If this says:

```text
is_uniform_degenerate = true
```

or if:

```text
mean_max_membership is close to 1/c
mean_relative_entropy is close to 1.0
```

then the fuzzy result is numerically uninformative. Do not interpret it as
"all calls are fully fuzzy".

This can happen because:

- distance concentration in high-dimensional representation
- centroids are too close
- `c=8` does not match the latent geometry
- the fuzzy update has collapsed to near-uniform membership

In that case, inspect:

```text
reports\animal2vec_fuzzy_diagnostics_by_c.csv
```

and compare `c=2,3,4,5,6,8,10,12`. The selected `c` for transition analysis
should be reported as an analytic setting, not as a label-supervised optimum.

If all or most fuzzy c-means settings remain near-uniform, pause fuzzy
interpretation and run the soft-membership robustness script:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\soft_membership_robustness.py
```

This compares:

- fuzzy c-means over `PCA dim x c x m`
- GMM posterior membership
- HDBSCAN soft membership

Outputs:

```text
reports\animal2vec_soft_membership_fcm_grid.csv
reports\animal2vec_soft_fcm_selected_by_sample.csv
reports\animal2vec_soft_fcm_selected_by_label.csv
reports\animal2vec_soft_fcm_selected_diagnostics.json
reports\animal2vec_soft_gmm_selected_by_sample.csv
reports\animal2vec_soft_gmm_selected_by_label.csv
reports\animal2vec_soft_gmm_selected_diagnostics.json
reports\animal2vec_soft_hdbscan_selected_by_sample.csv
reports\animal2vec_soft_hdbscan_selected_by_label.csv
reports\animal2vec_soft_hdbscan_selected_diagnostics.json
```

Interpretation:

- If only fuzzy c-means degenerates, but GMM/HDBSCAN soft memberships are
  structured, the problem is probably FCM distance concentration or fuzzifier
  behavior.
- If all soft methods are near-uniform, the current representation does not
  support interpretable soft components.
- If a lower-dimensional FCM setting works but PCA-50 fails, report fuzzy
  membership as dimension-sensitive.

Do not use GMM posterior or HDBSCAN soft membership to claim biological
categories directly. They are diagnostic soft assignments.

For the exact `c=8` fuzzy c-means degeneration check, run:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\debug_fuzzy_cmeans_degeneracy.py `
  --c 8 `
  --m 2.0 `
  --pca-dim 50
```

This writes:

```text
reports\animal2vec_fcm_debug_c8_m2p0_pca50_summary.json
reports\animal2vec_fcm_debug_c8_m2p0_pca50_iteration_trace.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_initial_sample_distance_cv.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_final_sample_distance_cv.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_membership_by_sample.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_initial_centroids.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_final_centroids.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_initial_centroid_distance_matrix.csv
reports\animal2vec_fcm_debug_c8_m2p0_pca50_final_centroid_distance_matrix.csv
```

It directly checks:

- 8 fuzzy centroid pairwise distance min / median / max
- each sample's distance CV to the 8 centroids
- initial membership min / max / std
- centroid displacement after every iteration

Interpretation:

- If initial membership is structured but final membership becomes uniform,
  the FCM update is collapsing.
- If centroid distances shrink heavily during iteration, centroids are
  converging toward the same region.
- If final sample distance CV is near zero, each sample is almost equally far
  from all centroids, so membership must become near-uniform.
- If initial membership is already near-uniform, the issue starts before
  iteration, likely from distance concentration or bad initialization.

To check whether `c=3` effectively collapses into a two-prototype solution, run:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\debug_fuzzy_cmeans_degeneracy.py `
  --c 3 `
  --m 2.0 `
  --pca-dim 50
```

Then inspect:

```text
reports\animal2vec_fcm_debug_c3_m2p0_pca50_final_centroid_distance_matrix.csv
```

If two centers have a near-zero distance, `c=3` should not be interpreted as a
reliable three-component fuzzy solution.

The robustness table:

```text
reports\animal2vec_soft_membership_fcm_grid.csv
```

contains:

- `pca_dim`
- `c`
- `m`
- `is_uniform_degenerate`
- `mean_max_membership`
- `mean_entropy`
- `mean_relative_entropy`
- `partition_coefficient`
- `min_center_distance`
- `xie_beni_index`
- `fcm_winner_label_ami`
- `fcm_winner_label_ari`
- component-label purity summaries

Xie-Beni index:

- Lower values usually indicate more compact and separated fuzzy clusters.
- Infinite or extremely large values indicate collapsed or near-overlapping
  centers.
- It should be compared across nearby settings, not used as a single magic
  cutoff.

The reporting rule is:

```text
Report parameter regions, not isolated parameter points.
```

Use:

```text
Stable under pca_dim = ..., c = ..., m = ...
```

only when nearby settings show similar non-degenerate membership, similar
center separation, and similar label correspondence. If only one setting works,
write it as sensitivity, not as a robust result.

**Supports**

- local overlap between specific labels
- candidate transitional pairs
- fuzzy rather than hard membership

**Cannot support**

- biological graded communication by itself
- stable clusters by itself
- UMAP islands as meaningful structures before artifact checks

## Step 2: Artifact / Nuisance-Factor Audit

**Question**

Are the far-left UMAP islands and selected HDBSCAN clusters explained by
recording, individual, duration, valid frames, or quality?

This step uses the output from Step 1.

**Run yourself**

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\audit_animal2vec_artifacts_and_clusters.py
```

For a focused padding/context audit, also run:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\audit_animal2vec_context_padding.py
```

Default selected HDBSCAN setting:

```text
min_cluster_size = 30
min_samples = 10
epsilon = 0.0
selection_method = eom
```

This is the same fixed setting used in the bootstrap stability run.

**Outputs**

```text
reports\animal2vec_left_island_samples_with_fuzzy.csv
reports\animal2vec_left_island_numeric_nuisance_audit.csv
reports\animal2vec_left_island_categorical_nuisance_audit.csv
reports\animal2vec_selected_hdbscan_assignments_with_metadata.csv
reports\animal2vec_selected_hdbscan_label_composition.csv
reports\animal2vec_selected_hdbscan_numeric_nuisance_summary.csv
reports\animal2vec_artifact_cluster_audit_summary.json
reports\animal2vec_left_island_context_padding_samples.csv
reports\animal2vec_left_island_context_padding_numeric_audit.csv
reports\animal2vec_left_island_context_padding_categorical_audit.csv
reports\animal2vec_left_island_context_padding_audit_summary.json
```

Depending on available metadata columns, it may also write:

```text
reports\animal2vec_selected_hdbscan_<column>_composition.csv
```

**How to read**

For the left UMAP islands:

- If one recording/file dominates, suspect recording artifact.
- If one individual dominates, suspect identity/caller artifact.
- If duration or focal frame count is extreme, suspect duration/context artifact.
- If SNR/RMS/clipping/validity differs strongly, suspect signal-quality artifact.

For selected HDBSCAN clusters:

- If clusters are mostly explained by recording or individual, do not interpret
  them as repertoire structure.
- If clusters are mixed across recording/individual and have normal quality,
  they are stronger candidates for acoustic structure.

**Supports**

- ruling out obvious nuisance explanations
- identifying artifact-driven islands or clusters
- deciding whether cluster/fuzzy results are interpretable

**Cannot support**

- graded transitions by itself
- label discreteness by itself

## Step 3: HDBSCAN Parameter Stability

**Question**

Is the animal2vec 3-cluster structure stable under HDBSCAN parameter
perturbation?

**Run yourself**

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\hdbscan_stability_grid.py `
  --npz features\animal2vec_context500ms_focal_mean_balanced_600.npz `
  --metadata features\animal2vec_context500ms_focal_mean_balanced_600_metadata.csv `
  --representation-name animal2vec_context500ms_focal_mean_balanced_600_zscore_context `
  --bootstrap `
  --n-bootstrap 20
```

**Outputs**

```text
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_parameter_grid.csv
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_parameter_solution_stability.csv
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_bootstrap_runs.csv
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_bootstrap_solution_stability.csv
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_summary.json
reports\hdbscan_stability_animal2vec_context500ms_focal_mean_balanced_600_zscore_context_parameter_grid.png
```

**How to read**

Parameter stability:

- Stable cluster count across grid supports algorithmic robustness.
- Strongly varying `n_clusters` argues against stable cluster recovery.
- Strongly varying `noise_rate` means the density structure is parameter
  sensitive.
- Low pairwise solution AMI/ARI means different settings recover different
  assignments.

Bootstrap stability:

- High bootstrap solution AMI/ARI means the same fixed setting recovers similar
  structure after sample perturbation.
- Low bootstrap solution AMI/ARI means the structure is sample-fragile.

Important:

- Label AMI/ARI are post-hoc only.
- Do not choose the setting with the highest label AMI and call it "best".
- Report it as "highest observed AMI".

**Supports**

- whether HDBSCAN solutions are stable under parameter/sample perturbation
- whether a 3-cluster solution is reproducible

**Cannot support**

- that the three clusters are biological classes
- that the eight manual labels are natural clusters

## Step 4: Summarize Stability Across Representations

After you run Step 3 for multiple representations, combine summaries:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\summarize_hdbscan_stability.py
```

**Output**

```text
reports\hdbscan_stability_summary_table.csv
```

For other representations, reuse the same script by changing:

```text
--npz
--metadata
--representation-name
```

The key scientific claim should be based on the cross-representation table, not
only animal2vec.

## Step 5: Transition-Axis Analysis

**Question**

For candidate confused or neighboring label pairs, is there evidence of
overlap, or stronger evidence for graded/transitional structure?

This step does not use UMAP as the main axis. It projects samples onto the
centroid-to-centroid axis between two labels in PCA-reduced animal2vec space.

Default pairs:

```text
ld:mo
agg:soc
cc:oth
al:mo
```

**Run yourself**

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\analyze_animal2vec_transition_axes.py
```

Custom pairs:

```powershell
& "C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe" `
  scripts\analyze_animal2vec_transition_axes.py `
  --pairs "ld:mo,agg:soc,cc:oth,al:mo,sn:cc"
```

**Outputs**

```text
reports\animal2vec_pair_transition_axis_by_sample.csv
reports\animal2vec_pair_transition_axis_bins.csv
reports\animal2vec_pair_transition_axis_summary.csv
```

**How to read**

Evidence for simple overlap:

- two labels occupy overlapping axis ranges
- kNN neighborhoods contain both labels
- fuzzy entropy is high in the overlap area

Stronger evidence for graded / transitional structure:

- label proportion changes gradually across the centroid axis
- fuzzy component membership changes gradually across bins
- middle-axis bins contain mixed labels rather than empty gaps
- the same candidate transition appears in pairwise distance, kNN mixing, fuzzy
  membership, and classifier confusion

Weak evidence:

- overlap exists but only because of low-quality recordings
- middle bins are dominated by one recording/individual
- transition appears only in UMAP, not PCA/latent-space axis

## Recommended Interpretation Order

Use this order when writing the result:

1. Manual labels are decodable from animal2vec embeddings.
2. HDBSCAN recovers a coarse structure under some settings.
3. The coarse structure is not strongly aligned with the eight manual labels.
4. Bootstrap asks whether one fixed solution is sample-stable.
5. Parameter grid asks whether the solution is algorithmically stable.
6. Artifact audit asks whether islands/clusters are recording/individual/quality
   artifacts.
7. Pairwise and fuzzy analyses identify candidate overlap or transition regions.
8. Transition-axis analysis tests whether overlap can be upgraded to graded /
   transitional evidence.

## Conservative Claim Template

If the stability and artifact checks are weak:

```text
No clear label-aligned clusters were recovered under the tested settings.
```

If parameter and bootstrap stability are weak across representations:

```text
Across tested representations and perturbation settings, HDBSCAN did not recover
stable natural clusters that consistently aligned with public MeerKAT call
labels.
```

If animal2vec has stable coarse clusters but weak label alignment:

```text
Animal2vec revealed a reproducible coarse acoustic structure, but this structure
was lower-dimensional than the eight manual labels and only weakly aligned with
them.
```

If pairwise/fuzzy/transition-axis evidence is consistent and artifacts are
excluded:

```text
The results support overlapping and potentially graded acoustic structure
between specific call-label pairs, rather than a repertoire organized as eight
fully discrete acoustic categories.
```
