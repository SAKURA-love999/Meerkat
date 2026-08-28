# Script I/O and Method Audit

This audit records what each script consumes, what it computes, what it writes, and where label leakage or analysis artifacts may enter the current MeerKAT repertoire geometry pipeline.

## Project-Level Data Flow

Input data are public MeerKAT 10-second `wav` files and paired HDF5 label files. The HDF5 files provide event boundaries, call labels, label categories, and focal/non-focal flags. The public release does not preserve original temporal order or individual identity in recoverable form, so the current public-data case study can test acoustic geometry of labelled calls, but cannot directly test true sequence history or individual identity without additional non-public metadata.

The current target labels are `sn`, `cc`, `ld`, `mo`, `al`, `soc`, `agg`, and `oth`. Labels are used to filter and balance pilot subsets, to color plots, and to evaluate cluster-label agreement or classifier performance. Labels are not used to fit PCA, UMAP, HDBSCAN, GMM, acoustic features, spectrogram vectors, or eGeMAPS features.

The main current inference is intentionally narrow: public call labels are partly decodable from several acoustic representations, but they do not map cleanly onto stable, label-aligned unsupervised clusters in the first balanced pilot.

## Leakage and Artifact Summary

Balanced sampling uses labels. This is acceptable for a strict geometry pilot, because it controls prevalence and gives rare labels enough samples, but conclusions are conditional on a label-balanced subset. A capped or natural-prevalence pilot is still needed to test robustness under real long-tail prevalence.

Current classifier probes have a mild distribution leakage risk. In `analyze_geometry_pilot.py`, median imputation and scaling are fit on all rows before cross-validation. In `analyze_embedding_pilot.py`, scaling and PCA are also fit before cross-validation. This does not leak labels, but it lets the test fold influence preprocessing. The current Macro-F1 and balanced accuracy should therefore be treated as preliminary. A stricter version should put imputation, scaling, and PCA inside each fold, preferably with `GroupKFold` or grouped stratification by `source_id` to avoid train/test calls from the same 10-second file.

Unsupervised PCA, UMAP, HDBSCAN, kNN purity, silhouette, and GMM+BIC are allowed to use the full matrix, because they are descriptive geometry analyses rather than supervised generalization estimates. Their interpretation is still sensitive to feature scaling, dimensionality, distance metric, and HDBSCAN parameters.

NaN handling differs by representation. Librosa features can produce NaN for very short or near-silent calls and are median-imputed before analysis. eGeMAPS often produces NaN for very short `sn` calls; valid-only eGeMAPS analyses are therefore essential. Spectrogram vectors do not generally require imputation because each call is converted to a fixed numeric image vector.

Amplitude is a potential nuisance factor. Current analyses preserve paired raw-amplitude and RMS-normalized versions. Raw amplitude tests whether loudness is part of the public label structure; RMS-normalized tests whether geometry remains when global loudness is reduced.

Duration is a potential nuisance factor. Raw-boundary features preserve duration. `logmel_call_resize` removes or distorts duration by warping every call to the same number of frames. `logmel_paper_pad500` preserves duration within a fixed 500 ms grid but adds right-padding, so duration can still be visible through padded regions.

UMAP plots are visualization diagnostics only. Apparent bridges, gaps, or clusters in 2D UMAP are not proof of continuity or discreteness by themselves.

HDBSCAN parameter sweeps must not select a setting because it maximizes AMI. AMI/ARI use labels and are evaluation statistics, not unsupervised model-selection criteria. Reports should distinguish unsupervised-screened settings from the highest observed AMI settings.

GMM+BIC estimates how many Gaussian components fit a chosen representation better under a parametric model. It does not by itself prove discrete or continuous vocal categories.

## Script Audit

### `scripts/build_meerkat_call_manifest.py`

Input: public MeerKAT dataset root containing `lbl/08000Hz/*.h5` and `wav/08000Hz/*.wav`.

Steps: reads HDF5 fields `start_time_lbl`, `end_time_lbl`, `start_frame_lbl`, `end_frame_lbl`, `lbl`, `lbl_cat`, and `foc`; decodes each event into one call-level row; matches the paired source wav; summarizes counts and duration distributions.

Output: `data_manifest/call_manifest.csv`, `data_manifest/label_summary.csv`, and `data_manifest/duration_summary.csv`.

Why reasonable: it converts public event annotations into the call-level table needed by every later analysis while preserving source file, boundary, label, and focal metadata.

Label use: labels are read as metadata only.

Leakage/artifacts: no model fitting occurs. Public filenames are randomized, so source order and identity should not be inferred from `source_id`.

### `scripts/make_pilot_manifest.py`

Input: `data_manifest/call_manifest.csv`.

Steps: filters to target labels, optionally focal-only, removes calls outside a duration range, shuffles with a fixed seed, and samples up to `max_per_label` per label.

Output: `data_manifest/pilot_manifest_<name>.csv` and `data_manifest/pilot_manifest_<name>_summary.csv`.

Why reasonable: the balanced pilot prevents common labels such as `cc` from dominating geometry diagnostics and gives rare labels enough representation for first-pass comparison.

Label use: labels are used directly for filtering and stratified sampling.

Leakage/artifacts: this creates an intentionally label-conditioned dataset. It is not leakage for unsupervised feature extraction, but it changes prevalence and can make rare-label structure look more visible than in the full public distribution.

### `scripts/extract_acoustic_features.py`

Input: `data_manifest/pilot_manifest_<pilot>.csv` and source wav files.

Steps: cuts each focal call using original boundaries; computes raw RMS, peak, crest factor, clipping fraction, zero-crossing rate, spectral centroid/bandwidth/rolloff/flatness, spectral entropy, and MFCC summary statistics; creates a paired RMS-normalized version targeting -25 dBFS; estimates an SNR proxy from 250 ms before/after context; writes QC flags.

Output: `features/acoustic_features_<pilot>.csv`, `features/feature_validity_QC_<pilot>.csv`, and `features/feature_validity_QC_<pilot>_summary.csv`.

Why reasonable: hand-crafted acoustic summaries are transparent, cheap, and make a useful first baseline before neural embeddings.

Label use: label is copied as metadata and not used to compute features.

Leakage/artifacts: raw features can encode loudness and duration; normalized features reduce global loudness but not all amplitude-related effects. The SNR proxy uses neighboring audio only for QC, not as a feature in the main geometry analysis.

### `scripts/extract_egemaps_features.py`

Input: `data_manifest/pilot_manifest_<pilot>.csv` and source wav files.

Steps: cuts each focal call using original boundaries; computes openSMILE eGeMAPSv02 Functionals for raw and RMS-normalized audio; records NaN/Inf counts and validity.

Output: `features/egemaps_features_<pilot>.csv`, `features/egemaps_feature_validity_QC_<pilot>.csv`, and `features/egemaps_feature_validity_QC_<pilot>_summary.csv`.

Why reasonable: eGeMAPS is a standardized, interpretable acoustic feature set commonly used for affective and paralinguistic audio analysis; it gives an external feature baseline beyond custom librosa summaries.

Label use: label is copied as metadata and not used to compute features.

Leakage/artifacts: eGeMAPS Functionals summarize the entire input segment. Therefore 500 ms context windows should not be fed into eGeMAPS if the intended object is the focal call. Very short calls, especially `sn`, often produce NaN, so valid-only analysis is necessary.

### `scripts/extract_spectrogram_representation.py`

Input: `data_manifest/pilot_manifest_<pilot>.csv` and source wav files.

Steps: cuts each focal call using original boundaries; creates raw and RMS-normalized audio variants; converts each call to log-mel spectrogram vectors. `logmel_call_resize` uses 64 mel bins and resizes the time axis to 64 frames. `logmel_paper_pad500` uses 40 mel bins, about 30 ms frames, 3.75 ms hop, per-spectrogram z-score, original duration, and right-padding to a 500 ms grid.

Output: `features/spectrogram_<representation>_<pilot>.npz` and matching metadata CSV.

Why reasonable: spectrogram vectors test whether weak clustering is caused by overly coarse summary features.

Label use: label is copied as metadata and not used to compute vectors.

Leakage/artifacts: resizing can distort temporal structure and remove true duration. Padding preserves duration cues but may introduce pad-pattern cues. Per-spectrogram z-score removes global loudness, so it cannot test whether absolute loudness carries label information.

### `scripts/extract_animal2vec_embeddings.py`

Input: `data_manifest/pilot_manifest_<pilot>.csv`, source wav files, official animal2vec repo root, and a `.pt` checkpoint.

Steps: loads the animal2vec/fairseq checkpoint; extracts either a fixed real-context window around the focal call or the full 10-second source file; z-normalizes the model input; obtains transformer layer embeddings; maps focal sample boundaries to model frame indices; pools only focal-call frames.

Output: `features/animal2vec_<mode>_<pilot>.npz` and matching metadata CSV.

Why reasonable: short calls such as `sn` may be too brief for a context-aware neural model if passed alone. Feeding real context while pooling only focal frames preserves model context without treating the entire context as the focal event.

Label use: labels are copied as metadata and not used for embedding extraction.

Leakage/artifacts: using the finetuned MeerKAT checkpoint risks conceptual label leakage because it was trained with MeerKAT labels. For unsupervised representation geometry, the pretrained self-supervised checkpoint is the cleaner first choice. Context length can change embeddings, so 250/500/1000 ms sensitivity analysis is needed.

### `scripts/analyze_geometry_pilot.py`

Input: CSV feature table, optionally a validity QC table.

Steps: selects numeric feature columns while excluding metadata and extraction parameters; imputes missing values with feature medians; standardizes features; computes PCA and UMAP visualizations; runs default HDBSCAN with `min_cluster_size=50`, `min_samples=10`; computes label silhouette, cluster silhouette, AMI, ARI, kNN-15 label purity, linear classifier probe, and GMM+BIC.

Output: PCA/UMAP/confusion figures, HDBSCAN assignments, GMM+BIC CSV, and `features/geometry_metrics_<representation>_<pilot>.csv`.

Why reasonable: it separates visual geometry, natural clustering, label-cluster alignment, local label mixing, supervised separability, and parametric component-count diagnostics.

Label use: labels are used for plot colors, silhouette-by-label, AMI/ARI, kNN purity, and classifier evaluation. Labels are not used for PCA, UMAP, HDBSCAN, or GMM fitting.

Leakage/artifacts: classifier metrics are currently preliminary because imputation and scaling are fit before CV. This should be corrected by moving preprocessing into the CV pipeline.

### `scripts/analyze_embedding_pilot.py`

Input: dense `.npz` matrix with `X`, `labels`, `call_ids`, and `amplitude_variants`.

Steps: splits by amplitude variant; standardizes vectors; reduces to an analysis PCA space; saves PCA/UMAP plots; runs default HDBSCAN; computes the same label-alignment, local-mixing, classifier, and GMM+BIC metrics.

Output: figures, HDBSCAN assignments, confusion matrices, GMM+BIC CSV, and `features/geometry_metrics_<representation>_<pilot>.csv`.

Why reasonable: high-dimensional spectrogram or neural embeddings need a shared analysis layer, especially PCA, before HDBSCAN/GMM are computationally stable.

Label use: labels are used only for evaluation and visualization, not for PCA/HDBSCAN/GMM fitting.

Leakage/artifacts: classifier metrics are preliminary because PCA is fit before CV. A stricter probe should fit scaler/PCA/classifier inside each fold.

### `scripts/hdbscan_sensitivity.py`

Input: `features/acoustic_features_<pilot>.csv`.

Steps: builds full and PCA-reduced spaces; sweeps `min_cluster_size`, `min_samples`, and PCA dimensions; computes noise rate, cluster count, AMI, ARI, cluster silhouette, majority purity, and membership probability.

Output: `features/hdbscan_sensitivity_<pilot>.csv`, highest-AMI top rows, and heatmaps.

Why reasonable: it tests whether the weak/default HDBSCAN result is a parameter artifact.

Label use: labels are used to compute AMI/ARI and purity after clustering.

Leakage/artifacts: highest AMI must not be called the best unsupervised setting. The script's `top20` filename is less careful than the later embedding sweep naming, so reports should phrase it as highest observed AMI.

### `scripts/hdbscan_sensitivity_embedding.py`

Input: dense `.npz` vectors such as spectrogram or animal2vec embeddings.

Steps: standardizes vectors; builds PCA spaces; sweeps HDBSCAN parameters; records unsupervised cluster stability/size/noise diagnostics and label-alignment metrics.

Output: `features/hdbscan_sensitivity_<representation>_<pilot>.csv`, highest-AMI summaries, and heatmaps.

Why reasonable: it checks whether dense representation clustering depends on HDBSCAN hyperparameters and analysis dimensionality.

Label use: labels are used only after clustering for AMI/ARI and purity.

Leakage/artifacts: PCA spaces are fit on all points, which is acceptable for unsupervised diagnostic geometry. Highest-AMI rows are descriptive, not model selection.

### `scripts/summarize_hdbscan_setting.py`

Input: a dense `.npz` representation and one chosen HDBSCAN setting.

Steps: recreates the PCA space, runs HDBSCAN once, computes per-label non-noise inclusion and per-cluster label composition.

Output: `features/hdbscan_setting_<...>_label_inclusion.csv` and `features/hdbscan_setting_<...>_cluster_composition.csv`.

Why reasonable: it explains whether a setting's clusters are driven by one label, a few labels, or mixed acoustic regions.

Label use: labels are used only to summarize inclusion and composition after clustering.

Leakage/artifacts: the chosen setting should be justified using unsupervised criteria or clearly described as a diagnostic setting. Cluster inclusion rate means the fraction of a label assigned to any non-noise cluster, not classification accuracy.

## Environment and Checkpoint Status

A dedicated Python 3.9 environment has been created locally for animal2vec. Keep the exact machine path out of public Git.

Confirmed installed versions:

- Python 3.9.23
- torch 1.13.1+cpu
- torchaudio 0.13.1+cpu
- librosa 0.10.1
- numpy 1.23.5
- pandas 1.5.3

`fairseq` is not currently installed. The PyPI `fairseq==0.12.2` build failed because Microsoft Visual C++ 14.0+ Build Tools are required. The official pinned GitHub commit also failed on Windows because the installer attempted to create a symlink and hit `WinError 1314`. To run animal2vec on this Windows machine, install MSVC Build Tools and enable symlink privileges/developer mode, or run the extraction in WSL/Linux.

The official model-weight DOI exposes two checkpoints:

- `animal2vec_large_finetuned_MeerKAT_240507.pt`, datafile `253219`, about 22.9 GB, MD5 `b377ea79700f3bbc98b6154f21545158`
- `animal2vec_large_pretrained_MeerKAT_240507.pt`, datafile `253220`, about 5.0 GB, MD5 `c0ae0cb16afd0501f00a5955fb6482ed`

For the current unsupervised repertoire-geometry experiment, the pretrained checkpoint is the preferred first animal2vec representation. The finetuned checkpoint can be useful later, but it should be reported separately because it was trained with MeerKAT labels.

## Reference Sources Used

Official animal2vec GitHub repository: `https://github.com/livingingroups/animal2vec`

Used for the installation constraints, the Python 3.6-3.9 warning, the `pip==24.0` instruction, the pinned fairseq commit, the need to import `nn` before loading fairseq checkpoints, the MeerKAT class-name list, and the expected 10-second 8 kHz input convention.

animal2vec and MeerKAT paper: `https://doi.org/10.1111/2041-210x.70218`

Used for the interpretation of animal2vec as a self-supervised transformer designed for sparse and imbalanced bioacoustic data, and for the distinction between pretrained and finetuned model use.

MeerKAT public dataset DOI: `https://doi.org/10.17617/3.0J0DYB`

Used as the source definition for public 10-second audio chunks and HDF5 call annotations.

animal2vec model-weight DOI: `https://doi.org/10.17617/3.ETPUKU`

Used to identify the official `.pt` checkpoints, file sizes, datafile IDs, and MD5 checksums.

## Result-Code Consistency

The existing reports match the current code structure and output file naming for the analyses already run: librosa summary features, spectrogram resize, paper-style spectrogram padding, HDBSCAN sensitivity, and eGeMAPS.

Two caveats should be kept with the current reported numbers:

1. Classifier-probe metrics are preliminary until preprocessing and PCA are moved inside cross-validation folds.
2. Balanced-pilot conclusions answer label-balanced acoustic geometry, not natural population prevalence.

The unsupervised cluster metrics, UMAP/PCA plots, HDBSCAN sensitivity tables, eGeMAPS validity tables, and spectrogram metadata remain consistent with the current scripts.
