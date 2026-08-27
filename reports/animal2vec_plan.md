# animal2vec Embedding Plan

## Why animal2vec

animal2vec is a self-supervised transformer for sparse bioacoustic events. It was released with the public MeerKAT dataset and is therefore the most relevant neural representation baseline for this project.

The role of animal2vec in Experiment 1 is not to classify calls directly. Instead, it should provide a learned acoustic representation that can be compared with:

- librosa summary features
- paper-style log-mel spectrogram vectors
- eGeMAPS

## Required External Assets

The current project does not store model weights in Git.

To run animal2vec extraction, the machine needs:

- the official animal2vec repository
- a compatible Python environment, ideally Python 3.9
- fairseq, PyTorch, and the animal2vec dependencies
- a MeerKAT-pretrained or fine-tuned animal2vec `.pt` checkpoint

The current project environment uses Python 3.12, while the animal2vec repository warns that Python 3.10+ is not compatible with its fairseq dependency stack. For that reason, animal2vec should be run in a separate environment rather than installed into the existing `.venv`.

The official README example loads checkpoints with fairseq, imports `nn` from the repository root to register model/task objects, normalizes 10 s audio chunks, and calls the model as `model(source=single_chunk)`. The local extraction script follows that loading pattern, while adding focal-frame pooling for known annotated calls.

## Extraction Design

The implemented script is:

- `scripts/extract_animal2vec_embeddings.py`

It implements the intended context-aware design:

1. Read focal calls from `pilot_manifest_balanced_600.csv`.
2. Extract a real acoustic context window around the focal call.
3. Z-normalize the context waveform.
4. Feed the context waveform to animal2vec.
5. Read transformer `layer_results` from the animal2vec model output.
6. Average the top transformer layers by default.
7. Map the focal-call sample range onto model frames.
8. Pool only focal-call frames.
9. Save one embedding vector per focal call.

The script supports two input modes:

- `context_window`: use a fixed local context window, such as 500 ms.
- `source_file`: use the original 10 s MeerKAT source chunk, matching the official animal2vec example more closely, then pool only annotated focal-call frames.

The default representation is:

```text
animal2vec_context500ms_focal_mean
```

This keeps the context as model input while avoiding the eGeMAPS problem where the final feature vector summarizes the entire context window.

If the model expects 10 s inputs in practice, use:

```powershell
python scripts/extract_animal2vec_embeddings.py `
  --pilot-name balanced_600 `
  --animal2vec-root C:\path\to\animal2vec `
  --checkpoint-path C:\path\to\animal2vec_large_finetuned_MeerKAT_240507.pt `
  --input-mode source_file `
  --average-top-k-layers 12 `
  --pooling mean
```

## Example Command

```powershell
python scripts/extract_animal2vec_embeddings.py `
  --pilot-name balanced_600 `
  --animal2vec-root C:\path\to\animal2vec `
  --checkpoint-path C:\path\to\animal2vec_large_finetuned_MeerKAT_240507.pt `
  --context-ms 500 `
  --average-top-k-layers 12 `
  --pooling mean
```

Then analyze the resulting embeddings with:

```powershell
python scripts/analyze_embedding_pilot.py `
  --pilot-name balanced_600 `
  --representation-name animal2vec_context500ms_focal_mean `
  --npz-name animal2vec_context500ms_focal_mean_balanced_600.npz
```

## Planned Sensitivity Analyses

After the first 500 ms run:

- context window: 250 ms, 500 ms, 1000 ms
- layer index: last layer vs middle transformer layer
- pooling: mean vs max
- model checkpoint: pretrained vs MeerKAT-finetuned, if both are available

## Interpretation Rules

If animal2vec yields higher classifier performance but weak HDBSCAN/AMI/ARI, that supports categorical decodability without strong natural cluster discreteness.

If animal2vec yields stable label-aligned HDBSCAN clusters across context windows and layers, that would be stronger evidence that public labels align with learned acoustic structure.

If animal2vec shows a continuous UMAP/manifold with local label enrichment, that would support the current geometric-continuity framing and motivate explicit continuity metrics.

## References

- Schäfer-Zimmermann et al. animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics.
- Official code: https://github.com/livingingroups/animal2vec
