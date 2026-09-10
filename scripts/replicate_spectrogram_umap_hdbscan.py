#!/usr/bin/env python
"""Minimal spectrogram -> UMAP -> HDBSCAN replication protocol.

This is a sanity-check replication, not a parameter search. Its goal is to
verify that the data, labels, spectrogram preprocessing, UMAP, and clustering
pipeline do not have obvious errors.

Protocol:

    raw call audio -> paper-style log-mel spectrogram -> flatten
    -> UMAP 2D -> HDBSCAN -> compare clusters with manual labels

Evaluation is reported in four forms when both original and balanced manifests
are provided:

    A. original distribution + include HDBSCAN noise
    B. original distribution + exclude HDBSCAN noise
    C. balanced_600 + include HDBSCAN noise
    D. balanced_600 + exclude HDBSCAN noise
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import hdbscan
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


LABEL_CANDIDATES = ("label", "call_type", "manual_label", "type")
AUDIO_CANDIDATES = ("audio_path", "wav_path", "file_path", "path", "recording_path", "source_path", "source_wav")
START_CANDIDATES = ("start_s", "start_sec", "start_time_sec", "onset_s", "begin_s", "t_start", "start")
END_CANDIDATES = ("end_s", "end_sec", "end_time_sec", "offset_s", "stop_s", "t_end", "end")
DURATION_CANDIDATES = ("duration_s", "duration_sec", "duration", "dur_s")
H5_ROW_CANDIDATES = ("h5_index", "row_index", "sample_index", "call_index", "index")


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def require_column(df: pd.DataFrame, explicit: str | None, candidates: Iterable[str], name: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{name} column '{explicit}' not found in manifest")
        return explicit
    found = first_existing(df, candidates)
    if found is None:
        raise ValueError(f"Could not infer {name} column. Tried: {', '.join(candidates)}")
    return found


def infer_or_none(df: pd.DataFrame, explicit: str | None, candidates: Iterable[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"column '{explicit}' not found in manifest")
        return explicit
    return first_existing(df, candidates)


def resolve_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def load_audio_from_file(path: Path, sr: int, start: float | None, end: float | None) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if start is not None and end is not None and np.isfinite(start) and np.isfinite(end):
        i0 = max(0, int(round(start * sr)))
        i1 = min(len(y), int(round(end * sr)))
        y = y[i0:i1]
    return y.astype(np.float32, copy=False)


def find_h5_dataset(h5: h5py.File, requested: str | None) -> h5py.Dataset:
    if requested:
        return h5[requested]
    candidates = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (1, 2) and np.issubdtype(obj.dtype, np.number):
            candidates.append((name, obj))

    h5.visititems(visit)
    if not candidates:
        raise ValueError("Could not infer numeric audio dataset from h5 file")
    candidates.sort(key=lambda item: item[1].size, reverse=True)
    return candidates[0][1]


def load_audio_from_h5(
    h5_path: Path,
    dataset_name: str | None,
    row_index: int | None,
    sr: int,
    start: float | None,
    end: float | None,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        ds = find_h5_dataset(h5, dataset_name)
        if ds.ndim == 2:
            if row_index is None:
                raise ValueError("2D h5 audio dataset requires a row index column")
            y = np.asarray(ds[row_index], dtype=np.float32)
        else:
            y = np.asarray(ds[:], dtype=np.float32)
            if start is not None and end is not None:
                i0 = max(0, int(round(start * sr)))
                i1 = min(len(y), int(round(end * sr)))
                y = y[i0:i1]
    return y


def safe_audio(y: np.ndarray, sr: int, min_duration: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = y[np.isfinite(y)]
    min_len = max(1, int(round(min_duration * sr)))
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    return y


def logmel_paper_vector(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    frame_ms: float,
    hop_ms: float,
    max_duration: float,
) -> tuple[np.ndarray, dict[str, float]]:
    y = safe_audio(y, sr=sr, min_duration=frame_ms / 1000.0)
    n_fft = max(16, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    target_frames = int(math.ceil(max_duration * sr / hop))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=False,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    mu = float(np.mean(db))
    sd = float(np.std(db))
    z = (db - mu) / sd if sd > 1e-8 else db - mu

    original_frames = z.shape[1]
    if original_frames >= target_frames:
        z_fixed = z[:, :target_frames]
        truncated = True
    else:
        z_fixed = np.pad(z, ((0, 0), (0, target_frames - original_frames)), mode="constant")
        truncated = False

    info = {
        "audio_duration_s": float(len(y) / sr),
        "spectrogram_frames_original": int(original_frames),
        "spectrogram_frames_target": int(target_frames),
        "spectrogram_truncated": bool(truncated),
        "spectrogram_padding_fraction": float(max(0, target_frames - original_frames) / target_frames),
    }
    return z_fixed.reshape(-1).astype(np.float32), info


def extract_features(
    manifest: pd.DataFrame,
    manifest_path: Path,
    args: argparse.Namespace,
    dataset_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    label_col = require_column(manifest, args.label_col, LABEL_CANDIDATES, "label")
    audio_col = infer_or_none(manifest, args.audio_col, AUDIO_CANDIDATES)
    start_col = infer_or_none(manifest, args.start_col, START_CANDIDATES)
    end_col = infer_or_none(manifest, args.end_col, END_CANDIDATES)
    duration_col = infer_or_none(manifest, args.duration_col, DURATION_CANDIDATES)
    h5_row_col = infer_or_none(manifest, args.h5_row_col, H5_ROW_CANDIDATES)

    if audio_col is None and args.h5_file is None:
        raise ValueError("Need either an audio path column or --h5-file")

    rows = []
    vectors = []
    root = manifest_path.parent

    for i, row in manifest.reset_index(drop=True).iterrows():
        start = float(row[start_col]) if start_col else None
        end = float(row[end_col]) if end_col else None
        if end is None and duration_col and start is not None:
            end = start + float(row[duration_col])

        if audio_col is not None:
            audio_path = resolve_path(str(row[audio_col]), root=root)
            y = load_audio_from_file(audio_path, sr=args.sample_rate, start=start, end=end)
        else:
            h5_path = resolve_path(str(args.h5_file), root=Path.cwd())
            h5_row = int(row[h5_row_col]) if h5_row_col else None
            y = load_audio_from_h5(h5_path, args.h5_audio_dataset, h5_row, args.sample_rate, start, end)

        vec, info = logmel_paper_vector(
            y,
            sr=args.sample_rate,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=args.fmax,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            max_duration=args.max_duration,
        )
        vectors.append(vec)
        out_row = row.to_dict()
        out_row["dataset_name"] = dataset_name
        out_row["label"] = str(row[label_col])
        out_row.update(info)
        rows.append(out_row)

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"{dataset_name}: extracted {i + 1}/{len(manifest)}")

    return np.vstack(vectors), pd.DataFrame(rows)


def maybe_load_or_extract(manifest_path: Path, dataset_name: str, args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    feature_path = args.feature_dir / f"replication_logmel_paper_{dataset_name}.npz"
    meta_path = args.feature_dir / f"replication_logmel_paper_{dataset_name}_metadata.csv"
    if feature_path.exists() and meta_path.exists() and not args.force_extract:
        print(f"loading existing features for {dataset_name}: {feature_path}")
        return np.load(feature_path, allow_pickle=False)["X"], pd.read_csv(meta_path)

    manifest = pd.read_csv(manifest_path)
    if args.max_calls and len(manifest) > args.max_calls:
        manifest = manifest.sample(n=args.max_calls, random_state=args.seed).reset_index(drop=True)
    X, meta = extract_features(manifest, manifest_path=manifest_path, args=args, dataset_name=dataset_name)
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, X=X)
    meta.to_csv(meta_path, index=False)
    return X, meta


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2) or len(labels) <= len(unique):
        return np.nan
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return np.nan


def plot_umap(coords: np.ndarray, values: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    cats = list(pd.unique(values))
    cmap = plt.get_cmap("tab20")
    for i, cat in enumerate(cats):
        mask = values == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.75, color=cmap(i % 20), label=str(cat))
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def composition_tables(assign: np.ndarray, labels: np.ndarray, out_prefix: Path, include_noise: bool) -> None:
    tmp = pd.DataFrame({"hdbscan_cluster": assign, "label": labels})
    if not include_noise:
        tmp = tmp.loc[tmp["hdbscan_cluster"] != -1]
    counts = pd.crosstab(tmp["hdbscan_cluster"], tmp["label"])
    frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    counts.to_csv(out_prefix.with_name(out_prefix.name + "_composition_counts.csv"))
    frac.to_csv(out_prefix.with_name(out_prefix.name + "_composition_fraction.csv"))


def analyze_dataset(X: np.ndarray, meta: pd.DataFrame, dataset_name: str, args: argparse.Namespace) -> list[dict[str, float]]:
    labels = meta["label"].astype(str).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    )
    coords = reducer.fit_transform(Xs)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        cluster_selection_method=args.hdbscan_selection_method,
        metric="euclidean",
    )
    pred = clusterer.fit_predict(coords)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    plot_umap(coords, labels, f"{dataset_name}: UMAP colored by manual label", args.figure_dir / f"replication_{dataset_name}_umap_manual_label.png")
    plot_umap(coords, pred.astype(str), f"{dataset_name}: UMAP colored by HDBSCAN", args.figure_dir / f"replication_{dataset_name}_umap_hdbscan_cluster.png")

    rows = []
    for include_noise in (True, False):
        eval_name = "include_noise" if include_noise else "exclude_noise"
        mask = np.ones(len(labels), dtype=bool) if include_noise else pred != -1
        eval_pred = pred[mask]
        eval_labels = labels[mask]
        eval_X = Xs[mask]
        eval_coords = coords[mask]

        prefix = args.report_dir / f"replication_{dataset_name}_{eval_name}"
        composition_tables(pred, labels, prefix, include_noise=include_noise)

        clusters = [c for c in np.unique(eval_pred) if c != -1]
        rows.append(
            {
                "dataset_name": dataset_name,
                "evaluation_protocol": eval_name,
                "n_calls": int(len(eval_labels)),
                "n_labels": int(len(np.unique(eval_labels))),
                "n_clusters": int(len(clusters)),
                "noise_rate_full": float(np.mean(pred == -1)),
                "noise_rate_evaluated": float(np.mean(eval_pred == -1)),
                "ami": float(adjusted_mutual_info_score(eval_labels, eval_pred)) if len(eval_labels) else np.nan,
                "ari": float(adjusted_rand_score(eval_labels, eval_pred)) if len(eval_labels) else np.nan,
                "label_silhouette_original_feature_space": safe_silhouette(eval_X, eval_labels),
                "label_silhouette_umap_space": safe_silhouette(eval_coords, eval_labels),
                "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
                "hdbscan_min_samples": args.hdbscan_min_samples,
                "umap_neighbors": args.umap_neighbors,
                "umap_min_dist": args.umap_min_dist,
            }
        )

    coords_out = meta.copy()
    coords_out["umap_x"] = coords[:, 0]
    coords_out["umap_y"] = coords[:, 1]
    coords_out["hdbscan_cluster"] = pred
    coords_out["hdbscan_probability"] = clusterer.probabilities_
    coords_out.to_csv(args.report_dir / f"replication_{dataset_name}_umap_hdbscan_assignments.csv", index=False)
    return rows


def default_balanced_manifest() -> Path:
    candidates = [
        Path("data_manifest/balanced_600_manifest.csv"),
        Path("data_manifest/balanced_600.csv"),
        Path("data_manifest/pilot_balanced_600.csv"),
        Path("data_manifest/balanced_600_calls.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find balanced_600 manifest. Pass --balanced-manifest explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--balanced-manifest", type=Path, default=None)
    parser.add_argument("--datasets", default="original,balanced", help="original, balanced, or original,balanced")
    parser.add_argument("--feature-dir", type=Path, default=Path("features"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--audio-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--h5-file", type=Path, default=None)
    parser.add_argument("--h5-audio-dataset", default=None)
    parser.add_argument("--h5-row-col", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=4000.0)
    parser.add_argument("--frame-ms", type=float, default=30.0)
    parser.add_argument("--hop-ms", type=float, default=3.75)
    parser.add_argument("--max-duration", type=float, default=0.5)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument("--hdbscan-selection-method", default="eom")
    parser.add_argument("--max-calls", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    all_rows = []
    if "original" in selected:
        X, meta = maybe_load_or_extract(args.original_manifest, "original_distribution", args)
        all_rows.extend(analyze_dataset(X, meta, "original_distribution", args))
    if "balanced" in selected:
        balanced = args.balanced_manifest if args.balanced_manifest else default_balanced_manifest()
        X, meta = maybe_load_or_extract(balanced, "balanced_600", args)
        all_rows.extend(analyze_dataset(X, meta, "balanced_600", args))

    out = pd.DataFrame(all_rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.report_dir / "replication_spectrogram_umap_hdbscan_metrics.csv", index=False)

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["original_manifest"] = str(args.original_manifest)
    config["balanced_manifest"] = str(args.balanced_manifest) if args.balanced_manifest else None
    with open(args.report_dir / "replication_spectrogram_umap_hdbscan_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
