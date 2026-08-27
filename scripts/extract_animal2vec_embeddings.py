from __future__ import annotations

import argparse
import csv
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")
EPS = 1e-8


@lru_cache(maxsize=256)
def read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    mean = float(np.mean(audio)) if audio.size else 0.0
    std = float(np.std(audio)) if audio.size else 0.0
    if not np.isfinite(std) or std <= EPS:
        return np.zeros_like(audio, dtype=np.float32)
    return ((audio - mean) / std).astype(np.float32)


def extract_context_window(
    row: pd.Series,
    context_ms: float,
    anchor: str,
) -> tuple[np.ndarray, int, int, int, int, int]:
    audio, sample_rate = read_audio(str(row["source_wav"]))
    call_start = int(round(float(row["start_time_sec"]) * sample_rate))
    call_end = int(round(float(row["end_time_sec"]) * sample_rate))
    context_samples = int(round(context_ms * sample_rate / 1000.0))
    if context_samples <= 0:
        raise ValueError("context window must contain at least one sample")

    if anchor == "center":
        call_mid = (call_start + call_end) // 2
        context_start = call_mid - context_samples // 2
    elif anchor == "call_start":
        context_start = call_start
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    context_end = context_start + context_samples

    src_start = max(0, context_start)
    src_end = min(len(audio), context_end)
    left_pad = max(0, -context_start)
    right_pad = max(0, context_end - len(audio))
    window = audio[src_start:src_end]
    if left_pad or right_pad:
        window = np.pad(window, (left_pad, right_pad), mode="constant")

    focal_start_in_window = max(0, call_start - context_start)
    focal_end_in_window = min(context_samples, call_end - context_start)
    if focal_end_in_window <= focal_start_in_window:
        raise ValueError("focal call does not overlap context window")

    return (
        np.asarray(window, dtype=np.float32),
        sample_rate,
        focal_start_in_window,
        focal_end_in_window,
        left_pad,
        right_pad,
    )


def extract_source_window(row: pd.Series) -> tuple[np.ndarray, int, int, int, int, int]:
    audio, sample_rate = read_audio(str(row["source_wav"]))
    call_start = int(round(float(row["start_time_sec"]) * sample_rate))
    call_end = int(round(float(row["end_time_sec"]) * sample_rate))
    return np.asarray(audio, dtype=np.float32), sample_rate, call_start, call_end, 0, 0


def load_animal2vec_model(animal2vec_root: Path, checkpoint_path: Path, device: str):
    sys.path.insert(0, str(animal2vec_root))
    importlib.import_module("nn")
    import torch
    from fairseq import checkpoint_utils

    models, _ = checkpoint_utils.load_model_ensemble([str(checkpoint_path)])
    model = models[0].to(device)
    model.eval()
    return model, torch


def get_layer_tensor(layer_result: Any) -> Any:
    if isinstance(layer_result, tuple):
        return layer_result[0]
    return layer_result


def make_tensor(torch: Any, audio: np.ndarray, device: str) -> Any:
    tensor = torch.from_numpy(audio).float().to(device)
    if tensor.dim() == 1:
        tensor = tensor.view(1, -1)
    return tensor


def layer_tensor_to_time_channel(layer_result: Any) -> Any:
    layer_tensor = get_layer_tensor(layer_result)
    if layer_tensor.dim() == 2:
        return layer_tensor
    if layer_tensor.dim() != 3:
        raise ValueError(f"expected 2D or 3D layer tensor, got shape {tuple(layer_tensor.shape)}")
    if layer_tensor.shape[0] == 1:
        return layer_tensor[0, :, :]
    if layer_tensor.shape[1] == 1:
        return layer_tensor[:, 0, :]
    raise ValueError(f"cannot identify batch/time dimensions in shape {tuple(layer_tensor.shape)}")


def extract_frame_embeddings(
    model: Any,
    torch: Any,
    audio: np.ndarray,
    device: str,
    layer_index: int | None,
    average_top_k_layers: int,
) -> np.ndarray:
    tensor = make_tensor(torch, audio, device)
    with torch.inference_mode():
        try:
            output = model.extract_features(source=tensor.to(device))
        except Exception:
            output = model(source=tensor.to(device))
    layer_results = output.get("layer_results")
    if layer_results is None:
        raise KeyError("animal2vec model output does not contain layer_results")
    if layer_index is not None:
        selected = [layer_results[layer_index]]
    else:
        selected = layer_results[-average_top_k_layers:]
    tensors = [layer_tensor_to_time_channel(layer_result) for layer_result in selected]
    frame_embeddings = sum(tensors) / len(tensors)
    return frame_embeddings.detach().cpu().numpy().astype(np.float32)


def pool_focal_frames(
    frame_embeddings: np.ndarray,
    focal_start_sample: int,
    focal_end_sample: int,
    n_samples: int,
    pooling: str,
) -> tuple[np.ndarray, int, int]:
    n_frames = frame_embeddings.shape[0]
    start_frame = int(np.floor(focal_start_sample / max(n_samples, 1) * n_frames))
    end_frame = int(np.ceil(focal_end_sample / max(n_samples, 1) * n_frames))
    start_frame = max(0, min(n_frames - 1, start_frame))
    end_frame = max(start_frame + 1, min(n_frames, end_frame))
    focal_embeddings = frame_embeddings[start_frame:end_frame]
    if pooling == "mean":
        pooled = np.mean(focal_embeddings, axis=0)
    elif pooling == "max":
        pooled = np.max(focal_embeddings, axis=0)
    else:
        raise ValueError(f"unknown pooling: {pooling}")
    return pooled.astype(np.float32), start_frame, end_frame


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract animal2vec focal-frame pooled embeddings.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--animal2vec-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--context-ms", type=float, default=500.0)
    parser.add_argument("--input-mode", choices=["context_window", "source_file"], default="context_window")
    parser.add_argument("--anchor", choices=["center", "call_start"], default="center")
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--average-top-k-layers", type=int, default=12)
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.project_root / "data_manifest" / f"pilot_manifest_{args.pilot_name}.csv"
    manifest = pd.read_csv(manifest_path)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows)

    model, torch = load_animal2vec_model(args.animal2vec_root, args.checkpoint_path, args.device)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    call_ids: list[str] = []
    variants: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    base_columns = [
        "call_id",
        "source_id",
        "label",
        "label_category",
        "event_index",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "focal",
        "sample_rate",
    ]

    for index, row in manifest.iterrows():
        (
            window,
            sample_rate,
            focal_start,
            focal_end,
            left_pad,
            right_pad,
        ) = (
            extract_context_window(row, context_ms=args.context_ms, anchor=args.anchor)
            if args.input_mode == "context_window"
            else extract_source_window(row)
        )
        normalized_window = normalize_waveform(window)
        frame_embeddings = extract_frame_embeddings(
            model=model,
            torch=torch,
            audio=normalized_window,
            device=args.device,
            layer_index=args.layer_index,
            average_top_k_layers=args.average_top_k_layers,
        )
        pooled, start_frame, end_frame = pool_focal_frames(
            frame_embeddings=frame_embeddings,
            focal_start_sample=focal_start,
            focal_end_sample=focal_end,
            n_samples=len(normalized_window),
            pooling=args.pooling,
        )

        base = {column: row[column] for column in base_columns if column in row}
        base.update(
            {
                "representation": (
                    f"animal2vec_context{int(args.context_ms)}ms_focal_{args.pooling}"
                    if args.input_mode == "context_window"
                    else f"animal2vec_source10s_focal_{args.pooling}"
                ),
                "amplitude_variant": "zscore_context",
                "input_mode": args.input_mode,
                "context_ms": args.context_ms,
                "context_anchor": args.anchor,
                "context_n_samples": int(len(window)),
                "context_left_pad_samples": int(left_pad),
                "context_right_pad_samples": int(right_pad),
                "focal_start_sample_in_context": int(focal_start),
                "focal_end_sample_in_context": int(focal_end),
                "model_layer_index": args.layer_index if args.layer_index is not None else "",
                "average_top_k_layers": args.average_top_k_layers if args.layer_index is None else "",
                "model_total_frames": int(frame_embeddings.shape[0]),
                "focal_start_frame": int(start_frame),
                "focal_end_frame": int(end_frame),
                "focal_pooled_frames": int(end_frame - start_frame),
                "embedding_dim": int(pooled.size),
            }
        )
        metadata_rows.append(base)
        vectors.append(pooled)
        labels.append(str(row["label"]))
        call_ids.append(str(row["call_id"]))
        variants.append("zscore_context")

        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"processed={index + 1}/{len(manifest)}", flush=True)

    x = np.vstack(vectors).astype(np.float32)
    representation_name = (
        f"animal2vec_context{int(args.context_ms)}ms_focal_{args.pooling}"
        if args.input_mode == "context_window"
        else f"animal2vec_source10s_focal_{args.pooling}"
    )
    out_base = args.project_root / "features" / f"{representation_name}_{args.pilot_name}"
    np.savez(
        out_base.with_suffix(".npz"),
        X=x,
        labels=np.asarray(labels),
        call_ids=np.asarray(call_ids),
        amplitude_variants=np.asarray(variants),
        representation=np.asarray([representation_name]),
    )
    write_metadata(out_base.with_name(out_base.name + "_metadata.csv"), metadata_rows)
    print(f"vectors: {x.shape}")
    print(f"npz: {out_base.with_suffix('.npz')}")
    print(f"metadata: {out_base.with_name(out_base.name + '_metadata.csv')}")


if __name__ == "__main__":
    main()
