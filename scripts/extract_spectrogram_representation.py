from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")
EPS = 1e-10
DEFAULT_SAMPLE_RATE = 8000


@lru_cache(maxsize=256)
def read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPS)))


def normalize_rms(audio: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float, bool]:
    current_rms = rms(audio)
    if not np.isfinite(current_rms) or current_rms <= EPS:
        return audio.copy(), 0.0, False
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / current_rms
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    limited = False
    if peak * gain > 0.99:
        gain = 0.99 / max(peak, EPS)
        limited = True
    return np.asarray(audio * gain, dtype=np.float32), db(gain), limited


def extract_segment(row: pd.Series) -> tuple[np.ndarray, int]:
    audio, sample_rate = read_audio(str(row["source_wav"]))
    start = max(0, int(round(float(row["start_time_sec"]) * sample_rate)))
    end = min(len(audio), int(round(float(row["end_time_sec"]) * sample_rate)))
    return audio[start:end], sample_rate


def resize_time_axis(spec: np.ndarray, target_frames: int) -> np.ndarray:
    if spec.shape[1] == target_frames:
        return spec
    if spec.shape[1] <= 1:
        return np.repeat(spec, target_frames, axis=1)
    old_x = np.linspace(0.0, 1.0, spec.shape[1], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    resized = np.empty((spec.shape[0], target_frames), dtype=np.float32)
    for mel_index in range(spec.shape[0]):
        resized[mel_index] = np.interp(new_x, old_x, spec[mel_index]).astype(np.float32)
    return resized


def zscore_spectrogram(spec: np.ndarray) -> np.ndarray:
    mean = float(np.mean(spec))
    std = float(np.std(spec))
    if not np.isfinite(std) or std <= EPS:
        return np.zeros_like(spec, dtype=np.float32)
    return ((spec - mean) / std).astype(np.float32)


def pad_time_axis(spec: np.ndarray, target_frames: int) -> tuple[np.ndarray, int]:
    if spec.shape[1] > target_frames:
        return spec[:, :target_frames].astype(np.float32), int(spec.shape[1] - target_frames)
    if spec.shape[1] == target_frames:
        return spec.astype(np.float32), 0
    padded = np.zeros((spec.shape[0], target_frames), dtype=np.float32)
    padded[:, : spec.shape[1]] = spec
    return padded, 0


def logmel_vector(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    representation_name: str,
    target_frames: int | None = None,
    max_duration_sec: float | None = None,
    per_spectrogram_zscore: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size == 0:
        raise ValueError("empty audio segment")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=min(4000, sample_rate / 2),
        power=2.0,
        center=True,
    )
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    spec = logmel.astype(np.float32)
    if per_spectrogram_zscore:
        spec = zscore_spectrogram(spec)

    truncated_frames = 0
    if representation_name == "logmel_call_resize":
        if target_frames is None:
            raise ValueError("target_frames is required for logmel_call_resize")
        spec = resize_time_axis(spec, target_frames=target_frames)
    elif representation_name == "logmel_paper_pad500":
        if max_duration_sec is None:
            raise ValueError("max_duration_sec is required for logmel_paper_pad500")
        max_samples = int(round(max_duration_sec * sample_rate))
        max_frames = int(np.ceil(max_samples / hop_length)) + 1
        spec, truncated_frames = pad_time_axis(spec, target_frames=max_frames)
    else:
        raise ValueError(f"Unknown representation_name: {representation_name}")

    vector = spec.reshape(-1).astype(np.float32)
    stats = {
        "orig_frames": int(logmel.shape[1]),
        "n_mels": n_mels,
        "target_frames": int(spec.shape[1]),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "max_duration_sec": max_duration_sec if max_duration_sec is not None else "",
        "per_spectrogram_zscore": per_spectrogram_zscore,
        "truncated_frames": truncated_frames,
        "logmel_min": float(np.min(logmel)),
        "logmel_max": float(np.max(logmel)),
        "logmel_mean": float(np.mean(logmel)),
        "logmel_std": float(np.std(logmel)),
        "vector_min": float(np.min(spec)),
        "vector_max": float(np.max(spec)),
        "vector_mean": float(np.mean(spec)),
        "vector_std": float(np.std(spec)),
    }
    return vector, stats


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
    parser = argparse.ArgumentParser(description="Extract fixed-size log-mel spectrogram representations.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--representation-name", choices=["logmel_call_resize", "logmel_paper_pad500"], default="logmel_call_resize")
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--frame-ms", type=float, default=30.0)
    parser.add_argument("--hop-ms", type=float, default=3.75)
    parser.add_argument("--max-duration-sec", type=float, default=0.5)
    parser.add_argument("--per-spectrogram-zscore", action="store_true")
    parser.add_argument("--target-dbfs", type=float, default=-25.0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.project_root / "data_manifest" / f"pilot_manifest_{args.pilot_name}.csv"
    manifest = pd.read_csv(manifest_path)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows)

    sample_rate_for_defaults = int(manifest["sample_rate"].dropna().iloc[0]) if "sample_rate" in manifest else DEFAULT_SAMPLE_RATE
    if args.representation_name == "logmel_call_resize":
        n_mels = args.n_mels or 64
        n_fft = args.n_fft or 256
        hop_length = args.hop_length or 16
        per_spectrogram_zscore = args.per_spectrogram_zscore
    else:
        n_mels = args.n_mels or 40
        n_fft = args.n_fft or max(1, int(round(args.frame_ms * sample_rate_for_defaults / 1000.0)))
        hop_length = args.hop_length or max(1, int(round(args.hop_ms * sample_rate_for_defaults / 1000.0)))
        per_spectrogram_zscore = True if not args.per_spectrogram_zscore else args.per_spectrogram_zscore

    vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    call_ids: list[str] = []
    variants: list[str] = []
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
        segment, sample_rate = extract_segment(row)
        norm_segment, norm_gain_db, norm_limited = normalize_rms(segment, args.target_dbfs)
        for variant, audio in (("raw", segment), ("rms_normalized", norm_segment)):
            vector, stats = logmel_vector(
                audio=audio,
                sample_rate=sample_rate,
                n_mels=n_mels,
                target_frames=args.target_frames,
                n_fft=n_fft,
                hop_length=hop_length,
                representation_name=args.representation_name,
                max_duration_sec=args.max_duration_sec,
                per_spectrogram_zscore=per_spectrogram_zscore,
            )
            base = {column: row[column] for column in base_columns if column in row}
            base.update(stats)
            base["amplitude_variant"] = variant
            base["representation"] = args.representation_name
            base["normalization_target_dbfs"] = args.target_dbfs if variant == "rms_normalized" else ""
            base["normalization_gain_db"] = norm_gain_db if variant == "rms_normalized" else 0.0
            base["normalization_peak_limited"] = norm_limited if variant == "rms_normalized" else False
            base["raw_rms_dbfs"] = db(rms(segment))
            base["vector_dim"] = int(vector.size)
            metadata_rows.append(base)
            vectors.append(vector)
            labels.append(str(row["label"]))
            call_ids.append(str(row["call_id"]))
            variants.append(variant)
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"processed={index + 1}/{len(manifest)}", flush=True)

    x = np.vstack(vectors).astype(np.float32)
    out_base = args.project_root / "features" / f"spectrogram_{args.representation_name}_{args.pilot_name}"
    np.savez(
        out_base.with_suffix(".npz"),
        X=x,
        labels=np.asarray(labels),
        call_ids=np.asarray(call_ids),
        amplitude_variants=np.asarray(variants),
        representation=np.asarray([args.representation_name]),
    )
    write_metadata(out_base.with_name(out_base.name + "_metadata.csv"), metadata_rows)
    print(f"vectors: {x.shape}")
    print(f"npz: {out_base.with_suffix('.npz')}")
    print(f"metadata: {out_base.with_name(out_base.name + '_metadata.csv')}")


if __name__ == "__main__":
    main()
