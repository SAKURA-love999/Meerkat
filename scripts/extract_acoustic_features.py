from __future__ import annotations

import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")
EPS = 1e-10


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
    return float(20.0 * math.log10(max(value, EPS)))


def clip_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    if audio.size == 0:
        return float("nan")
    return float(np.mean(np.abs(audio) >= threshold))


def safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def spectral_entropy(power_spectrum: np.ndarray) -> float:
    power = np.asarray(power_spectrum, dtype=float)
    total = float(np.sum(power))
    if total <= EPS:
        return float("nan")
    probs = power / total
    entropy = -float(np.sum(probs * np.log2(probs + EPS)))
    return entropy / math.log2(max(len(probs), 2))


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


def extract_segment(row: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    audio, sample_rate = read_audio(str(row["source_wav"]))
    start = max(0, int(round(float(row["start_time_sec"]) * sample_rate)))
    end = min(len(audio), int(round(float(row["end_time_sec"]) * sample_rate)))
    segment = audio[start:end]

    context_margin = int(round(float(row.get("context_margin_sec", 0.25)) * sample_rate))
    context_start = max(0, start - context_margin)
    context_end = min(len(audio), end + context_margin)
    before = audio[context_start:start]
    after = audio[end:context_end]
    context = np.concatenate([before, after]) if before.size or after.size else np.array([], dtype=np.float32)
    return segment, context, sample_rate


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    prefix: str = "",
    include_pitch: bool = False,
) -> dict[str, float]:
    features: dict[str, float] = {}
    n_samples = int(audio.size)
    duration = n_samples / float(sample_rate)
    audio_rms = rms(audio)
    peak = float(np.max(np.abs(audio))) if n_samples else float("nan")
    features.update(
        {
            f"{prefix}n_samples": n_samples,
            f"{prefix}duration_sec_from_audio": duration,
            f"{prefix}rms": audio_rms,
            f"{prefix}rms_dbfs": db(audio_rms) if np.isfinite(audio_rms) else float("nan"),
            f"{prefix}peak_abs": peak,
            f"{prefix}crest_factor": peak / max(audio_rms, EPS) if np.isfinite(peak) else float("nan"),
            f"{prefix}clipping_fraction": clip_fraction(audio),
        }
    )
    if n_samples < 4 or audio_rms <= EPS:
        return features

    n_fft = min(512, max(64, 2 ** int(math.floor(math.log2(max(n_samples, 64))))))
    hop_length = max(16, n_fft // 4)
    win_length = min(n_fft, max(16, n_samples))

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length, center=True)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    n_mels = min(32, max(8, n_fft // 4))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True))
    mean_power = np.mean(np.square(stft), axis=1)

    features.update(safe_stats(zcr, f"{prefix}zcr"))
    features.update(safe_stats(centroid, f"{prefix}spectral_centroid_hz"))
    features.update(safe_stats(bandwidth, f"{prefix}spectral_bandwidth_hz"))
    features.update(safe_stats(rolloff, f"{prefix}spectral_rolloff_hz"))
    features.update(safe_stats(flatness, f"{prefix}spectral_flatness"))
    features[f"{prefix}spectral_entropy"] = spectral_entropy(mean_power)
    features[f"{prefix}n_fft"] = float(n_fft)
    features[f"{prefix}hop_length"] = float(hop_length)
    features[f"{prefix}n_mels"] = float(n_mels)

    for index in range(mfcc.shape[0]):
        coeff = mfcc[index]
        features.update(safe_stats(coeff, f"{prefix}mfcc{index + 1:02d}"))

    if include_pitch:
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=200,
                fmax=min(3500, sample_rate / 2 - 50),
                sr=sample_rate,
                frame_length=n_fft,
                hop_length=hop_length,
                center=True,
            )
            features.update(safe_stats(f0, f"{prefix}f0_hz"))
            features[f"{prefix}voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag.size else float("nan")
        except Exception:
            features.update(
                {
                    f"{prefix}f0_hz_mean": float("nan"),
                    f"{prefix}f0_hz_std": float("nan"),
                    f"{prefix}f0_hz_min": float("nan"),
                    f"{prefix}f0_hz_max": float("nan"),
                    f"{prefix}voiced_fraction": float("nan"),
                }
            )
    return features


def build_feature_rows(
    manifest: pd.DataFrame,
    target_dbfs: float,
    progress_every: int,
    include_pitch: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    metadata_columns = [
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
        base = {column: row[column] for column in metadata_columns if column in row}
        status = "ok"
        error = ""
        try:
            segment, context, sample_rate = extract_segment(row)
            raw_rms = rms(segment)
            context_rms = rms(context)
            snr_proxy_db = db(raw_rms) - db(context_rms) if np.isfinite(context_rms) and context_rms > EPS else float("nan")
            raw_features = extract_features(segment, sample_rate, include_pitch=include_pitch)
            norm_segment, norm_gain_db, norm_limited = normalize_rms(segment, target_dbfs=target_dbfs)
            norm_features = extract_features(norm_segment, sample_rate, include_pitch=include_pitch)

            for variant, features in (("raw", raw_features), ("rms_normalized", norm_features)):
                out = dict(base)
                out["amplitude_variant"] = variant
                out["normalization_target_dbfs"] = target_dbfs if variant == "rms_normalized" else ""
                out["normalization_gain_db"] = norm_gain_db if variant == "rms_normalized" else 0.0
                out["normalization_peak_limited"] = norm_limited if variant == "rms_normalized" else False
                out.update(features)
                feature_rows.append(out)

            values = np.array(list(raw_features.values()) + list(norm_features.values()), dtype=float)
            has_nan = bool(np.any(~np.isfinite(values)))
            qc_rows.append(
                {
                    **base,
                    "status": status,
                    "error": error,
                    "audio_n_samples": int(segment.size),
                    "audio_duration_sec": segment.size / sample_rate if sample_rate else float("nan"),
                    "raw_rms": raw_rms,
                    "raw_rms_dbfs": db(raw_rms) if np.isfinite(raw_rms) else float("nan"),
                    "context_rms": context_rms,
                    "snr_proxy_db": snr_proxy_db,
                    "raw_clipping_fraction": clip_fraction(segment),
                    "norm_gain_db": norm_gain_db,
                    "norm_peak_limited": norm_limited,
                    "has_nan_or_inf": has_nan,
                    "feature_valid": not has_nan,
                }
            )
        except Exception as exc:
            status = "error"
            error = repr(exc)
            qc_rows.append(
                {
                    **base,
                    "status": status,
                    "error": error,
                    "audio_n_samples": "",
                    "audio_duration_sec": "",
                    "raw_rms": "",
                    "raw_rms_dbfs": "",
                    "context_rms": "",
                    "snr_proxy_db": "",
                    "raw_clipping_fraction": "",
                    "norm_gain_db": "",
                    "norm_peak_limited": "",
                    "has_nan_or_inf": True,
                    "feature_valid": False,
                }
            )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"processed={index + 1}/{len(manifest)}", flush=True)
    return feature_rows, qc_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_qc(qc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(qc_rows)
    summaries: list[dict[str, Any]] = []
    for label, group in df.groupby("label", dropna=False):
        durations = pd.to_numeric(group["audio_duration_sec"], errors="coerce")
        rms_db = pd.to_numeric(group["raw_rms_dbfs"], errors="coerce")
        snr = pd.to_numeric(group["snr_proxy_db"], errors="coerce")
        clipping = pd.to_numeric(group["raw_clipping_fraction"], errors="coerce")
        summaries.append(
            {
                "label": label,
                "n_calls": len(group),
                "n_valid": int(group["feature_valid"].astype(bool).sum()),
                "duration_mean_sec": float(durations.mean()),
                "duration_median_sec": float(durations.median()),
                "duration_min_sec": float(durations.min()),
                "duration_max_sec": float(durations.max()),
                "raw_rms_dbfs_mean": float(rms_db.mean()),
                "raw_rms_dbfs_median": float(rms_db.median()),
                "snr_proxy_db_mean": float(snr.mean()),
                "snr_proxy_db_median": float(snr.median()),
                "clipping_fraction_max": float(clipping.max()),
                "n_nan_or_inf": int(group["has_nan_or_inf"].astype(bool).sum()),
                "n_errors": int((group["status"] != "ok").sum()),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract acoustic features for MeerKAT pilot manifests.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--target-dbfs", type=float, default=-25.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--include-pitch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.project_root / "data_manifest" / f"pilot_manifest_{args.pilot_name}.csv"
    manifest = pd.read_csv(manifest_path)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows)
    manifest["context_margin_sec"] = 0.25

    feature_rows, qc_rows = build_feature_rows(
        manifest=manifest,
        target_dbfs=args.target_dbfs,
        progress_every=args.progress_every,
        include_pitch=args.include_pitch,
    )

    feature_path = args.project_root / "features" / f"acoustic_features_{args.pilot_name}.csv"
    qc_path = args.project_root / "features" / f"feature_validity_QC_{args.pilot_name}.csv"
    qc_summary_path = args.project_root / "features" / f"feature_validity_QC_{args.pilot_name}_summary.csv"
    write_csv(feature_path, feature_rows)
    write_csv(qc_path, qc_rows)
    write_csv(qc_summary_path, summarize_qc(qc_rows))

    print(f"feature_rows: {len(feature_rows)}")
    print(f"qc_rows: {len(qc_rows)}")
    print(f"features: {feature_path}")
    print(f"qc: {qc_path}")
    print(f"qc_summary: {qc_summary_path}")


if __name__ == "__main__":
    main()
