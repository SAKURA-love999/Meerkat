from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Any
import warnings

import numpy as np
import opensmile
import opensmile.core.smile as opensmile_smile
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")
OPENSMILE_CONFIG_CACHE = Path(r"C:\Users\ccccc\Documents\Codex\opensmile_config_cache")
EPS = 1e-10

warnings.filterwarnings(
    "ignore",
    message="Segment too short, filling with NaN.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not determine a version for module '__main__'.",
    category=RuntimeWarning,
)


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


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = values.astype(float, copy=False)
    return {
        "n_features": int(values.size),
        "n_nan": int(np.isnan(values).sum()),
        "n_inf": int(np.isinf(values).sum()),
        "n_finite": int(np.isfinite(values).sum()),
        "feature_min": float(np.nanmin(values)) if np.isfinite(values).any() else float("nan"),
        "feature_max": float(np.nanmax(values)) if np.isfinite(values).any() else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def ensure_ascii_opensmile_config() -> Path:
    source_root = Path(opensmile_smile.__file__).resolve().parent / "config"
    target_root = OPENSMILE_CONFIG_CACHE / "config"
    if not target_root.exists():
        target_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, target_root)
    return target_root


def make_smile() -> opensmile.Smile:
    config_root = ensure_ascii_opensmile_config()

    class AsciiConfigSmile(opensmile.Smile):
        @property
        def default_config_root(self) -> str:
            return str(config_root)

    return AsciiConfigSmile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps(smile: opensmile.Smile, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size == 0:
        raise ValueError("empty audio segment")
    features = smile.process_signal(audio, sampling_rate=sample_rate)
    if features.empty:
        raise ValueError("openSMILE returned no features")
    row = features.iloc[0]
    return {str(key): float(value) for key, value in row.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract eGeMAPSv02 Functionals for MeerKAT pilot calls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--target-dbfs", type=float, default=-25.0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.project_root / "data_manifest" / f"pilot_manifest_{args.pilot_name}.csv"
    manifest = pd.read_csv(manifest_path)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows)

    smile = make_smile()
    feature_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
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
            base = {column: row[column] for column in base_columns if column in row}
            base["amplitude_variant"] = variant
            base["representation"] = "egemaps"
            base["normalization_target_dbfs"] = args.target_dbfs if variant == "rms_normalized" else ""
            base["normalization_gain_db"] = norm_gain_db if variant == "rms_normalized" else 0.0
            base["normalization_peak_limited"] = norm_limited if variant == "rms_normalized" else False
            base["raw_rms_dbfs"] = db(rms(segment))
            base["segment_n_samples"] = int(segment.size)

            try:
                features = extract_egemaps(smile, audio, sample_rate)
                feature_values = np.asarray(list(features.values()), dtype=float)
                stats = finite_stats(feature_values)
                valid = stats["n_features"] > 0 and stats["n_nan"] == 0 and stats["n_inf"] == 0
                error = ""
            except Exception as exc:
                features = {}
                stats = {
                    "n_features": 0,
                    "n_nan": 0,
                    "n_inf": 0,
                    "n_finite": 0,
                    "feature_min": float("nan"),
                    "feature_max": float("nan"),
                }
                valid = False
                error = str(exc)

            feature_row = dict(base)
            feature_row.update(features)
            feature_rows.append(feature_row)

            qc_row = dict(base)
            qc_row.update(stats)
            qc_row["valid"] = valid
            qc_row["error"] = error
            qc_rows.append(qc_row)

        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"processed={index + 1}/{len(manifest)}", flush=True)

    feature_path = args.project_root / "features" / f"egemaps_features_{args.pilot_name}.csv"
    qc_path = args.project_root / "features" / f"egemaps_feature_validity_QC_{args.pilot_name}.csv"
    write_csv(feature_path, feature_rows)
    write_csv(qc_path, qc_rows)

    qc = pd.DataFrame(qc_rows)
    summary_path = args.project_root / "features" / f"egemaps_feature_validity_QC_{args.pilot_name}_summary.csv"
    summary = (
        qc.groupby(["label", "amplitude_variant"], dropna=False)
        .agg(
            n_rows=("call_id", "count"),
            valid_rows=("valid", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            mean_raw_rms_dbfs=("raw_rms_dbfs", "mean"),
            mean_n_nan=("n_nan", "mean"),
            mean_n_inf=("n_inf", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"features: {feature_path}")
    print(f"qc: {qc_path}")
    print(f"qc_summary: {summary_path}")


if __name__ == "__main__":
    main()
