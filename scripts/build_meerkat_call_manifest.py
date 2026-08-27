from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASET_ROOT = Path(r"C:\学校\MeerKAT\MeerKAT_10s_2024-06-12\MeerKAT_10s_2024-06-12")
DEFAULT_PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")


def _as_list(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.shape == ():
        return [array.item()]
    return array.reshape(-1).tolist()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _read_dataset(handle: h5py.File, name: str) -> list[Any]:
    if name not in handle:
        return []
    return _as_list(handle[name][()])


def read_label_file(h5_path: Path, wav_path: Path, sample_rate: int) -> list[dict[str, Any]]:
    with h5py.File(h5_path, "r") as handle:
        starts = _read_dataset(handle, "start_time_lbl")
        ends = _read_dataset(handle, "end_time_lbl")
        start_frames = _read_dataset(handle, "start_frame_lbl")
        end_frames = _read_dataset(handle, "end_frame_lbl")
        labels = [_decode(x) for x in _read_dataset(handle, "lbl")]
        label_categories = [_decode(x) for x in _read_dataset(handle, "lbl_cat")]
        focal_values = _read_dataset(handle, "foc")

    event_count = max(
        len(starts),
        len(ends),
        len(start_frames),
        len(end_frames),
        len(labels),
        len(label_categories),
        len(focal_values),
    )
    rows: list[dict[str, Any]] = []
    for event_index in range(event_count):
        start_sec = float(starts[event_index]) if event_index < len(starts) else ""
        end_sec = float(ends[event_index]) if event_index < len(ends) else ""
        duration_sec = (
            max(0.0, end_sec - start_sec)
            if isinstance(start_sec, float) and isinstance(end_sec, float)
            else ""
        )
        focal_raw = focal_values[event_index] if event_index < len(focal_values) else ""
        rows.append(
            {
                "call_id": f"{h5_path.stem}__{event_index:03d}",
                "source_id": h5_path.stem,
                "source_wav": str(wav_path),
                "source_h5": str(h5_path),
                "event_index": event_index,
                "label": labels[event_index] if event_index < len(labels) else "",
                "label_category": label_categories[event_index]
                if event_index < len(label_categories)
                else "",
                "start_time_sec": start_sec,
                "end_time_sec": end_sec,
                "duration_sec": duration_sec,
                "start_frame": int(start_frames[event_index])
                if event_index < len(start_frames)
                else "",
                "end_frame": int(end_frames[event_index]) if event_index < len(end_frames) else "",
                "focal": bool(focal_raw) if focal_raw != "" else "",
                "sample_rate": sample_rate,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    dataset_root: Path,
    project_root: Path,
    sample_rate: int,
    limit_files: int | None,
    progress_every: int,
    skip_empty_size: int | None,
) -> tuple[Path, Path, Path, int, int]:
    label_dir = dataset_root / "lbl" / f"{sample_rate:05d}Hz"
    wav_dir = dataset_root / "wav" / f"{sample_rate:05d}Hz"
    out_dir = project_root / "data_manifest"
    manifest_path = out_dir / "call_manifest.csv"
    summary_path = out_dir / "label_summary.csv"
    duration_path = out_dir / "duration_summary.csv"

    fieldnames = [
        "call_id",
        "source_id",
        "source_wav",
        "source_h5",
        "event_index",
        "label",
        "label_category",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "start_frame",
        "end_frame",
        "focal",
        "sample_rate",
    ]

    label_files = sorted(label_dir.glob("*.h5"))
    if skip_empty_size is not None:
        label_files = [p for p in label_files if p.stat().st_size != skip_empty_size]
    if limit_files is not None:
        label_files = label_files[:limit_files]

    counts: Counter[tuple[str, str, str]] = Counter()
    durations: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    n_rows = 0
    missing_wavs = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for file_index, h5_path in enumerate(label_files, start=1):
            wav_path = wav_dir / f"{h5_path.stem}.wav"
            if not wav_path.exists():
                missing_wavs += 1
                continue
            event_rows = read_label_file(h5_path, wav_path, sample_rate)
            writer.writerows(event_rows)
            for row in event_rows:
                focal = str(row["focal"])
                label = str(row["label"])
                label_category = str(row["label_category"])
                counts[(label, label_category, focal)] += 1
                if isinstance(row["duration_sec"], float):
                    durations[(label, focal)].append(row["duration_sec"])
            n_rows += len(event_rows)
            if progress_every and file_index % progress_every == 0:
                print(
                    f"processed_files={file_index}/{len(label_files)} events={n_rows}",
                    flush=True,
                    file=sys.stderr,
                )

    summary_rows = [
        {
            "label": label,
            "label_category": label_category,
            "focal": focal,
            "n_calls": n_calls,
        }
        for (label, label_category, focal), n_calls in sorted(counts.items())
    ]
    write_csv(summary_path, summary_rows, ["label", "label_category", "focal", "n_calls"])

    duration_rows = []
    for (label, focal), values in sorted(durations.items()):
        arr = np.asarray(values, dtype=float)
        duration_rows.append(
            {
                "label": label,
                "focal": focal,
                "n_calls": int(arr.size),
                "duration_mean_sec": float(np.mean(arr)),
                "duration_sd_sec": float(np.std(arr)),
                "duration_min_sec": float(np.min(arr)),
                "duration_p25_sec": float(np.quantile(arr, 0.25)),
                "duration_median_sec": float(np.median(arr)),
                "duration_p75_sec": float(np.quantile(arr, 0.75)),
                "duration_max_sec": float(np.max(arr)),
            }
        )
    write_csv(
        duration_path,
        duration_rows,
        [
            "label",
            "focal",
            "n_calls",
            "duration_mean_sec",
            "duration_sd_sec",
            "duration_min_sec",
            "duration_p25_sec",
            "duration_median_sec",
            "duration_p75_sec",
            "duration_max_sec",
        ],
    )
    return manifest_path, summary_path, duration_path, n_rows, missing_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a call-level manifest for public MeerKAT.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--skip-empty-size",
        type=int,
        default=None,
        help="Skip HDF5 files with this byte size. In public MeerKAT, 3208-byte label files are empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path, summary_path, duration_path, n_rows, missing_wavs = build_manifest(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        sample_rate=args.sample_rate,
        limit_files=args.limit_files,
        progress_every=args.progress_every,
        skip_empty_size=args.skip_empty_size,
    )
    print(f"events: {n_rows}")
    print(f"missing wavs: {missing_wavs}")
    print(f"manifest: {manifest_path}")
    print(f"label summary: {summary_path}")
    print(f"duration summary: {duration_path}")


if __name__ == "__main__":
    main()
