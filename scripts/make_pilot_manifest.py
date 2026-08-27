from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")
DEFAULT_LABELS = ("sn", "cc", "ld", "mo", "al", "soc", "agg", "oth")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def read_filtered_rows(
    manifest_path: Path,
    labels: set[str],
    min_duration: float,
    max_duration: float,
    focal_only: bool,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    n_seen = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            n_seen += 1
            label = row["label"]
            if label not in labels:
                continue
            if focal_only and not parse_bool(row["focal"]):
                continue
            try:
                duration = float(row["duration_sec"])
            except ValueError:
                continue
            if duration < min_duration or duration > max_duration:
                continue
            rows_by_label[label].append(row)
    return rows_by_label, n_seen


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows_by_label: dict[str, list[dict[str, str]]], selected: list[dict[str, str]]) -> None:
    selected_counts = defaultdict(int)
    selected_durations = defaultdict(list)
    for row in selected:
        label = row["label"]
        selected_counts[label] += 1
        selected_durations[label].append(float(row["duration_sec"]))

    summary_rows: list[dict[str, str | int | float]] = []
    for label in sorted(rows_by_label):
        durations = selected_durations[label]
        if durations:
            mean_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            mean_duration = ""
            min_duration = ""
            max_duration = ""
        summary_rows.append(
            {
                "label": label,
                "available_after_filters": len(rows_by_label[label]),
                "selected": selected_counts[label],
                "selected_duration_mean_sec": mean_duration,
                "selected_duration_min_sec": min_duration,
                "selected_duration_max_sec": max_duration,
            }
        )

    fieldnames = [
        "label",
        "available_after_filters",
        "selected",
        "selected_duration_mean_sec",
        "selected_duration_min_sec",
        "selected_duration_max_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class-balanced pilot manifests from MeerKAT calls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--max-per-label", type=int, default=600)
    parser.add_argument("--min-duration", type=float, default=0.01)
    parser.add_argument("--max-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--name", default="balanced_600")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.project_root / "data_manifest" / "call_manifest.csv"
    rows_by_label, n_seen = read_filtered_rows(
        manifest_path=manifest_path,
        labels=set(args.labels),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        focal_only=not args.include_background,
    )

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in sorted(args.labels):
        label_rows = list(rows_by_label.get(label, []))
        rng.shuffle(label_rows)
        selected.extend(label_rows[: args.max_per_label])
    rng.shuffle(selected)

    out_dir = args.project_root / "data_manifest"
    out_path = out_dir / f"pilot_manifest_{args.name}.csv"
    summary_path = out_dir / f"pilot_manifest_{args.name}_summary.csv"
    fieldnames = list(selected[0].keys()) if selected else []
    write_rows(out_path, selected, fieldnames)
    write_summary(summary_path, rows_by_label, selected)

    print(f"rows_seen: {n_seen}")
    print(f"selected: {len(selected)}")
    print(f"pilot_manifest: {out_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
