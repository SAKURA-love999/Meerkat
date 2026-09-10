#!/usr/bin/env python
"""Collect HDBSCAN stability summaries into one CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, list):
            out[key] = ";".join(str(x) for x in value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/hdbscan_stability_summary_table.csv"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.reports_dir.glob("hdbscan_stability_*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = flatten(json.load(f))
        row["source_file"] = path.name
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No hdbscan_stability_*_summary.json files found in {args.reports_dir}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()
