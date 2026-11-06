#!/usr/bin/env python
"""Add human-readable Egyptian fruit bat context labels to audit outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BAT_CONTEXT_LABELS = {
    "0": "Unknown",
    "1": "Separation",
    "2": "Biting",
    "3": "Feeding",
    "4": "Fighting",
    "5": "Grooming",
    "6": "Isolation",
    "7": "Kissing",
    "8": "Landing",
    "9": "Mating protest",
    "10": "Threat-like",
    "11": "General",
    "12": "Sleeping",
}


def add_context_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["context_label"] = out["context"].astype(str).map(BAT_CONTEXT_LABELS).fillna(out["context"])
    out.loc[out["context"].astype(str).eq("ALL"), "context_label"] = "ALL"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/bat_proxy_audit/bat_temporal_adjacency_reliability_curve.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/bat_proxy_audit/bat_temporal_adjacency_reliability_curve_labeled.csv"),
    )
    parser.add_argument(
        "--context-table",
        type=Path,
        default=Path("reports/bat_context_codebook.csv"),
    )
    args = parser.parse_args()

    curve = pd.read_csv(args.input)
    labeled = add_context_labels(curve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)

    table = pd.DataFrame(
        [{"context": int(code), "context_label": label} for code, label in BAT_CONTEXT_LABELS.items()]
    )
    args.context_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.context_table, index=False)
    print(f"wrote {args.output}")
    print(f"wrote {args.context_table}")


if __name__ == "__main__":
    main()

