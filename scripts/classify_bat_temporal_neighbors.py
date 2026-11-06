#!/usr/bin/env python
"""Classify Egyptian fruit bat temporal-neighbor edges.

The input edge table is produced by audit_bat_interaction_mapping.py. Each row
links one voiced segment to the next voiced segment in the same file. This
script classifies whether that temporal neighbor stays in the same annotated
interaction, preserves emitter-addressee structure, preserves context, or crosses
interaction boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def clean_id(x: object) -> str:
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


def abs_id(x: object) -> str:
    val = clean_id(x)
    if val == "":
        return ""
    try:
        return str(abs(int(val)))
    except ValueError:
        return val.lstrip("-")


def add_neighbor_categories(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    out["both_unique"] = as_bool(out["both_unique"])
    out["same_interaction_interval"] = as_bool(out["same_interaction_interval"])

    for col in ["context_i", "context_j", "emitter_i", "emitter_j", "addressee_i", "addressee_j"]:
        out[col] = out[col].map(clean_id)

    out["context_i_label"] = out["context_i"].map(BAT_CONTEXT_LABELS).fillna(out["context_i"])
    out["context_j_label"] = out["context_j"].map(BAT_CONTEXT_LABELS).fillna(out["context_j"])
    out["same_context"] = out["context_i"].eq(out["context_j"])
    out["same_directed_pair"] = out["emitter_i"].eq(out["emitter_j"]) & out["addressee_i"].eq(out["addressee_j"])

    dyad_i = [
        tuple(sorted([abs_id(e), abs_id(a)]))
        for e, a in zip(out["emitter_i"], out["addressee_i"])
    ]
    dyad_j = [
        tuple(sorted([abs_id(e), abs_id(a)]))
        for e, a in zip(out["emitter_j"], out["addressee_j"])
    ]
    out["same_unordered_dyad"] = [a == b and "" not in a for a, b in zip(dyad_i, dyad_j)]

    pair_status = pd.Series(
        np.where(
        out["same_directed_pair"],
        "same_directed_pair",
        np.where(out["same_unordered_dyad"], "same_unordered_dyad", "different_pair"),
        ),
        index=out.index,
    )
    context_status = pd.Series(np.where(out["same_context"], "same_context", "different_context"), index=out.index)
    cross_category = "cross_interaction__" + pair_status + "__" + context_status
    out["neighbor_category"] = np.where(
        ~out["both_unique"],
        "mapping_unresolved",
        np.where(
            out["same_interaction_interval"],
            "same_interaction",
            cross_category,
        ),
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    out = (
        df.groupby("neighbor_category", dropna=False)
        .size()
        .rename("n_edges")
        .reset_index()
        .sort_values("n_edges", ascending=False)
    )
    out["fraction"] = out["n_edges"] / total if total else np.nan
    return out


def summarize_by_window(df: pd.DataFrame, windows_ms: list[float]) -> pd.DataFrame:
    rows = []
    for tau_ms in windows_ms:
        sub = df.loc[df["delta_t_s"] <= tau_ms / 1000.0]
        if sub.empty:
            continue
        counts = summarize(sub)
        counts.insert(0, "window_ms", tau_ms)
        counts.insert(1, "n_candidate_edges", len(sub))
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("reports/bat_proxy_audit/bat_next_voiced_segment_edges.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/bat_proxy_audit"),
    )
    parser.add_argument("--windows-ms", default="50,100,250,500,1000,2000")
    parser.add_argument("--write-classified-edges", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.edges)
    classified = add_neighbor_categories(edges)

    if args.write_classified_edges:
        classified.to_csv(args.output_dir / "bat_temporal_neighbor_classified_edges.csv", index=False)

    summary = summarize(classified)
    summary.to_csv(args.output_dir / "bat_temporal_neighbor_category_summary.csv", index=False)

    windows_ms = [float(x) for x in args.windows_ms.split(",") if x.strip()]
    by_window = summarize_by_window(classified, windows_ms)
    by_window.to_csv(args.output_dir / "bat_temporal_neighbor_category_by_window.csv", index=False)

    payload = {
        "n_edges": int(len(classified)),
        "n_both_unique": int(classified["both_unique"].sum()),
        "same_interaction_fraction": float(classified["same_interaction_interval"].mean()),
        "same_context_fraction": float(classified.loc[classified["both_unique"], "same_context"].mean()),
        "same_directed_pair_fraction": float(classified.loc[classified["both_unique"], "same_directed_pair"].mean()),
        "same_unordered_dyad_fraction": float(classified.loc[classified["both_unique"], "same_unordered_dyad"].mean()),
        "outputs": {
            "summary": str(args.output_dir / "bat_temporal_neighbor_category_summary.csv"),
            "by_window": str(args.output_dir / "bat_temporal_neighbor_category_by_window.csv"),
            "classified_edges": str(args.output_dir / "bat_temporal_neighbor_classified_edges.csv")
            if args.write_classified_edges
            else None,
        },
    }
    with open(args.output_dir / "bat_temporal_neighbor_category_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
