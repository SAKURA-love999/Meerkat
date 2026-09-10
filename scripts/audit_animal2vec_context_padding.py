#!/usr/bin/env python
"""Audit whether animal2vec UMAP islands are driven by context/padding factors.

This script uses the diagnostic sample table produced by
diagnose_animal2vec_umap_fuzzy.py. It does not recompute embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_PATTERNS = {
    "duration": re.compile(r"duration|dur|length", re.I),
    "frame": re.compile(r"frame|pooled|model_total|valid", re.I),
    "context": re.compile(r"context|window|pad|padding|left|right|start|end|offset|focal", re.I),
    "quality": re.compile(r"snr|rms|energy|loud|clip|quality|nan|finite", re.I),
    "recording": re.compile(r"recording|file|wav|audio|session|date|group|location|mic", re.I),
    "individual": re.compile(r"individual|meerkat|subject|animal|caller|identity", re.I),
}


def matching_columns(df: pd.DataFrame, groups: list[str]) -> list[str]:
    cols = []
    for col in df.columns:
        for group in groups:
            if CONTEXT_PATTERNS[group].search(col):
                cols.append(col)
                break
    return list(dict.fromkeys(cols))


def numeric_summary(df: pd.DataFrame, flag: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() == 0:
            continue
        island = x[flag]
        rest = x[~flag]
        sd = x.std(ddof=0)
        rows.append(
            {
                "variable": col,
                "left_island_mean": island.mean(),
                "rest_mean": rest.mean(),
                "left_island_median": island.median(),
                "rest_median": rest.median(),
                "left_island_p05": island.quantile(0.05),
                "left_island_p95": island.quantile(0.95),
                "rest_p05": rest.quantile(0.05),
                "rest_p95": rest.quantile(0.95),
                "left_island_missing_rate": island.isna().mean(),
                "rest_missing_rate": rest.isna().mean(),
                "standardized_mean_difference": (island.mean() - rest.mean()) / sd if sd and np.isfinite(sd) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_difference", key=lambda s: s.abs(), ascending=False)


def categorical_summary(df: pd.DataFrame, flag: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        x = df[col].astype(str).fillna("NA")
        island = x[flag]
        rest = x[~flag]
        if island.empty:
            continue
        island_top = island.value_counts()
        rest_top = rest.value_counts()
        rows.append(
            {
                "variable": col,
                "left_island_unique": int(island.nunique()),
                "rest_unique": int(rest.nunique()),
                "left_island_top_value": island_top.index[0],
                "left_island_top_fraction": float(island_top.iloc[0] / len(island)),
                "rest_top_value": rest_top.index[0],
                "rest_top_fraction": float(rest_top.iloc[0] / len(rest)),
                "left_island_top10": json.dumps(island_top.head(10).to_dict(), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("left_island_top_fraction", ascending=False)


def add_derived_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "focal_pooled_frames" in out.columns and "model_total_frames" in out.columns:
        focal = pd.to_numeric(out["focal_pooled_frames"], errors="coerce")
        total = pd.to_numeric(out["model_total_frames"], errors="coerce")
        out["derived_focal_frame_ratio"] = focal / total.replace(0, np.nan)
        out["derived_context_frame_ratio"] = 1.0 - out["derived_focal_frame_ratio"]
    if "duration" in out.columns:
        dur = pd.to_numeric(out["duration"], errors="coerce")
        out["derived_duration_lt_50ms"] = dur < 0.050
        out["derived_duration_lt_100ms"] = dur < 0.100
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-table", type=Path, default=Path("reports/animal2vec_fuzzy_membership_by_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--left-quantile", type=float, default=0.03)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sample_table)
    if "umap_x" not in df.columns:
        raise ValueError("sample table must contain umap_x; rerun diagnose_animal2vec_umap_fuzzy.py")

    df = add_derived_context_columns(df)
    left_cut = df["umap_x"].quantile(args.left_quantile)
    flag = df["umap_x"] <= left_cut
    df["left_umap_island"] = flag

    numeric_cols = matching_columns(df, ["duration", "frame", "context", "quality"])
    categorical_cols = matching_columns(df, ["recording", "individual"]) + ["label"]
    categorical_cols = [c for c in dict.fromkeys(categorical_cols) if c in df.columns]

    numeric_summary(df, flag, numeric_cols).to_csv(
        args.output_dir / "animal2vec_left_island_context_padding_numeric_audit.csv", index=False
    )
    categorical_summary(df, flag, categorical_cols).to_csv(
        args.output_dir / "animal2vec_left_island_context_padding_categorical_audit.csv", index=False
    )
    df.loc[flag].to_csv(args.output_dir / "animal2vec_left_island_context_padding_samples.csv", index=False)

    summary = {
        "sample_table": str(args.sample_table),
        "left_quantile": args.left_quantile,
        "left_umap_x_cutoff": float(left_cut),
        "n_left_island_samples": int(flag.sum()),
        "n_total_samples": int(len(df)),
        "numeric_columns_checked": numeric_cols,
        "categorical_columns_checked": categorical_cols,
    }
    with open(args.output_dir / "animal2vec_left_island_context_padding_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
