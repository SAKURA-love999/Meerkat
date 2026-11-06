#!/usr/bin/env python
"""Create source/clip-normalized acoustic feature tables.

Purpose:
    Reduce recording-condition and distance-proxy effects that are shared
    within a 10-second source clip.

This is a sensitivity control, not a magic fix. If most source clips contain
only one selected focal call, within-source z-scoring will collapse many rows
to zero. The group-size diagnostics written by this script should always be
reported with the downstream M1 result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fit_meerkat_m1_acoustic_residual import DEFAULT_DROP_FEATURE_REGEX, first_existing


EPS = 1e-12


def acoustic_columns(df: pd.DataFrame, drop_regex: str) -> list[str]:
    drop_re = re.compile(drop_regex)
    return [
        c
        for c in df.columns
        if not drop_re.search(c)
        and c != "call_id"
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def normalize_group(g: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    out = g.copy()
    values = out[cols].astype(float)
    means = values.mean(axis=0, skipna=True)
    if method == "center":
        out.loc[:, cols] = values - means
        return out
    if method == "zscore":
        stds = values.std(axis=0, skipna=True, ddof=0)
        stds = stds.where(stds > EPS, np.nan)
        z = (values - means) / stds
        out.loc[:, cols] = z.fillna(0.0)
        return out
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["center", "zscore"], default="zscore")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--drop-feature-regex", default=DEFAULT_DROP_FEATURE_REGEX)
    parser.add_argument("--amplitude-variant", default=None)
    parser.add_argument("--write-diagnostics", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    group_col = args.group_col or first_existing(df, ["source_id", "clip_id", "file_id", "source_wav", "recording"])
    if args.amplitude_variant and "amplitude_variant" in df.columns:
        other = df.loc[~df["amplitude_variant"].astype(str).eq(args.amplitude_variant)].copy()
        work = df.loc[df["amplitude_variant"].astype(str).eq(args.amplitude_variant)].copy()
    else:
        other = df.iloc[0:0].copy()
        work = df.copy()

    cols = acoustic_columns(work, args.drop_feature_regex)
    if not cols:
        raise ValueError("No acoustic numeric columns selected.")

    group_sizes = work.groupby(group_col, dropna=False).size()
    normalized = (
        work.groupby(group_col, dropna=False, group_keys=False)
        .apply(lambda g: normalize_group(g, cols, args.method))
        .sort_index()
    )
    if not other.empty:
        out = pd.concat([normalized, other], ignore_index=True, sort=False)
    else:
        out = normalized

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    diag = {
        "features": str(args.features),
        "output": str(args.output),
        "method": args.method,
        "group_col": group_col,
        "amplitude_variant": args.amplitude_variant,
        "n_rows_normalized": int(len(work)),
        "n_groups": int(group_sizes.shape[0]),
        "n_acoustic_columns": int(len(cols)),
        "group_size_min": int(group_sizes.min()) if len(group_sizes) else 0,
        "group_size_median": float(group_sizes.median()) if len(group_sizes) else None,
        "group_size_mean": float(group_sizes.mean()) if len(group_sizes) else None,
        "group_size_max": int(group_sizes.max()) if len(group_sizes) else 0,
        "singleton_group_fraction": float((group_sizes == 1).mean()) if len(group_sizes) else None,
        "selected_columns": cols,
    }
    diag_path = args.write_diagnostics or args.output.with_suffix(".diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    print(json.dumps(diag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
