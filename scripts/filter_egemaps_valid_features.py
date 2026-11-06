#!/usr/bin/env python
"""Filter eGeMAPS features to valid rows for a chosen amplitude variant.

openSMILE can return invalid/NaN feature rows for very short segments. For M1
residual acoustic analyses, using only valid rows is the cleaner primary
sensitivity because otherwise imputation can quietly encode extraction failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--call-id-col", default="call_id")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    qc = pd.read_csv(args.qc)
    if "valid" not in qc.columns:
        raise ValueError(f"QC file has no 'valid' column: {args.qc}")
    if "amplitude_variant" not in features.columns or "amplitude_variant" not in qc.columns:
        raise ValueError("Both feature and QC files must contain amplitude_variant.")

    qc_keep = qc.loc[
        qc["amplitude_variant"].astype(str).eq(args.variant) & truthy(qc["valid"]),
        [args.call_id_col, "amplitude_variant"],
    ].drop_duplicates()
    feat_keep = features.loc[features["amplitude_variant"].astype(str).eq(args.variant)].copy()
    out = feat_keep.merge(qc_keep, on=[args.call_id_col, "amplitude_variant"], how="inner", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = {
        "features": str(args.features),
        "qc": str(args.qc),
        "output": str(args.output),
        "variant": args.variant,
        "n_input_variant_rows": int(len(feat_keep)),
        "n_valid_output_rows": int(len(out)),
        "valid_fraction": float(len(out) / len(feat_keep)) if len(feat_keep) else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
