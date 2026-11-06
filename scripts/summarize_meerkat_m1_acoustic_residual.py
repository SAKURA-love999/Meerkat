#!/usr/bin/env python
"""Summarize MeerKAT M1 acoustic-residual outputs across representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    headers = [str(c) for c in work.columns]
    rows = [[str(v) for v in row] for row in work.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("reports/meerkat_m1_acoustic_residual"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_m1_acoustic_residual"))
    parser.add_argument("--focus-labels", default="cc,sn")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contrast_files = sorted(args.input_dir.glob("*_m1_post_pre_contrast.csv"))
    overall_files = sorted(args.input_dir.glob("*_m1_overall_metrics.csv"))
    fold_files = sorted(args.input_dir.glob("*_m1_fold_metrics.csv"))
    per_label_files = sorted(args.input_dir.glob("*_m1_per_label_delta.csv"))

    if not contrast_files:
        raise FileNotFoundError(f"No *_m1_post_pre_contrast.csv files found in {args.input_dir}")

    contrasts = pd.concat([pd.read_csv(p) for p in contrast_files], ignore_index=True)
    overall = pd.concat([pd.read_csv(p) for p in overall_files], ignore_index=True) if overall_files else pd.DataFrame()
    folds = pd.concat([pd.read_csv(p) for p in fold_files], ignore_index=True) if fold_files else pd.DataFrame()
    per_label = pd.concat([pd.read_csv(p) for p in per_label_files], ignore_index=True) if per_label_files else pd.DataFrame()

    contrasts.to_csv(args.output_dir / "m1_all_representations_post_pre_contrast.csv", index=False)
    if not overall.empty:
        overall.to_csv(args.output_dir / "m1_all_representations_overall_metrics.csv", index=False)
    if not folds.empty:
        fold_summary = (
            folds.groupby(["representation", "window_ms", "target"], dropna=False)[
                ["delta_roc_auc", "delta_average_precision", "delta_log_loss_reduction"]
            ]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        fold_summary.columns = [
            "_".join(str(x) for x in col if str(x) != "")
            for col in fold_summary.columns.to_flat_index()
        ]
        fold_summary.to_csv(args.output_dir / "m1_all_representations_fold_stability_summary.csv", index=False)
    if not per_label.empty:
        focus = {x.strip() for x in args.focus_labels.split(",") if x.strip()}
        per_label.to_csv(args.output_dir / "m1_all_representations_per_label_delta.csv", index=False)
        per_label.loc[per_label["manual_label"].isin(focus)].to_csv(
            args.output_dir / "m1_focus_labels_cc_sn_delta.csv", index=False
        )

    lines = [
        "# M1 Acoustic Residual Combined Summary",
        "",
        "## Main Contrast",
        "",
        "Positive `post_minus_pre_log_loss_reduction` means the acoustic representation improves",
        "`NF_post` more than the PRE control.",
        "",
        df_to_markdown(contrasts[
            [
                "representation",
                "window_ms",
                "delta_post_log_loss_reduction",
                "delta_pre_log_loss_reduction",
                "post_minus_pre_log_loss_reduction",
                "delta_post_roc_auc",
                "delta_pre_roc_auc",
                "post_minus_pre_roc_auc",
            ]
        ]),
    ]
    if not per_label.empty:
        focus_table = per_label.loc[per_label["manual_label"].isin({"cc", "sn"})].copy()
        lines += [
            "",
            "## Focus Labels: cc / sn",
            "",
            df_to_markdown(focus_table[
                [
                    "representation",
                    "window_ms",
                    "target",
                    "manual_label",
                    "n",
                    "positive_rate",
                    "delta_roc_auc",
                    "delta_average_precision",
                    "delta_log_loss_reduction",
                ]
            ]),
        ]
    (args.output_dir / "m1_combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'm1_all_representations_post_pre_contrast.csv'}")
    print(f"wrote {args.output_dir / 'm1_combined_summary.md'}")


if __name__ == "__main__":
    main()
