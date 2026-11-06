#!/usr/bin/env python
"""Feature-mask ablation for MeerKAT M1 acoustic residual analysis.

The M1 question is whether acoustic realization adds information beyond
manual labels, QC/boundary variables, and temporal history. This script asks
which acoustic feature family carries that residual information.

It runs the same M0/M1 protocol repeatedly with different acoustic feature
subsets:
    - full acoustic set
    - remove energy/amplitude
    - remove MFCCs
    - remove spectral/ZCR
    - keep only energy/amplitude
    - keep only spectral/ZCR
    - keep only MFCCs

This is an observed ablation, not a permutation test. Use it after the
within-label permutation result identifies a label/window worth explaining
(currently sn at 400 ms).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fit_meerkat_m1_acoustic_residual import build_targets, evaluate_pair, load_feature_table


@dataclass(frozen=True)
class MaskSpec:
    name: str
    mode: str
    patterns: tuple[str, ...]
    description: str


MASK_SPECS = [
    MaskSpec(
        name="full",
        mode="all",
        patterns=(),
        description="All acoustic features selected by the base feature loader.",
    ),
    MaskSpec(
        name="full_no_analysis_params",
        mode="exclude",
        patterns=(r"^(n_fft|hop_length|n_mels)$",),
        description="Full acoustic set without constant feature-extraction parameter columns.",
    ),
    MaskSpec(
        name="remove_energy_amplitude",
        mode="exclude",
        patterns=(r"^(rms|rms_dbfs|peak_abs|crest_factor|clipping_fraction)$",),
        description="Remove amplitude, loudness, peak, crest, and clipping-related features.",
    ),
    MaskSpec(
        name="remove_mfcc",
        mode="exclude",
        patterns=(r"^mfcc\d+",),
        description="Remove MFCC features.",
    ),
    MaskSpec(
        name="remove_spectral_zcr",
        mode="exclude",
        patterns=(r"^spectral_", r"^zcr_"),
        description="Remove spectral-shape and zero-crossing features.",
    ),
    MaskSpec(
        name="remove_quality_like",
        mode="exclude",
        patterns=(r"^(clipping_fraction|crest_factor|peak_abs)$",),
        description="Remove simple quality/peak-shape features.",
    ),
    MaskSpec(
        name="keep_energy_amplitude_only",
        mode="include",
        patterns=(r"^(rms|rms_dbfs|peak_abs|crest_factor|clipping_fraction)$",),
        description="Use only amplitude/loudness/peak/clipping features.",
    ),
    MaskSpec(
        name="keep_spectral_zcr_only",
        mode="include",
        patterns=(r"^spectral_", r"^zcr_"),
        description="Use only spectral-shape and zero-crossing features.",
    ),
    MaskSpec(
        name="keep_mfcc_only",
        mode="include",
        patterns=(r"^mfcc\d+",),
        description="Use only MFCC features.",
    ),
]


def matches_any(column: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, column) for pattern in patterns)


def select_mask_columns(all_cols: list[str], spec: MaskSpec) -> list[str]:
    if spec.mode == "all":
        return list(all_cols)
    if spec.mode == "include":
        return [c for c in all_cols if matches_any(c, spec.patterns)]
    if spec.mode == "exclude":
        return [c for c in all_cols if not matches_any(c, spec.patterns)]
    raise ValueError(f"Unknown mask mode: {spec.mode}")


def contrast_from_overall(overall_df: pd.DataFrame, representation: str, mask_name: str) -> pd.DataFrame:
    post = overall_df.loc[overall_df["target"] == "NF_post"].set_index("window_ms")
    pre = overall_df.loc[overall_df["target"] == "NF_pre"].set_index("window_ms")
    rows = []
    for tau_ms in sorted(set(post.index).intersection(pre.index)):
        rows.append(
            {
                "representation": representation,
                "mask": mask_name,
                "window_ms": tau_ms,
                "delta_post_log_loss_reduction": float(post.loc[tau_ms, "delta_log_loss_reduction"]),
                "delta_pre_log_loss_reduction": float(pre.loc[tau_ms, "delta_log_loss_reduction"]),
                "post_minus_pre_log_loss_reduction": float(
                    post.loc[tau_ms, "delta_log_loss_reduction"] - pre.loc[tau_ms, "delta_log_loss_reduction"]
                ),
                "delta_post_roc_auc": float(post.loc[tau_ms, "delta_roc_auc"]),
                "delta_pre_roc_auc": float(pre.loc[tau_ms, "delta_roc_auc"]),
                "post_minus_pre_roc_auc": float(post.loc[tau_ms, "delta_roc_auc"] - pre.loc[tau_ms, "delta_roc_auc"]),
                "delta_post_average_precision": float(post.loc[tau_ms, "delta_average_precision"]),
                "delta_pre_average_precision": float(pre.loc[tau_ms, "delta_average_precision"]),
                "post_minus_pre_average_precision": float(
                    post.loc[tau_ms, "delta_average_precision"] - pre.loc[tau_ms, "delta_average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def contrast_from_per_label(per_label_df: pd.DataFrame, representation: str, mask_name: str) -> pd.DataFrame:
    rows = []
    keys = ["window_ms", "manual_label"]
    post = per_label_df.loc[per_label_df["target"] == "NF_post"].set_index(keys)
    pre = per_label_df.loc[per_label_df["target"] == "NF_pre"].set_index(keys)
    for key in sorted(set(post.index).intersection(pre.index)):
        tau_ms, label = key
        rows.append(
            {
                "representation": representation,
                "mask": mask_name,
                "window_ms": tau_ms,
                "manual_label": label,
                "post_n": int(post.loc[key, "n"]),
                "pre_n": int(pre.loc[key, "n"]),
                "post_positive_rate": float(post.loc[key, "positive_rate"]),
                "pre_positive_rate": float(pre.loc[key, "positive_rate"]),
                "delta_post_log_loss_reduction": float(post.loc[key, "delta_log_loss_reduction"]),
                "delta_pre_log_loss_reduction": float(pre.loc[key, "delta_log_loss_reduction"]),
                "post_minus_pre_log_loss_reduction": float(
                    post.loc[key, "delta_log_loss_reduction"] - pre.loc[key, "delta_log_loss_reduction"]
                ),
                "delta_post_roc_auc": float(post.loc[key, "delta_roc_auc"]),
                "delta_pre_roc_auc": float(pre.loc[key, "delta_roc_auc"]),
                "post_minus_pre_roc_auc": float(post.loc[key, "delta_roc_auc"] - pre.loc[key, "delta_roc_auc"]),
                "delta_post_average_precision": float(post.loc[key, "delta_average_precision"]),
                "delta_pre_average_precision": float(pre.loc[key, "delta_average_precision"]),
                "post_minus_pre_average_precision": float(
                    post.loc[key, "delta_average_precision"] - pre.loc[key, "delta_average_precision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_one_mask(
    data: pd.DataFrame,
    acoustic_cols: list[str],
    mask_name: str,
    representation: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_args = copy.copy(args)
    if local_args.pca_dim and local_args.pca_dim > 0:
        local_args.pca_dim = min(local_args.pca_dim, len(acoustic_cols))
    overall_rows = []
    fold_tables = []
    per_label_tables = []
    for tau_ms, g in data.groupby("window_ms", sort=True):
        for target in ["NF_post", "NF_pre"]:
            overall, folds, per_label = evaluate_pair(g, target, acoustic_cols, local_args)
            overall["window_ms"] = tau_ms
            overall["representation"] = representation
            overall["mask"] = mask_name
            overall["n_acoustic_features"] = len(acoustic_cols)
            folds.insert(0, "mask", mask_name)
            folds.insert(0, "window_ms", tau_ms)
            folds.insert(0, "representation", representation)
            per_label.insert(0, "mask", mask_name)
            per_label.insert(0, "window_ms", tau_ms)
            per_label.insert(0, "representation", representation)
            overall_rows.append(overall)
            fold_tables.append(folds)
            per_label_tables.append(per_label)

    overall_df = pd.DataFrame(overall_rows)
    per_label_df = pd.concat(per_label_tables, ignore_index=True)
    contrast_df = contrast_from_overall(overall_df, representation, mask_name)
    per_label_contrast_df = contrast_from_per_label(per_label_df, representation, mask_name)
    fold_df = pd.concat(fold_tables, ignore_index=True)
    return overall_df, fold_df, per_label_df, pd.concat([contrast_df, per_label_contrast_df], keys=["overall", "per_label"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--representation-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_m1_feature_mask_ablation"))
    parser.add_argument("--windows-ms", default="400,800,1500")
    parser.add_argument("--clip-duration-s", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--amplitude-variant", default=None)
    parser.add_argument(
        "--temporal-control-mode",
        choices=["none", "opposite_proxy", "symmetric_history"],
        default="symmetric_history",
    )
    parser.add_argument("--history-windows-ms", default="400,800,1500")
    parser.add_argument("--drop-feature-regex", default=None)
    parser.add_argument("--clip-col", default=None)
    parser.add_argument("--call-id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--focal-col", default=None)
    parser.add_argument("--feature-call-id-col", default=None)
    parser.add_argument(
        "--masks",
        default=",".join(spec.name for spec in MASK_SPECS),
        help="Comma-separated mask names. Default: all predefined masks.",
    )
    args = parser.parse_args()

    if args.drop_feature_regex is None:
        from fit_meerkat_m1_acoustic_residual import DEFAULT_DROP_FEATURE_REGEX

        args.drop_feature_regex = DEFAULT_DROP_FEATURE_REGEX

    mask_by_name = {spec.name: spec for spec in MASK_SPECS}
    requested_masks = [m.strip() for m in args.masks.split(",") if m.strip()]
    unknown = sorted(set(requested_masks) - set(mask_by_name))
    if unknown:
        raise ValueError(f"Unknown masks: {unknown}. Known masks: {sorted(mask_by_name)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    targets = build_targets(manifest, args)
    feature_table, all_acoustic_cols = load_feature_table(args)
    data = targets.merge(feature_table, on="call_id", how="inner", validate="many_to_one")
    if data.empty:
        raise ValueError("No focal calls matched between manifest targets and acoustic features.")

    all_overall = []
    all_folds = []
    all_per_label = []
    all_overall_contrasts = []
    all_per_label_contrasts = []
    mask_rows = []

    for mask_name in requested_masks:
        spec = mask_by_name[mask_name]
        cols = select_mask_columns(all_acoustic_cols, spec)
        if not cols:
            raise ValueError(f"Mask {mask_name} selected zero acoustic columns.")
        print(f"running mask={mask_name} n_features={len(cols)}", flush=True)
        overall, folds, per_label, joined_contrast = run_one_mask(
            data,
            cols,
            mask_name,
            args.representation_name,
            args,
        )
        overall_contrast = joined_contrast.loc["overall"].reset_index(drop=True)
        per_label_contrast = joined_contrast.loc["per_label"].reset_index(drop=True)
        all_overall.append(overall)
        all_folds.append(folds)
        all_per_label.append(per_label)
        all_overall_contrasts.append(overall_contrast)
        all_per_label_contrasts.append(per_label_contrast)
        mask_rows.append(
            {
                "mask": mask_name,
                "mode": spec.mode,
                "n_features": len(cols),
                "description": spec.description,
                "features": ";".join(cols),
            }
        )

    stem = args.representation_name
    pd.DataFrame(mask_rows).to_csv(args.output_dir / f"{stem}_mask_definitions.csv", index=False)
    pd.concat(all_overall, ignore_index=True).to_csv(args.output_dir / f"{stem}_mask_overall_metrics.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(args.output_dir / f"{stem}_mask_fold_metrics.csv", index=False)
    pd.concat(all_per_label, ignore_index=True).to_csv(args.output_dir / f"{stem}_mask_per_label_delta.csv", index=False)
    pd.concat(all_overall_contrasts, ignore_index=True).to_csv(
        args.output_dir / f"{stem}_mask_post_pre_contrast.csv",
        index=False,
    )
    pd.concat(all_per_label_contrasts, ignore_index=True).to_csv(
        args.output_dir / f"{stem}_mask_per_label_post_pre_contrast.csv",
        index=False,
    )

    summary = {
        "representation": args.representation_name,
        "manifest": str(args.manifest),
        "feature_csv": str(args.feature_csv),
        "n_unique_matched_calls": int(data["call_id"].nunique()),
        "n_all_acoustic_features": int(len(all_acoustic_cols)),
        "pca_dim": args.pca_dim,
        "temporal_control_mode": args.temporal_control_mode,
        "masks": requested_masks,
        "outputs": {
            "mask_definitions": str(args.output_dir / f"{stem}_mask_definitions.csv"),
            "overall_contrast": str(args.output_dir / f"{stem}_mask_post_pre_contrast.csv"),
            "per_label_contrast": str(args.output_dir / f"{stem}_mask_per_label_post_pre_contrast.csv"),
        },
    }
    with open(args.output_dir / f"{stem}_mask_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
