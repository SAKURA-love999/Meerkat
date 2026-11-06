#!/usr/bin/env python
"""Within-label permutation test for MeerKAT M1 acoustic residuals.

This script keeps the event table, manual labels, clip groups, durations,
boundary variables, temporal-history controls, and train/test grouping fixed.
It only shuffles whole acoustic feature vectors among calls with the same
manual label.

Interpretation:
    If the observed M1 improvement is not larger than this within-label null,
    the apparent acoustic gain should not be interpreted as evidence that
    continuous within-type acoustic realization carries residual future
    interaction information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_meerkat_m1_acoustic_residual import build_targets, evaluate_pair, load_feature_table


def contrast_from_overall(overall_df: pd.DataFrame, representation: str, permutation: int) -> pd.DataFrame:
    post = overall_df.loc[overall_df["target"] == "NF_post"].set_index("window_ms")
    pre = overall_df.loc[overall_df["target"] == "NF_pre"].set_index("window_ms")
    rows = []
    for tau_ms in sorted(set(post.index).intersection(pre.index)):
        rows.append(
            {
                "representation": representation,
                "permutation": permutation,
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


def contrast_from_per_label(per_label_df: pd.DataFrame, representation: str, permutation: int) -> pd.DataFrame:
    """Build per-label POST-minus-PRE contrasts from target-specific rows."""
    rows = []
    keys = ["window_ms", "manual_label"]
    post = per_label_df.loc[per_label_df["target"] == "NF_post"].set_index(keys)
    pre = per_label_df.loc[per_label_df["target"] == "NF_pre"].set_index(keys)
    for key in sorted(set(post.index).intersection(pre.index)):
        tau_ms, label = key
        rows.append(
            {
                "representation": representation,
                "permutation": permutation,
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


def run_m1(data: pd.DataFrame, acoustic_cols: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    per_label_tables = []
    for tau_ms, g in data.groupby("window_ms", sort=True):
        for target in ["NF_post", "NF_pre"]:
            overall, _, per_label = evaluate_pair(g, target, acoustic_cols, args)
            overall["window_ms"] = tau_ms
            overall["representation"] = args.representation_name
            overall["n_acoustic_features"] = len(acoustic_cols)
            per_label.insert(0, "window_ms", tau_ms)
            per_label.insert(0, "representation", args.representation_name)
            overall_rows.append(overall)
            per_label_tables.append(per_label)
    return pd.DataFrame(overall_rows), pd.concat(per_label_tables, ignore_index=True)


def shuffle_features_within_label(
    feature_table: pd.DataFrame,
    labels_by_call: pd.DataFrame,
    acoustic_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    work = feature_table.merge(labels_by_call, on="call_id", how="inner", validate="one_to_one")
    shuffled = work.copy()
    for _, idx in work.groupby("manual_label", sort=False).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) <= 1:
            continue
        permuted_idx = rng.permutation(idx)
        shuffled.loc[idx, acoustic_cols] = work.loc[permuted_idx, acoustic_cols].to_numpy()
    return shuffled[["call_id"] + acoustic_cols]


def summarize_null(observed: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, obs in observed.iterrows():
        tau = obs["window_ms"]
        null = null_df.loc[null_df["window_ms"] == tau]
        row = {
            "window_ms": tau,
            "n_permutations": int(len(null)),
        }
        for metric in [
            "delta_post_log_loss_reduction",
            "delta_pre_log_loss_reduction",
            "post_minus_pre_log_loss_reduction",
            "delta_post_roc_auc",
            "delta_pre_roc_auc",
            "post_minus_pre_roc_auc",
            "delta_post_average_precision",
            "delta_pre_average_precision",
            "post_minus_pre_average_precision",
        ]:
            obs_value = float(obs[metric])
            null_values = null[metric].to_numpy(float)
            row[f"observed_{metric}"] = obs_value
            row[f"null_mean_{metric}"] = float(np.nanmean(null_values))
            row[f"null_sd_{metric}"] = float(np.nanstd(null_values, ddof=1)) if len(null_values) > 1 else np.nan
            row[f"null_q025_{metric}"] = float(np.nanquantile(null_values, 0.025))
            row[f"null_q975_{metric}"] = float(np.nanquantile(null_values, 0.975))
            row[f"one_sided_p_ge_observed_{metric}"] = float((1.0 + np.sum(null_values >= obs_value)) / (len(null_values) + 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_null_by_keys(observed: pd.DataFrame, null_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Summarize a permutation null for rows identified by `keys`.

    This is used for per-label contrasts, where each row is a
    (window_ms, manual_label) result rather than only a window-level result.
    """
    rows = []
    metrics = [
        "delta_post_log_loss_reduction",
        "delta_pre_log_loss_reduction",
        "post_minus_pre_log_loss_reduction",
        "delta_post_roc_auc",
        "delta_pre_roc_auc",
        "post_minus_pre_roc_auc",
        "delta_post_average_precision",
        "delta_pre_average_precision",
        "post_minus_pre_average_precision",
    ]
    for _, obs in observed.iterrows():
        mask = pd.Series(True, index=null_df.index)
        row = {}
        for key in keys:
            row[key] = obs[key]
            mask &= null_df[key].eq(obs[key])
        null = null_df.loc[mask]
        row["n_permutations"] = int(null["permutation"].nunique())
        for optional in ["post_n", "pre_n", "post_positive_rate", "pre_positive_rate"]:
            if optional in obs:
                row[f"observed_{optional}"] = obs[optional]
        for metric in metrics:
            obs_value = float(obs[metric])
            null_values = null[metric].to_numpy(float)
            row[f"observed_{metric}"] = obs_value
            row[f"null_mean_{metric}"] = float(np.nanmean(null_values))
            row[f"null_sd_{metric}"] = float(np.nanstd(null_values, ddof=1)) if len(null_values) > 1 else np.nan
            row[f"null_q025_{metric}"] = float(np.nanquantile(null_values, 0.025))
            row[f"null_q975_{metric}"] = float(np.nanquantile(null_values, 0.975))
            row[f"one_sided_p_ge_observed_{metric}"] = float(
                (1.0 + np.sum(null_values >= obs_value)) / (len(null_values) + 1.0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--representation-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_m1_within_label_permutation"))
    parser.add_argument("--windows-ms", default="400,800,1500")
    parser.add_argument("--clip-duration-s", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=100)
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
    args = parser.parse_args()

    if args.drop_feature_regex is None:
        from fit_meerkat_m1_acoustic_residual import DEFAULT_DROP_FEATURE_REGEX

        args.drop_feature_regex = DEFAULT_DROP_FEATURE_REGEX

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    targets = build_targets(manifest, args)
    feature_table, acoustic_cols = load_feature_table(args)
    labels_by_call = targets.drop_duplicates("call_id")[["call_id", "manual_label"]]
    observed_data = targets.merge(feature_table, on="call_id", how="inner", validate="many_to_one")
    if observed_data.empty:
        raise ValueError("No focal calls matched between manifest targets and acoustic features.")

    observed_overall, observed_per_label = run_m1(observed_data, acoustic_cols, args)
    observed_contrast = contrast_from_overall(observed_overall, args.representation_name, permutation=-1)
    observed_per_label_contrast = contrast_from_per_label(
        observed_per_label,
        args.representation_name,
        permutation=-1,
    )

    rng = np.random.default_rng(args.random_state)
    null_tables = []
    per_label_null_tables = []
    for perm in range(args.n_permutations):
        shuffled_features = shuffle_features_within_label(feature_table, labels_by_call, acoustic_cols, rng)
        shuffled_data = targets.merge(shuffled_features, on="call_id", how="inner", validate="many_to_one")
        perm_overall, perm_per_label = run_m1(shuffled_data, acoustic_cols, args)
        null_tables.append(contrast_from_overall(perm_overall, args.representation_name, permutation=perm))
        per_label_null_tables.append(contrast_from_per_label(perm_per_label, args.representation_name, permutation=perm))
        if (perm + 1) % 10 == 0 or perm + 1 == args.n_permutations:
            print(f"finished permutation {perm + 1}/{args.n_permutations}", flush=True)

    null_df = pd.concat(null_tables, ignore_index=True)
    per_label_null_df = pd.concat(per_label_null_tables, ignore_index=True)
    null_summary = summarize_null(observed_contrast, null_df)
    per_label_null_summary = summarize_null_by_keys(
        observed_per_label_contrast,
        per_label_null_df,
        keys=["window_ms", "manual_label"],
    )

    stem = args.representation_name
    observed_overall.to_csv(args.output_dir / f"{stem}_observed_overall_metrics.csv", index=False)
    observed_per_label.to_csv(args.output_dir / f"{stem}_observed_per_label_delta.csv", index=False)
    observed_per_label_contrast.to_csv(args.output_dir / f"{stem}_observed_per_label_post_pre_contrast.csv", index=False)
    observed_contrast.to_csv(args.output_dir / f"{stem}_observed_post_pre_contrast.csv", index=False)
    null_df.to_csv(args.output_dir / f"{stem}_within_label_permutation_null.csv", index=False)
    null_summary.to_csv(args.output_dir / f"{stem}_within_label_permutation_summary.csv", index=False)
    per_label_null_df.to_csv(args.output_dir / f"{stem}_per_label_permutation_null.csv", index=False)
    per_label_null_summary.to_csv(args.output_dir / f"{stem}_per_label_permutation_summary.csv", index=False)

    summary = {
        "representation": args.representation_name,
        "n_permutations": args.n_permutations,
        "n_unique_matched_calls": int(observed_data["call_id"].nunique()),
        "n_acoustic_features": int(len(acoustic_cols)),
        "pca_dim": args.pca_dim,
        "temporal_control_mode": args.temporal_control_mode,
        "outputs": {
            "observed_contrast": str(args.output_dir / f"{stem}_observed_post_pre_contrast.csv"),
            "null": str(args.output_dir / f"{stem}_within_label_permutation_null.csv"),
            "null_summary": str(args.output_dir / f"{stem}_within_label_permutation_summary.csv"),
            "observed_per_label": str(args.output_dir / f"{stem}_observed_per_label_delta.csv"),
            "observed_per_label_contrast": str(args.output_dir / f"{stem}_observed_per_label_post_pre_contrast.csv"),
            "per_label_null": str(args.output_dir / f"{stem}_per_label_permutation_null.csv"),
            "per_label_null_summary": str(args.output_dir / f"{stem}_per_label_permutation_summary.csv"),
        },
    }
    with open(args.output_dir / f"{stem}_within_label_permutation_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
