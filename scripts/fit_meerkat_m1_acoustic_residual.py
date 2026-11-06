#!/usr/bin/env python
"""Fit M1 acoustic-residual baselines for MeerKAT PRE/POST proxies.

Question:
    Given manual call type, duration, boundary/QC variables, and a simple
    temporal-neighborhood control, does focal-call acoustic realization add
    residual information about clip-local non-focal vocal activity?

This script compares:
    M0: target ~ manual label + duration + boundary/QC (+ temporal control)
    M1: target ~ M0 + acoustic representation

It reports Delta_post and Delta_pre. Positive log-loss delta means M1 improves
over M0. A post-specific interpretation requires Delta_post > Delta_pre, not
merely Delta_post > 0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DROP_FEATURE_REGEX = (
    r"^(call_id|source_id|source_wav|source_h5|label|label_category|event_index|"
    r"start_time_sec|end_time_sec|duration_sec|duration_s|focal|sample_rate|"
    r"amplitude_variant|representation|normalization_.*|segment_n_samples|"
    r"n_samples|duration_sec_from_audio)$"
)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Could not find any of {candidates}. Columns: {list(df.columns)}")


def add_symmetric_history_features(
    base: pd.DataFrame,
    work: pd.DataFrame,
    history_windows_ms: list[float],
) -> pd.DataFrame:
    """Add clip-local temporal-history features around each focal call.

    `pre_*` features only use events ending before focal onset. `post_*` features
    only use events starting after focal offset. During model fitting,
    `NF_post` uses the pre-history side and `NF_pre` uses the post-history side.
    This keeps the PRE control symmetric without leaking the target itself.
    """
    out = base.copy()
    for prefix in ["pre", "post"]:
        out[f"{prefix}_time_to_nearest_event_s"] = np.nan
        out[f"{prefix}_time_to_nearest_nonfocal_s"] = np.nan
        out[f"{prefix}_nearest_label_same_as_focal"] = 0.0
        for hist_ms in history_windows_ms:
            stem = f"{prefix}_{int(hist_ms)}ms"
            out[f"{stem}_all_count"] = 0.0
            out[f"{stem}_focal_count"] = 0.0
            out[f"{stem}_nonfocal_count"] = 0.0
            out[f"{stem}_same_label_count"] = 0.0
            out[f"{stem}_distinct_label_count"] = 0.0

    work_sorted = work.sort_values(["_clip", "_start", "_end"]).reset_index(drop=True)
    for _, events in work_sorted.groupby("_clip", sort=False):
        idx = out.index[out["_clip"].eq(events["_clip"].iloc[0])]
        if len(idx) == 0:
            continue
        focal_rows = out.loc[idx]
        starts = events["_start"].to_numpy(float)
        ends = events["_end"].to_numpy(float)
        is_focal = events["_focal"].to_numpy(bool)
        labels = events["manual_label"].astype(str).to_numpy()

        end_order = np.argsort(ends)
        ends_sorted = ends[end_order]
        end_labels = labels[end_order]
        end_focal = is_focal[end_order]

        start_order = np.argsort(starts)
        starts_sorted = starts[start_order]
        start_labels = labels[start_order]
        start_focal = is_focal[start_order]

        for row_index, row in focal_rows.iterrows():
            focal_start = float(row["_start"])
            focal_end = float(row["_end"])
            focal_label = str(row["manual_label"])

            pre_right = np.searchsorted(ends_sorted, focal_start, side="right")
            if pre_right > 0:
                nearest_pos = pre_right - 1
                out.at[row_index, "pre_time_to_nearest_event_s"] = focal_start - ends_sorted[nearest_pos]
                out.at[row_index, "pre_nearest_label_same_as_focal"] = float(end_labels[nearest_pos] == focal_label)
                nf_positions = np.flatnonzero(~end_focal[:pre_right])
                if nf_positions.size:
                    nf_pos = nf_positions[-1]
                    out.at[row_index, "pre_time_to_nearest_nonfocal_s"] = focal_start - ends_sorted[nf_pos]

            post_left = np.searchsorted(starts_sorted, focal_end, side="left")
            if post_left < len(starts_sorted):
                nearest_pos = post_left
                out.at[row_index, "post_time_to_nearest_event_s"] = starts_sorted[nearest_pos] - focal_end
                out.at[row_index, "post_nearest_label_same_as_focal"] = float(start_labels[nearest_pos] == focal_label)
                nf_positions = np.flatnonzero(~start_focal[post_left:])
                if nf_positions.size:
                    nf_pos = post_left + nf_positions[0]
                    out.at[row_index, "post_time_to_nearest_nonfocal_s"] = starts_sorted[nf_pos] - focal_end

            for hist_ms in history_windows_ms:
                tau = hist_ms / 1000.0
                pre_left = np.searchsorted(ends_sorted, focal_start - tau, side="left")
                pre_slice = slice(pre_left, pre_right)
                pre_labels = end_labels[pre_slice]
                pre_focal = end_focal[pre_slice]
                pre_stem = f"pre_{int(hist_ms)}ms"
                out.at[row_index, f"{pre_stem}_all_count"] = float(len(pre_labels))
                out.at[row_index, f"{pre_stem}_focal_count"] = float(np.sum(pre_focal))
                out.at[row_index, f"{pre_stem}_nonfocal_count"] = float(np.sum(~pre_focal))
                out.at[row_index, f"{pre_stem}_same_label_count"] = float(np.sum(pre_labels == focal_label))
                out.at[row_index, f"{pre_stem}_distinct_label_count"] = float(len(set(pre_labels.tolist())))

                post_right = np.searchsorted(starts_sorted, focal_end + tau, side="right")
                post_slice = slice(post_left, post_right)
                post_labels = start_labels[post_slice]
                post_focal = start_focal[post_slice]
                post_stem = f"post_{int(hist_ms)}ms"
                out.at[row_index, f"{post_stem}_all_count"] = float(len(post_labels))
                out.at[row_index, f"{post_stem}_focal_count"] = float(np.sum(post_focal))
                out.at[row_index, f"{post_stem}_nonfocal_count"] = float(np.sum(~post_focal))
                out.at[row_index, f"{post_stem}_same_label_count"] = float(np.sum(post_labels == focal_label))
                out.at[row_index, f"{post_stem}_distinct_label_count"] = float(len(set(post_labels.tolist())))
    return out


def build_targets(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    clip_col = args.clip_col or first_existing(df, ["source_id", "clip_id", "file_id", "source_wav", "recording"])
    call_col = args.call_id_col or first_existing(df, ["call_id"])
    label_col = args.label_col or first_existing(df, ["label", "call_type", "manual_label", "type"])
    start_col = args.start_col or first_existing(df, ["start_time_sec", "start_s", "start_sec", "onset_s", "start"])
    end_col = args.end_col or first_existing(df, ["end_time_sec", "end_s", "end_sec", "offset_s", "end"])
    duration_col = args.duration_col or first_existing(df, ["duration_sec", "duration_s", "duration"])
    focal_col = args.focal_col or first_existing(df, ["focal", "is_focal"])

    work = pd.DataFrame(
        {
            "call_id": df[call_col].astype(str),
            "_clip": df[clip_col].astype(str),
            "manual_label": df[label_col].astype(str),
            "_start": pd.to_numeric(df[start_col], errors="coerce"),
            "_end": pd.to_numeric(df[end_col], errors="coerce"),
            "duration_s": pd.to_numeric(df[duration_col], errors="coerce"),
            "_focal": bool_series(df[focal_col]),
        }
    ).dropna(subset=["_start", "_end", "duration_s"])

    focal = work.loc[work["_focal"], ["call_id", "_clip", "manual_label", "_start", "_end", "duration_s"]].copy()
    focal["row_index"] = focal.index
    nonfocal = work.loc[~work["_focal"], ["_clip", "_start", "_end"]].copy()

    left_post = focal.sort_values(["_end", "_clip"]).reset_index(drop=True)
    right_post = (
        nonfocal[["_clip", "_start"]]
        .rename(columns={"_start": "next_nonfocal_start"})
        .sort_values(["next_nonfocal_start", "_clip"])
        .reset_index(drop=True)
    )
    post = pd.merge_asof(
        left_post,
        right_post,
        by="_clip",
        left_on="_end",
        right_on="next_nonfocal_start",
        direction="forward",
        allow_exact_matches=True,
    )

    left_pre = focal.sort_values(["_start", "_clip"]).reset_index(drop=True)
    right_pre = (
        nonfocal[["_clip", "_end"]]
        .rename(columns={"_end": "prev_nonfocal_end"})
        .sort_values(["prev_nonfocal_end", "_clip"])
        .reset_index(drop=True)
    )
    pre = pd.merge_asof(
        left_pre,
        right_pre,
        by="_clip",
        left_on="_start",
        right_on="prev_nonfocal_end",
        direction="backward",
        allow_exact_matches=True,
    )

    base = post[
        [
            "call_id",
            "_clip",
            "manual_label",
            "_start",
            "_end",
            "duration_s",
            "row_index",
            "next_nonfocal_start",
        ]
    ].merge(pre[["row_index", "prev_nonfocal_end"]], on="row_index", how="left")
    base["log_duration_s"] = np.log1p(base["duration_s"].clip(lower=0))
    base["clip_start_margin_s"] = base["_start"]
    base["clip_end_margin_s"] = args.clip_duration_s - base["_end"]
    base["clip_center_s"] = (base["_start"] + base["_end"]) / 2.0
    history_windows_ms = [float(x) for x in args.history_windows_ms.split(",") if x.strip()]
    if args.temporal_control_mode == "symmetric_history":
        base = add_symmetric_history_features(base, work, history_windows_ms)

    rows = []
    for tau_ms in [float(x) for x in args.windows_ms.split(",") if x.strip()]:
        tau = tau_ms / 1000.0
        tmp = base.copy()
        tmp["window_ms"] = tau_ms
        tmp["post_censored"] = tmp["_end"] > (args.clip_duration_s - tau)
        tmp["pre_censored"] = tmp["_start"] < tau
        tmp["NF_post"] = (~tmp["post_censored"]) & tmp["next_nonfocal_start"].notna() & (
            tmp["next_nonfocal_start"] <= tmp["_end"] + tau
        )
        tmp["NF_pre"] = (~tmp["pre_censored"]) & tmp["prev_nonfocal_end"].notna() & (
            tmp["prev_nonfocal_end"] >= tmp["_start"] - tau
        )
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_feature_table(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    if args.feature_csv:
        feat = pd.read_csv(args.feature_csv)
        call_col = args.feature_call_id_col or first_existing(feat, ["call_id"])
        feat = feat.rename(columns={call_col: "call_id"})
        drop_re = re.compile(args.drop_feature_regex)
        numeric_cols = [
            c
            for c in feat.columns
            if c != "call_id" and not drop_re.search(c) and pd.api.types.is_numeric_dtype(feat[c])
        ]
        if args.amplitude_variant and "amplitude_variant" in feat.columns:
            feat = feat.loc[feat["amplitude_variant"].astype(str) == args.amplitude_variant].copy()
        return feat[["call_id"] + numeric_cols], numeric_cols

    if args.feature_npz:
        arr = np.load(args.feature_npz, allow_pickle=False)
        X = arr["X"]
        call_ids = arr["call_ids"].astype(str)
        if args.amplitude_variant and "amplitude_variants" in arr.files:
            keep = arr["amplitude_variants"].astype(str) == args.amplitude_variant
            X = X[keep]
            call_ids = call_ids[keep]
        cols = [f"acoustic_{i:04d}" for i in range(X.shape[1])]
        feat = pd.DataFrame(X, columns=cols)
        feat.insert(0, "call_id", call_ids)
        return feat, cols

    raise ValueError("Provide either --feature-csv or --feature-npz.")


def make_pipeline(
    numeric_cols: list[str],
    acoustic_cols: list[str],
    include_acoustic: bool,
    pca_dim: int | None,
    random_state: int,
) -> Pipeline:
    categorical = ["manual_label"]

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
    ]
    if include_acoustic:
        acoustic_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if pca_dim and pca_dim > 0:
            acoustic_steps.append(("pca", PCA(n_components=pca_dim, random_state=random_state)))
        transformers.append(("acoustic", Pipeline(steps=acoustic_steps), acoustic_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=args_class_weight_placeholder(),
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def args_class_weight_placeholder():
    # Kept as a function so the model definition stays pickle-safe and easy to patch.
    return None


def score_predictions(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, pred)),
        "log_loss": float(log_loss(y, pred, labels=[0, 1])),
    }


def evaluate_pair(
    data: pd.DataFrame,
    target: str,
    acoustic_cols: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    censored_col = "post_censored" if target == "NF_post" else "pre_censored"
    eligible = data.loc[~data[censored_col]].copy()
    temporal_cols: list[str] = []
    if args.temporal_control_mode == "opposite_proxy":
        eligible["opposite_side_proxy"] = eligible["NF_pre"].astype(int) if target == "NF_post" else eligible["NF_post"].astype(int)
        temporal_cols.append("opposite_side_proxy")
    elif args.temporal_control_mode == "symmetric_history":
        prefix = "pre_" if target == "NF_post" else "post_"
        temporal_cols = [
            c
            for c in eligible.columns
            if c.startswith(prefix)
            and (
                c.endswith("_count")
                or c.endswith("_count")
                or c.endswith("_s")
                or c.endswith("_same_as_focal")
            )
        ]

    y = eligible[target].astype(int).to_numpy()
    groups = eligible["_clip"].astype(str).to_numpy()
    numeric_cols = [
        "duration_s",
        "log_duration_s",
        "clip_start_margin_s",
        "clip_end_margin_s",
        "clip_center_s",
    ] + temporal_cols
    base_cols = ["manual_label"] + numeric_cols
    X_m0 = eligible[base_cols]
    X_m1 = eligible[base_cols + acoustic_cols]

    n_groups = len(pd.unique(groups))
    n_splits = min(args.n_splits, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    pred_m0 = np.full(len(eligible), np.nan)
    pred_m1 = np.full(len(eligible), np.nan)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_m0, y, groups), start=1):
        m0 = make_pipeline(numeric_cols, [], False, None, args.random_state)
        m1 = make_pipeline(numeric_cols, acoustic_cols, True, args.pca_dim, args.random_state)
        m0.fit(X_m0.iloc[train_idx], y[train_idx])
        m1.fit(X_m1.iloc[train_idx], y[train_idx])
        p0 = m0.predict_proba(X_m0.iloc[test_idx])[:, 1]
        p1 = m1.predict_proba(X_m1.iloc[test_idx])[:, 1]
        pred_m0[test_idx] = p0
        pred_m1[test_idx] = p1
        s0 = score_predictions(y[test_idx], p0)
        s1 = score_predictions(y[test_idx], p1)
        fold_rows.append(
            {
                "fold": fold,
                "target": target,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_positive_rate": float(y[test_idx].mean()),
                "m0_roc_auc": s0["roc_auc"],
                "m1_roc_auc": s1["roc_auc"],
                "delta_roc_auc": s1["roc_auc"] - s0["roc_auc"],
                "m0_average_precision": s0["average_precision"],
                "m1_average_precision": s1["average_precision"],
                "delta_average_precision": s1["average_precision"] - s0["average_precision"],
                "m0_log_loss": s0["log_loss"],
                "m1_log_loss": s1["log_loss"],
                "delta_log_loss_reduction": s0["log_loss"] - s1["log_loss"],
            }
        )

    s0 = score_predictions(y, pred_m0)
    s1 = score_predictions(y, pred_m1)
    overall = {
        "target": target,
        "n": int(len(eligible)),
        "positive_n": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_clip_groups": int(n_groups),
        "n_temporal_history_features": int(len(temporal_cols)),
        "m0_roc_auc": s0["roc_auc"],
        "m1_roc_auc": s1["roc_auc"],
        "delta_roc_auc": s1["roc_auc"] - s0["roc_auc"],
        "m0_average_precision": s0["average_precision"],
        "m1_average_precision": s1["average_precision"],
        "delta_average_precision": s1["average_precision"] - s0["average_precision"],
        "m0_log_loss": s0["log_loss"],
        "m1_log_loss": s1["log_loss"],
        "delta_log_loss_reduction": s0["log_loss"] - s1["log_loss"],
    }

    pred_df = eligible[["call_id", "_clip", "manual_label", "window_ms", "NF_post", "NF_pre"]].copy()
    pred_df["target"] = target
    pred_df["y_true"] = y
    pred_df["m0_pred"] = pred_m0
    pred_df["m1_pred"] = pred_m1

    per_label_rows = []
    for label, g in pred_df.groupby("manual_label", sort=True):
        yy = g["y_true"].to_numpy(int)
        if len(np.unique(yy)) < 2:
            continue
        s0_label = score_predictions(yy, g["m0_pred"].to_numpy(float))
        s1_label = score_predictions(yy, g["m1_pred"].to_numpy(float))
        per_label_rows.append(
            {
                "target": target,
                "manual_label": label,
                "n": int(len(g)),
                "positive_rate": float(yy.mean()),
                "delta_roc_auc": s1_label["roc_auc"] - s0_label["roc_auc"],
                "delta_average_precision": s1_label["average_precision"] - s0_label["average_precision"],
                "delta_log_loss_reduction": s0_label["log_loss"] - s1_label["log_loss"],
            }
        )
    return overall, pd.DataFrame(fold_rows), pd.DataFrame(per_label_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--representation-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_m1_acoustic_residual"))
    parser.add_argument("--windows-ms", default="400,800,1500")
    parser.add_argument("--clip-duration-s", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--amplitude-variant", default=None)
    parser.add_argument(
        "--temporal-control-mode",
        choices=["none", "opposite_proxy", "symmetric_history"],
        default="symmetric_history",
    )
    parser.add_argument("--history-windows-ms", default="400,800,1500")
    parser.add_argument("--drop-feature-regex", default=DEFAULT_DROP_FEATURE_REGEX)
    parser.add_argument("--clip-col", default=None)
    parser.add_argument("--call-id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--focal-col", default=None)
    parser.add_argument("--feature-call-id-col", default=None)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    if bool(args.feature_csv) == bool(args.feature_npz):
        raise ValueError("Use exactly one of --feature-csv or --feature-npz.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    targets = build_targets(manifest, args)
    feature_table, acoustic_cols = load_feature_table(args)
    data = targets.merge(feature_table, on="call_id", how="inner", validate="many_to_one")
    if data.empty:
        raise ValueError("No focal calls matched between manifest targets and acoustic features.")

    overall_rows = []
    fold_tables = []
    per_label_tables = []
    prediction_tables = []
    for tau_ms, g in data.groupby("window_ms", sort=True):
        for target in ["NF_post", "NF_pre"]:
            overall, folds, per_label = evaluate_pair(g, target, acoustic_cols, args)
            overall["window_ms"] = tau_ms
            overall["representation"] = args.representation_name
            overall["n_acoustic_features"] = len(acoustic_cols)
            folds.insert(0, "window_ms", tau_ms)
            folds.insert(0, "representation", args.representation_name)
            per_label.insert(0, "window_ms", tau_ms)
            per_label.insert(0, "representation", args.representation_name)
            overall_rows.append(overall)
            fold_tables.append(folds)
            per_label_tables.append(per_label)

    overall_df = pd.DataFrame(overall_rows)
    post = overall_df.loc[overall_df["target"] == "NF_post"].set_index("window_ms")
    pre = overall_df.loc[overall_df["target"] == "NF_pre"].set_index("window_ms")
    contrasts = []
    for tau_ms in sorted(set(post.index).intersection(pre.index)):
        contrasts.append(
            {
                "representation": args.representation_name,
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
    contrast_df = pd.DataFrame(contrasts)

    stem = args.representation_name
    overall_df.to_csv(args.output_dir / f"{stem}_m1_overall_metrics.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(args.output_dir / f"{stem}_m1_fold_metrics.csv", index=False)
    pd.concat(per_label_tables, ignore_index=True).to_csv(args.output_dir / f"{stem}_m1_per_label_delta.csv", index=False)
    contrast_df.to_csv(args.output_dir / f"{stem}_m1_post_pre_contrast.csv", index=False)

    summary = {
        "representation": args.representation_name,
        "manifest": str(args.manifest),
        "feature_csv": str(args.feature_csv) if args.feature_csv else None,
        "feature_npz": str(args.feature_npz) if args.feature_npz else None,
        "n_matched_focal_call_window_rows": int(len(data)),
        "n_unique_matched_calls": int(data["call_id"].nunique()),
        "n_acoustic_features": int(len(acoustic_cols)),
        "pca_dim": args.pca_dim,
        "temporal_control_mode": args.temporal_control_mode,
        "history_windows_ms": [float(x) for x in args.history_windows_ms.split(",") if x.strip()],
        "outputs": {
            "overall": str(args.output_dir / f"{stem}_m1_overall_metrics.csv"),
            "folds": str(args.output_dir / f"{stem}_m1_fold_metrics.csv"),
            "per_label": str(args.output_dir / f"{stem}_m1_per_label_delta.csv"),
            "post_pre_contrast": str(args.output_dir / f"{stem}_m1_post_pre_contrast.csv"),
        },
    }
    with open(args.output_dir / f"{stem}_m1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
