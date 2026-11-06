#!/usr/bin/env python
"""Fit simple MeerKAT PRE/POST proxy baselines.

This baseline deliberately excludes acoustic embeddings. It tests whether manual
labels plus simple duration/boundary/QC variables can predict clip-local
non-focal vocal activity before or after a focal call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def nearest_nonfocal_targets(
    work: pd.DataFrame,
    clip_duration_s: float,
    windows_ms: list[float],
) -> pd.DataFrame:
    """Construct focal rows with nearest pre/post non-focal events per clip."""
    focal = work.loc[work["_focal"], ["_clip", "_label", "_start", "_end", "_duration"]].copy()
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

    base = post[["_clip", "_label", "_start", "_end", "_duration", "row_index", "next_nonfocal_start"]].merge(
        pre[["row_index", "prev_nonfocal_end"]], on="row_index", how="left"
    )
    base["clip_start_margin_s"] = base["_start"]
    base["clip_end_margin_s"] = clip_duration_s - base["_end"]
    base["clip_center_s"] = (base["_start"] + base["_end"]) / 2.0
    base["log_duration_s"] = np.log1p(base["_duration"].clip(lower=0))

    rows = []
    for tau_ms in windows_ms:
        tau = tau_ms / 1000.0
        tmp = base.copy()
        tmp["window_ms"] = tau_ms
        tmp["post_censored"] = tmp["_end"] > (clip_duration_s - tau)
        tmp["pre_censored"] = tmp["_start"] < tau
        tmp["NF_post"] = (~tmp["post_censored"]) & tmp["next_nonfocal_start"].notna() & (
            tmp["next_nonfocal_start"] <= tmp["_end"] + tau
        )
        tmp["NF_pre"] = (~tmp["pre_censored"]) & tmp["prev_nonfocal_end"].notna() & (
            tmp["prev_nonfocal_end"] >= tmp["_start"] - tau
        )
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def make_model(random_state: int) -> Pipeline:
    categorical = ["manual_label"]
    numeric = [
        "duration_s",
        "log_duration_s",
        "clip_start_margin_s",
        "clip_end_margin_s",
        "clip_center_s",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def evaluate_one(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_col = "post_censored" if target_col == "NF_post" else "pre_censored"
    data = df.loc[~df[eligible_col]].copy()
    data = data.rename(
        columns={
            "_label": "manual_label",
            "_duration": "duration_s",
        }
    )
    y = data[target_col].astype(int).to_numpy()
    groups = data["_clip"].astype(str).to_numpy()
    X = data[
        [
            "manual_label",
            "duration_s",
            "log_duration_s",
            "clip_start_margin_s",
            "clip_end_margin_s",
            "clip_center_s",
        ]
    ]

    n_groups = len(pd.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError("Need at least two clip groups for grouped CV.")
    cv = GroupKFold(n_splits=splits)

    pred = np.full(len(data), np.nan)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = make_model(random_state=random_state)
        model.fit(X.iloc[train_idx], y[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred[test_idx] = prob
        threshold = float(y[train_idx].mean())
        y_hat = (prob >= threshold).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_positive_rate": float(y[train_idx].mean()),
                "test_positive_rate": float(y[test_idx].mean()),
                "roc_auc": float(roc_auc_score(y[test_idx], prob)) if len(np.unique(y[test_idx])) == 2 else np.nan,
                "average_precision": float(average_precision_score(y[test_idx], prob)),
                "log_loss": float(log_loss(y[test_idx], prob, labels=[0, 1])),
                "balanced_accuracy_prevalence_threshold": float(balanced_accuracy_score(y[test_idx], y_hat)),
                "macro_f1_prevalence_threshold": float(f1_score(y[test_idx], y_hat, average="macro")),
            }
        )

    threshold_all = float(y.mean())
    pred_label = (pred >= threshold_all).astype(int)
    overall = pd.DataFrame(
        [
            {
                "target": target_col,
                "n": int(len(data)),
                "positive_n": int(y.sum()),
                "positive_rate": float(y.mean()),
                "n_clip_groups": int(n_groups),
                "roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
                "average_precision": float(average_precision_score(y, pred)),
                "log_loss": float(log_loss(y, pred, labels=[0, 1])),
                "balanced_accuracy_prevalence_threshold": float(balanced_accuracy_score(y, pred_label)),
                "macro_f1_prevalence_threshold": float(f1_score(y, pred_label, average="macro")),
            }
        ]
    )
    folds = pd.DataFrame(fold_rows)
    return overall, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_proxy_baseline"))
    parser.add_argument("--windows-ms", default="400,800,1500")
    parser.add_argument("--clip-duration-s", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--clip-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--focal-col", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    clip_col = args.clip_col or first_existing(df, ["source_id", "clip_id", "file_id", "source_wav", "recording"])
    label_col = args.label_col or first_existing(df, ["label", "call_type", "manual_label", "type"])
    start_col = args.start_col or first_existing(df, ["start_time_sec", "start_s", "start_sec", "onset_s", "start"])
    end_col = args.end_col or first_existing(df, ["end_time_sec", "end_s", "end_sec", "offset_s", "end"])
    duration_col = args.duration_col or first_existing(df, ["duration_sec", "duration_s", "duration"])
    focal_col = args.focal_col or first_existing(df, ["focal", "is_focal"])

    work = pd.DataFrame(
        {
            "_clip": df[clip_col].astype(str),
            "_label": df[label_col].astype(str),
            "_start": pd.to_numeric(df[start_col], errors="coerce"),
            "_end": pd.to_numeric(df[end_col], errors="coerce"),
            "_duration": pd.to_numeric(df[duration_col], errors="coerce"),
            "_focal": bool_series(df[focal_col]),
        }
    ).dropna(subset=["_start", "_end", "_duration"])

    windows_ms = [float(x) for x in args.windows_ms.split(",") if x.strip()]
    targets = nearest_nonfocal_targets(work, args.clip_duration_s, windows_ms)
    targets.to_csv(args.output_dir / "meerkat_prepost_baseline_targets.csv", index=False)

    overall_rows = []
    fold_rows = []
    for tau_ms, g in targets.groupby("window_ms", sort=True):
        for target_col in ["NF_post", "NF_pre"]:
            overall, folds = evaluate_one(g, target_col, args.n_splits, args.random_state)
            overall.insert(0, "window_ms", tau_ms)
            folds.insert(0, "window_ms", tau_ms)
            folds.insert(1, "target", target_col)
            overall_rows.append(overall)
            fold_rows.append(folds)

    overall_df = pd.concat(overall_rows, ignore_index=True)
    folds_df = pd.concat(fold_rows, ignore_index=True)
    overall_df.to_csv(args.output_dir / "meerkat_m0_prepost_baseline_metrics.csv", index=False)
    folds_df.to_csv(args.output_dir / "meerkat_m0_prepost_baseline_fold_metrics.csv", index=False)

    summary = {
        "manifest": str(args.manifest),
        "n_events": int(len(df)),
        "n_focal_events": int(targets["row_index"].nunique()),
        "windows_ms": windows_ms,
        "model": "LogisticRegression",
        "features": [
            "manual_label one-hot",
            "duration_s",
            "log_duration_s",
            "clip_start_margin_s",
            "clip_end_margin_s",
            "clip_center_s",
        ],
        "cv": f"GroupKFold(n_splits={args.n_splits}) grouped by 10-s clip/source_id",
        "outputs": {
            "targets": str(args.output_dir / "meerkat_prepost_baseline_targets.csv"),
            "overall_metrics": str(args.output_dir / "meerkat_m0_prepost_baseline_metrics.csv"),
            "fold_metrics": str(args.output_dir / "meerkat_m0_prepost_baseline_fold_metrics.csv"),
        },
    }
    with open(args.output_dir / "meerkat_m0_prepost_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

