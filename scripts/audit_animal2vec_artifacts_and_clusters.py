#!/usr/bin/env python
"""Audit animal2vec UMAP islands and selected HDBSCAN clusters for artifacts.

This script is intentionally descriptive. It checks whether visually isolated
UMAP structures or HDBSCAN clusters are dominated by recording, individual,
duration, frame-count, or signal-quality variables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


NUISANCE_PATTERNS = {
    "sample": re.compile(r"(sample|call).*id|uuid|index", re.I),
    "individual": re.compile(r"individual|meerkat|subject|animal|caller|identity|id$", re.I),
    "recording": re.compile(r"recording|file|wav|audio|session|date|group|location|mic", re.I),
    "duration": re.compile(r"duration|dur|length", re.I),
    "frames": re.compile(r"frame|pooled", re.I),
    "quality": re.compile(r"snr|rms|energy|loud|clip|valid|quality|nan|finite", re.I),
}


def load_x(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "X" not in data:
        raise ValueError(f"{path} does not contain X")
    X = np.asarray(data["X"], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def preprocess(X: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > pca_dim:
        return PCA(n_components=pca_dim, random_state=seed).fit_transform(Xs)
    return Xs


def candidate_columns(df: pd.DataFrame, groups: list[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col.startswith("fuzzy_") or col in {"umap_x", "umap_y"}:
            continue
        for group in groups:
            if NUISANCE_PATTERNS[group].search(col):
                cols.append(col)
                break
    return list(dict.fromkeys(cols))


def numeric_audit(df: pd.DataFrame, flag: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        a = values[flag]
        b = values[~flag]
        pooled = values.std(ddof=0)
        rows.append(
            {
                "variable": col,
                "island_mean": a.mean(),
                "rest_mean": b.mean(),
                "island_median": a.median(),
                "rest_median": b.median(),
                "island_missing_rate": a.isna().mean(),
                "rest_missing_rate": b.isna().mean(),
                "standardized_mean_difference": (a.mean() - b.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_difference", key=lambda x: x.abs(), ascending=False)


def categorical_audit(df: pd.DataFrame, flag: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        values = df[col].astype(str).fillna("NA")
        island = values[flag]
        rest = values[~flag]
        if island.empty:
            continue
        island_counts = island.value_counts()
        rest_counts = rest.value_counts()
        rows.append(
            {
                "variable": col,
                "island_unique": int(island.nunique()),
                "rest_unique": int(rest.nunique()),
                "island_top_value": island_counts.index[0],
                "island_top_fraction": float(island_counts.iloc[0] / len(island)),
                "rest_top_value": rest_counts.index[0],
                "rest_top_fraction": float(rest_counts.iloc[0] / len(rest)),
                "island_top5": json.dumps(island_counts.head(5).to_dict(), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("island_top_fraction", ascending=False)


def composition_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    tab = pd.crosstab(df[group_col], df[value_col])
    frac = tab.div(tab.sum(axis=1), axis=0).add_suffix("_fraction")
    return pd.concat([tab.add_suffix("_n"), frac], axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600.npz"))
    parser.add_argument("--diagnostic-samples", type=Path, default=Path("reports/animal2vec_fuzzy_membership_by_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--left-quantile", type=float, default=0.03)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--selection-method", default="eom")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = load_x(args.npz)
    df = pd.read_csv(args.diagnostic_samples)
    if len(df) != len(X):
        raise ValueError(f"diagnostic rows ({len(df)}) != X rows ({len(X)})")
    if "label" not in df.columns:
        raise ValueError("diagnostic samples must contain a label column")
    if "umap_x" not in df.columns:
        raise ValueError("diagnostic samples must contain umap_x; run diagnose_animal2vec_umap_fuzzy.py first")

    left_cut = df["umap_x"].quantile(args.left_quantile)
    left_flag = df["umap_x"] <= left_cut
    df["left_umap_island"] = left_flag

    numeric_cols = candidate_columns(df, ["duration", "frames", "quality"])
    categorical_cols = candidate_columns(df, ["sample", "individual", "recording"])

    numeric_audit(df, left_flag, numeric_cols).to_csv(
        args.output_dir / "animal2vec_left_island_numeric_nuisance_audit.csv", index=False
    )
    categorical_audit(df, left_flag, categorical_cols + ["label"]).to_csv(
        args.output_dir / "animal2vec_left_island_categorical_nuisance_audit.csv", index=False
    )
    df.loc[left_flag].to_csv(args.output_dir / "animal2vec_left_island_samples_with_fuzzy.csv", index=False)

    Xp = preprocess(X, pca_dim=args.pca_dim, seed=args.seed)
    model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.epsilon,
        cluster_selection_method=args.selection_method,
        metric="euclidean",
    )
    clusters = model.fit_predict(Xp)
    df["hdbscan_cluster"] = clusters
    df["hdbscan_probability"] = model.probabilities_
    df.to_csv(args.output_dir / "animal2vec_selected_hdbscan_assignments_with_metadata.csv", index=False)

    composition_table(df, "hdbscan_cluster", "label").to_csv(
        args.output_dir / "animal2vec_selected_hdbscan_label_composition.csv", index=False
    )

    for col in categorical_cols:
        composition_table(df, "hdbscan_cluster", col).to_csv(
            args.output_dir / f"animal2vec_selected_hdbscan_{col}_composition.csv", index=False
        )

    cluster_numeric = df.groupby("hdbscan_cluster")[numeric_cols + ["hdbscan_probability"]].agg(["count", "mean", "median", "std"])
    cluster_numeric.to_csv(args.output_dir / "animal2vec_selected_hdbscan_numeric_nuisance_summary.csv")

    summary = {
        "n_calls": int(len(df)),
        "left_quantile": args.left_quantile,
        "left_umap_x_cutoff": float(left_cut),
        "n_left_island_samples": int(left_flag.sum()),
        "selected_hdbscan_setting": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "epsilon": args.epsilon,
            "selection_method": args.selection_method,
        },
        "selected_hdbscan_n_clusters": int(len([c for c in np.unique(clusters) if c != -1])),
        "selected_hdbscan_noise_rate": float(np.mean(clusters == -1)),
        "selected_hdbscan_label_ami": float(adjusted_mutual_info_score(df["label"].astype(str), clusters)),
        "selected_hdbscan_label_ari": float(adjusted_rand_score(df["label"].astype(str), clusters)),
        "n_numeric_nuisance_columns_checked": len(numeric_cols),
        "n_categorical_nuisance_columns_checked": len(categorical_cols),
    }
    with open(args.output_dir / "animal2vec_artifact_cluster_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
