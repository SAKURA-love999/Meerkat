from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")


METADATA_COLUMNS = {
    "call_id",
    "source_id",
    "label",
    "label_category",
    "event_index",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "focal",
    "sample_rate",
    "amplitude_variant",
    "normalization_target_dbfs",
    "normalization_gain_db",
    "normalization_peak_limited",
}

EXTRACTION_PARAMETER_COLUMNS = {
    "n_fft",
    "hop_length",
    "n_mels",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        if column in METADATA_COLUMNS or column in EXTRACTION_PARAMETER_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def prepare_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = df[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    x = SimpleImputer(strategy="median").fit_transform(x)
    return StandardScaler().fit_transform(x)


def make_spaces(x: np.ndarray, pca_dims: list[int], random_state: int) -> dict[str, np.ndarray]:
    spaces = {"full": x}
    for dim in pca_dims:
        if dim < min(x.shape):
            spaces[f"pca{dim}"] = PCA(n_components=dim, random_state=random_state).fit_transform(x)
    return spaces


def cluster_majority_purity(cluster_labels: np.ndarray, labels: np.ndarray) -> float:
    mask = cluster_labels != -1
    if not np.any(mask):
        return float("nan")
    total = 0
    majority = 0
    for cluster_id in sorted(set(cluster_labels[mask])):
        cluster_mask = cluster_labels == cluster_id
        cluster_label_values, counts = np.unique(labels[cluster_mask], return_counts=True)
        del cluster_label_values
        total += int(np.sum(counts))
        majority += int(np.max(counts))
    return float(majority / total) if total else float("nan")


def run_one(
    x: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> dict[str, Any]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(x)
    clusters = sorted(set(cluster_labels) - {-1})
    n_clusters = len(clusters)
    noise_rate = float(np.mean(cluster_labels == -1))
    clustered_mask = cluster_labels != -1

    if n_clusters > 1 and np.sum(clustered_mask) > n_clusters:
        cluster_silhouette = float(silhouette_score(x[clustered_mask], cluster_labels[clustered_mask]))
    else:
        cluster_silhouette = float("nan")

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "noise_rate": noise_rate,
        "clustered_fraction": 1.0 - noise_rate,
        "ami_all_points": float(adjusted_mutual_info_score(labels, cluster_labels)),
        "ari_all_points": float(adjusted_rand_score(labels, cluster_labels)),
        "cluster_silhouette_no_noise": cluster_silhouette,
        "cluster_majority_purity_no_noise": cluster_majority_purity(cluster_labels, labels),
        "mean_membership_probability": float(np.mean(clusterer.probabilities_[clustered_mask]))
        if np.any(clustered_mask)
        else float("nan"),
    }


def save_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    path: Path,
    fmt: str = ".2f",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="min_samples", columns="min_cluster_size", values=metric)
    plt.figure(figsize=(8, 5.5), dpi=160)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="viridis")
    plt.title(title)
    plt.xlabel("min_cluster_size")
    plt.ylabel("min_samples")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HDBSCAN parameter sensitivity analysis.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--min-cluster-sizes", default="10,25,50,100,200")
    parser.add_argument("--min-samples", default="1,5,10,25,50")
    parser.add_argument("--pca-dims", default="10,20,50")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = args.project_root / "features" / f"acoustic_features_{args.pilot_name}.csv"
    features = pd.read_csv(feature_path)
    min_cluster_sizes = parse_int_list(args.min_cluster_sizes)
    min_samples_values = parse_int_list(args.min_samples)
    pca_dims = parse_int_list(args.pca_dims)

    all_rows: list[dict[str, Any]] = []
    for variant in sorted(features["amplitude_variant"].unique()):
        print(f"variant={variant}", flush=True)
        variant_df = features[features["amplitude_variant"] == variant].copy()
        variant_df = variant_df.sort_values("call_id").reset_index(drop=True)
        labels = variant_df["label"].astype(str).to_numpy()
        x = prepare_matrix(variant_df, get_feature_columns(variant_df))
        spaces = make_spaces(x, pca_dims=pca_dims, random_state=args.random_state)

        for space_name, space_x in spaces.items():
            print(f"  space={space_name}", flush=True)
            space_rows = []
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_values:
                    result = run_one(
                        x=space_x,
                        labels=labels,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                    )
                    result.update(
                        {
                            "pilot_name": args.pilot_name,
                            "amplitude_variant": variant,
                            "space": space_name,
                            "n_calls": len(variant_df),
                            "n_labels": len(np.unique(labels)),
                            "n_dimensions": space_x.shape[1],
                        }
                    )
                    all_rows.append(result)
                    space_rows.append(result)
            space_df = pd.DataFrame(space_rows)
            prefix = f"hdbscan_sensitivity_{args.pilot_name}_{variant}_{space_name}"
            save_heatmap(
                space_df,
                "ami_all_points",
                f"AMI: {variant}, {space_name}",
                args.project_root / "figures" / f"{prefix}_ami.png",
            )
            save_heatmap(
                space_df,
                "noise_rate",
                f"Noise rate: {variant}, {space_name}",
                args.project_root / "figures" / f"{prefix}_noise_rate.png",
            )
            save_heatmap(
                space_df,
                "n_clusters",
                f"Number of clusters: {variant}, {space_name}",
                args.project_root / "figures" / f"{prefix}_n_clusters.png",
                fmt=".0f",
            )

    out_path = args.project_root / "features" / f"hdbscan_sensitivity_{args.pilot_name}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = pd.DataFrame(all_rows)
    best = summary.sort_values(["ami_all_points", "ari_all_points"], ascending=False).head(20)
    best_path = args.project_root / "features" / f"hdbscan_sensitivity_{args.pilot_name}_top20.csv"
    best.to_csv(best_path, index=False)
    print(f"results: {out_path}")
    print(f"top20: {best_path}")


if __name__ == "__main__":
    main()
