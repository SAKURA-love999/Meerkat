from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")


def parse_space(value: str) -> int:
    if not value.startswith("pca"):
        raise ValueError("Only PCA spaces are supported, for example pca5 or pca10.")
    return int(value.replace("pca", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize label inclusion and cluster composition for one HDBSCAN setting.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--representation-name", default="logmel_paper_pad500")
    parser.add_argument("--npz-name", default=None)
    parser.add_argument("--amplitude-variant", default="raw")
    parser.add_argument("--space", default="pca5")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_name = args.npz_name or f"spectrogram_{args.representation_name}_{args.pilot_name}.npz"
    data = np.load(args.project_root / "features" / npz_name, allow_pickle=False)
    x_all = data["X"]
    labels_all = data["labels"].astype(str)
    variants_all = data["amplitude_variants"].astype(str)
    mask = variants_all == args.amplitude_variant

    labels = labels_all[mask]
    pca_dim = parse_space(args.space)
    x_scaled = StandardScaler().fit_transform(x_all[mask])
    x = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(x_scaled)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    clusters = clusterer.fit_predict(x)
    non_noise = clusters != -1

    inclusion_rows = []
    for label in sorted(np.unique(labels)):
        label_mask = labels == label
        label_clusters = clusters[label_mask]
        label_non_noise = label_clusters != -1
        clustered = label_clusters[label_non_noise]
        if clustered.size:
            counts = pd.Series(clustered).value_counts()
            largest_cluster = int(counts.index[0])
            largest_cluster_count = int(counts.iloc[0])
            largest_cluster_fraction_within_label = largest_cluster_count / int(np.sum(label_mask))
            clusters_touched = int(counts.size)
        else:
            largest_cluster = -1
            largest_cluster_count = 0
            largest_cluster_fraction_within_label = 0.0
            clusters_touched = 0
        inclusion_rows.append(
            {
                "label": label,
                "total_count": int(np.sum(label_mask)),
                "non_noise_count": int(np.sum(label_non_noise)),
                "noise_count": int(np.sum(~label_non_noise)),
                "cluster_inclusion_rate": float(np.mean(label_non_noise)),
                "n_clusters_touched": clusters_touched,
                "largest_cluster_for_label": largest_cluster,
                "largest_cluster_count_within_label": largest_cluster_count,
                "largest_cluster_fraction_within_label": largest_cluster_fraction_within_label,
            }
        )

    composition_rows = []
    for cluster_id in sorted(set(clusters[non_noise])):
        cluster_mask = clusters == cluster_id
        cluster_labels = labels[cluster_mask]
        counts = pd.Series(cluster_labels).value_counts()
        composition_rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(np.sum(cluster_mask)),
                "majority_label": str(counts.index[0]),
                "majority_fraction": float(counts.iloc[0] / np.sum(cluster_mask)),
                "label_counts": ";".join(f"{label}:{count}" for label, count in counts.items()),
            }
        )

    prefix = (
        f"hdbscan_setting_{args.representation_name}_{args.pilot_name}_{args.amplitude_variant}_"
        f"{args.space}_mcs{args.min_cluster_size}_ms{args.min_samples}"
    )
    inclusion_path = args.project_root / "features" / f"{prefix}_label_inclusion.csv"
    composition_path = args.project_root / "features" / f"{prefix}_cluster_composition.csv"
    pd.DataFrame(inclusion_rows).to_csv(inclusion_path, index=False)
    pd.DataFrame(composition_rows).sort_values("size", ascending=False).to_csv(composition_path, index=False)

    print(f"n_clusters={len(set(clusters[non_noise]))}")
    print(f"noise_rate={float(np.mean(~non_noise)):.4f}")
    print(f"label_inclusion: {inclusion_path}")
    print(pd.DataFrame(inclusion_rows).round(4).to_string(index=False))
    print(f"cluster_composition: {composition_path}")


if __name__ == "__main__":
    main()
