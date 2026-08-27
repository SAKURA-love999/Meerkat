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
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(r"C:\学校\MeerKAT\repertoire_geometry")


def knn_label_purity(x: np.ndarray, y: np.ndarray, k: int) -> float:
    n_neighbors = min(k + 1, len(y))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    indices = neighbors.kneighbors(return_distance=False)
    scores = []
    for i, row in enumerate(indices):
        neighbor_labels = y[row[row != i]]
        if neighbor_labels.size:
            scores.append(float(np.mean(neighbor_labels == y[i])))
    return float(np.mean(scores))


def run_gmm_bic(x: np.ndarray, max_components: int, random_state: int) -> tuple[pd.DataFrame, int]:
    pca_dims = min(50, x.shape[1], x.shape[0] - 1)
    x_reduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(x)
    rows = []
    for n_components in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6,
            max_iter=300,
        )
        model.fit(x_reduced)
        rows.append({"n_components": n_components, "bic": float(model.bic(x_reduced))})
    bic_df = pd.DataFrame(rows)
    best_k = int(bic_df.loc[bic_df["bic"].idxmin(), "n_components"])
    return bic_df, best_k


def save_embedding_plot(embedding: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=160)
    plot_df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "label": labels})
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        s=8,
        linewidth=0,
        alpha=0.75,
        palette="tab10",
    )
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_plot(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=160)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def analyze_variant(
    x_all: np.ndarray,
    labels_all: np.ndarray,
    call_ids_all: np.ndarray,
    variants_all: np.ndarray,
    variant: str,
    project_root: Path,
    pilot_name: str,
    representation_name: str,
    pca_dims: int,
    random_state: int,
) -> dict[str, Any]:
    mask = variants_all == variant
    x_raw = x_all[mask]
    labels = labels_all[mask].astype(str)
    call_ids = call_ids_all[mask].astype(str)

    x_scaled = StandardScaler().fit_transform(x_raw)
    x = PCA(n_components=min(pca_dims, x_scaled.shape[1], x_scaled.shape[0] - 1), random_state=random_state).fit_transform(x_scaled)

    pca2 = x[:, :2]
    prefix = f"{representation_name}_{pilot_name}_{variant}"
    save_embedding_plot(
        pca2,
        labels,
        project_root / "figures" / f"pca_{prefix}.png",
        f"PCA: {representation_name}, {pilot_name}, {variant}",
    )

    umap2 = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(x)
    save_embedding_plot(
        umap2,
        labels,
        project_root / "figures" / f"umap_{prefix}.png",
        f"UMAP: {representation_name}, {pilot_name}, {variant}",
    )

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    cluster_labels = clusterer.fit_predict(x)
    n_clusters = len(set(cluster_labels) - {-1})
    noise_rate = float(np.mean(cluster_labels == -1))
    ami = adjusted_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    label_silhouette = float(silhouette_score(x, labels))

    cluster_mask = cluster_labels != -1
    if n_clusters > 1 and np.sum(cluster_mask) > n_clusters:
        cluster_silhouette = float(silhouette_score(x[cluster_mask], cluster_labels[cluster_mask]))
    else:
        cluster_silhouette = float("nan")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(classifier, x, y, cv=cv)
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    balanced_acc = float(balanced_accuracy_score(y, y_pred))

    cm = confusion_matrix(y, y_pred, normalize="true")
    save_confusion_plot(
        cm,
        list(label_encoder.classes_),
        project_root / "figures" / f"confusion_linear_probe_{prefix}.png",
        f"Linear probe: {representation_name}, {variant}",
    )
    pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
        project_root / "features" / f"confusion_linear_probe_{prefix}.csv"
    )

    knn_purity = knn_label_purity(x, labels, k=15)
    bic_df, best_k = run_gmm_bic(x, max_components=12, random_state=random_state)
    bic_df["pilot_name"] = pilot_name
    bic_df["representation"] = representation_name
    bic_df["amplitude_variant"] = variant
    bic_df.to_csv(project_root / "features" / f"gmm_bic_{prefix}.csv", index=False)

    assignments = pd.DataFrame(
        {
            "call_id": call_ids,
            "label": labels,
            "hdbscan_cluster": cluster_labels,
            "hdbscan_probability": clusterer.probabilities_,
        }
    )
    assignments.to_csv(project_root / "features" / f"hdbscan_assignments_{prefix}.csv", index=False)

    return {
        "pilot_name": pilot_name,
        "representation": representation_name,
        "amplitude_variant": variant,
        "n_calls": int(x.shape[0]),
        "n_labels": int(len(np.unique(labels))),
        "original_vector_dim": int(x_raw.shape[1]),
        "analysis_pca_dim": int(x.shape[1]),
        "hdbscan_n_clusters": n_clusters,
        "hdbscan_noise_rate": noise_rate,
        "hdbscan_label_ami": float(ami),
        "hdbscan_label_ari": float(ari),
        "label_silhouette": label_silhouette,
        "cluster_silhouette_no_noise": cluster_silhouette,
        "knn15_label_purity": knn_purity,
        "knn15_label_mixing": 1.0 - knn_purity,
        "linear_probe_macro_f1": macro_f1,
        "linear_probe_balanced_accuracy": balanced_acc,
        "gmm_bic_best_k": best_k,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dense embedding representations with the pilot geometry metrics.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pilot-name", default="balanced_600")
    parser.add_argument("--representation-name", default="logmel_call_resize")
    parser.add_argument("--npz-name", default=None)
    parser.add_argument("--analysis-pca-dim", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_name = args.npz_name or f"spectrogram_logmel_{args.pilot_name}.npz"
    npz_path = args.project_root / "features" / npz_name
    data = np.load(npz_path, allow_pickle=False)
    x = data["X"]
    labels = data["labels"]
    call_ids = data["call_ids"]
    variants = data["amplitude_variants"]

    results = []
    for variant in sorted(np.unique(variants.astype(str))):
        print(f"analyzing {variant}", flush=True)
        results.append(
            analyze_variant(
                x_all=x,
                labels_all=labels,
                call_ids_all=call_ids,
                variants_all=variants.astype(str),
                variant=variant,
                project_root=args.project_root,
                pilot_name=args.pilot_name,
                representation_name=args.representation_name,
                pca_dims=args.analysis_pca_dim,
                random_state=args.random_state,
            )
        )

    out_path = args.project_root / "features" / f"geometry_metrics_{args.representation_name}_{args.pilot_name}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"metrics: {out_path}")


if __name__ == "__main__":
    main()
