#!/usr/bin/env python
"""HDBSCAN parameter and bootstrap stability diagnostics.

This script asks a narrower question than the main geometry analysis:

    Are the recovered HDBSCAN clusters stable under small algorithmic or
    sampling perturbations?

It does not choose a "best" setting by AMI/ARI. Label metrics are reported only
as post-hoc descriptions of each unsupervised solution.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


LABEL_COLUMNS = ("label", "call_type", "manual_label", "type")


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def find_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Tried: {', '.join(LABEL_COLUMNS)}")


def load_matrix(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    if "X" not in data:
        raise ValueError(f"{npz_path} does not contain an 'X' array.")
    X = np.asarray(data["X"], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{npz_path} contains NaN or infinite values.")
    return X


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    label_col = find_label_column(df)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    return df


def preprocess(X: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    if pca_dim > 0 and Xs.shape[1] > pca_dim:
        return PCA(n_components=pca_dim, random_state=seed).fit_transform(Xs)
    return Xs


def relabel(labels: np.ndarray) -> np.ndarray:
    """Relabel non-noise cluster ids to 0..k-1 while keeping noise as -1."""
    out = np.full(labels.shape, -1, dtype=int)
    clusters = sorted(c for c in np.unique(labels) if c != -1)
    for new_id, old_id in enumerate(clusters):
        out[labels == old_id] = new_id
    return out


def cluster_stats(X: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray) -> dict[str, float]:
    cluster_labels = relabel(cluster_labels)
    non_noise = cluster_labels != -1
    clusters = sorted(c for c in np.unique(cluster_labels) if c != -1)
    sizes = np.array([np.sum(cluster_labels == c) for c in clusters], dtype=float)

    row: dict[str, float] = {
        "n_clusters": float(len(clusters)),
        "noise_rate": float(np.mean(~non_noise)),
        "largest_cluster_fraction": float(sizes.max() / len(cluster_labels)) if len(sizes) else 0.0,
        "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "min_cluster_size_observed": float(sizes.min()) if len(sizes) else 0.0,
        "max_cluster_size_observed": float(sizes.max()) if len(sizes) else 0.0,
        "label_ami": float(adjusted_mutual_info_score(true_labels, cluster_labels)),
        "label_ari": float(adjusted_rand_score(true_labels, cluster_labels)),
    }

    if len(clusters) >= 2 and np.sum(non_noise) > len(clusters):
        row["cluster_silhouette_no_noise"] = float(silhouette_score(X[non_noise], cluster_labels[non_noise]))
    else:
        row["cluster_silhouette_no_noise"] = np.nan
    return row


def assignment_info(labels: np.ndarray) -> dict[str, float]:
    clusters = [c for c in np.unique(labels) if c != -1]
    noise_rate = float(np.mean(labels == -1))
    return {
        "n_clusters": float(len(clusters)),
        "noise_rate": noise_rate,
        "nontrivial": bool(len(clusters) >= 2 and noise_rate < 0.95),
    }


def pairwise_assignment_stability(assignments: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(range(len(assignments)), 2):
        info_i = assignment_info(assignments[i])
        info_j = assignment_info(assignments[j])
        rows.append(
            {
                "solution_i": i,
                "solution_j": j,
                "n_clusters_i": info_i["n_clusters"],
                "n_clusters_j": info_j["n_clusters"],
                "noise_rate_i": info_i["noise_rate"],
                "noise_rate_j": info_j["noise_rate"],
                "both_nontrivial": bool(info_i["nontrivial"] and info_j["nontrivial"]),
                "solution_ami": adjusted_mutual_info_score(assignments[i], assignments[j]),
                "solution_ari": adjusted_rand_score(assignments[i], assignments[j]),
            }
        )
    return pd.DataFrame(rows)


def run_parameter_grid(
    X: np.ndarray,
    labels: np.ndarray,
    min_cluster_sizes: list[int],
    min_samples_values: list[int],
    epsilons: list[float],
    selection_methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    assignments = []

    for min_cluster_size, min_samples, epsilon, selection_method in itertools.product(
        min_cluster_sizes, min_samples_values, epsilons, selection_methods
    ):
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            cluster_selection_method=selection_method,
            metric="euclidean",
        )
        pred = model.fit_predict(X)
        assignments.append(relabel(pred))
        stats = cluster_stats(X, labels, pred)

        persistence = np.asarray(getattr(model, "cluster_persistence_", []), dtype=float)
        stats.update(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": epsilon,
                "cluster_selection_method": selection_method,
                "mean_persistence": float(np.mean(persistence)) if persistence.size else np.nan,
                "max_persistence": float(np.max(persistence)) if persistence.size else np.nan,
            }
        )
        rows.append(stats)

    grid = pd.DataFrame(rows)
    stability = pairwise_assignment_stability(assignments)
    return grid, stability


def bootstrap_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for lab in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == lab)
        n = max(2, int(round(len(idx) * fraction)))
        selected.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(selected))


def run_bootstrap_stability(
    X: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    selection_method: str,
    n_bootstrap: int,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_assignments = []
    rows = []

    for b in range(n_bootstrap):
        idx = bootstrap_indices(labels, fraction=fraction, seed=seed + b)
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            cluster_selection_method=selection_method,
            metric="euclidean",
        )
        pred_sub = relabel(model.fit_predict(X[idx]))
        pred_full = np.full(len(labels), -999, dtype=int)
        pred_full[idx] = pred_sub
        full_assignments.append(pred_full)

        stats = cluster_stats(X[idx], labels[idx], pred_sub)
        stats.update({"bootstrap": b, "n_sampled": len(idx)})
        rows.append(stats)

    pair_rows = []
    for i, j in itertools.combinations(range(n_bootstrap), 2):
        both = (full_assignments[i] != -999) & (full_assignments[j] != -999)
        info_i = assignment_info(full_assignments[i][both])
        info_j = assignment_info(full_assignments[j][both])
        pair_rows.append(
            {
                "bootstrap_i": i,
                "bootstrap_j": j,
                "n_overlap": int(np.sum(both)),
                "n_clusters_i": info_i["n_clusters"],
                "n_clusters_j": info_j["n_clusters"],
                "noise_rate_i": info_i["noise_rate"],
                "noise_rate_j": info_j["noise_rate"],
                "both_nontrivial": bool(info_i["nontrivial"] and info_j["nontrivial"]),
                "bootstrap_solution_ami": float(adjusted_mutual_info_score(full_assignments[i][both], full_assignments[j][both])),
                "bootstrap_solution_ari": float(adjusted_rand_score(full_assignments[i][both], full_assignments[j][both])),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def plot_grid(grid: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    axes[0].scatter(grid["noise_rate"], grid["n_clusters"], c=grid["label_ami"], cmap="viridis", s=45)
    axes[0].set_xlabel("noise rate")
    axes[0].set_ylabel("n clusters")
    axes[0].set_title("Grid solutions colored by label AMI")

    axes[1].hist(grid["n_clusters"], bins=range(int(grid["n_clusters"].max()) + 3), color="#5B8FF9")
    axes[1].set_xlabel("n clusters")
    axes[1].set_ylabel("count")
    axes[1].set_title("Cluster-count stability")

    axes[2].scatter(grid["mean_persistence"], grid["label_ami"], c=grid["noise_rate"], cmap="magma", s=45)
    axes[2].set_xlabel("mean persistence")
    axes[2].set_ylabel("post-hoc label AMI")
    axes[2].set_title("Persistence vs label alignment")

    axes[3].scatter(grid["largest_cluster_fraction"], grid["label_ari"], c=grid["n_clusters"], cmap="plasma", s=45)
    axes[3].set_xlabel("largest cluster fraction")
    axes[3].set_ylabel("post-hoc label ARI")
    axes[3].set_title("Cluster imbalance vs label ARI")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--representation-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-cluster-sizes", default="10,20,30,50,75,100,150,200")
    parser.add_argument("--min-samples", default="1,5,10,20,30,50")
    parser.add_argument("--epsilons", default="0,0.05,0.1,0.2,0.5")
    parser.add_argument("--selection-methods", default="eom,leaf")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-min-cluster-size", type=int, default=30)
    parser.add_argument("--bootstrap-min-samples", type=int, default=10)
    parser.add_argument("--bootstrap-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap-selection-method", default="eom")
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = load_matrix(args.npz)
    meta = load_metadata(args.metadata)
    if len(meta) != len(X):
        raise ValueError(f"metadata rows ({len(meta)}) != X rows ({len(X)})")
    labels = meta["label"].astype(str).to_numpy()
    Xp = preprocess(X, pca_dim=args.pca_dim, seed=args.seed)

    mcs = parse_int_list(args.min_cluster_sizes)
    ms = parse_int_list(args.min_samples)
    eps = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    methods = [x.strip() for x in args.selection_methods.split(",") if x.strip()]

    grid, grid_stability = run_parameter_grid(Xp, labels, mcs, ms, eps, methods)
    prefix = args.output_dir / f"hdbscan_stability_{args.representation_name}"
    grid.to_csv(prefix.with_name(prefix.name + "_parameter_grid.csv"), index=False)
    grid_stability.to_csv(prefix.with_name(prefix.name + "_parameter_solution_stability.csv"), index=False)
    plot_grid(grid, prefix.with_name(prefix.name + "_parameter_grid.png"))

    nontrivial_grid_stability = grid_stability.loc[grid_stability["both_nontrivial"]].copy()
    summary = {
        "representation": args.representation_name,
        "n_calls": int(len(X)),
        "pca_dim": int(args.pca_dim),
        "n_parameter_settings": int(len(grid)),
        "n_nontrivial_parameter_settings": int(np.sum((grid["n_clusters"] >= 2) & (grid["noise_rate"] < 0.95))),
        "n_clusters_median": float(grid["n_clusters"].median()),
        "n_clusters_iqr": [
            float(grid["n_clusters"].quantile(0.25)),
            float(grid["n_clusters"].quantile(0.75)),
        ],
        "noise_rate_median": float(grid["noise_rate"].median()),
        "noise_rate_iqr": [
            float(grid["noise_rate"].quantile(0.25)),
            float(grid["noise_rate"].quantile(0.75)),
        ],
        "posthoc_label_ami_median": float(grid["label_ami"].median()),
        "posthoc_label_ami_max_observed": float(grid["label_ami"].max()),
        "posthoc_label_ari_median": float(grid["label_ari"].median()),
        "parameter_solution_ami_median": float(grid_stability["solution_ami"].median()) if len(grid_stability) else np.nan,
        "parameter_solution_ari_median": float(grid_stability["solution_ari"].median()) if len(grid_stability) else np.nan,
        "nontrivial_parameter_solution_ami_median": float(nontrivial_grid_stability["solution_ami"].median()) if len(nontrivial_grid_stability) else np.nan,
        "nontrivial_parameter_solution_ari_median": float(nontrivial_grid_stability["solution_ari"].median()) if len(nontrivial_grid_stability) else np.nan,
    }

    if args.bootstrap:
        boot, boot_stability = run_bootstrap_stability(
            Xp,
            labels,
            min_cluster_size=args.bootstrap_min_cluster_size,
            min_samples=args.bootstrap_min_samples,
            epsilon=args.bootstrap_epsilon,
            selection_method=args.bootstrap_selection_method,
            n_bootstrap=args.n_bootstrap,
            fraction=args.bootstrap_fraction,
            seed=args.seed,
        )
        boot.to_csv(prefix.with_name(prefix.name + "_bootstrap_runs.csv"), index=False)
        boot_stability.to_csv(prefix.with_name(prefix.name + "_bootstrap_solution_stability.csv"), index=False)
        nontrivial_boot_stability = boot_stability.loc[boot_stability["both_nontrivial"]].copy()
        summary.update(
            {
                "bootstrap_setting": {
                    "min_cluster_size": args.bootstrap_min_cluster_size,
                    "min_samples": args.bootstrap_min_samples,
                    "epsilon": args.bootstrap_epsilon,
                    "selection_method": args.bootstrap_selection_method,
                    "n_bootstrap": args.n_bootstrap,
                    "fraction": args.bootstrap_fraction,
                },
                "bootstrap_n_clusters_median": float(boot["n_clusters"].median()),
                "bootstrap_noise_rate_median": float(boot["noise_rate"].median()),
                "bootstrap_solution_ami_median": float(boot_stability["bootstrap_solution_ami"].median()),
                "bootstrap_solution_ari_median": float(boot_stability["bootstrap_solution_ari"].median()),
                "nontrivial_bootstrap_solution_ami_median": float(nontrivial_boot_stability["bootstrap_solution_ami"].median()) if len(nontrivial_boot_stability) else np.nan,
                "nontrivial_bootstrap_solution_ari_median": float(nontrivial_boot_stability["bootstrap_solution_ari"].median()) if len(nontrivial_boot_stability) else np.nan,
            }
        )

    with open(prefix.with_name(prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
