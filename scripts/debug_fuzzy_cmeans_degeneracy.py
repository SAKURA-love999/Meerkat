#!/usr/bin/env python
"""Debug fuzzy c-means degeneration for animal2vec embeddings.

Outputs requested diagnostics for c=8:

1. Pairwise centroid distance min / median / max.
2. Per-sample coefficient of variation of distances to the 8 centroids.
3. Initial membership min / max / std.
4. Centroid displacement after each iteration.

This script is diagnostic only. It should not be used as evidence for graded
calls unless the membership solution passes validity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def load_inputs(npz_path: Path, metadata_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(npz_path, allow_pickle=False)
    if "X" not in data:
        raise ValueError(f"{npz_path} does not contain X")
    X = np.asarray(data["X"], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    meta = pd.read_csv(metadata_path)
    if len(meta) != len(X):
        raise ValueError(f"metadata rows ({len(meta)}) != X rows ({len(X)})")
    return X, meta


def make_space(X: np.ndarray, pca_dim: int, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    Xs = StandardScaler().fit_transform(X)
    info = {"original_dim": int(X.shape[1]), "standardized_dim": int(Xs.shape[1])}
    if pca_dim > 0 and Xs.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xp = pca.fit_transform(Xs)
        info["pca_dim"] = int(pca_dim)
        info["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        return Xp, info
    info["pca_dim"] = int(Xs.shape[1])
    info["pca_explained_variance_ratio_sum"] = np.nan
    return Xs, info


def update_membership(X: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    dist = np.maximum(pairwise_distances(X, centers), 1e-12)
    inv = dist ** (-2.0 / (m - 1.0))
    return inv / np.sum(inv, axis=1, keepdims=True)


def centroid_distance_stats(centers: np.ndarray) -> dict[str, float]:
    D = pairwise_distances(centers)
    off = D[np.triu_indices_from(D, k=1)]
    return {
        "centroid_pairwise_min": float(np.min(off)),
        "centroid_pairwise_median": float(np.median(off)),
        "centroid_pairwise_max": float(np.max(off)),
    }


def membership_stats(U: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_membership_min": float(np.min(U)),
        f"{prefix}_membership_max": float(np.max(U)),
        f"{prefix}_membership_mean": float(np.mean(U)),
        f"{prefix}_membership_std": float(np.std(U)),
        f"{prefix}_max_membership_mean": float(np.mean(np.max(U, axis=1))),
        f"{prefix}_max_membership_median": float(np.median(np.max(U, axis=1))),
    }


def per_sample_distance_cv(X: np.ndarray, centers: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    D = pairwise_distances(X, centers)
    mean = np.mean(D, axis=1)
    std = np.std(D, axis=1)
    cv = std / np.maximum(mean, 1e-12)
    sample = pd.DataFrame(
        {
            "distance_mean_to_centroids": mean,
            "distance_std_to_centroids": std,
            "distance_cv_to_centroids": cv,
            "nearest_centroid": np.argmin(D, axis=1),
            "nearest_centroid_distance": np.min(D, axis=1),
            "farthest_centroid_distance": np.max(D, axis=1),
        }
    )
    summary = {
        "distance_cv_min": float(np.min(cv)),
        "distance_cv_p05": float(np.quantile(cv, 0.05)),
        "distance_cv_median": float(np.median(cv)),
        "distance_cv_mean": float(np.mean(cv)),
        "distance_cv_p95": float(np.quantile(cv, 0.95)),
        "distance_cv_max": float(np.max(cv)),
    }
    return sample, summary


def run_debug(X: np.ndarray, c: int, m: float, seed: int, max_iter: int, tol: float) -> dict[str, object]:
    kmeans = KMeans(n_clusters=c, n_init=1, random_state=seed)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.astype(float)
    U = update_membership(X, centers, m=m)

    initial_centers = centers.copy()
    initial_membership = U.copy()
    trace = []

    for iteration in range(1, max_iter + 1):
        old_U = U.copy()
        old_centers = centers.copy()
        Um = U ** m
        centers = (Um.T @ X) / np.maximum(np.sum(Um, axis=0)[:, None], 1e-12)
        U = update_membership(X, centers, m=m)

        displacement = np.linalg.norm(centers - old_centers, axis=1)
        max_delta_membership = np.max(np.abs(U - old_U))
        row = {
            "iteration": iteration,
            "centroid_displacement_min": float(np.min(displacement)),
            "centroid_displacement_median": float(np.median(displacement)),
            "centroid_displacement_max": float(np.max(displacement)),
            "centroid_displacement_mean": float(np.mean(displacement)),
            "max_delta_membership": float(max_delta_membership),
        }
        row.update(centroid_distance_stats(centers))
        row.update(membership_stats(U, "current"))
        trace.append(row)
        if max_delta_membership < tol:
            break

    final_sample_cv, final_cv_summary = per_sample_distance_cv(X, centers)
    initial_sample_cv, initial_cv_summary = per_sample_distance_cv(X, initial_centers)

    summary = {
        "c": int(c),
        "m": float(m),
        "seed": int(seed),
        "n_iterations": int(len(trace)),
        "converged": bool(trace[-1]["max_delta_membership"] < tol) if trace else False,
        **centroid_distance_stats(centers),
        **membership_stats(initial_membership, "initial"),
        **membership_stats(U, "final"),
        **{f"initial_{k}": v for k, v in initial_cv_summary.items()},
        **{f"final_{k}": v for k, v in final_cv_summary.items()},
        "initial_to_final_centroid_displacement_min": float(np.min(np.linalg.norm(centers - initial_centers, axis=1))),
        "initial_to_final_centroid_displacement_median": float(np.median(np.linalg.norm(centers - initial_centers, axis=1))),
        "initial_to_final_centroid_displacement_max": float(np.max(np.linalg.norm(centers - initial_centers, axis=1))),
    }

    return {
        "summary": summary,
        "trace": pd.DataFrame(trace),
        "final_sample_cv": final_sample_cv,
        "initial_sample_cv": initial_sample_cv,
        "initial_membership": initial_membership,
        "final_membership": U,
        "initial_centers": initial_centers,
        "final_centers": centers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--c", type=int, default=8)
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, meta = load_inputs(args.npz, args.metadata)
    Xp, space_info = make_space(X, pca_dim=args.pca_dim, seed=args.seed)
    debug = run_debug(Xp, c=args.c, m=args.m, seed=args.seed, max_iter=args.max_iter, tol=args.tol)

    prefix = f"animal2vec_fcm_debug_c{args.c}_m{str(args.m).replace('.', 'p')}_pca{args.pca_dim}"
    debug["trace"].to_csv(args.output_dir / f"{prefix}_iteration_trace.csv", index=False)

    sample_base = meta.reset_index(drop=True).copy()
    sample_base["row_index"] = np.arange(len(sample_base))
    pd.concat([sample_base, debug["initial_sample_cv"]], axis=1).to_csv(
        args.output_dir / f"{prefix}_initial_sample_distance_cv.csv", index=False
    )
    pd.concat([sample_base, debug["final_sample_cv"]], axis=1).to_csv(
        args.output_dir / f"{prefix}_final_sample_distance_cv.csv", index=False
    )

    init_mem = pd.DataFrame(debug["initial_membership"], columns=[f"initial_fuzzy_{i}" for i in range(args.c)])
    final_mem = pd.DataFrame(debug["final_membership"], columns=[f"final_fuzzy_{i}" for i in range(args.c)])
    pd.concat([sample_base, init_mem, final_mem], axis=1).to_csv(
        args.output_dir / f"{prefix}_membership_by_sample.csv", index=False
    )

    pd.DataFrame(debug["initial_centers"]).to_csv(args.output_dir / f"{prefix}_initial_centroids.csv", index=False)
    pd.DataFrame(debug["final_centers"]).to_csv(args.output_dir / f"{prefix}_final_centroids.csv", index=False)
    pd.DataFrame(pairwise_distances(debug["initial_centers"])).to_csv(
        args.output_dir / f"{prefix}_initial_centroid_distance_matrix.csv", index=False
    )
    pd.DataFrame(pairwise_distances(debug["final_centers"])).to_csv(
        args.output_dir / f"{prefix}_final_centroid_distance_matrix.csv", index=False
    )

    summary = {**space_info, **debug["summary"]}
    with open(args.output_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
