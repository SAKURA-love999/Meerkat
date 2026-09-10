#!/usr/bin/env python
"""Animal2vec UMAP island, label-neighborhood, and fuzzy-membership diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def update_memberships(X: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    dist = pairwise_distances(X, centers)
    dist = np.maximum(dist, 1e-12)
    power = 2.0 / (m - 1.0)
    inv = dist ** (-power)
    return inv / inv.sum(axis=1, keepdims=True)


def fuzzy_objective(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float) -> float:
    dist2 = pairwise_distances(X, centers) ** 2
    return float(np.sum((U ** m) * dist2))


def fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    m: float,
    seed: int,
    max_iter: int,
    tol: float,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fuzzy c-means with KMeans initialization and degeneration diagnostics."""
    best = None

    for init_id in range(n_init):
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed + init_id)
        km.fit(X)
        centers = km.cluster_centers_.astype(float)
        U = update_memberships(X, centers, m)

        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            U_old = U.copy()
            Um = U ** m
            centers = (Um.T @ X) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)
            U = update_memberships(X, centers, m)
            if np.max(np.abs(U - U_old)) < tol:
                break

        obj = fuzzy_objective(X, centers, U, m)
        if best is None or obj < best[0]:
            best = (obj, centers, U, n_iter, init_id)

    assert best is not None
    obj, centers, U, n_iter, init_id = best
    diag = fuzzy_diagnostics(X, centers, U, m)
    diag.update({"objective": obj, "n_iter": n_iter, "best_init": init_id})
    return centers, U, diag



def entropy(U: np.ndarray) -> np.ndarray:
    return -np.sum(U * np.log(np.maximum(U, 1e-12)), axis=1)


def fuzzy_diagnostics(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float) -> dict[str, float]:
    c = U.shape[1]
    max_mem = U.max(axis=1)
    ent = entropy(U)
    center_dist = pairwise_distances(centers)
    off_diag = center_dist[np.triu_indices_from(center_dist, k=1)]
    rel_entropy = ent / np.log(c)
    return {
        "fuzzy_c": int(c),
        "fuzzifier_m": float(m),
        "mean_max_membership": float(max_mem.mean()),
        "median_max_membership": float(np.median(max_mem)),
        "p05_max_membership": float(np.quantile(max_mem, 0.05)),
        "p95_max_membership": float(np.quantile(max_mem, 0.95)),
        "mean_entropy": float(ent.mean()),
        "mean_relative_entropy": float(rel_entropy.mean()),
        "median_relative_entropy": float(np.median(rel_entropy)),
        "fuzzy_partition_coefficient": float(np.mean(np.sum(U ** 2, axis=1))),
        "min_center_distance": float(off_diag.min()) if len(off_diag) else np.nan,
        "median_center_distance": float(np.median(off_diag)) if len(off_diag) else np.nan,
        "is_uniform_degenerate": bool(max_mem.mean() < (1.0 / c + 0.03) and rel_entropy.mean() > 0.98),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--left-quantile", type=float, default=0.03)
    parser.add_argument("--fuzzy-clusters", type=int, default=8)
    parser.add_argument("--fuzzy-candidates", default="2,3,4,5,6,8,10,12")
    parser.add_argument("--fuzzy-m", type=float, default=2.0)
    parser.add_argument("--fuzzy-n-init", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=False)
    X = np.asarray(data["X"], dtype=np.float32)
    meta = pd.read_csv(args.metadata)
    labels = meta["label"].astype(str).to_numpy()
    label_names = sorted(pd.unique(labels))

    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(Xs)
    UMAP = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=args.seed).fit_transform(Xp)

    meta = meta.copy()
    meta["umap_x"] = UMAP[:, 0]
    meta["umap_y"] = UMAP[:, 1]

    # Left UMAP islands: operational diagnostic set, not a biological cluster.
    left_cut = meta["umap_x"].quantile(args.left_quantile)
    left = meta.loc[meta["umap_x"] <= left_cut].copy()
    left.to_csv(args.output_dir / "animal2vec_left_umap_islands_samples.csv", index=False)

    centroids = np.vstack([Xp[labels == lab].mean(axis=0) for lab in label_names])
    D = pairwise_distances(centroids, metric="euclidean")
    pd.DataFrame(D, index=label_names, columns=label_names).to_csv(
        args.output_dir / "animal2vec_pairwise_label_centroid_distances.csv"
    )

    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(Xp)
    neigh = nn.kneighbors(Xp, return_distance=False)[:, 1:]
    mix = pd.DataFrame(0.0, index=label_names, columns=label_names)
    for i, row in enumerate(neigh):
        for dst in label_names:
            mix.loc[labels[i], dst] += np.mean(labels[row] == dst)
    mix = mix.div(pd.Series(labels).value_counts().reindex(label_names), axis=0)
    mix.to_csv(args.output_dir / f"animal2vec_knn{args.knn}_label_mixing_matrix.csv")

    candidate_cs = [int(x.strip()) for x in args.fuzzy_candidates.split(",") if x.strip()]
    diag_rows = []
    selected = None
    for c in candidate_cs:
        centers_c, memberships_c, diag = fuzzy_c_means(
            Xp,
            n_clusters=c,
            m=args.fuzzy_m,
            seed=args.seed,
            max_iter=300,
            tol=1e-5,
            n_init=args.fuzzy_n_init,
        )
        diag_rows.append(diag)
        if c == args.fuzzy_clusters:
            selected = (centers_c, memberships_c, diag)

    if selected is None:
        centers_c, memberships_c, diag = fuzzy_c_means(
            Xp,
            n_clusters=args.fuzzy_clusters,
            m=args.fuzzy_m,
            seed=args.seed,
            max_iter=300,
            tol=1e-5,
            n_init=args.fuzzy_n_init,
        )
        diag_rows.append(diag)
        selected = (centers_c, memberships_c, diag)

    pd.DataFrame(diag_rows).to_csv(args.output_dir / "animal2vec_fuzzy_diagnostics_by_c.csv", index=False)
    centers, memberships, selected_diag = selected
    mem_cols = [f"fuzzy_{i}" for i in range(args.fuzzy_clusters)]
    mem = pd.DataFrame(memberships, columns=mem_cols)
    mem["fuzzy_entropy"] = entropy(memberships)
    mem["fuzzy_winner"] = memberships.argmax(axis=1)
    out = pd.concat([meta.reset_index(drop=True), mem], axis=1)
    out.to_csv(args.output_dir / "animal2vec_fuzzy_membership_by_sample.csv", index=False)

    label_summary = out.groupby("label")[mem_cols + ["fuzzy_entropy"]].mean()
    label_summary.to_csv(args.output_dir / "animal2vec_fuzzy_membership_by_label.csv")

    with open(args.output_dir / "animal2vec_fuzzy_selected_diagnostics.json", "w", encoding="utf-8") as f:
        import json

        json.dump(selected_diag, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(9, 7))
    plt.scatter(UMAP[:, 0], UMAP[:, 1], c=mem["fuzzy_entropy"], s=8, cmap="magma")
    plt.colorbar(label="fuzzy membership entropy")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("animal2vec fuzzy uncertainty on UMAP")
    plt.tight_layout()
    plt.savefig(args.figure_dir / "umap_animal2vec_fuzzy_entropy.png", dpi=180)
    plt.close()

    print(f"wrote diagnostics to {args.output_dir}")
    print(f"left island samples: {len(left)}")
    print(f"left island x cutoff: {left_cut:.4f}")
    if selected_diag["is_uniform_degenerate"]:
        print("WARNING: selected fuzzy solution is near-uniform and should not be interpreted.")
    print(
        "selected fuzzy:",
        f"c={args.fuzzy_clusters}",
        f"mean_max_membership={selected_diag['mean_max_membership']:.4f}",
        f"mean_relative_entropy={selected_diag['mean_relative_entropy']:.4f}",
    )


if __name__ == "__main__":
    main()
