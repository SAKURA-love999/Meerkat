#!/usr/bin/env python
"""Robust soft-membership diagnostics for animal2vec embeddings.

The previous c=8 fuzzy c-means result can collapse to a uniform 1/c solution in
PCA-50 space. This script treats that as a numerical diagnostic, not a
biological result. It compares several soft assignment strategies:

1. Fuzzy c-means across PCA dimensions, c values, and fuzzifier m values.
2. Gaussian mixture posterior probabilities.
3. HDBSCAN soft membership vectors for one fixed density-clustering setting.

Labels are used only for post-hoc summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from hdbscan.prediction import all_points_membership_vectors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler


def load_inputs(npz_path: Path, metadata_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.asarray(np.load(npz_path, allow_pickle=False)["X"], dtype=np.float32)
    meta = pd.read_csv(metadata_path)
    if len(meta) != len(X):
        raise ValueError(f"metadata rows ({len(meta)}) != X rows ({len(X)})")
    if "label" not in meta.columns:
        raise ValueError("metadata must contain a label column")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X, meta


def pca_space(X: np.ndarray, dim: int, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    if dim > 0 and Xs.shape[1] > dim:
        return PCA(n_components=dim, random_state=seed).fit_transform(Xs)
    return Xs


def entropy(U: np.ndarray) -> np.ndarray:
    return -np.sum(U * np.log(np.maximum(U, 1e-12)), axis=1)


def membership_diagnostics(U: np.ndarray, centers: np.ndarray | None = None, X: np.ndarray | None = None) -> dict[str, float]:
    c = U.shape[1]
    max_mem = np.max(U, axis=1)
    ent = entropy(U)
    rel_ent = ent / np.log(c) if c > 1 else np.zeros(len(U))
    out = {
        "c": int(c),
        "mean_max_membership": float(np.mean(max_mem)),
        "median_max_membership": float(np.median(max_mem)),
        "p05_max_membership": float(np.quantile(max_mem, 0.05)),
        "p95_max_membership": float(np.quantile(max_mem, 0.95)),
        "mean_entropy": float(np.mean(ent)),
        "median_entropy": float(np.median(ent)),
        "mean_relative_entropy": float(np.mean(rel_ent)),
        "median_relative_entropy": float(np.median(rel_ent)),
        "partition_coefficient": float(np.mean(np.sum(U * U, axis=1))),
        "is_uniform_degenerate": bool(np.mean(max_mem) < (1.0 / c + 0.03) and np.mean(rel_ent) > 0.98),
    }
    if centers is not None:
        center_dist = pairwise_distances(centers)
        off_diag = center_dist[np.triu_indices_from(center_dist, k=1)]
        out["min_center_distance"] = float(np.min(off_diag)) if len(off_diag) else np.nan
        out["median_center_distance"] = float(np.median(off_diag)) if len(off_diag) else np.nan
    if centers is not None and X is not None:
        nearest = np.min(pairwise_distances(X, centers), axis=1)
        out["median_nearest_center_distance"] = float(np.median(nearest))
        out["xie_beni_index_m2"] = float(xie_beni_index(X, centers, U, m=2.0))
    return out


def xie_beni_index(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float) -> float:
    center_dist2 = pairwise_distances(centers) ** 2
    off = center_dist2[np.triu_indices_from(center_dist2, k=1)]
    min_center_dist2 = float(np.min(off)) if len(off) else 0.0
    if min_center_dist2 <= 1e-20:
        return np.inf
    dist2 = pairwise_distances(X, centers) ** 2
    return float(np.sum((U ** m) * dist2) / (len(X) * min_center_dist2))


def label_correspondence(meta: pd.DataFrame, U: np.ndarray, prefix: str) -> dict[str, float]:
    labels = meta["label"].astype(str).to_numpy()
    winner = np.argmax(U, axis=1)
    tab = pd.crosstab(pd.Series(winner, name="component"), pd.Series(labels, name="label"))
    component_purity = tab.max(axis=1) / tab.sum(axis=1)
    label_tab = pd.crosstab(pd.Series(labels, name="label"), pd.Series(winner, name="component"))
    label_recall = label_tab.max(axis=1) / label_tab.sum(axis=1)
    return {
        f"{prefix}_winner_label_ami": float(adjusted_mutual_info_score(labels, winner)),
        f"{prefix}_winner_label_ari": float(adjusted_rand_score(labels, winner)),
        f"{prefix}_mean_component_label_purity": float(component_purity.mean()),
        f"{prefix}_weighted_component_label_purity": float(np.average(component_purity, weights=tab.sum(axis=1))),
        f"{prefix}_mean_label_best_component_fraction": float(label_recall.mean()),
    }


def update_fcm_membership(X: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    dist = np.maximum(pairwise_distances(X, centers), 1e-12)
    inv = dist ** (-2.0 / (m - 1.0))
    return inv / np.sum(inv, axis=1, keepdims=True)


def run_fcm(X: np.ndarray, c: int, m: float, seed: int, n_init: int, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for i in range(n_init):
        km = KMeans(n_clusters=c, n_init=1, random_state=seed + i).fit(X)
        centers = km.cluster_centers_.astype(float)
        U = update_fcm_membership(X, centers, m=m)
        for _ in range(max_iter):
            old = U.copy()
            Um = U ** m
            centers = (Um.T @ X) / np.maximum(np.sum(Um, axis=0)[:, None], 1e-12)
            U = update_fcm_membership(X, centers, m=m)
            if np.max(np.abs(U - old)) < tol:
                break
        obj = float(np.sum((U ** m) * (pairwise_distances(X, centers) ** 2)))
        if best is None or obj < best[0]:
            best = (obj, centers, U)
    assert best is not None
    return best[1], best[2], best[0]


def label_membership_summary(meta: pd.DataFrame, U: np.ndarray, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(U.shape[1])]
    df = pd.concat([meta[["label"]].reset_index(drop=True), pd.DataFrame(U, columns=cols)], axis=1)
    return df.groupby("label")[cols].mean().reset_index()


def write_component_label_tables(meta: pd.DataFrame, U: np.ndarray, output_dir: Path, prefix: str) -> None:
    labels = meta["label"].astype(str).to_numpy()
    winner = np.argmax(U, axis=1)
    hard = pd.crosstab(pd.Series(winner, name="component"), pd.Series(labels, name="label"))
    hard.to_csv(output_dir / f"animal2vec_{prefix}_component_label_crosstab.csv")
    hard_fraction = hard.div(hard.sum(axis=1), axis=0)
    hard_fraction.to_csv(output_dir / f"animal2vec_{prefix}_component_label_fraction.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pca-dims", default="2,3,5,10,20,50")
    parser.add_argument("--c-values", default="2,3,4,5,6,8,10,12")
    parser.add_argument("--m-values", default="1.2,1.5,2.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--selected-pca-dim", type=int, default=10)
    parser.add_argument("--selected-c", type=int, default=3)
    parser.add_argument("--selected-m", type=float, default=1.5)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, meta = load_inputs(args.npz, args.metadata)
    pca_dims = [int(x) for x in args.pca_dims.split(",") if x.strip()]
    c_values = [int(x) for x in args.c_values.split(",") if x.strip()]
    m_values = [float(x) for x in args.m_values.split(",") if x.strip()]

    rows = []
    selected_U = None
    selected_X = None
    selected_centers = None

    for dim in pca_dims:
        Xp = pca_space(X, dim=dim, seed=args.seed)
        for c in c_values:
            for m in m_values:
                centers, U, obj = run_fcm(Xp, c=c, m=m, seed=args.seed, n_init=args.n_init, max_iter=300, tol=1e-5)
                row = membership_diagnostics(U, centers=centers, X=Xp)
                row.update({"method": "fuzzy_c_means", "pca_dim": dim, "m": m, "objective": obj})
                row.update(label_correspondence(meta, U, "fcm"))
                row["xie_beni_index"] = float(xie_beni_index(Xp, centers, U, m=m))
                rows.append(row)
                if dim == args.selected_pca_dim and c == args.selected_c and abs(m - args.selected_m) < 1e-9:
                    selected_U = U
                    selected_X = Xp
                    selected_centers = centers

    pd.DataFrame(rows).to_csv(args.output_dir / "animal2vec_soft_membership_fcm_grid.csv", index=False)

    if selected_U is not None:
        cols = [f"soft_fcm_{i}" for i in range(selected_U.shape[1])]
        out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(selected_U, columns=cols)], axis=1)
        out["soft_fcm_entropy"] = entropy(selected_U)
        out["soft_fcm_winner"] = selected_U.argmax(axis=1)
        out.to_csv(args.output_dir / "animal2vec_soft_fcm_selected_by_sample.csv", index=False)
        label_membership_summary(meta, selected_U, "soft_fcm").to_csv(
            args.output_dir / "animal2vec_soft_fcm_selected_by_label.csv", index=False
        )
        write_component_label_tables(meta, selected_U, args.output_dir, "soft_fcm_selected")
        pd.DataFrame(pairwise_distances(selected_centers)).to_csv(
            args.output_dir / "animal2vec_soft_fcm_selected_centroid_distance_matrix.csv", index=False
        )
        selected_diag = membership_diagnostics(selected_U, centers=selected_centers, X=selected_X)
        selected_diag.update({"method": "fuzzy_c_means", "pca_dim": args.selected_pca_dim, "m": args.selected_m})
        selected_diag.update(label_correspondence(meta, selected_U, "fcm"))
        selected_diag["xie_beni_index"] = float(xie_beni_index(selected_X, selected_centers, selected_U, m=args.selected_m))
        with open(args.output_dir / "animal2vec_soft_fcm_selected_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(selected_diag, f, indent=2, ensure_ascii=False)

    # GMM posterior as a model-based soft-clustering baseline.
    Xg = pca_space(X, dim=args.selected_pca_dim, seed=args.seed)
    gmm = GaussianMixture(n_components=args.selected_c, covariance_type="full", random_state=args.seed, n_init=5)
    gmm.fit(Xg)
    Ug = gmm.predict_proba(Xg)
    gmm_cols = [f"soft_gmm_{i}" for i in range(Ug.shape[1])]
    gmm_out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(Ug, columns=gmm_cols)], axis=1)
    gmm_out["soft_gmm_entropy"] = entropy(Ug)
    gmm_out["soft_gmm_winner"] = Ug.argmax(axis=1)
    gmm_out.to_csv(args.output_dir / "animal2vec_soft_gmm_selected_by_sample.csv", index=False)
    label_membership_summary(meta, Ug, "soft_gmm").to_csv(
        args.output_dir / "animal2vec_soft_gmm_selected_by_label.csv", index=False
    )
    write_component_label_tables(meta, Ug, args.output_dir, "soft_gmm_selected")
    pd.DataFrame(pairwise_distances(gmm.means_)).to_csv(
        args.output_dir / "animal2vec_soft_gmm_selected_centroid_distance_matrix.csv", index=False
    )
    gmm_diag = membership_diagnostics(Ug, centers=gmm.means_, X=Xg)
    gmm_diag.update({"method": "gmm_posterior", "pca_dim": args.selected_pca_dim, "bic": float(gmm.bic(Xg))})
    gmm_diag.update(label_correspondence(meta, Ug, "gmm"))
    with open(args.output_dir / "animal2vec_soft_gmm_selected_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(gmm_diag, f, indent=2, ensure_ascii=False)

    # HDBSCAN soft membership for the fixed 3-cluster-like setting.
    Xh = pca_space(X, dim=50, seed=args.seed)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        prediction_data=True,
        metric="euclidean",
    ).fit(Xh)
    labels = clusterer.labels_
    n_clusters = len([c for c in np.unique(labels) if c != -1])
    if n_clusters > 1:
        Uh = all_points_membership_vectors(clusterer)
        h_cols = [f"soft_hdbscan_{i}" for i in range(Uh.shape[1])]
        h_out = pd.concat([meta.reset_index(drop=True), pd.DataFrame(Uh, columns=h_cols)], axis=1)
        h_out["hdbscan_label"] = labels
        h_out["soft_hdbscan_entropy"] = entropy(Uh)
        h_out["soft_hdbscan_winner"] = Uh.argmax(axis=1)
        h_out.to_csv(args.output_dir / "animal2vec_soft_hdbscan_selected_by_sample.csv", index=False)
        label_membership_summary(meta, Uh, "soft_hdbscan").to_csv(
            args.output_dir / "animal2vec_soft_hdbscan_selected_by_label.csv", index=False
        )
        write_component_label_tables(meta, Uh, args.output_dir, "soft_hdbscan_selected")
        h_diag = membership_diagnostics(Uh)
        h_diag.update(label_correspondence(meta, Uh, "hdbscan"))
    else:
        h_diag = {"method": "hdbscan_soft_membership", "n_clusters": n_clusters, "warning": "not enough clusters for soft membership"}
    h_diag.update(
        {
            "method": "hdbscan_soft_membership",
            "pca_dim": 50,
            "n_clusters": int(n_clusters),
            "noise_rate": float(np.mean(labels == -1)),
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
        }
    )
    with open(args.output_dir / "animal2vec_soft_hdbscan_selected_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(h_diag, f, indent=2, ensure_ascii=False)

    print("wrote soft membership robustness outputs")
    if selected_U is not None:
        print("selected FCM:", json.dumps(selected_diag, ensure_ascii=False))
    print("selected GMM:", json.dumps(gmm_diag, ensure_ascii=False))
    print("selected HDBSCAN:", json.dumps(h_diag, ensure_ascii=False))


if __name__ == "__main__":
    main()
