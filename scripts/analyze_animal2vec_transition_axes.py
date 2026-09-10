#!/usr/bin/env python
"""Test pairwise label transitions along centroid axes in animal2vec space.

This is a post-hoc analysis for candidate transitional pairs. It should be run
after fuzzy memberships have been generated.

The main axis is not UMAP. For each pair A/B, samples are projected onto the
centroid-to-centroid axis in PCA-reduced animal2vec space. We then ask whether
label composition, neighborhood mixing, and fuzzy membership change gradually
along that axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for part in text.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        pairs.append((a.strip(), b.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("features/animal2vec_context500ms_focal_mean_balanced_600.npz"))
    parser.add_argument("--membership", type=Path, default=Path("reports/animal2vec_fuzzy_membership_by_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--pairs", default="ld:mo,agg:soc,cc:oth,al:mo")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(np.load(args.npz, allow_pickle=False)["X"], dtype=np.float32)
    df = pd.read_csv(args.membership)
    if len(df) != len(X):
        raise ValueError(f"membership rows ({len(df)}) != X rows ({len(X)})")
    if "label" not in df.columns:
        raise ValueError("membership table must contain label")

    labels = df["label"].astype(str).to_numpy()
    fuzzy_cols = [c for c in df.columns if c.startswith("fuzzy_") and c != "fuzzy_entropy" and c != "fuzzy_winner"]
    fuzzy_degenerate = False
    fuzzy_mean_max = np.nan
    fuzzy_relative_entropy = np.nan
    if fuzzy_cols:
        U_fuzzy = df[fuzzy_cols].to_numpy(dtype=float)
        c_fuzzy = U_fuzzy.shape[1]
        fuzzy_mean_max = float(np.nanmean(np.nanmax(U_fuzzy, axis=1)))
        fuzzy_relative_entropy = float(
            np.nanmean(-np.sum(U_fuzzy * np.log(np.maximum(U_fuzzy, 1e-12)), axis=1) / np.log(c_fuzzy))
        )
        fuzzy_degenerate = bool(fuzzy_mean_max < (1.0 / c_fuzzy + 0.03) and fuzzy_relative_entropy > 0.98)

    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(Xs)
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(Xp)
    neigh = nn.kneighbors(Xp, return_distance=False)[:, 1:]

    sample_rows = []
    summary_rows = []
    bin_rows = []

    for a, b in parse_pairs(args.pairs):
        pair_mask = np.isin(labels, [a, b])
        if not pair_mask.any():
            continue
        if np.sum(labels == a) == 0 or np.sum(labels == b) == 0:
            continue

        ca = Xp[labels == a].mean(axis=0)
        cb = Xp[labels == b].mean(axis=0)
        axis = cb - ca
        denom = float(np.dot(axis, axis))
        if denom <= 0:
            continue

        t = ((Xp - ca) @ axis) / denom
        pair_idx = np.flatnonzero(pair_mask)
        pair_t = t[pair_idx]
        pair_labels = labels[pair_idx]

        for idx in pair_idx:
            neigh_labels = labels[neigh[idx]]
            row = {
                "pair": f"{a}:{b}",
                "row_index": int(idx),
                "label": labels[idx],
                "axis_position": float(t[idx]),
                "axis_position_clipped": float(np.clip(t[idx], 0.0, 1.0)),
                f"knn{args.knn}_{a}_fraction": float(np.mean(neigh_labels == a)),
                f"knn{args.knn}_{b}_fraction": float(np.mean(neigh_labels == b)),
                "fuzzy_entropy": float(df.loc[idx, "fuzzy_entropy"]) if "fuzzy_entropy" in df.columns else np.nan,
            }
            for col in fuzzy_cols:
                row[col] = float(df.loc[idx, col])
            sample_rows.append(row)

        tmp = pd.DataFrame([r for r in sample_rows if r["pair"] == f"{a}:{b}"])
        tmp["bin"] = pd.cut(tmp["axis_position_clipped"], bins=np.linspace(0, 1, args.n_bins + 1), include_lowest=True)
        grouped = tmp.groupby("bin", observed=False)
        for bin_name, g in grouped:
            if len(g) == 0:
                continue
            row = {
                "pair": f"{a}:{b}",
                "bin": str(bin_name),
                "n": int(len(g)),
                f"prop_{b}": float(np.mean(g["label"] == b)),
                "axis_mean": float(g["axis_position"].mean()),
                "fuzzy_entropy_mean": float(g["fuzzy_entropy"].mean()) if "fuzzy_entropy" in g.columns else np.nan,
                f"knn{args.knn}_{a}_fraction_mean": float(g[f"knn{args.knn}_{a}_fraction"].mean()),
                f"knn{args.knn}_{b}_fraction_mean": float(g[f"knn{args.knn}_{b}_fraction"].mean()),
            }
            for col in fuzzy_cols:
                row[f"{col}_mean"] = float(g[col].mean())
            bin_rows.append(row)

        # Spearman correlation uses label order only for interpretation.
        label_binary = (pair_labels == b).astype(float)
        axis_series = pd.Series(pair_t)
        summary_rows.append(
            {
                "pair": f"{a}:{b}",
                "label_a": a,
                "label_b": b,
                "n_pair_samples": int(len(pair_idx)),
                "centroid_axis_distance": float(np.linalg.norm(axis)),
                "axis_label_spearman": float(axis_series.corr(pd.Series(label_binary), method="spearman")),
                "middle_axis_fraction_0.25_0.75": float(np.mean((pair_t >= 0.25) & (pair_t <= 0.75))),
                "a_median_axis": float(np.median(pair_t[pair_labels == a])),
                "b_median_axis": float(np.median(pair_t[pair_labels == b])),
                "axis_overlap_fraction_a_above_0.25": float(np.mean(pair_t[pair_labels == a] > 0.25)),
                "axis_overlap_fraction_b_below_0.75": float(np.mean(pair_t[pair_labels == b] < 0.75)),
                "fuzzy_degenerate_warning": fuzzy_degenerate,
                "fuzzy_mean_max_membership": fuzzy_mean_max,
                "fuzzy_mean_relative_entropy": fuzzy_relative_entropy,
            }
        )

    pd.DataFrame(sample_rows).to_csv(args.output_dir / "animal2vec_pair_transition_axis_by_sample.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(args.output_dir / "animal2vec_pair_transition_axis_bins.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "animal2vec_pair_transition_axis_summary.csv", index=False)
    if fuzzy_degenerate:
        with open(args.output_dir / "animal2vec_pair_transition_axis_fuzzy_warning.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "warning": "Fuzzy memberships are near-uniform; do not interpret fuzzy transition evidence.",
                    "fuzzy_mean_max_membership": fuzzy_mean_max,
                    "fuzzy_mean_relative_entropy": fuzzy_relative_entropy,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("WARNING: fuzzy memberships are near-uniform; transition-axis fuzzy evidence is not interpretable.")
    print(f"wrote transition-axis diagnostics for {len(summary_rows)} pairs")


if __name__ == "__main__":
    main()
