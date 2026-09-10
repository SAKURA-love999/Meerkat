#!/usr/bin/env python
"""Compare GMM and HDBSCAN coarse groups in animal2vec space.

Questions answered:

1. Which manual labels compose each GMM component?
2. Which manual labels compose each HDBSCAN cluster?
3. Do GMM and HDBSCAN assign the same samples to the same coarse groups?

The comparison is unsupervised with respect to fitting. Manual labels are used
only for post-hoc interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def choose_label_col(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    candidates = [c for c in df.columns if "label" in c.lower() or "type" in c.lower()]
    if not candidates:
        raise ValueError("No label column found")
    return candidates[0]


def find_gmm_winner(df: pd.DataFrame) -> pd.Series:
    if "soft_gmm_winner" in df.columns:
        return df["soft_gmm_winner"].astype(int)
    cols = [c for c in df.columns if c.startswith("soft_gmm_") and c[len("soft_gmm_") :].isdigit()]
    if not cols:
        raise ValueError("No GMM posterior columns found")
    return df[cols].to_numpy().argmax(axis=1)


def find_hdbscan_winner(df: pd.DataFrame) -> pd.Series:
    if "hdbscan_label" in df.columns:
        return df["hdbscan_label"].astype(int)
    if "hdbscan_cluster" in df.columns:
        return df["hdbscan_cluster"].astype(int)
    if "soft_hdbscan_winner" in df.columns:
        return df["soft_hdbscan_winner"].astype(int)
    cols = [c for c in df.columns if c.startswith("soft_hdbscan_") and c[len("soft_hdbscan_") :].isdigit()]
    if not cols:
        raise ValueError("No HDBSCAN assignment or soft-membership columns found")
    return df[cols].to_numpy().argmax(axis=1)


def composition(assign: pd.Series, labels: pd.Series, name: str) -> pd.DataFrame:
    tab = pd.crosstab(assign.rename(name), labels.rename("label"))
    frac = tab.div(tab.sum(axis=1), axis=0).add_suffix("_fraction")
    out = pd.concat([tab.add_suffix("_n"), frac], axis=1).reset_index()
    out["n_samples"] = tab.sum(axis=1).to_numpy()
    out["top_label"] = tab.idxmax(axis=1).to_numpy()
    out["top_label_fraction"] = (tab.max(axis=1) / tab.sum(axis=1)).to_numpy()
    return out


def coassignment_matrix(a: np.ndarray, b: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(pd.Series(a, name="gmm_component"), pd.Series(b, name="hdbscan_cluster"))
    row_frac = counts.div(counts.sum(axis=1), axis=0)
    return counts, row_frac


def best_component_matching(counts: pd.DataFrame) -> pd.DataFrame:
    arr = counts.to_numpy()
    if arr.size == 0:
        return pd.DataFrame()
    row_ind, col_ind = linear_sum_assignment(-arr)
    rows = []
    for r, c in zip(row_ind, col_ind):
        rows.append(
            {
                "gmm_component": counts.index[r],
                "matched_hdbscan_cluster": counts.columns[c],
                "overlap_n": int(arr[r, c]),
                "gmm_component_n": int(arr[r, :].sum()),
                "hdbscan_cluster_n": int(arr[:, c].sum()),
                "gmm_fraction_in_matched_hdbscan": float(arr[r, c] / arr[r, :].sum()) if arr[r, :].sum() else np.nan,
                "hdbscan_fraction_from_matched_gmm": float(arr[r, c] / arr[:, c].sum()) if arr[:, c].sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pairwise_coassignment_agreement(a: np.ndarray, b: np.ndarray, max_pairs: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i = i[keep]
        j = j[keep]
    same_a = a[i] == a[j]
    same_b = b[i] == b[j]
    return {
        "pairwise_same_group_agreement": float(np.mean(same_a == same_b)),
        "pairwise_both_same_fraction": float(np.mean(same_a & same_b)),
        "pairwise_gmm_same_hdbscan_different_fraction": float(np.mean(same_a & ~same_b)),
        "pairwise_hdbscan_same_gmm_different_fraction": float(np.mean(~same_a & same_b)),
        "n_pairs_evaluated": int(len(i)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm-samples", type=Path, default=Path("reports/animal2vec_soft_gmm_selected_by_sample.csv"))
    parser.add_argument("--hdbscan-samples", type=Path, default=Path("reports/animal2vec_soft_hdbscan_selected_by_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--exclude-hdbscan-noise", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gmm = read_table(args.gmm_samples)
    hdb = read_table(args.hdbscan_samples)
    if len(gmm) != len(hdb):
        raise ValueError(f"GMM rows ({len(gmm)}) != HDBSCAN rows ({len(hdb)})")

    label_col = choose_label_col(gmm)
    labels = gmm[label_col].astype(str)
    gmm_assign = pd.Series(find_gmm_winner(gmm), name="gmm_component")
    hdb_assign = pd.Series(find_hdbscan_winner(hdb), name="hdbscan_cluster")

    if args.exclude_hdbscan_noise:
        mask = hdb_assign.to_numpy() != -1
        labels = labels.loc[mask].reset_index(drop=True)
        gmm_assign = gmm_assign.loc[mask].reset_index(drop=True)
        hdb_assign = hdb_assign.loc[mask].reset_index(drop=True)

    gmm_comp = composition(gmm_assign, labels, "gmm_component")
    hdb_comp = composition(hdb_assign, labels, "hdbscan_cluster")
    gmm_comp.to_csv(args.output_dir / "animal2vec_gmm3_manual_label_composition.csv", index=False)
    hdb_comp.to_csv(args.output_dir / "animal2vec_hdbscan_manual_label_composition.csv", index=False)

    counts, row_frac = coassignment_matrix(gmm_assign.to_numpy(), hdb_assign.to_numpy())
    counts.to_csv(args.output_dir / "animal2vec_gmm_hdbscan_component_crosstab_counts.csv")
    row_frac.to_csv(args.output_dir / "animal2vec_gmm_hdbscan_component_crosstab_row_fraction.csv")

    matching = best_component_matching(counts)
    matching.to_csv(args.output_dir / "animal2vec_gmm_hdbscan_best_component_matching.csv", index=False)

    g = gmm_assign.to_numpy()
    h = hdb_assign.to_numpy()
    summary = {
        "n_samples_compared": int(len(g)),
        "exclude_hdbscan_noise": bool(args.exclude_hdbscan_noise),
        "gmm_n_components": int(len(np.unique(g))),
        "hdbscan_n_clusters_including_noise": int(len(np.unique(h))),
        "hdbscan_noise_rate_in_compared_samples": float(np.mean(h == -1)),
        "gmm_hdbscan_ami": float(adjusted_mutual_info_score(g, h)),
        "gmm_hdbscan_nmi": float(normalized_mutual_info_score(g, h)),
        "gmm_hdbscan_ari": float(adjusted_rand_score(g, h)),
        "gmm_label_ami": float(adjusted_mutual_info_score(labels, g)),
        "gmm_label_ari": float(adjusted_rand_score(labels, g)),
        "hdbscan_label_ami": float(adjusted_mutual_info_score(labels, h)),
        "hdbscan_label_ari": float(adjusted_rand_score(labels, h)),
    }
    summary.update(pairwise_coassignment_agreement(g, h, max_pairs=args.max_pairs, seed=args.seed))
    if len(matching):
        summary["matched_overlap_total_n"] = int(matching["overlap_n"].sum())
        summary["matched_overlap_fraction_of_samples"] = float(matching["overlap_n"].sum() / len(g))
        summary["mean_gmm_fraction_in_matched_hdbscan"] = float(matching["gmm_fraction_in_matched_hdbscan"].mean())

    with open(args.output_dir / "animal2vec_gmm_hdbscan_coarse_group_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
