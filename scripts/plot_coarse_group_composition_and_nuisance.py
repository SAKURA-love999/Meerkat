#!/usr/bin/env python
"""Plot and audit coarse-group composition for GMM and HDBSCAN.

This script answers:

1. What manual labels compose each GMM component and HDBSCAN cluster?
2. Do the algorithms cut the space along similar biological/acoustic directions?
3. Are groups explained by duration, quality, recording, or individual?
4. Which manual labels span multiple components?
5. Which candidate label pairs frequently share components?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_ORDER = ["sn", "cc", "ld", "mo", "al", "soc", "agg", "oth"]
PAIR_DEFAULT = "ld:mo,al:mo,agg:soc,cc:oth"

NUMERIC_PATTERNS = re.compile(
    r"duration|dur|rms|energy|loud|snr|clip|quality|frame|pooled|valid|f0|pitch|centroid|rolloff|bandwidth|zcr|mfcc",
    re.I,
)
CATEGORICAL_PATTERNS = re.compile(
    r"recording|file|wav|audio|session|date|group|location|mic|individual|subject|animal|caller|identity",
    re.I,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def component_from_soft(df: pd.DataFrame, prefix: str, winner_col: str) -> pd.Series:
    if winner_col in df.columns:
        return df[winner_col].astype(int)
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]
    if not cols:
        raise ValueError(f"No {prefix} membership columns found")
    return pd.Series(df[cols].to_numpy().argmax(axis=1), index=df.index)


def hdbscan_component(df: pd.DataFrame) -> pd.Series:
    if "hdbscan_label" in df.columns:
        return df["hdbscan_label"].astype(int)
    if "hdbscan_cluster" in df.columns:
        return df["hdbscan_cluster"].astype(int)
    return component_from_soft(df, "soft_hdbscan_", "soft_hdbscan_winner")


def ordered_labels(labels: pd.Series) -> list[str]:
    present = list(pd.unique(labels.astype(str)))
    ordered = [x for x in LABEL_ORDER if x in present]
    ordered += sorted(x for x in present if x not in ordered)
    return ordered


def composition_matrix(assign: pd.Series, labels: pd.Series, include_noise: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = pd.DataFrame({"component": assign.astype(int), "label": labels.astype(str)})
    if not include_noise:
        tmp = tmp.loc[tmp["component"] != -1]
    tab = pd.crosstab(tmp["component"], tmp["label"])
    labels_order = ordered_labels(tmp["label"])
    tab = tab.reindex(columns=labels_order, fill_value=0)
    frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)
    return tab, frac


def plot_heatmap(frac: pd.DataFrame, counts: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.7 * len(frac) + 1.8)))
    im = ax.imshow(frac.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(frac.shape[1]))
    ax.set_xticklabels(frac.columns, rotation=0)
    ax.set_yticks(np.arange(frac.shape[0]))
    ax.set_yticklabels(frac.index)
    ax.set_xlabel("manual label")
    ax.set_ylabel("component / cluster")
    ax.set_title(title)

    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            value = frac.iloc[i, j]
            n = counts.iloc[i, j]
            text_color = "white" if value >= 0.5 else "black"
            ax.text(j, i, f"{value:.2f}\n{int(n)}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("row fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def entropy_from_counts(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def label_spread(assign: pd.Series, labels: pd.Series, method: str, include_noise: bool) -> pd.DataFrame:
    rows = []
    tmp = pd.DataFrame({"component": assign.astype(int), "label": labels.astype(str)})
    if not include_noise:
        tmp = tmp.loc[tmp["component"] != -1]
    for lab, g in tmp.groupby("label"):
        counts = g["component"].value_counts().sort_index()
        rows.append(
            {
                "method": method,
                "label": lab,
                "n": int(len(g)),
                "n_components_used": int(counts.size),
                "largest_component": int(counts.idxmax()),
                "largest_component_fraction": float(counts.max() / counts.sum()),
                "component_entropy": entropy_from_counts(counts),
                "component_counts": json.dumps({int(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    out = []
    for part in text.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        out.append((a.strip(), b.strip()))
    return out


def pair_sharing(assign: pd.Series, labels: pd.Series, method: str, pairs: list[tuple[str, str]], include_noise: bool) -> pd.DataFrame:
    tmp = pd.DataFrame({"component": assign.astype(int), "label": labels.astype(str)})
    if not include_noise:
        tmp = tmp.loc[tmp["component"] != -1]
    rows = []
    for a, b in pairs:
        a_comp = tmp.loc[tmp["label"] == a, "component"]
        b_comp = tmp.loc[tmp["label"] == b, "component"]
        if len(a_comp) == 0 or len(b_comp) == 0:
            continue
        a_dist = a_comp.value_counts(normalize=True)
        b_dist = b_comp.value_counts(normalize=True)
        all_components = sorted(set(a_dist.index).union(b_dist.index))
        overlap = sum(min(float(a_dist.get(c, 0.0)), float(b_dist.get(c, 0.0))) for c in all_components)
        rows.append(
            {
                "method": method,
                "pair": f"{a}:{b}",
                "n_a": int(len(a_comp)),
                "n_b": int(len(b_comp)),
                "component_distribution_overlap": float(overlap),
                "a_top_component": int(a_dist.idxmax()),
                "a_top_fraction": float(a_dist.max()),
                "b_top_component": int(b_dist.idxmax()),
                "b_top_fraction": float(b_dist.max()),
                "same_top_component": bool(a_dist.idxmax() == b_dist.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def numeric_audit(df: pd.DataFrame, assign: pd.Series, method: str, include_noise: bool) -> pd.DataFrame:
    tmp = df.copy()
    tmp["component"] = assign.astype(int)
    if not include_noise:
        tmp = tmp.loc[tmp["component"] != -1]
    numeric_cols = []
    for col in tmp.columns:
        if col.startswith("soft_") or col in {"component", "label"}:
            continue
        if NUMERIC_PATTERNS.search(col):
            values = pd.to_numeric(tmp[col], errors="coerce")
            if values.notna().sum() > 0:
                numeric_cols.append(col)
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(tmp[col], errors="coerce")
        overall_sd = values.std(ddof=0)
        grouped = tmp.assign(_value=values).groupby("component")["_value"]
        means = grouped.mean()
        medians = grouped.median()
        rows.append(
            {
                "method": method,
                "variable": col,
                "n_nonmissing": int(values.notna().sum()),
                "overall_mean": float(values.mean()),
                "overall_median": float(values.median()),
                "component_mean_min": float(means.min()),
                "component_mean_max": float(means.max()),
                "component_mean_range": float(means.max() - means.min()),
                "component_mean_range_in_sd": float((means.max() - means.min()) / overall_sd) if overall_sd and np.isfinite(overall_sd) else np.nan,
                "component_means": json.dumps({int(k): float(v) for k, v in means.items()}, ensure_ascii=False),
                "component_medians": json.dumps({int(k): float(v) for k, v in medians.items()}, ensure_ascii=False),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "variable",
                "n_nonmissing",
                "overall_mean",
                "overall_median",
                "component_mean_min",
                "component_mean_max",
                "component_mean_range",
                "component_mean_range_in_sd",
                "component_means",
                "component_medians",
            ]
        )
    return pd.DataFrame(rows).sort_values("component_mean_range_in_sd", key=lambda s: s.abs(), ascending=False)


def categorical_audit(df: pd.DataFrame, assign: pd.Series, method: str, include_noise: bool) -> pd.DataFrame:
    tmp = df.copy()
    tmp["component"] = assign.astype(int)
    if not include_noise:
        tmp = tmp.loc[tmp["component"] != -1]
    cols = [c for c in tmp.columns if CATEGORICAL_PATTERNS.search(c)]
    rows = []
    for col in cols:
        values = tmp[col].astype(str).fillna("NA")
        component_rows = []
        for comp, g in tmp.assign(_value=values).groupby("component"):
            counts = g["_value"].value_counts()
            component_rows.append(
                {
                    "component": int(comp),
                    "top_value": counts.index[0],
                    "top_fraction": float(counts.iloc[0] / counts.sum()),
                    "n_unique": int(counts.size),
                }
            )
        top_fraction_max = max(r["top_fraction"] for r in component_rows) if component_rows else np.nan
        rows.append(
            {
                "method": method,
                "variable": col,
                "max_component_top_fraction": top_fraction_max,
                "component_dominance": json.dumps(component_rows, ensure_ascii=False),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "variable",
                "max_component_top_fraction",
                "component_dominance",
            ]
        )
    return pd.DataFrame(rows).sort_values("max_component_top_fraction", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm-samples", type=Path, default=Path("reports/animal2vec_soft_gmm_selected_by_sample.csv"))
    parser.add_argument("--hdbscan-samples", type=Path, default=Path("reports/animal2vec_soft_hdbscan_selected_by_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--pairs", default=PAIR_DEFAULT)
    parser.add_argument("--include-noise", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    gmm = read_csv(args.gmm_samples)
    hdb = read_csv(args.hdbscan_samples)
    if len(gmm) != len(hdb):
        raise ValueError(f"GMM rows ({len(gmm)}) != HDBSCAN rows ({len(hdb)})")
    if "label" not in gmm.columns:
        raise ValueError("GMM sample table must contain label")

    labels = gmm["label"].astype(str)
    gmm_assign = component_from_soft(gmm, "soft_gmm_", "soft_gmm_winner")
    hdb_assign = hdbscan_component(hdb)
    pairs = parse_pairs(args.pairs)

    outputs = {}
    for method, df, assign in [
        ("gmm", gmm, gmm_assign),
        ("hdbscan", hdb, hdb_assign),
    ]:
        counts, frac = composition_matrix(assign, labels, include_noise=args.include_noise)
        counts.to_csv(args.output_dir / f"animal2vec_{method}_coarse_group_label_counts_matrix.csv")
        frac.to_csv(args.output_dir / f"animal2vec_{method}_coarse_group_label_fraction_matrix.csv")
        plot_heatmap(
            frac,
            counts,
            title=f"animal2vec {method.upper()} coarse group label composition",
            out_path=args.figure_dir / f"animal2vec_{method}_coarse_group_label_composition_heatmap.png",
        )
        outputs[f"{method}_n_groups"] = int(len(counts))

    pd.concat(
        [
            label_spread(gmm_assign, labels, "gmm", include_noise=args.include_noise),
            label_spread(hdb_assign, labels, "hdbscan", include_noise=args.include_noise),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "animal2vec_coarse_group_label_spread.csv", index=False)

    pd.concat(
        [
            pair_sharing(gmm_assign, labels, "gmm", pairs, include_noise=args.include_noise),
            pair_sharing(hdb_assign, labels, "hdbscan", pairs, include_noise=args.include_noise),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "animal2vec_coarse_group_pair_sharing.csv", index=False)

    pd.concat(
        [
            numeric_audit(gmm, gmm_assign, "gmm", include_noise=args.include_noise),
            numeric_audit(hdb, hdb_assign, "hdbscan", include_noise=args.include_noise),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "animal2vec_coarse_group_numeric_nuisance_audit.csv", index=False)

    pd.concat(
        [
            categorical_audit(gmm, gmm_assign, "gmm", include_noise=args.include_noise),
            categorical_audit(hdb, hdb_assign, "hdbscan", include_noise=args.include_noise),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "animal2vec_coarse_group_categorical_nuisance_audit.csv", index=False)

    outputs.update(
        {
            "include_noise": bool(args.include_noise),
            "figures": [
                str(args.figure_dir / "animal2vec_gmm_coarse_group_label_composition_heatmap.png"),
                str(args.figure_dir / "animal2vec_hdbscan_coarse_group_label_composition_heatmap.png"),
            ],
            "tables": [
                str(args.output_dir / "animal2vec_coarse_group_label_spread.csv"),
                str(args.output_dir / "animal2vec_coarse_group_pair_sharing.csv"),
                str(args.output_dir / "animal2vec_coarse_group_numeric_nuisance_audit.csv"),
                str(args.output_dir / "animal2vec_coarse_group_categorical_nuisance_audit.csv"),
            ],
        }
    )
    with open(args.output_dir / "animal2vec_coarse_group_composition_nuisance_summary.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
