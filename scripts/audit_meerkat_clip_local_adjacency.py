#!/usr/bin/env python
"""Audit clip-local temporal-adjacency rates in public MeerKAT annotations.

This script does not infer vocal responses. It estimates, for focal=True events,
whether a non-focal event occurs before/after the focal event within clip-local
windows. Events near the 10-s clip boundary are right/left censored and excluded
from the corresponding rate denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Could not find any of {candidates}. Columns: {list(df.columns)}")


def event_window_has_nonfocal(
    nonfocal_starts: np.ndarray,
    nonfocal_ends: np.ndarray,
    start: float,
    end: float,
    tau: float,
    direction: str,
) -> bool:
    if direction == "post":
        idx = np.searchsorted(nonfocal_starts, end, side="left")
        return bool(idx < len(nonfocal_starts) and nonfocal_starts[idx] <= end + tau)
    if direction == "pre":
        idx = np.searchsorted(nonfocal_ends, start, side="right") - 1
        return bool(idx >= 0 and nonfocal_ends[idx] >= start - tau)
    raise ValueError(direction)


def audit(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    clip_col = args.clip_col or first_existing(df, ["source_id", "clip_id", "file_id", "source_wav", "recording"])
    label_col = args.label_col or first_existing(df, ["label", "call_type", "manual_label", "type"])
    start_col = args.start_col or first_existing(df, ["start_time_sec", "start_s", "start_sec", "onset_s", "start"])
    end_col = args.end_col or first_existing(df, ["end_time_sec", "end_s", "end_sec", "offset_s", "end"])
    focal_col = args.focal_col or first_existing(df, ["focal", "is_focal"])

    work = df.copy()
    work["_clip"] = work[clip_col].astype(str)
    work["_label"] = work[label_col].astype(str)
    work["_start"] = pd.to_numeric(work[start_col], errors="coerce")
    work["_end"] = pd.to_numeric(work[end_col], errors="coerce")
    work["_focal"] = bool_series(work[focal_col])
    work = work.dropna(subset=["_start", "_end"])

    windows_ms = [float(x) for x in args.windows_ms.split(",") if x.strip()]
    if args.skip_detail_csv:
        focal = work.loc[work["_focal"], ["_clip", "_label", "_start", "_end"]].copy()
        focal["row_index"] = focal.index
        nonfocal = work.loc[~work["_focal"], ["_clip", "_start", "_end"]].copy()

        left_post = focal.sort_values(["_end", "_clip"]).reset_index(drop=True)
        right_post = (
            nonfocal[["_clip", "_start"]]
            .rename(columns={"_start": "next_nonfocal_start"})
            .sort_values(["next_nonfocal_start", "_clip"])
            .reset_index(drop=True)
        )
        post = pd.merge_asof(
            left_post,
            right_post,
            by="_clip",
            left_on="_end",
            right_on="next_nonfocal_start",
            direction="forward",
            allow_exact_matches=True,
        )

        left_pre = focal.sort_values(["_start", "_clip"]).reset_index(drop=True)
        right_pre = (
            nonfocal[["_clip", "_end"]]
            .rename(columns={"_end": "prev_nonfocal_end"})
            .sort_values(["prev_nonfocal_end", "_clip"])
            .reset_index(drop=True)
        )
        pre = pd.merge_asof(
            left_pre,
            right_pre,
            by="_clip",
            left_on="_start",
            right_on="prev_nonfocal_end",
            direction="backward",
            allow_exact_matches=True,
        )

        nearest = post[["row_index", "_clip", "_label", "_start", "_end", "next_nonfocal_start"]].merge(
            pre[["row_index", "prev_nonfocal_end"]], on="row_index", how="left"
        )
        rows = []
        for tau_ms in windows_ms:
            tau = tau_ms / 1000.0
            tmp = nearest.copy()
            tmp["post_censored"] = tmp["_end"] > (args.clip_duration_s - tau)
            tmp["pre_censored"] = tmp["_start"] < tau
            tmp["nf_post"] = (~tmp["post_censored"]) & tmp["next_nonfocal_start"].notna() & (
                tmp["next_nonfocal_start"] <= tmp["_end"] + tau
            )
            tmp["nf_pre"] = (~tmp["pre_censored"]) & tmp["prev_nonfocal_end"].notna() & (
                tmp["prev_nonfocal_end"] >= tmp["_start"] - tau
            )
            for label, g in tmp.groupby("_label", sort=True):
                post_eligible = int((~g["post_censored"]).sum())
                pre_eligible = int((~g["pre_censored"]).sum())
                post_positive = int(g["nf_post"].sum())
                pre_positive = int(g["nf_pre"].sum())
                p_post = post_positive / post_eligible if post_eligible else np.nan
                p_pre = pre_positive / pre_eligible if pre_eligible else np.nan
                n_focal_targets = int(len(g))
                rows.append(
                    {
                        "label": label,
                        "window_ms": tau_ms,
                        "n_focal_targets": n_focal_targets,
                        "post_censored_n": int(g["post_censored"].sum()),
                        "post_censored_fraction": float(g["post_censored"].mean()),
                        "post_eligible_n": post_eligible,
                        "post_positive_n": post_positive,
                        "p_nf_post": float(p_post),
                        "pre_censored_n": int(g["pre_censored"].sum()),
                        "pre_censored_fraction": float(g["pre_censored"].mean()),
                        "pre_eligible_n": pre_eligible,
                        "pre_positive_n": pre_positive,
                        "p_nf_pre": float(p_pre),
                        "post_minus_pre": float(p_post - p_pre) if post_eligible and pre_eligible else np.nan,
                    }
                )
        summary = pd.DataFrame(rows).sort_values(["window_ms", "label"])
        detail = focal[["row_index"]].copy()
        return detail, summary

    detail_rows = []
    for clip_id, g in work.groupby("_clip", sort=False):
        g = g.sort_values("_start")
        nonfocal = g.loc[~g["_focal"]]
        nf_starts = np.sort(nonfocal["_start"].to_numpy(float))
        nf_ends = np.sort(nonfocal["_end"].to_numpy(float))
        for row_idx, row in g.loc[g["_focal"]].iterrows():
            for tau_ms in windows_ms:
                tau = tau_ms / 1000.0
                post_censored = row["_end"] > (args.clip_duration_s - tau)
                pre_censored = row["_start"] < tau
                detail_rows.append(
                    {
                        "row_index": int(row_idx),
                        "clip_id": clip_id,
                        "label": row["_label"],
                        "start_time_sec": float(row["_start"]),
                        "end_time_sec": float(row["_end"]),
                        "window_ms": tau_ms,
                        "post_censored": bool(post_censored),
                        "pre_censored": bool(pre_censored),
                        "nf_post": False if post_censored else event_window_has_nonfocal(
                            nf_starts, nf_ends, row["_start"], row["_end"], tau, "post"
                        ),
                        "nf_pre": False if pre_censored else event_window_has_nonfocal(
                            nf_starts, nf_ends, row["_start"], row["_end"], tau, "pre"
                        ),
                    }
                )
    detail = pd.DataFrame(detail_rows)

    rows = []
    for (label, tau), g in detail.groupby(["label", "window_ms"]):
        post_eligible = g.loc[~g["post_censored"]]
        pre_eligible = g.loc[~g["pre_censored"]]
        rows.append(
            {
                "label": label,
                "window_ms": tau,
                "n_focal_targets": int(len(g)),
                "post_censored_n": int(g["post_censored"].sum()),
                "post_censored_fraction": float(g["post_censored"].mean()),
                "post_eligible_n": int(len(post_eligible)),
                "post_positive_n": int(post_eligible["nf_post"].sum()),
                "p_nf_post": float(post_eligible["nf_post"].mean()) if len(post_eligible) else np.nan,
                "pre_censored_n": int(g["pre_censored"].sum()),
                "pre_censored_fraction": float(g["pre_censored"].mean()),
                "pre_eligible_n": int(len(pre_eligible)),
                "pre_positive_n": int(pre_eligible["nf_pre"].sum()),
                "p_nf_pre": float(pre_eligible["nf_pre"].mean()) if len(pre_eligible) else np.nan,
                "post_minus_pre": (
                    float(post_eligible["nf_post"].mean()) - float(pre_eligible["nf_pre"].mean())
                    if len(post_eligible) and len(pre_eligible)
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["window_ms", "label"])
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest/call_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/meerkat_proxy_audit"))
    parser.add_argument("--windows-ms", default="400,800,1500")
    parser.add_argument("--clip-duration-s", type=float, default=10.0)
    parser.add_argument("--clip-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    parser.add_argument("--focal-col", default=None)
    parser.add_argument(
        "--skip-detail-csv",
        action="store_true",
        help="Only write label/window summary rates, not the per-focal-event detail CSV.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    detail, summary = audit(df, args)
    detail_path = args.output_dir / "meerkat_clip_local_prepost_nf_by_focal_event.csv"
    if not args.skip_detail_csv:
        detail.to_csv(detail_path, index=False)
    summary.to_csv(args.output_dir / "meerkat_clip_local_prepost_nf_rates_by_label.csv", index=False)

    overall = {
        "manifest": str(args.manifest),
        "n_events": int(len(df)),
        "n_focal_events": int(len(detail[["row_index"]].drop_duplicates())),
        "windows_ms": [float(x) for x in args.windows_ms.split(",") if x.strip()],
        "clip_duration_s": args.clip_duration_s,
        "output_detail": None if args.skip_detail_csv else str(detail_path),
        "output_summary": str(args.output_dir / "meerkat_clip_local_prepost_nf_rates_by_label.csv"),
    }
    with open(args.output_dir / "meerkat_clip_local_adjacency_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
