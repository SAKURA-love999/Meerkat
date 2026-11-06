#!/usr/bin/env python
"""Audit Egyptian fruit bat segment-to-interaction mapping.

This is the first feasibility check for calibrated proxy validity. It does not
fit a response model. It asks whether voiced segments in FileInfo.csv can be
reliably mapped to annotated interaction intervals in Annotations.csv.

Ground truth in this audit:

    same annotated interaction interval

not:

    whether the second vocalization is a response to the first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd


FILE_ID_CANDIDATES = ("File ID", "FileID", "File Id", "file_id", "file")
CONTEXT_CANDIDATES = ("Context", "context")
EMITTER_CANDIDATES = ("Emitter", "emitter")
ADDRESSEE_CANDIDATES = ("Addressee", "addressee")
ANN_START_CANDIDATES = ("Start sample", "Annotation start sample", "annotation_start_sample", "start_sample")
ANN_END_CANDIDATES = ("End sample", "Annotation end sample", "annotation_end_sample", "end_sample")
SEG_START_CANDIDATES = (
    "Voiced segment start sample",
    "Segment start sample",
    "Start sample",
    "start_sample",
    "voiced_start_sample",
    "segment_start_sample",
)
SEG_END_CANDIDATES = (
    "Voiced segment end sample",
    "Segment end sample",
    "End sample",
    "end_sample",
    "voiced_end_sample",
    "segment_end_sample",
)

WIDE_VOICE_START_RE = re.compile(r"^Voice start sample \((\d+)\)$", re.I)
WIDE_VOICE_END_RE = re.compile(r"^Voice end sample \((\d+)\)$", re.I)

BAT_CONTEXT_LABELS = {
    "0": "Unknown",
    "1": "Separation",
    "2": "Biting",
    "3": "Feeding",
    "4": "Fighting",
    "5": "Grooming",
    "6": "Isolation",
    "7": "Kissing",
    "8": "Landing",
    "9": "Mating protest",
    "10": "Threat-like",
    "11": "General",
    "12": "Sleeping",
}


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in norm:
            return norm[key]
    return None


def require_col(df: pd.DataFrame, explicit: str | None, candidates: Iterable[str], role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column '{explicit}' not found. Columns: {list(df.columns)}")
        return explicit
    found = first_existing(df, candidates)
    if not found:
        raise ValueError(f"Could not infer {role} column. Tried {list(candidates)}. Columns: {list(df.columns)}")
    return found


def optional_col(df: pd.DataFrame, explicit: str | None, candidates: Iterable[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"column '{explicit}' not found. Columns: {list(df.columns)}")
        return explicit
    return first_existing(df, candidates)


def expand_wide_fileinfo_segments(fileinfo: pd.DataFrame, file_col: str) -> pd.DataFrame:
    """Convert FileInfo wide voice columns to one row per voiced segment."""
    start_cols = {}
    end_cols = {}
    for col in fileinfo.columns:
        m = WIDE_VOICE_START_RE.match(col)
        if m:
            start_cols[int(m.group(1))] = col
        m = WIDE_VOICE_END_RE.match(col)
        if m:
            end_cols[int(m.group(1))] = col
    common = sorted(set(start_cols).intersection(end_cols))
    if not common:
        return fileinfo

    rows = []
    keep_meta = [c for c in fileinfo.columns if not WIDE_VOICE_START_RE.match(c) and not WIDE_VOICE_END_RE.match(c)]
    for _, row in fileinfo.iterrows():
        base = {c: row[c] for c in keep_meta}
        for k in common:
            s = row[start_cols[k]]
            e = row[end_cols[k]]
            if pd.isna(s) or pd.isna(e):
                continue
            out = dict(base)
            out["voice_segment_number"] = int(k)
            out["segment_start_sample"] = float(s)
            out["segment_end_sample"] = float(e)
            rows.append(out)
    return pd.DataFrame(rows)


def read_fileinfo(path: Path) -> pd.DataFrame:
    """Read FileInfo.csv, whose rows can contain variable numbers of voice pairs."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        rows = []
        fixed_cols = [
            "FileID",
            "Treatment ID",
            "File name",
            "File folder",
            "Recording channel",
            "Recording time",
        ]
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[:6] != fixed_cols:
                fixed_cols = header[:6]
            for values in reader:
                if not values:
                    continue
                row = {col: values[i] if i < len(values) else None for i, col in enumerate(fixed_cols)}
                voice_values = values[len(fixed_cols) :]
                for k in range(0, len(voice_values), 2):
                    pair_id = (k // 2) + 1
                    if k < len(voice_values):
                        row[f"Voice start sample ({pair_id})"] = voice_values[k]
                    if k + 1 < len(voice_values):
                        row[f"Voice end sample ({pair_id})"] = voice_values[k + 1]
                rows.append(row)
        return pd.DataFrame(rows)


def read_fileinfo_long(path: Path) -> pd.DataFrame:
    """Read FileInfo.csv directly as one row per voiced segment."""
    fixed_cols = [
        "FileID",
        "Treatment ID",
        "File name",
        "File folder",
        "Recording channel",
        "Recording time",
    ]
    rows = []
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:6] != fixed_cols:
            fixed_cols = header[:6]
        for values in reader:
            if len(values) < len(fixed_cols):
                continue
            base = {col: values[i] for i, col in enumerate(fixed_cols)}
            voice_values = values[len(fixed_cols) :]
            for k in range(0, len(voice_values), 2):
                if k + 1 >= len(voice_values):
                    continue
                start = voice_values[k]
                end = voice_values[k + 1]
                if start == "" or end == "":
                    continue
                out = dict(base)
                out["voice_segment_number"] = (k // 2) + 1
                out["segment_start_sample"] = float(start)
                out["segment_end_sample"] = float(end)
                rows.append(out)
    return pd.DataFrame(rows)


def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_segments(
    annotations: pd.DataFrame,
    segments: pd.DataFrame,
    file_col_ann: str,
    file_col_seg: str,
    ann_start_col: str,
    ann_end_col: str,
    seg_start_col: str,
    seg_end_col: str,
    context_col: str | None,
    emitter_col: str | None,
    addressee_col: str | None,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    ann = annotations.copy().reset_index(drop=True)
    ann["_file_id_str"] = ann[file_col_ann].astype(str)
    annotated_files = set(ann["_file_id_str"])
    seg = segments.loc[segments[file_col_seg].astype(str).isin(annotated_files)].copy().reset_index(drop=True)
    seg["_file_id_str"] = seg[file_col_seg].astype(str)
    ann["_annotation_index"] = np.arange(len(ann))
    seg["_segment_index"] = np.arange(len(seg))
    rows = []

    ann_counts = ann.groupby("_file_id_str").size()
    single_files = set(ann_counts.index[ann_counts == 1])
    single_ann = ann.loc[ann["_file_id_str"].isin(single_files)].copy()
    single_seg = seg.loc[seg["_file_id_str"].isin(single_files)].copy()
    if not single_seg.empty:
        cols = [
            "_file_id_str",
            "_annotation_index",
            ann_start_col,
            ann_end_col,
        ]
        if context_col:
            cols.append(context_col)
        if emitter_col:
            cols.append(emitter_col)
        if addressee_col:
            cols.append(addressee_col)
        merged = single_seg.merge(single_ann[cols], on="_file_id_str", how="left", validate="many_to_one")
        s0 = pd.to_numeric(merged[seg_start_col], errors="coerce").to_numpy(float)
        s1 = pd.to_numeric(merged[seg_end_col], errors="coerce").to_numpy(float)
        a0 = pd.to_numeric(merged[ann_start_col], errors="coerce").to_numpy(float)
        a1 = pd.to_numeric(merged[ann_end_col], errors="coerce").to_numpy(float)
        durations = np.maximum(s1 - s0, 1.0)
        overlaps = np.maximum(0.0, np.minimum(s1, a1) - np.maximum(s0, a0))
        overlap_frac = overlaps / durations
        centers = (s0 + s1) / 2.0
        hit = (overlap_frac >= min_overlap_fraction) | ((a0 <= centers) & (centers <= a1))
        out = pd.DataFrame(
            {
                "segment_index": merged["_segment_index"].astype(int),
                "file_id": merged["_file_id_str"].astype(str),
                "segment_start_sample": s0,
                "segment_end_sample": s1,
                "segment_duration_samples": s1 - s0,
                "mapping_status": np.where(hit, "unique", "zero"),
                "n_matching_intervals": hit.astype(int),
                "best_overlap_fraction": np.where(hit, overlap_frac, 0.0),
                "annotation_index": np.where(hit, merged["_annotation_index"].to_numpy(int), np.nan),
                "annotation_start_sample": np.where(hit, a0, np.nan),
                "annotation_end_sample": np.where(hit, a1, np.nan),
            }
        )
        if context_col:
            out["context"] = np.where(hit, merged[context_col].astype(object), None)
        if emitter_col:
            out["emitter"] = np.where(hit, merged[emitter_col].astype(object), None)
        if addressee_col:
            out["addressee"] = np.where(hit, merged[addressee_col].astype(object), None)
        rows.append(out)

    multi_seg = seg.loc[~seg["_file_id_str"].isin(single_files)].copy()
    grouped_ann = {str(k): g.reset_index(drop=True) for k, g in ann.loc[~ann["_file_id_str"].isin(single_files)].groupby("_file_id_str")}
    for file_id, g in multi_seg.groupby("_file_id_str", sort=False):
        candidates = grouped_ann.get(str(file_id))
        if candidates is None or candidates.empty:
            continue

        s0 = pd.to_numeric(g[seg_start_col], errors="coerce").to_numpy(float)
        s1 = pd.to_numeric(g[seg_end_col], errors="coerce").to_numpy(float)
        a0 = pd.to_numeric(candidates[ann_start_col], errors="coerce").to_numpy(float)
        a1 = pd.to_numeric(candidates[ann_end_col], errors="coerce").to_numpy(float)
        durations = np.maximum(s1 - s0, 1.0)
        overlaps = np.maximum(0.0, np.minimum(s1[:, None], a1[None, :]) - np.maximum(s0[:, None], a0[None, :]))
        overlap_frac = overlaps / durations[:, None]
        centers = ((s0 + s1) / 2.0)[:, None]
        center_inside = (a0[None, :] <= centers) & (centers <= a1[None, :])
        hit_matrix = (overlap_frac >= min_overlap_fraction) | center_inside
        hit_counts = hit_matrix.sum(axis=1)
        best_idx = np.argmax(overlap_frac, axis=1)
        best_overlap = overlap_frac[np.arange(len(g)), best_idx]

        for local_i, (_, row) in enumerate(g.iterrows()):
            n_hits = int(hit_counts[local_i])
            if n_hits == 0:
                status = "zero"
                best = None
                ov = 0.0
            elif n_hits == 1:
                status = "unique"
                idx = int(np.flatnonzero(hit_matrix[local_i])[0])
                best = candidates.iloc[idx]
                ov = float(overlap_frac[local_i, idx])
            else:
                status = "multiple"
                idx = int(best_idx[local_i])
                best = candidates.iloc[idx]
                ov = float(best_overlap[local_i])

            out = {
                "segment_index": int(row["_segment_index"]),
                "file_id": str(file_id),
                "segment_start_sample": float(s0[local_i]),
                "segment_end_sample": float(s1[local_i]),
                "segment_duration_samples": float(s1[local_i] - s0[local_i]),
                "mapping_status": status,
                "n_matching_intervals": n_hits,
                "best_overlap_fraction": ov,
                "annotation_index": int(best["_annotation_index"]) if best is not None else np.nan,
                "annotation_start_sample": float(best[ann_start_col]) if best is not None else np.nan,
                "annotation_end_sample": float(best[ann_end_col]) if best is not None else np.nan,
            }
            if context_col:
                out["context"] = best[context_col] if best is not None else None
            if emitter_col:
                out["emitter"] = best[emitter_col] if best is not None else None
            if addressee_col:
                out["addressee"] = best[addressee_col] if best is not None else None
            rows.append(pd.DataFrame([out]))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def file_summary(mapping: pd.DataFrame, annotations: pd.DataFrame, file_col_ann: str) -> pd.DataFrame:
    ann_counts = annotations.groupby(file_col_ann).size().rename("n_annotation_intervals")
    seg_counts = mapping.groupby("file_id").size().rename("n_voiced_segments")
    status_counts = pd.crosstab(mapping["file_id"], mapping["mapping_status"])
    out = pd.concat([seg_counts, ann_counts], axis=1).fillna(0)
    out = out.join(status_counts, how="left").fillna(0)
    for col in ("zero", "unique", "multiple"):
        if col not in out.columns:
            out[col] = 0
    out["unique_mapping_rate"] = out["unique"] / out["n_voiced_segments"].replace(0, np.nan)
    out["can_define_next_segment"] = out["n_voiced_segments"] >= 2
    return out.reset_index().rename(columns={"index": "file_id"})


def adjacency_edges(mapping: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame()
    g = mapping.sort_values(["file_id", "segment_start_sample", "segment_end_sample"]).reset_index(drop=True)
    nxt = g.shift(-1)
    same_file = g["file_id"].eq(nxt["file_id"])
    a = g.loc[same_file].reset_index(drop=True)
    b = nxt.loc[same_file].reset_index(drop=True)
    both_unique = a["mapping_status"].eq("unique") & b["mapping_status"].eq("unique")
    same_interval = both_unique & a["annotation_index"].eq(b["annotation_index"])
    out = pd.DataFrame(
        {
            "file_id": a["file_id"].astype(str),
            "segment_i": a["segment_index"].astype(int),
            "segment_j": b["segment_index"].astype(int),
            "delta_t_s": (b["segment_start_sample"].to_numpy(float) - a["segment_end_sample"].to_numpy(float)) / sample_rate,
            "both_unique": both_unique.to_numpy(bool),
            "same_interaction_interval": same_interval.to_numpy(bool),
        }
    )
    for col in ("context", "emitter", "addressee"):
        if col in a.columns:
            out[f"{col}_i"] = a[col].to_numpy()
            out[f"{col}_j"] = b[col].to_numpy()
    return out


def reliability_curve(edges: pd.DataFrame, windows_ms: list[float]) -> pd.DataFrame:
    rows = []
    if edges.empty:
        return pd.DataFrame(rows)
    known = edges.loc[edges["both_unique"]].copy()
    contexts = ["ALL"]
    if "context_i" in known.columns:
        contexts += sorted(x for x in known["context_i"].dropna().astype(str).unique())
    for context in contexts:
        sub = known if context == "ALL" else known.loc[known["context_i"].astype(str) == context]
        true_total = int(sub["same_interaction_interval"].sum())
        for tau_ms in windows_ms:
            tau = tau_ms / 1000.0
            cand = sub.loc[sub["delta_t_s"] <= tau]
            same = int(cand["same_interaction_interval"].sum()) if len(cand) else 0
            n_cand = int(len(cand))
            rows.append(
                {
                    "context": context,
                    "window_ms": tau_ms,
                    "n_known_edges": int(len(sub)),
                    "n_candidate_edges": n_cand,
                    "n_same_interaction_edges": same,
                    "precision_p_same_given_dt_le_tau": same / n_cand if n_cand else np.nan,
                    "false_link_rate": 1.0 - (same / n_cand) if n_cand else np.nan,
                    "coverage_candidate_fraction": n_cand / len(sub) if len(sub) else np.nan,
                    "recall_of_same_interaction_edges": same / true_total if true_total else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["context_label"] = out["context"].astype(str).map(BAT_CONTEXT_LABELS).fillna(out["context"])
        out.loc[out["context"].astype(str).eq("ALL"), "context_label"] = "ALL"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--fileinfo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/bat_proxy_audit"))
    parser.add_argument("--sample-rate", type=float, default=250000.0)
    parser.add_argument("--windows-ms", default="50,100,250,500,1000,2000")
    parser.add_argument("--min-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--annotation-file-col", default=None)
    parser.add_argument("--segment-file-col", default=None)
    parser.add_argument("--annotation-start-col", default=None)
    parser.add_argument("--annotation-end-col", default=None)
    parser.add_argument("--segment-start-col", default=None)
    parser.add_argument("--segment-end-col", default=None)
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--emitter-col", default=None)
    parser.add_argument("--addressee-col", default=None)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional quick-test limit: keep only the first N annotated files.",
    )
    parser.add_argument(
        "--skip-full-mapping-csv",
        action="store_true",
        help="Do not write the large segment-to-interaction mapping CSV.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.annotations)
    fileinfo = read_fileinfo_long(args.fileinfo)

    ann_file_col = require_col(annotations, args.annotation_file_col, FILE_ID_CANDIDATES, "annotation file id")
    seg_file_col = require_col(fileinfo, args.segment_file_col, FILE_ID_CANDIDATES, "segment file id")
    if "segment_start_sample" in fileinfo.columns and "segment_end_sample" in fileinfo.columns:
        args.segment_start_col = args.segment_start_col or "segment_start_sample"
        args.segment_end_col = args.segment_end_col or "segment_end_sample"
    ann_start_col = require_col(annotations, args.annotation_start_col, ANN_START_CANDIDATES, "annotation start")
    ann_end_col = require_col(annotations, args.annotation_end_col, ANN_END_CANDIDATES, "annotation end")
    seg_start_col = require_col(fileinfo, args.segment_start_col, SEG_START_CANDIDATES, "segment start")
    seg_end_col = require_col(fileinfo, args.segment_end_col, SEG_END_CANDIDATES, "segment end")
    context_col = optional_col(annotations, args.context_col, CONTEXT_CANDIDATES)
    emitter_col = optional_col(annotations, args.emitter_col, EMITTER_CANDIDATES)
    addressee_col = optional_col(annotations, args.addressee_col, ADDRESSEE_CANDIDATES)

    if args.max_files is not None:
        keep_files = annotations[ann_file_col].astype(str).drop_duplicates().head(args.max_files)
        keep_files = set(keep_files)
        annotations = annotations.loc[annotations[ann_file_col].astype(str).isin(keep_files)].copy()
        fileinfo = fileinfo.loc[fileinfo[seg_file_col].astype(str).isin(keep_files)].copy()

    mapping = map_segments(
        annotations,
        fileinfo,
        ann_file_col,
        seg_file_col,
        ann_start_col,
        ann_end_col,
        seg_start_col,
        seg_end_col,
        context_col,
        emitter_col,
        addressee_col,
        args.min_overlap_fraction,
    )
    if not args.skip_full_mapping_csv:
        mapping.to_csv(args.output_dir / "bat_segment_to_interaction_mapping.csv", index=False)
    file_summary(mapping, annotations, ann_file_col).to_csv(args.output_dir / "bat_file_mapping_summary.csv", index=False)

    edges = adjacency_edges(mapping, sample_rate=args.sample_rate)
    edges.to_csv(args.output_dir / "bat_next_voiced_segment_edges.csv", index=False)
    windows = [float(x) for x in args.windows_ms.split(",") if x.strip()]
    reliability_curve(edges, windows).to_csv(args.output_dir / "bat_temporal_adjacency_reliability_curve.csv", index=False)

    status_counts = mapping["mapping_status"].value_counts().to_dict()
    summary = {
        "n_annotations": int(len(annotations)),
        "n_voiced_segments": int(len(fileinfo)),
        "annotation_file_col": ann_file_col,
        "segment_file_col": seg_file_col,
        "annotation_start_col": ann_start_col,
        "annotation_end_col": ann_end_col,
        "segment_start_col": seg_start_col,
        "segment_end_col": seg_end_col,
        "context_col": context_col,
        "emitter_col": emitter_col,
        "addressee_col": addressee_col,
        "mapping_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "unique_mapping_rate": float(np.mean(mapping["mapping_status"] == "unique")),
        "zero_mapping_rate": float(np.mean(mapping["mapping_status"] == "zero")),
        "multiple_mapping_rate": float(np.mean(mapping["mapping_status"] == "multiple")),
        "n_next_segment_edges": int(len(edges)),
        "n_known_next_segment_edges": int(edges["both_unique"].sum()) if len(edges) else 0,
    }
    with open(args.output_dir / "bat_proxy_mapping_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
