#!/usr/bin/env python
"""Compare original versus current control inputs; no model fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import extend_shared_cohort_controls as ext
import run_control_block_attribution as current


def comparison(left, right, columns):
    rows = []
    for col in columns:
        a, b = left[col].to_numpy(), right[col].to_numpy()
        same = (a == b) | (pd.isna(a) & pd.isna(b))
        rows.append(dict(column=col, n_different=int((~same).sum()),
            max_absolute_difference=float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))),
            original_dtype=str(a.dtype), compared_dtype=str(b.dtype)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--controls-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.window_ms = 400
    new, _, columns = current.load(args, ext, ext.shared)
    original, _, old_columns, _ = ext.load_inputs(args)
    if columns != old_columns:
        raise ValueError('Acoustic column set/order differs.')
    order_same = original.call_id.tolist() == new.call_id.tolist()
    new = new.set_index('call_id').loc[original.call_id]
    controls = pd.read_csv(args.controls_dir / 'cohort_controls.csv', float_precision='round_trip').set_index('call_id')
    controls = controls.loc[original.call_id]
    baseline = ext.shared.model_numeric_columns(original, 'NF_post')
    rows = comparison(original, new, baseline + columns + ext.CONTROLS)
    rt = comparison(original, controls, ext.CONTROLS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / 'current_input_differences.csv', index=False)
    pd.DataFrame(rt).to_csv(args.output_dir / 'round_trip_control_differences.csv', index=False)
    summary = dict(n_calls=len(original), row_order_identical=order_same,
        current_loader_sha256=ext.shared.digest(Path(current.__file__)),
        original_vs_current_control_differences=[r for r in rows if r['column'] in ext.CONTROLS],
        original_vs_round_trip_control_differences=rt,
        n_changed_noncontrol_columns=sum(r['n_different'] > 0 for r in rows if r['column'] not in ext.CONTROLS),
        model_fits=0, interpretation='Input reconstruction audit only; does not explain prediction differences by itself.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
