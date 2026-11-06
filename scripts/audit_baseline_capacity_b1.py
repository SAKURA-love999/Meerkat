"""Read-only B1 OOF audit; no estimator fitting or source-result replacement."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import run_baseline_capacity as b1


def loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1-1e-15)
    y = np.asarray(y, float)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root', 'shared-dir', 'controls-dir', 'reference-run',
                 'run-dir', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.seeds, args.n_bootstrap, args.threads = [431], 1000, 4
    if args.output_dir.exists():
        raise ValueError('Use a fresh audit directory, not an old result path.')
    protocol = json.loads((args.run_dir/'protocol.json').read_text(encoding='utf-8'))
    for section in ['source_hashes', 'input_hashes']:
        for path, expected in protocol[section].items():
            if b1.core.digest(path) != expected:
                raise ValueError('Frozen source/input hash differs: '+path)
    current, _, _, _, _ = b1.load_design(args)
    manifest = pd.read_csv(args.run_dir/'all_oof_manifest.csv')
    expected = {(family, fold, side) for family in
                ['logistic_linear', 'known_h_reference', 'spline_rms', 'hist_gbdt']
                for fold in range(1, 6) for side in ['post', 'pre']}
    if len(manifest) != 40 or set(zip(manifest.model_family, manifest.fold,
                                      manifest.target)) != expected:
        raise ValueError('OOF manifest coverage differs.')
    records = []
    protocol_sha = b1.core.digest(args.run_dir/'protocol.json')
    for item in manifest.itertuples(index=False):
        if b1.core.digest(item.path) != item.sha256:
            raise ValueError('OOF hash differs: '+item.path)
        frame = b1.read(item.path)
        test = current.loc[current.fold.eq(item.fold)]
        if len(frame) != item.n_rows:
            raise ValueError('OOF row count differs.')
        b1.check_record(frame, test, item.model_family, item.fold, item.target,
                        len(current)-len(test),
                        protocol_sha if item.role == 'new_capacity_fit' else None)
        frame['model_family'] = item.model_family
        records.append(frame)
    raw = pd.concat(records, ignore_index=True)
    if len(raw) != 146760 or raw.duplicated(['model_family', 'target', 'call_id']).any():
        raise ValueError('OOF completeness/uniqueness differs.')
    for _, group in raw.groupby(['target', 'call_id']):
        if any(group[c].nunique(dropna=False) != 1 for c in
               ['y_sim', '_clip', 'fold', 'manual_label', 'egemaps_valid']):
            raise ValueError('Matched target/metadata differs across learners.')
    tables = []
    for view, rows in [('all', raw), ('original_valid_rows', raw.loc[raw.egemaps_valid])]:
        part = rows.copy()
        part['view'] = view
        for model in ['m0', 'm1']:
            part[model+'_loss'] = loss(part.y_sim, part[model+'_pred'])
        part['gain'] = part.m0_loss-part.m1_loss
        tables.append(part)
    table = pd.concat(tables, ignore_index=True)
    reported = b1.read(args.run_dir/'comparison_metrics_and_ci.csv')
    secondary, checks, folds = [], [], []
    key = ['model_family', 'view', 'manual_label']
    points = reported.set_index(key)
    for (family, view), group in table.groupby(['model_family', 'view']):
        for label, scope in [('overall', group)]+list(group.groupby('manual_label')):
            row = points.loc[(family, view, label)]
            calculated = {}
            extras = dict(model_family=family, view=view, manual_label=label)
            for side in ['post', 'pre']:
                part = scope.loc[scope.target.eq('sim_'+side)]
                calculated['delta_'+side] = part.gain.mean()
                calculated[side+'_positive_n'] = int(part.y_sim.sum())
                for model in ['m0', 'm1']:
                    calculated[model+'_'+side+'_logloss'] = part[model+'_loss'].mean()
                    y, p = part.y_sim.to_numpy(int), part[model+'_pred'].to_numpy()
                    extras[model+'_'+side+'_auc'] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
                    extras[model+'_'+side+'_ap'] = average_precision_score(y, p) if y.sum() else np.nan
            calculated['post_minus_pre'] = calculated['delta_post']-calculated['delta_pre']
            for metric, value in calculated.items():
                checks.append(dict(model_family=family, view=view, manual_label=label,
                                   metric=metric, abs_error=abs(value-row[metric])))
            secondary.append(extras)
        for fold, scope in group.groupby('fold'):
            item = dict(model_family=family, view=view, fold=fold)
            for side in ['post', 'pre']:
                part = scope.loc[scope.target.eq('sim_'+side)]
                item['delta_'+side] = part.gain.mean()
                for model in ['m0', 'm1']:
                    item[model+'_'+side+'_logloss'] = part[model+'_loss'].mean()
            item['post_minus_pre'] = item['delta_post']-item['delta_pre']
            folds.append(item)
    checked = pd.DataFrame(checks)
    if checked.abs_error.max() > 1e-12:
        raise ValueError('Primary point metric mismatch.')
    # This recomputes intervals from unchanged OOF, never fitting models.
    stats = b1.statistics_table(raw)
    ci = b1.core.bootstrap_overall(stats, 1000, 7452).rename(columns={'analysis_arm':'model_family'})
    ci_cols = [c for c in ci if '_ci_' in c]
    merged = reported.merge(ci, on=['scenario']+key, suffixes=('_old', '_new'))
    ci_error = max(float((merged[c+'_old']-merged[c+'_new']).abs().max()) for c in ci_cols)
    old_changes = b1.read(args.run_dir/'paired_changes_vs_linear.csv')
    new_changes = b1.paired_changes(stats, 1000)
    change_key = ['model_family', 'contrast', 'view', 'manual_label']
    old_changes = old_changes.set_index(change_key).sort_index()
    new_changes = new_changes.set_index(change_key).sort_index()
    pd.testing.assert_index_equal(old_changes.index, new_changes.index)
    change_error = float((old_changes-new_changes).abs().max().max())
    if ci_error > 1e-12 or change_error > 1e-12:
        raise ValueError('Paired interval/contrast mismatch.')
    args.output_dir.mkdir(parents=True)
    checked.to_csv(args.output_dir/'point_metric_checks.csv', index=False)
    pd.DataFrame(secondary).to_csv(args.output_dir/'secondary_metrics_from_oof.csv', index=False)
    pd.DataFrame(folds).to_csv(args.output_dir/'fold_metrics.csv', index=False)
    summary = dict(status='passed', model_fits=0, raw_oof_rows=len(raw), manifest_jobs=len(manifest),
                   primary_rows=len(reported), point_scalars_checked=len(checked),
                   point_max_abs_error=float(checked.abs_error.max()), ci_max_abs_error=ci_error,
                   paired_changes_max_abs_error=change_error,
                   original_secondary_columns_all_missing=bool(reported.filter(regex='_(auc|ap)$').isna().all().all()),
                   secondary_correction='row.view resolves to Series.view; use bracket column access. Original files unchanged.',
                   source_protocol_sha256=protocol_sha, scope='B1 known-null only, seed431; no real-data determination',
                   interval_scope='1000 paired clips conditional on stored predictions, no refitting uncertainty')
    b1.core.write_json(args.output_dir/'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
