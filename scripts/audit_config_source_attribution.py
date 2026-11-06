"""Read-only configuration-source OOF audit; writes only to a fresh audit folder."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_baseline_capacity_b3 as check
import run_config_source_attribution as runner


def rename_roles(frame):
    return frame.rename(columns={c:c.replace('m0_', 'left_', 1) if c.startswith('m0_')
        else c.replace('m1_', 'right_', 1) for c in frame
        if c.startswith('m0_') or c.startswith('m1_')})


def wide_metrics(long):
    rows = []
    for tags, part in long.groupby(['model_family', 'view', 'manual_label'], sort=True):
        lookup = part.set_index('comparison')
        total, cfg = lookup.loc['total'], lookup.loc['config']
        row = dict(zip(['model_family', 'view', 'manual_label'], tags))
        for col in ['n', 'n_clips', 'n_folds', 'post_positive_n', 'pre_positive_n']:
            row[col] = int(total[col])
        for side in ['post', 'pre']:
            for model, source, prefix in [('M0', total, 'left'), ('Mconfig', cfg, 'right'),
                    ('M1', total, 'right')]:
                for metric in ['logloss', 'auc', 'ap']:
                    row[model+'_'+side+'_'+metric] = source[prefix+'_'+side+'_'+metric]
        for comparison in runner.COMPARISONS:
            for col in ['delta_post', 'delta_pre', 'post_minus_pre']+[c for c in part
                    if '_ci_' in c or c.endswith('_finite_draws')]:
                row[comparison+'_'+col] = lookup.loc[comparison, col]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root', 'shared-dir', 'controls-dir', 'b1-run', 'b2-run',
            'b3-run', 'pca-run', 'audit-dir', 'run-dir', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.run_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError('Use a fresh independent audit output directory.')
    with threadpool_limits(limits=4), runner.pca.fixed.zero_fit_guard():
        protocol_path = args.run_dir/'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        runner.b3.verify_hashes(protocol)
        sha = runner.b3.shared.digest(protocol_path)
        data, cols, _, _ = runner.b3.load_design(args)
        runner.validate_audit(args.audit_dir, data)
        reconstruction = runner.reconstruct_from_controls(data, 8000)
        if not reconstruction.config_match.all():
            raise ValueError('Configuration reconstruction no longer matches.')
        references, reference_manifest = runner.read_references(args, data, cols)
        plan = runner.make_plan(data, reference_manifest)
        pd.testing.assert_frame_equal(runner.b3.read(args.run_dir/'fit_plan.csv'), plan,
            check_dtype=False)
        manifest = runner.b3.read(args.run_dir/'all_oof_manifest.csv')
        if len(manifest)!=60 or manifest.path.duplicated().any():
            raise ValueError('Expected 30 existing and 30 new OOF sources.')
        if manifest.role.value_counts().to_dict()!=dict(existing_M0_M1_noPCA=30,
                new_M0_plus_config=30):
            raise ValueError('Source roles differ.')
        for item in manifest.to_dict('records'):
            for path_key, hash_key in [('path', 'sha256'), ('metadata_path', 'metadata_sha256')]:
                if runner.b3.shared.digest(Path(item[path_key]))!=item[hash_key]:
                    raise ValueError('OOF/metadata source hash differs.')
        jobs = args.run_dir/'jobs'
        expected = set(plan.relative_path)
        if ({p.relative_to(args.run_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
                or {p.relative_to(args.run_dir).with_suffix('.csv').as_posix()
                    for p in jobs.glob('*.json')}!=expected):
            raise ValueError('Missing or unexpected configuration jobs.')
        parts, metadata, diagnostics = [], [], []
        source_args = argparse.Namespace(output_dir=args.run_dir)
        for spec in plan.to_dict('records'):
            train, test = data.loc[data.fold.ne(spec['fold'])], data.loc[data.fold.eq(spec['fold'])]
            if set(train['_clip']) & set(test['_clip']):
                raise ValueError('Train/test clip overlap.')
            ref = runner.pca.reference_for(references, spec, test)
            frame, meta = runner.load_job(source_args, spec, train, test, ref, sha)
            parts.append(frame)
            metadata.append(meta)
            diff = frame.mconfig_pred.to_numpy()-frame.m0_pred.to_numpy()
            diagnostics.append(dict(model_family=spec['family'], target=spec['target'],
                fold=spec['fold'], changed_predictions=int(np.count_nonzero(diff)),
                max_abs_prediction_change=float(np.max(np.abs(diff))),
                mean_abs_prediction_change=float(np.mean(np.abs(diff))),
                fit_start_utc=meta['fit_start_utc'], fit_end_utc=meta['fit_end_utc'],
                convergence_warning=bool(meta['convergence_warning'])))
        raw = pd.concat(parts, ignore_index=True)
        if len(raw)!=110070 or raw.duplicated(['model_family', 'target', 'call_id']).any():
            raise ValueError('OOF coverage differs.')
        identity_columns = ['y_true', '_clip', 'fold', 'manual_label', 'duration_s', 'egemaps_valid']
        if not raw.groupby(['target', 'call_id'])[identity_columns].nunique(dropna=False).eq(1).all().all():
            raise ValueError('Cross-family sample metadata differs.')
        pairs, decomposition_error = runner.make_pairs(raw)
        table = runner.b3.score_table(pairs)
        points = rename_roles(runner.decorate(runner.b3.point_metrics(table)))
        reported = runner.b3.read(args.run_dir/'paired_metrics_and_ci.csv')
        keys = ['model_family', 'comparison', 'view', 'manual_label']
        numeric = [c for c in points if c not in keys+['setting', 'left_model', 'right_model']]
        errors = dict(points=check.compare_tables(reported, points, keys, numeric))
        values = table[['call_id', '_clip', 'manual_label', 'target', 'model_family', 'view']].copy()
        values['contrast'], values['value'] = 'gain', table.gain.to_numpy()
        intervals = runner.b3.clip_intervals(values, 1000, 7452).drop(columns='contrast')
        decorated_ci = runner.decorate(intervals)
        ci_cols = [c for c in intervals if '_ci_' in c or c.endswith('_finite_draws')]
        errors['intervals'] = check.compare_tables(reported, decorated_ci, keys, ci_cols)
        long = points.merge(decorated_ci[keys+ci_cols], on=keys, validate='one_to_one')
        wide = wide_metrics(long)
        wide_keys = ['model_family', 'view', 'manual_label']
        errors['three_models'] = check.compare_tables(
            runner.b3.read(args.run_dir/'three_model_metrics.csv'), wide, wide_keys,
            [c for c in wide if c not in wide_keys])
        fold_rows = []
        for tags, part in table.groupby(['model_family', 'view', 'fold'], sort=True):
            row = dict(zip(['model_family', 'view', 'fold'], tags))
            for target, side in zip(runner.TARGETS, ['post', 'pre']):
                group = part.loc[part.target.eq(target)]
                row['left_'+side+'_logloss'] = group.m0_loss.mean()
                row['right_'+side+'_logloss'] = group.m1_loss.mean()
                row['delta_'+side] = group.gain.mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            fold_rows.append(row)
        folds = runner.decorate(pd.DataFrame(fold_rows))
        errors['folds'] = check.compare_tables(runner.b3.read(args.run_dir/'fold_metrics.csv'),
            folds, ['model_family', 'comparison', 'view', 'fold'],
            [c for c in folds if c.endswith('_logloss') or c in ['delta_post', 'delta_pre', 'post_minus_pre']])
        # Positive intervals are descriptive, unadjusted selections, not new confirmatory tests.
        scan = long.copy()
        for metric in ['delta_post', 'delta_pre', 'post_minus_pre']:
            scan[metric+'_positive_interval'] = scan[metric+'_ci_low'].gt(0)
        warnings_n = sum(bool(m['convergence_warning']) for m in metadata)
        if warnings_n:
            raise ValueError('Unexpected convergence warning; retain and inspect before interpretation.')
        args.output_dir.mkdir(parents=True)
        long.to_csv(args.output_dir/'recomputed_metrics_and_ci.csv', index=False)
        wide.to_csv(args.output_dir/'recomputed_three_model_metrics.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(args.output_dir/'configuration_prediction_changes.csv', index=False)
        scan.to_csv(args.output_dir/'all_label_interval_scan.csv', index=False)
        result = dict(status='passed', model_fits=0, zero_fit_guard=True, source_oof_files=60,
            source_metadata_files=60, raw_oof_rows=len(raw), metric_rows=len(long),
            wide_rows=len(wide), source_protocol_sha256=sha, max_abs_errors=errors,
            per_call_decomposition_max_error=decomposition_error,
            configurations_reconstructed_from_M0=True, existing_M0_M1_exactly_reused=True,
            convergence_warning_jobs=warnings_n,
            fit_start_utc=min(m['fit_start_utc'] for m in metadata),
            fit_end_utc=max(m['fit_end_utc'] for m in metadata),
            intervals='Unadjusted paired clips, conditional on stored OOF; no training uncertainty.',
            acceptance_status='Local deterministic audit only; no new reviewer or novelty adjudication.')
        runner.b3.shared.write_json(args.output_dir/'verification.json', result)
        print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
