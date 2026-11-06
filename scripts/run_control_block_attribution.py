#!/usr/bin/env python
"""Fit only missing amplitude/overlap control arms; reuse frozen shared OOF."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AMPLITUDE = ['control_raw_rms_dbfs', 'control_raw_peak_dbfs',
             'control_raw_crest', 'control_raw_clipping']
OVERLAP = ['control_overlap_fraction', 'control_nonfocal_overlap_fraction', 'control_overlap_count']
PLANS = [
    ('30_raw_amplitude', 'raw', AMPLITUDE),
    ('31_normalized_amplitude', 'rms_normalized', AMPLITUDE),
    ('32_raw_overlap', 'raw', OVERLAP),
    ('33_normalized_overlap', 'rms_normalized', OVERLAP),
]


def assert_job(record, test, condition, target, fold):
    full = record.loc[record.condition.eq(condition)].set_index('call_id')
    selected = record.loc[record.condition.eq(condition + '_validtest')].set_index('call_id')
    if len(record) != len(test) + int(test.egemaps_valid.sum()):
        raise ValueError('Incomplete/duplicated cached job.')
    if full.index.duplicated().any() or set(full.index) != set(test.call_id):
        raise ValueError('Cached job test-call mismatch.')
    full = full.loc[test.call_id]
    if not np.array_equal(full.y_true.to_numpy(int), test[target].to_numpy(int)):
        raise ValueError('Cached job target mismatch.')
    if not record.fold.eq(fold).all() or not record.target.eq(target).all():
        raise ValueError('Cached job fold/target mismatch.')
    valid_ids = test.loc[test.egemaps_valid, 'call_id']
    if set(selected.index) != set(valid_ids):
        raise ValueError('Valid-test calls changed.')
    if not np.array_equal(selected.loc[valid_ids, ['m0_pred', 'm1_pred']],
                          full.loc[valid_ids, ['m0_pred', 'm1_pred']]):
        raise ValueError('Valid-test predictions were not reused exactly.')
    p = record[['m0_pred', 'm1_pred']].to_numpy(float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid cached probabilities.')


def load(args, ext, shared):
    old_protocol = json.loads((args.controls_dir / 'protocol.json').read_text(encoding='utf-8'))
    if shared.digest(Path(ext.__file__)) != old_protocol['source_sha256']:
        raise ValueError('Original control helper changed; investigate, do not silently refit.')
    for name, sha in old_protocol['input_hashes'].items():
        if shared.digest(Path(name)) != sha:
            raise ValueError('Frozen source changed: ' + name)
    for name, field in [('protocol.json', 'shared_protocol_sha256'),
                        ('cohort_with_history.pkl', 'cache_sha256'),
                        ('clip_fold_mapping.csv', 'folds_sha256'),
                        ('oof_predictions.csv', 'previous_oof_sha256')]:
        if shared.digest(args.shared_dir / name) != old_protocol[field]:
            raise ValueError('Shared artifact changed: ' + name)
    data = pd.read_pickle(args.shared_dir / 'cohort_with_history.pkl')
    data = data.loc[data.window_ms.eq(400) & ~data.pre_censored & ~data.post_censored].copy()
    if len(data) != old_protocol['n_calls'] or data.call_id.duplicated().any():
        raise ValueError('Common cohort changed.')
    # These values were computed in memory in the original fits. The default
    # CSV converter changes their last bits; restore the saved doubles exactly.
    controls = pd.read_csv(args.controls_dir / 'cohort_controls.csv',
        dtype={'_clip': str}, float_precision='round_trip').set_index('call_id')
    controls = controls.loc[data.call_id]
    for col in ['_clip', 'fold', 'manual_label']:
        if not np.array_equal(data[col].to_numpy(), controls[col].to_numpy()):
            raise ValueError('Control membership differs: ' + col)
    for col in AMPLITUDE + OVERLAP:
        data[col] = controls[col].to_numpy()
    data = data.reset_index(drop=True)
    if not np.isfinite(data[AMPLITUDE + OVERLAP]).all().all():
        raise ValueError('Nonfinite measured controls.')
    folds = pd.read_csv(args.shared_dir / 'clip_fold_mapping.csv', dtype={'_clip': str}).set_index('_clip').fold
    if not np.array_equal(data['_clip'].map(folds), data.fold):
        raise ValueError('Fold assignment changed.')
    cols = old_protocol['acoustic_columns']
    loader = argparse.Namespace(feature_csv=args.project_root / 'features/acoustic_features_capped_3000.csv',
        feature_npz=None, feature_call_id_col=None,
        drop_feature_regex=shared.base.DEFAULT_DROP_FEATURE_REGEX, amplitude_variant='raw')
    raw, raw_cols = shared.base.load_feature_table(loader)
    loader.amplitude_variant = 'rms_normalized'
    norm, norm_cols = shared.base.load_feature_table(loader)
    if cols != raw_cols or cols != norm_cols or len(cols) != 81:
        raise ValueError('Acoustic feature set changed.')
    if not np.allclose(raw.set_index('call_id').loc[data.call_id, cols], data[cols],
                       rtol=0, atol=0, equal_nan=True):
        raise ValueError('Cached raw acoustic values changed.')
    normalized = data.copy()
    normalized.loc[:, cols] = norm.set_index('call_id').loc[data.call_id, cols].to_numpy()
    return data, normalized, cols


def freeze(args, shared, data, cols):
    here = Path(__file__).parent
    code = [Path(__file__), here / 'extend_shared_cohort_controls.py',
            here / 'run_shared_cohort_selection_control.py', here / 'fit_meerkat_m1_acoustic_residual.py',
            here / 'audit_egemaps_selection_on_librosa.py']
    inputs = [args.shared_dir / n for n in ['protocol.json', 'cohort_with_history.pkl', 'clip_fold_mapping.csv', 'oof_predictions.csv']]
    inputs += [args.controls_dir / n for n in ['protocol.json', 'cohort_controls.csv', 'oof_predictions_added.csv']]
    protocol = dict(window_ms=400, n_calls=len(data), n_valid=int(data.egemaps_valid.sum()),
        n_folds=5, acoustic_columns=cols, pca_dim=30, random_state=0,
        threads=args.threads, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        amplitude_controls=AMPLITUDE, overlap_controls=OVERLAP, plans=PLANS,
        code_hashes={str(p): shared.digest(p) for p in code},
        input_hashes={str(p): shared.digest(p) for p in inputs},
        environment=dict(python=sys.version.split()[0], **{p: importlib.metadata.version(p)
            for p in ['numpy', 'pandas', 'scikit-learn', 'scipy']}),
        controls_in='Both M0 and M1; raw received amplitude, focal-boundary-clipped annotation overlap.',
        control_reload='round_trip CSV floats; reset row index; original variant copy before fold split.',
        snr='Not controlled: existing SNR reads before-and-after audio.',
        estimand='Finite-model held-out logloss increment, not causal source decomposition.',
        preflight=dict(fold=1, condition='20_raw_shared_full', max_abs_probability_tolerance=1e-6),
        interval_scope='Exploratory paired clip bootstrap conditional on stored OOF; not full refit uncertainty.')
    # Normalize tuples before strict resumed-run comparison.
    protocol = json.loads(json.dumps(protocol))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol/runtime/code changed. Use a new directory and repeat preflight.')
    shared.write_json(path, protocol)
    return shared.digest(path)


def preflight(args, data, cols, ext, shared, protocol_sha):
    variant = data.copy()
    train, test = variant.loc[variant.fold.ne(1)], variant.loc[variant.fold.eq(1)]
    old = pd.read_csv(args.controls_dir / 'oof_predictions_added.csv')
    checks, records = [], []
    for target in ['NF_post', 'NF_pre']:
        ref = old.loc[old.condition.eq('20_raw_shared_full') & old.target.eq(target)
                      & old.fold.eq(1)].set_index('call_id').loc[test.call_id]
        if not np.array_equal(ref.y_true.to_numpy(int), test[target].to_numpy(int)):
            raise ValueError('Preflight saved target differs from cached target.')
        p0, p1 = ext.fit_control_pair(train, test, target, cols, AMPLITUDE + OVERLAP, args)
        for model, prob in [('m0', p0), ('m1', p1)]:
            error = float(np.max(np.abs(prob - ref[model + '_pred'].to_numpy())))
            difference = prob - ref[model + '_pred'].to_numpy()
            checks.append(dict(target=target, model=model, max_abs_probability_error=error,
                               mean_abs_probability_error=float(np.mean(np.abs(difference)))))
            records.append(pd.DataFrame(dict(call_id=test.call_id.to_numpy(), _clip=test['_clip'].to_numpy(),
                target=target, model=model, y_true=test[target].to_numpy(int),
                reference_prediction=ref[model + '_pred'].to_numpy(), prediction=prob,
                prediction_difference=difference)))
    pd.concat(records, ignore_index=True).to_csv(args.output_dir / 'preflight_predictions.csv', index=False)
    passed = all(c['max_abs_probability_error'] <= 1e-6 for c in checks)
    result = dict(status='passed' if passed else 'failed', protocol_sha256=protocol_sha,
                  tolerance=1e-6, checks=checks, note='Real-data CPU fits on first saved fold, user-run only.')
    shared.write_json(args.output_dir / 'preflight.json', result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise ValueError('Saved-fold reproduction failed. Do not launch missing arms.')


def fit_missing(args, data, normalized, cols, ext, shared):
    jobs = args.output_dir / 'jobs'
    jobs.mkdir(exist_ok=True)
    memberships, counts = [], []
    for fold in range(1, 6):
        cache = {}
        for condition, variant, controls in PLANS:
            current = (data if variant == 'raw' else normalized).copy()
            train, test = current.loc[current.fold.ne(fold)], current.loc[current.fold.eq(fold)]
            if set(train['_clip']) & set(test['_clip']):
                raise ValueError('Training/test clips overlap.')
            tm = train[['call_id', '_clip', 'manual_label', 'duration_bin']].copy()
            tm['heldout_fold'], tm['condition'] = fold, condition
            memberships.append(tm)
            counts += [ext.report_counts(train, fold, condition, 'train'),
                       ext.report_counts(test, fold, condition, 'test')]
            for target in ['NF_post', 'NF_pre']:
                path = jobs / f'{condition}_fold{fold}_{target}.csv'
                if path.exists():
                    saved = pd.read_csv(path, dtype={'call_id': str, '_clip': str})
                    assert_job(saved, test, condition, target, fold)
                    continue
                key = (target, tuple(controls))
                p0, p1 = ext.fit_control_pair(train, test, target, cols, controls,
                                             args, m0_cached=cache.get(key))
                cache[key] = p0
                record = shared.predict_record(test, target, condition, condition, p0, p1, len(train))
                valid = record.loc[record.egemaps_valid].copy()
                valid['condition'] += '_validtest'
                record = pd.concat([record, valid], ignore_index=True)
                assert_job(record, test, condition, target, fold)
                temporary = path.with_suffix('.tmp')
                record.to_csv(temporary, index=False)
                temporary.replace(path)
            print(f'completed fold={fold}/5 {condition}', flush=True)
    pd.concat(memberships, ignore_index=True).to_csv(args.output_dir / 'training_membership.csv', index=False)
    pd.concat(counts, ignore_index=True).to_csv(args.output_dir / 'fold_label_duration_counts.csv', index=False)


def summarize(args, data, shared):
    parts = []
    for fold in range(1, 6):
        test = data.loc[data.fold.eq(fold)]
        for condition, _, _ in PLANS:
            for target in ['NF_post', 'NF_pre']:
                path = args.output_dir / 'jobs' / f'{condition}_fold{fold}_{target}.csv'
                if not path.exists():
                    raise ValueError('Missing job; summary refused: ' + str(path))
                saved = pd.read_csv(path, dtype={'call_id': str, '_clip': str})
                assert_job(saved, test, condition, target, fold)
                parts.append(saved)
    extra = pd.concat(parts, ignore_index=True)
    extra.to_csv(args.output_dir / 'oof_predictions_added.csv', index=False)
    old = pd.read_csv(args.shared_dir / 'oof_predictions.csv')
    old = old.loc[old.condition.isin(['01_full_all', '02_full_valid']) & old.window_ms.eq(400)]
    both = pd.read_csv(args.controls_dir / 'oof_predictions_added.csv')
    both = both.loc[both.condition.isin(['20_raw_shared_full', '20_raw_shared_full_validtest',
        '21_normalized_shared_full', '21_normalized_shared_full_validtest',
        '24_normalized_original_full', '24_normalized_original_full_validtest'])]
    table = pd.concat([old, both, extra], ignore_index=True)
    args.reference_map = {}
    aliases = []
    for suffix in ['', '_validtest']:
        raw_none = '02_full_valid' if suffix else '01_full_all'
        norm_none = '24_normalized_original_full' + suffix
        for variant, number, baseline, joint in [
            ('raw', '30', raw_none, '20_raw_shared_full'),
            ('normalized', '31', norm_none, '21_normalized_shared_full')]:
            amp = number + '_' + variant + '_amplitude' + suffix
            overlap = ('32' if variant == 'raw' else '33') + '_' + variant + '_overlap' + suffix
            joint += suffix
            args.reference_map[amp] = baseline
            args.reference_map[overlap] = baseline
            args.reference_map[joint] = amp
            # Aliases reuse identical predictions solely to obtain additional
            # paired contrasts with the established shared bootstrap routine.
            for name, ref in [('vs_none', baseline), ('vs_overlap', overlap)]:
                alias = joint + '__' + name
                part = table.loc[table.condition.eq(joint)].copy()
                part['condition'] = alias
                aliases.append(part)
                args.reference_map[alias] = ref
    expanded = pd.concat([table] + aliases, ignore_index=True)
    result = shared.summarize_and_write(expanded, args)
    decomposition = []
    lookup = result.set_index(['condition', 'manual_label'])
    for row in result.to_dict('records'):
        ref = row.get('change_reference')
        if not isinstance(ref, str):
            continue
        r = lookup.loc[(ref, row['manual_label'])]
        out = dict(condition=row['condition'], reference=ref, manual_label=row['manual_label'])
        for target in ['post', 'pre']:
            for model in ['m0', 'm1']:
                col = model + '_' + target + '_logloss'
                out[col + '_change'] = row[col] - r[col]
            out['delta_' + target + '_change'] = row['delta_' + target] - r['delta_' + target]
        decomposition.append(out)
    pd.DataFrame(decomposition).to_csv(args.output_dir / 'block_effect_decomposition.csv', index=False)
    shared.write_json(args.output_dir / 'verification.json', dict(status='passed',
        n_jobs=40, same_calls=True, fixed_clip_folds=True, original_acoustic_features=81,
        existing_no_control_and_joint_predictions_reused=True,
        comparison_aliases='__vs_none/__vs_overlap are same OOF, not additional fits.'))
    print(result.loc[result.manual_label.eq('overall'),
        ['condition', 'delta_post', 'delta_pre', 'post_minus_pre']].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path(r'C:\学校\MeerKAT\repertoire_geometry'))
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--controls-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--run-models', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--bootstrap-seed', type=int, default=20261105)
    args = p.parse_args()
    if args.output_dir.resolve() in {args.shared_dir.resolve(), args.controls_dir.resolve()}:
        raise ValueError('Use a new output folder, not either input folder.')
    import extend_shared_cohort_controls as ext
    import run_shared_cohort_selection_control as shared
    from threadpoolctl import threadpool_limits
    data, normalized, cols = load(args, ext, shared)
    protocol_sha = freeze(args, shared, data, cols)
    with threadpool_limits(limits=args.threads):
        if args.preflight:
            preflight(args, data, cols, ext, shared, protocol_sha)
        elif args.run_models or args.summarize_only:
            check = json.loads((args.output_dir / 'preflight.json').read_text(encoding='utf-8'))
            if check['status'] != 'passed' or check['protocol_sha256'] != protocol_sha:
                raise ValueError('A matching passing preflight is required.')
            if args.run_models:
                fit_missing(args, data, normalized, cols, ext, shared)
            summarize(args, data, shared)
        else:
            print(json.dumps(dict(status='prepared_only', n_calls=len(data), n_valid=int(data.egemaps_valid.sum()),
                missing_fit_arms=[p[0] for p in PLANS], protocol_sha256=protocol_sha,
                next='User runs --preflight, then --run-models. No models fitted now.'), indent=2))


if __name__ == '__main__':
    main()
