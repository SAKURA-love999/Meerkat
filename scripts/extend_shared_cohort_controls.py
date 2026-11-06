#!/usr/bin/env python
"""Reuse frozen shared-cohort OOF; add label-quota and amplitude controls on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_shared_cohort_selection_control as shared

DIRECT_AMPLITUDE = ['rms', 'rms_dbfs', 'peak_abs', 'crest_factor', 'clipping_fraction']
CONTROLS = ['control_raw_rms_dbfs', 'control_raw_peak_dbfs',
            'control_raw_crest', 'control_raw_clipping',
            'control_overlap_fraction', 'control_nonfocal_overlap_fraction',
            'control_overlap_count']
CALL_LABELS = {'sn', 'cc', 'ld', 'mo', 'al', 'soc', 'agg', 'oth'}


def sample_label_quotas(train: pd.DataFrame, valid: pd.DataFrame,
                       seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    quotas = valid.groupby('manual_label', sort=True).size()
    for label, n in quotas.items():
        pool = train.loc[train.manual_label.eq(label)].sort_values('call_id')
        if len(pool) < n:
            raise ValueError('Insufficient training support for label quota.')
        parts.append(pool.iloc[rng.choice(len(pool), size=int(n), replace=False)])
    result = pd.concat(parts).sort_values('call_id')
    assert result.groupby('manual_label', sort=True).size().equals(quotas)
    assert not result.call_id.duplicated().any()
    return result


def union_duration(intervals: list[tuple[float, float]]) -> float:
    total, right = 0., -np.inf
    for start, end in sorted(intervals):
        total += max(0., end-max(start, right))
        right = max(right, end)
    return total


def make_overlap_table(calls: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    groups = {str(k): v for k, v in events.loc[events.label.isin(CALL_LABELS)].groupby('source_id')}
    rows = []
    for row in calls.drop_duplicates('call_id').to_dict('records'):
        group = groups[str(row['_clip'])]
        start, end = float(row['_start']), float(row['_end'])
        other = group.loc[group.call_id.ne(row['call_id']) &
                          group.start_time_sec.lt(end) & group.end_time_sec.gt(start)]
        intervals, nonfocal = [], []
        for event in other.itertuples():
            interval = (max(start, float(event.start_time_sec)), min(end, float(event.end_time_sec)))
            intervals.append(interval)
            focal = str(event.focal).strip().lower() in {'true', '1'}
            if not focal:
                nonfocal.append(interval)
        duration = end-start
        rows.append(dict(call_id=row['call_id'],
            control_overlap_fraction=union_duration(intervals)/duration,
            control_nonfocal_overlap_fraction=union_duration(nonfocal)/duration,
            control_overlap_count=len(intervals)))
    return pd.DataFrame(rows)


def fit_control_pair(train, test, target, acoustic, numeric_extra, args, m0_cached=None):
    if set(train['_clip']) & set(test['_clip']):
        raise ValueError('Train/test clip overlap.')
    numeric = shared.model_numeric_columns(train, target) + list(numeric_extra)
    predictions = []
    for use_audio in [False, True]:
        if not use_audio and m0_cached is not None:
            predictions.append(np.asarray(m0_cached))
            continue
        cols = acoustic if use_audio else []
        model = shared.base.make_pipeline(numeric, cols, use_audio,
                                         30 if use_audio else None, 0)
        model.fit(train[['manual_label']+numeric+cols], train[target].astype(int))
        predictions.append(model.predict_proba(test[['manual_label']+numeric+cols])[:, 1])
    return predictions


def load_inputs(args):
    protocol = json.loads((args.shared_dir/'protocol.json').read_text(encoding='utf-8'))
    for name, sha in protocol['input_hashes'].items():
        if shared.digest(Path(name)) != sha:
            raise ValueError('Frozen input changed: '+name)
    data = pd.read_pickle(args.shared_dir/'cohort_with_history.pkl')
    data = data.loc[data.window_ms.eq(args.window_ms) & ~data.pre_censored & ~data.post_censored].copy()
    if data.empty or data.call_id.duplicated().any():
        raise ValueError('Require one complete cached row per call/window.')
    folds = pd.read_csv(args.shared_dir/'clip_fold_mapping.csv', dtype={'_clip': str})
    assert (data['_clip'].map(folds.set_index('_clip').fold).to_numpy() == data.fold.to_numpy()).all()
    feature_path = args.project_root/'features/acoustic_features_capped_3000.csv'
    loader = argparse.Namespace(feature_csv=feature_path, feature_npz=None,
        feature_call_id_col=None, drop_feature_regex=shared.base.DEFAULT_DROP_FEATURE_REGEX,
        amplitude_variant='raw')
    raw, cols = shared.base.load_feature_table(loader)
    loader.amplitude_variant = 'rms_normalized'
    normalized, cols_normalized = shared.base.load_feature_table(loader)
    assert cols == cols_normalized and len(cols) == 81
    raw, normalized = raw.set_index('call_id'), normalized.set_index('call_id')
    raw = raw.loc[data.call_id]
    normalized = normalized.loc[data.call_id]
    assert np.allclose(raw[cols], data[cols], rtol=0, atol=0, equal_nan=True)
    data['control_raw_rms_dbfs'] = raw.rms_dbfs.to_numpy()
    data['control_raw_peak_dbfs'] = 20*np.log10(np.maximum(raw.peak_abs.to_numpy(), 1e-12))
    data['control_raw_crest'] = raw.crest_factor.to_numpy()
    data['control_raw_clipping'] = raw.clipping_fraction.to_numpy()
    events = pd.read_csv(args.project_root/'data_manifest/call_manifest.csv')
    data = data.merge(make_overlap_table(data, events), on='call_id', validate='one_to_one')
    normalized = normalized.loc[data.call_id].reset_index()
    if not np.isfinite(data[CONTROLS].to_numpy()).all():
        raise ValueError('Nonfinite shared amplitude/overlap controls.')
    return data, normalized, cols, protocol


def report_counts(data, fold, arm, split):
    out = data.groupby(['manual_label', 'duration_bin']).agg(n=('call_id', 'size'),
        n_clips=('_clip', 'nunique'), duration_min=('duration_s', 'min'),
        duration_median=('duration_s', 'median'), duration_max=('duration_s', 'max'),
        post_positive_n=('NF_post', 'sum'), pre_positive_n=('NF_pre', 'sum')).reset_index()
    out['fold'], out['arm'], out['split'] = fold, arm, split
    out['post_positive_rate'] = out.post_positive_n/out.n
    out['pre_positive_rate'] = out.pre_positive_n/out.n
    return out


def summarize(args, extra):
    old = pd.read_csv(args.shared_dir/'oof_predictions.csv')
    old = old.loc[old.window_ms.eq(args.window_ms) &
                  old.condition.isin(['01_full_all', '02_full_valid', '03_valid_valid'])]
    table = pd.concat([old, extra], ignore_index=True)
    args.reference_map = {}
    for condition in table.condition.unique():
        if condition.startswith('05_randomLabel_'):
            args.reference_map[condition] = '03_valid_valid'
        if condition.startswith(('20_', '21_', '22_', '23_', '24_')):
            valid = condition.endswith('_validtest')
            stem = condition.removesuffix('_validtest')
            refs = {'20_raw_shared_full':'01_full_all',
                    '21_normalized_shared_full':'20_raw_shared_full',
                    '22_raw_shared_shape':'20_raw_shared_full',
                    '23_normalized_shared_shape':'22_raw_shared_shape',
                    '24_normalized_original_full':'01_full_all'}
            ref = refs[stem]
            args.reference_map[condition] = ('02_full_valid' if ref=='01_full_all'
                else ref+'_validtest') if valid else ref
    result = shared.summarize_and_write(table, args)
    random = result.loc[result.condition.str.startswith('05_randomLabel_')]
    if len(random):
        random.groupby(['window_ms', 'manual_label']).agg(n_seeds=('condition', 'nunique'),
            delta_post_mean=('delta_post', 'mean'), delta_post_min=('delta_post', 'min'),
            delta_post_max=('delta_post', 'max'), delta_pre_mean=('delta_pre', 'mean'),
            contrast_mean=('post_minus_pre', 'mean')).to_csv(args.output_dir/'label_quota_seed_range.csv')
    print(result.loc[result.manual_label.isin(['overall','sn','cc']),
        ['condition','manual_label','n','post_positive_n','delta_post','delta_pre','post_minus_pre']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(r'C:\学校\MeerKAT\repertoire_geometry'))
    parser.add_argument('--shared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['label_control','amplitude','all'], default='all')
    parser.add_argument('--window-ms', type=int, default=400)
    parser.add_argument('--seeds', default='20261101,20261102,20261103')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261104)
    parser.add_argument('--run-models', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.output_dir.resolve() == args.shared_dir.resolve():
        raise ValueError('Use a new extension directory; never overwrite shared400.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = args.output_dir/'jobs'
    jobs.mkdir(exist_ok=True)
    if args.summarize_only:
        summarize(args, pd.concat([pd.read_csv(p) for p in sorted(jobs.glob('*.csv'))], ignore_index=True))
        return
    data, normalized, cols, protocol = load_inputs(args)
    freeze = dict(stage=args.stage, window_ms=args.window_ms, seeds=args.seeds,
        shared_protocol_sha256=shared.digest(args.shared_dir/'protocol.json'),
        cache_sha256=shared.digest(args.shared_dir/'cohort_with_history.pkl'),
        folds_sha256=shared.digest(args.shared_dir/'clip_fold_mapping.csv'),
        previous_oof_sha256=shared.digest(args.shared_dir/'oof_predictions.csv'),
        source_sha256=shared.digest(Path(__file__)), input_hashes=protocol['input_hashes'],
        n_calls=len(data), n_valid=int(data.egemaps_valid.sum()), acoustic_columns=cols,
        shared_controls=CONTROLS, shape_exclusions=DIRECT_AMPLITUDE,
        snr_status='Not controlled: existing SNR uses before-and-after waveform.',
        overlap_scope='Annotated call overlap inside focal boundaries, clipped at focal offset.',
        model='Original logistic/PCA30; train-only preprocessing; same hyperparameters.',
        random_rule='Uniform without replacement per training-fold manual label; exact valid quotas.',
        interpretation='Received amplitude, not source level/distance. Shape is operational, not amplitude-free.',
        ci_scope='Conditional OOF paired clip bootstrap; seed range separate; exploratory, not preregistered.')
    path = args.output_dir/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != freeze:
        raise ValueError('Extension protocol changed; use a new output directory.')
    shared.write_json(path, freeze)
    data[['call_id','_clip','fold','manual_label','duration_s','duration_bin','egemaps_valid']+CONTROLS].to_csv(
        args.output_dir/'cohort_controls.csv', index=False)
    print(json.dumps({k: freeze[k] for k in ['stage','n_calls','n_valid','shared_controls','snr_status']}, indent=2))
    if not args.run_models:
        print('Prepared only. Add --run-models for CPU fits. No encoder loaded.')
        return
    counts, membership = [], []
    previous = pd.read_csv(args.shared_dir/'oof_predictions.csv')
    previous = previous.loc[previous.condition.eq('01_full_all') & previous.window_ms.eq(args.window_ms)]
    m0_cache = {}
    with threadpool_limits(limits=args.threads):
        for fold in range(1,6):
            train, test = data.loc[data.fold.ne(fold)], data.loc[data.fold.eq(fold)]
            valid_train, valid_test = train.loc[train.egemaps_valid], test.loc[test.egemaps_valid]
            plans = []
            if args.stage in ['all','label_control']:
                for seed in map(int,args.seeds.split(',')):
                    sampled = sample_label_quotas(train, valid_train, seed+1009*fold+args.window_ms)
                    plans.append((f'05_randomLabel_{seed}', sampled, valid_test, cols, [], seed))
            if args.stage in ['all','amplitude']:
                shape = [c for c in cols if c not in DIRECT_AMPLITUDE]
                for condition, audio, controls in [
                    ('20_raw_shared_full',cols,CONTROLS), ('21_normalized_shared_full',cols,CONTROLS),
                    ('22_raw_shared_shape',shape,CONTROLS), ('23_normalized_shared_shape',shape,CONTROLS),
                    ('24_normalized_original_full',cols,[])]:
                    variant = data.copy()
                    if 'normalized' in condition:
                        variant.loc[:,cols] = normalized.set_index('call_id').loc[variant.call_id,cols].to_numpy()
                    plans.append((condition,variant.loc[variant.fold.ne(fold)],
                                  variant.loc[variant.fold.eq(fold)],audio,controls,-1))
            for condition,tr,te,audio,controls,seed in plans:
                assert not set(tr['_clip']) & set(te['_clip'])
                assert not tr.call_id.duplicated().any()
                tm = tr[['call_id','_clip','manual_label','duration_bin']].copy()
                tm['heldout_fold'], tm['condition'] = fold, condition
                membership.append(tm)
                counts += [report_counts(tr,fold,condition,'train'),report_counts(te,fold,condition,'test')]
                for target in ['NF_post','NF_pre']:
                    output = jobs/f'{condition}_fold{fold}_{target}.csv'
                    if output.exists():
                        saved = pd.read_csv(output)
                        assert set(saved.call_id)==set(te.call_id) and saved.fold.eq(fold).all()
                        continue
                    key = (fold,target,tuple(controls),condition if condition.startswith('05_') else 'all_train')
                    cached = m0_cache.get(key)
                    if not controls and not condition.startswith('05_'):
                        old = previous.loc[previous.target.eq(target)].set_index('call_id').loc[te.call_id]
                        assert np.array_equal(old.y_true.to_numpy(int),te[target].to_numpy(int))
                        cached = old.m0_pred.to_numpy()
                    p0,p1 = fit_control_pair(tr,te,target,audio,controls,args,m0_cached=cached)
                    m0_cache[key] = p0
                    record = shared.predict_record(te,target,condition,condition,p0,p1,len(tr),seed)
                    if not condition.startswith('05_'):
                        subset = record.loc[record.egemaps_valid].copy()
                        subset['condition'] += '_validtest'
                        record = pd.concat([record,subset],ignore_index=True)
                    record.to_csv(output,index=False)
                print(f'completed fold={fold}/5 {condition}',flush=True)
    pd.concat(counts,ignore_index=True).to_csv(args.output_dir/'fold_label_duration_counts.csv',index=False)
    pd.concat(membership,ignore_index=True).to_csv(args.output_dir/'training_membership.csv',index=False)
    extra = pd.concat([pd.read_csv(p) for p in sorted(jobs.glob('*.csv'))],ignore_index=True)
    for (condition,target),part in extra.groupby(['condition','target']):
        assert not part.call_id.duplicated().any()
        expected = data.loc[data.egemaps_valid] if condition.startswith('05_') or condition.endswith('_validtest') else data
        assert set(part.call_id)==set(expected.call_id)
    extra.to_csv(args.output_dir/'oof_predictions_added.csv',index=False)
    shared.write_json(args.output_dir/'verification.json',dict(status='passed',
        fixed_clip_folds=True,train_test_disjoint=True,exact_label_quotas=True,
        pre_post_same_calls=True,raw_normalized_same_calls=True,n_jobs=len(list(jobs.glob('*.csv')))))
    summarize(args,extra)


if __name__ == '__main__':
    main()
