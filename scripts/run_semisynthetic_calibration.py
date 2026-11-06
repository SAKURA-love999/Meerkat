#!/usr/bin/env python
"""Semi-synthetic fixed-design calibration; repeated fits are explicitly opt-in."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calibration_dgp import CONFIG, MARK_COLUMNS, SCENARIOS, TRUTH, generate

DEFAULT_SEEDS = [431, 907, 1601, 2203, 3251]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def binary_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -np.asarray(y) * np.log(p) - (1 - np.asarray(y)) * np.log1p(-p)


def load_real_design(args):
    import extend_shared_cohort_controls as ext
    import run_control_block_attribution as controls
    import run_shared_cohort_selection_control as shared
    data, _, cols = controls.load(args, ext, shared)
    # Real outcomes are never inputs or targets of the simulation assay.
    data = data.drop(columns=['NF_post', 'NF_pre']).copy()
    if len(cols) != 81 or not set(MARK_COLUMNS) <= set(cols):
        raise ValueError('Require original81 acoustic columns and declared mark channels.')
    if set(data.fold) != {1, 2, 3, 4, 5} or data.groupby('_clip').fold.nunique().max() != 1:
        raise ValueError('Frozen five-fold design changed.')
    return data, cols, ext.CONTROLS, shared


def protocol(args, data, cols, numeric_controls):
    here = Path(__file__).parent
    sources = [here / n for n in ['run_semisynthetic_calibration.py', 'calibration_dgp.py',
        'run_control_block_attribution.py', 'extend_shared_cohort_controls.py',
        'run_shared_cohort_selection_control.py', 'fit_meerkat_m1_acoustic_residual.py',
        'audit_egemaps_selection_on_librosa.py']]
    inputs = [args.shared_dir / n for n in ['cohort_with_history.pkl', 'clip_fold_mapping.csv', 'protocol.json']]
    inputs += [args.controls_dir / 'cohort_controls.csv', args.controls_dir / 'protocol.json',
               args.project_root / 'features/acoustic_features_capped_3000.csv']
    return dict(kind='SEMI_SYNTHETIC_DEVELOPMENT_PILOT', version=1,
        n_calls=len(data), n_clips=data['_clip'].nunique(), n_folds=5,
        seeds=args.seeds, scenarios=SCENARIOS, generator=CONFIG, truth=TRUTH,
        run_mode='quick_feasibility' if len(args.seeds)==1 else 'development_repeats',
        acoustic_columns=cols, measured_controls=numeric_controls,
        model='Original make_pipeline, median/scaler/PCA30/L2-logistic, seed0, train-only fitting.',
        bootstrap=dict(draws=args.n_bootstrap, seed_offset=7021,
                       scope='Paired clip resampling conditional on synthetic OOF; no refit uncertainty.'),
        threads=args.threads,
        source_hashes={str(p): digest(p) for p in sources},
        input_hashes={str(p): digest(p) for p in inputs},
        environment={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        history='Recorded real history is fixed covariate design, not regenerated synthetic temporal events.',
        selection='Original eGeMAPS validity flag is evaluation-only, not simulated feature extraction validity.',
        interpretation='Operational interpretation stress test; not validated CPI, nominal type-I control, or animal evidence.',
        deferred=['Conditional-importance comparator','Same-information reparameterization',
                  'Held-out generator families','Independent animal data'],
        reserved_evaluation=dict(expected_rate=.03, copula_clip_weight=.65,
            family='Nonlinear duration-history interaction rather than quadratic RMS; do not evaluate yet.'))


def freeze(args, data, cols, controls):
    value = protocol(args, data, cols, controls)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Protocol/code/runtime changed; preserve results and use a new directory.')
    write_json(path, value)
    return value


def design_path(args, seed, scenario):
    return args.output_dir / 'designs' / f'sim_seed{seed}_{scenario}.npz'


def prepare(args, data):
    folder = args.output_dir / 'designs'
    folder.mkdir(exist_ok=True)
    records, manifest = [], []
    for seed in args.seeds:
        for scenario in SCENARIOS:
            packet, info = generate(data, seed, scenario)
            path = design_path(args, seed, scenario)
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    if set(old.files) != set(packet) or any(not np.array_equal(old[k], packet[k]) for k in packet):
                        raise ValueError('Existing synthetic design differs: ' + str(path))
            else:
                temporary = path.with_name(path.stem + '.tmp.npz')
                np.savez_compressed(temporary, **packet)
                temporary.replace(path)
            info = {**info, 'truth': json.dumps(info['truth'], sort_keys=True)}
            records.append(info)
            manifest.append(dict(seed=seed, scenario=scenario, path=str(path), sha256=digest(path)))
    pd.DataFrame(records).to_csv(args.output_dir / 'synthetic_design_summary.csv', index=False)
    write_json(args.output_dir / 'synthetic_design_manifest.json', manifest)
    data[['call_id','_clip','fold','manual_label','duration_s','egemaps_valid']].to_csv(
        args.output_dir / 'fixed_design_membership.csv', index=False)
    print(json.dumps(dict(status='prepared_only', model_fits=0, n_designs=len(records),
        n_calls=len(data), synthetic=True, next='User runs -Stage Fit; no calibration result yet.'), indent=2))


def read_packet(args, data, seed, scenario, manifest):
    path = design_path(args, seed, scenario)
    entry = manifest[(seed, scenario)]
    if digest(path) != entry['sha256']:
        raise ValueError('Synthetic design changed.')
    with np.load(path, allow_pickle=False) as saved:
        packet = {k: saved[k] for k in saved.files}
    if not np.array_equal(packet['call_ids'], data.call_id.to_numpy()):
        raise ValueError('Synthetic/real design row order differs.')
    return packet


def materialize(data, packet):
    out = data.copy()
    out.loc[:, MARK_COLUMNS] = packet['sim_feature_values']
    for name in ['sim_mark','sim_h','sim_state','sim_p_post','sim_p_pre','sim_y_post','sim_y_pre']:
        out[name] = packet[name]
    assert 'NF_post' not in out and 'NF_pre' not in out
    return out


def arms(scenario):
    return ['standard', 'oracle_parent_reference'] if TRUTH[scenario]['oracle_parent'] else ['standard']


def fit_pair(train, test, side, scenario, arm, cols, controls, shared):
    if set(train['_clip']) & set(test['_clip']):
        raise ValueError('Train/test clips overlap.')
    target = 'sim_y_' + side
    if train[target].nunique() < 2:
        raise ValueError('Unfittable single-class training cell; no silent constant fallback.')
    numeric = shared.model_numeric_columns(train, 'NF_' + side) + controls
    if arm == 'oracle_parent_reference':
        parent = TRUTH[scenario]['oracle_parent']
        if not parent:
            raise ValueError('No declared generator-parent reference.')
        numeric = numeric + [parent]
    predictions = []
    for audio in [False, True]:
        acoustic = cols if audio else []
        inputs = ['manual_label'] + numeric + acoustic
        model = shared.base.make_pipeline(numeric, acoustic, audio, 30 if audio else None, 0)
        model.fit(train[inputs], train[target].to_numpy(int))
        predictions.append(model.predict_proba(test[inputs])[:, 1])
    return predictions


def job_path(args, seed, scenario, arm, fold, side):
    return args.output_dir / 'jobs' / f'sim_{seed}_{scenario}_{arm}_fold{fold}_{side}.csv'


def check_job(saved, test, seed, scenario, arm, fold, side, expected_n_train=None):
    if saved.call_id.duplicated().any() or set(saved.call_id) != set(test.call_id):
        raise ValueError('Synthetic OOF calls differ.')
    saved = saved.set_index('call_id').loc[test.call_id]
    expected = dict(replicate_seed=seed, scenario=scenario, analysis_arm=arm,
                    fold=fold, target='sim_' + side)
    if any(not saved[k].eq(v).all() for k, v in expected.items()):
        raise ValueError('Synthetic OOF key metadata differs.')
    if not saved.synthetic.eq(True).all():
        raise ValueError('Simulation flag missing or false.')
    if expected_n_train is not None and not saved.n_train.eq(expected_n_train).all():
        raise ValueError('Training-size metadata differs.')
    if not np.array_equal(saved.y_sim, test['sim_y_' + side]):
        raise ValueError('Wrong synthetic target, real labels must not be reused.')
    if not np.array_equal(saved.generator_probability, test['sim_p_' + side]):
        raise ValueError('Generator probability mismatch.')
    if not np.array_equal(saved['_clip'], test['_clip']):
        raise ValueError('Clip metadata mismatch.')
    p = saved[['m0_pred','m1_pred']].to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid synthetic OOF probabilities.')


def fit_all(args, data, cols, controls, shared, manifest):
    (args.output_dir / 'jobs').mkdir(exist_ok=True)
    for seed in args.seeds:
        for scenario in SCENARIOS:
            current = materialize(data, read_packet(args, data, seed, scenario, manifest))
            for arm in arms(scenario):
                for fold in range(1, 6):
                    train, test = current.loc[current.fold.ne(fold)], current.loc[current.fold.eq(fold)]
                    for side in ['post','pre']:
                        path = job_path(args, seed, scenario, arm, fold, side)
                        if path.exists():
                            check_job(pd.read_csv(path, float_precision='round_trip'), test,
                                      seed, scenario, arm, fold, side, len(train))
                            continue
                        p0, p1 = fit_pair(train, test, side, scenario, arm, cols, controls, shared)
                        record = test[['call_id','_clip','fold','manual_label','duration_s','egemaps_valid']].copy()
                        for key, value in dict(replicate_seed=seed, scenario=scenario, analysis_arm=arm,
                            target='sim_' + side, synthetic=True, n_train=len(train)).items():
                            record[key] = value
                        record['y_sim'] = test['sim_y_' + side]
                        record['generator_probability'] = test['sim_p_' + side]
                        record['m0_pred'], record['m1_pred'] = p0, p1
                        check_job(record, test, seed, scenario, arm, fold, side, len(train))
                        temporary = path.with_suffix('.tmp')
                        record.to_csv(temporary, index=False)
                        temporary.replace(path)
                    print(f'completed seed={seed} scenario={scenario} arm={arm} fold={fold}/5', flush=True)


def point_metrics(table):
    rows = []
    keys = ['scenario','analysis_arm','view']
    for tags, group in table.groupby(keys, sort=True):
        scenario, arm, view = tags
        extra_post = TRUTH[scenario]['extra_post'] if arm == 'standard' else False
        extra_pre = TRUTH[scenario]['extra_pre'] if arm == 'standard' else False
        for label, part in [('overall', group)] + list(group.groupby('manual_label', sort=True)):
            row = dict(scenario=scenario, analysis_arm=arm, view=view, manual_label=label,
                synthetic=True, n=part.call_id.nunique(), n_clips=part['_clip'].nunique(),
                truth_extra_post=extra_post, truth_extra_pre=extra_pre,
                direct_post_mark_in_generator=TRUTH[scenario]['direct_post_mark'])
            for side in ['post','pre']:
                g = part.loc[part.target.eq('sim_' + side)]
                if len(g) != row['n']:
                    raise ValueError('Simulated PRE/POST call sets differ.')
                row[side + '_positive_n'] = int(g.y_sim.sum())
                row[side + '_positive_rate'] = float(g.y_sim.mean())
                for model in ['m0','m1']:
                    row[model + '_' + side + '_logloss'] = float(g[model + '_loss'].mean())
                row['delta_' + side] = row['m0_' + side + '_logloss'] - row['m1_' + side + '_logloss']
                row['generator_parent_' + side + '_logloss'] = float(binary_loss(g.y_sim, g.generator_probability).mean())
            row['post_minus_pre'] = row['delta_post'] - row['delta_pre']
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_overall(table, n_draws, seed):
    rng = np.random.default_rng(seed)
    clip_ids = pd.Index(sorted(table['_clip'].unique()))
    keys = ['scenario','analysis_arm','view','target']
    cells, blocks = [], []
    for tags, group in table.groupby(keys, sort=True):
        g = group.groupby('_clip').agg(numerator=('gain','sum'), n=('call_id','size'))
        blocks.append(g.reindex(clip_ids, fill_value=0).to_numpy())
        cells.append(tags)
    block = np.stack(blocks, axis=1)
    samples = []
    for start in range(0, n_draws, 32):
        weights = rng.multinomial(len(clip_ids), np.full(len(clip_ids), 1 / len(clip_ids)),
                                  size=min(32, n_draws-start))
        totals = (weights @ block.reshape(len(clip_ids), -1)).reshape(len(weights), len(cells), 2)
        samples.append(totals[:,:,0] / np.where(totals[:,:,1] > 0, totals[:,:,1], np.nan))
    draws = np.concatenate(samples)
    rows = []
    for scenario, arm, view in sorted({c[:3] for c in cells}):
        ip = cells.index((scenario,arm,view,'sim_post'))
        ir = cells.index((scenario,arm,view,'sim_pre'))
        row = dict(scenario=scenario,analysis_arm=arm,view=view,manual_label='overall')
        estimates = [('delta_post',draws[:,ip]),('delta_pre',draws[:,ir]),
                     ('post_minus_pre',draws[:,ip]-draws[:,ir])]
        if arm == 'oracle_parent_reference':
            rp = cells.index((scenario,'standard',view,'sim_post'))
            rr = cells.index((scenario,'standard',view,'sim_pre'))
            estimates += [('delta_post_change_vs_standard',draws[:,ip]-draws[:,rp]),
                ('delta_pre_change_vs_standard',draws[:,ir]-draws[:,rr]),
                ('direction_change_vs_standard',draws[:,ip]-draws[:,ir]-draws[:,rp]+draws[:,rr])]
        for name, values in estimates:
            finite = values[np.isfinite(values)]
            row[name+'_ci_low'],row[name+'_ci_high'] = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    center = (k/n + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return center-half, center+half


def diagnostic_rates(metrics):
    overall = metrics.loc[metrics.manual_label.eq('overall')].copy()
    overall['positive_post_gain'] = overall.delta_post_ci_low.gt(0)
    overall['positive_pre_gain'] = overall.delta_pre_ci_low.gt(0)
    overall['positive_direction'] = overall.post_minus_pre_ci_low.gt(0)
    overall['post_specific_heuristic'] = overall.positive_post_gain & overall.positive_direction
    rows = []
    for tags, g in overall.groupby(['scenario','analysis_arm','view'], sort=True):
        checks = {name:g[name] for name in ['positive_post_gain','positive_pre_gain',
                                           'positive_direction','post_specific_heuristic']}
        if not bool(g.truth_extra_post.iloc[0]):
            checks['positive_gain_under_conditional_null'] = g.positive_post_gain
        if tags[0] == 'signal_post' and tags[1] == 'standard':
            checks['known_post_signal_detected'] = g.positive_post_gain
        if tags[0] == 'state_bilateral' and tags[1] == 'standard':
            checks['common_state_passes_post_specific_heuristic'] = g.post_specific_heuristic
        for name, values in checks.items():
            k, n = int(values.sum()), len(values)
            lo, hi = wilson(k,n)
            rows.append(dict(scenario=tags[0],analysis_arm=tags[1],view=tags[2],
                diagnostic=name,n_replicates=n,n_detected=k,rate=k/n,
                wilson_ci_low=lo,wilson_ci_high=hi,
                interpretation=('Single-replicate feasibility only; no error-rate calibration.' if n==1
                    else 'Development Monte Carlo rate, not a proven nominal type-I guarantee.')))
    return pd.DataFrame(rows)


def summarize(args, data, manifest):
    expected = {job_path(args,seed,scenario,arm,fold,side).name
        for seed in args.seeds for scenario in SCENARIOS for arm in arms(scenario)
        for fold in range(1,6) for side in ['post','pre']}
    actual = {p.name for p in (args.output_dir/'jobs').glob('*.csv')}
    if expected != actual:
        raise ValueError(f'Incomplete/stale jobs. Missing={len(expected-actual)}, extra={len(actual-expected)}; no summary written.')
    all_metrics, job_index = [], []
    for seed in args.seeds:
        parts = []
        for scenario in SCENARIOS:
            current = materialize(data, read_packet(args,data,seed,scenario,manifest))
            for arm in arms(scenario):
                for fold in range(1,6):
                    test = current.loc[current.fold.eq(fold)]
                    for side in ['post','pre']:
                        path = job_path(args,seed,scenario,arm,fold,side)
                        saved = pd.read_csv(path,float_precision='round_trip',dtype={'call_id':str,'_clip':str})
                        check_job(saved,test,seed,scenario,arm,fold,side,len(current)-len(test))
                        job_index.append(dict(path=str(path),sha256=digest(path),seed=seed,
                            scenario=scenario,analysis_arm=arm,fold=fold,target=side,n_rows=len(saved)))
                        saved['view'] = 'all'
                        selected = saved.loc[saved.egemaps_valid].copy()
                        selected['view'] = 'original_valid_rows'
                        parts += [saved,selected]
        table = pd.concat(parts,ignore_index=True)
        for model in ['m0','m1']:
            table[model+'_loss'] = binary_loss(table.y_sim,table[model+'_pred'])
        table['gain'] = table.m0_loss-table.m1_loss
        metrics = point_metrics(table).merge(bootstrap_overall(table,args.n_bootstrap,7021+seed),
            on=['scenario','analysis_arm','view','manual_label'],how='left',validate='one_to_one')
        metrics['replicate_seed'] = seed
        all_metrics.append(metrics)
    metrics = pd.concat(all_metrics,ignore_index=True)
    metrics.to_csv(args.output_dir/'synthetic_metrics.csv',index=False)
    diagnostic_rates(metrics).to_csv(args.output_dir/'replicate_diagnostic_rates.csv',index=False)
    pd.DataFrame(job_index).to_csv(args.output_dir/'synthetic_oof_manifest.csv',index=False)
    write_json(args.output_dir/'verification.json',dict(status='passed',synthetic=True,
        expected_jobs=len(expected),n_replicates=len(args.seeds),n_calls=len(data),
        run_mode='quick_feasibility' if len(args.seeds)==1 else 'development_repeats',
        validated_error_control=False,
        real_outcomes_not_used=True,fixed_clip_folds=True,train_test_clips_disjoint=True,
        original_classifier_hyperparameters=True,per_label_point_metrics=True,
        intervals='Overall only; conditional synthetic OOF paired clip bootstrap.',
        warning='A feasibility assay is not a validated method, nominal5% error control, or an animal finding.'))
    print('Synthetic pilot summary complete. Interpret repeated outcomes, not individual favorable seeds.')


def smoke(args, data, cols, controls, shared):
    chosen = []
    for _, part in data.groupby('fold'):
        clips = sorted(part['_clip'].unique(), key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())[:50]
        chosen.append(part.loc[part['_clip'].isin(clips)])
    small = pd.concat(chosen).reset_index(drop=True)
    packet, info = generate(small, 431, 'signal_post')
    current = materialize(small, packet)
    train, test = current.loc[current.fold.ne(1)], current.loc[current.fold.eq(1)]
    p0, p1 = fit_pair(train, test, 'post', 'signal_post', 'standard', cols, controls, shared)
    if len(p0) != len(test) or not np.isfinite(p1).all():
        raise ValueError('Technical model smoke failed.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='technical_smoke_passed', complete_calibration=False,
        scientific_results=False, synthetic=True, model_fits=2, n_calls=len(current),
        n_train=len(train), n_test=len(test), n_acoustic_features=len(cols),
        train_test_clips_disjoint=True, actual_model='Original make_pipeline/PCA30/logistic')
    write_json(args.output_dir / 'SMOKE_ONLY.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path(r'C:\学校\MeerKAT\repertoire_geometry'))
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--controls-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stage', choices=['inspect','prepare','smoke','fit','summary'], default='prepare')
    p.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(',')]
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Unique replicate seeds required.')
    if args.n_bootstrap < 20 or args.threads < 1:
        raise ValueError('Require at least20 bootstrap draws and positive thread count.')
    if args.output_dir.resolve() in {args.shared_dir.resolve(), args.controls_dir.resolve()}:
        raise ValueError('Never write simulation artifacts over real-data results.')
    data, cols, controls, shared = load_real_design(args)
    with threadpool_limits(limits=args.threads):
        if args.stage == 'inspect':
            args.output_dir.mkdir(parents=True, exist_ok=True)
            info = dict(status='design_check_only', n_calls=len(data), n_clips=data['_clip'].nunique(),
                n_acoustic_features=len(cols), actual_outcomes_dropped=True, model_fits=0)
            write_json(args.output_dir / 'DESIGN_CHECK_ONLY.json', info)
            print(json.dumps(info, indent=2))
            return
        if args.stage == 'smoke':
            smoke(args, data, cols, controls, shared)
            return
        freeze(args, data, cols, controls)
        if args.stage == 'prepare':
            prepare(args, data)
            return
        manifest = {(r['seed'],r['scenario']):r for r in json.loads(
            (args.output_dir / 'synthetic_design_manifest.json').read_text(encoding='utf-8'))}
        if args.stage == 'fit':
            fit_all(args, data, cols, controls, shared, manifest)
        summarize(args, data, manifest)


if __name__ == '__main__':
    main()
