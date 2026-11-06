"""Add one paired asymmetric-null task without changing frozen v1 sources."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import calibration_dgp_asymmetric as dgp
import run_semisynthetic_calibration as core


def read(path):
    return pd.read_csv(path, float_precision='round_trip', dtype={'call_id': str, '_clip': str})


def source_post_entries(base_run):
    manifest = pd.read_csv(base_run / 'synthetic_oof_manifest.csv')
    selected = manifest.loc[manifest.scenario.eq(dgp.SOURCE_SCENARIO)
                            & manifest.target.eq('post') & manifest.seed.eq(431)].copy()
    expected = {(arm, fold) for arm in ['standard', 'oracle_parent_reference'] for fold in range(1, 6)}
    if len(selected) != 10 or set(zip(selected.analysis_arm, selected.fold)) != expected:
        raise ValueError('Require the ten declared source POST jobs.')
    for entry in selected.itertuples(index=False):
        if core.digest(entry.path) != entry.sha256:
            raise ValueError('Source POST job hash differs: ' + entry.path)
    return selected


def validate_original(base_run):
    original = json.loads((base_run / 'protocol.json').read_text(encoding='utf-8'))
    status = json.loads((base_run / 'verification.json').read_text(encoding='utf-8'))
    if original['seeds'] != [431] or status['status'] != 'passed' or status['expected_jobs'] != 60:
        raise ValueError('Require the completed original QUICK seed431 run.')
    for key in ['source_hashes', 'input_hashes']:
        for path, expected in original[key].items():
            if core.digest(path) != expected:
                raise ValueError('Frozen original source/input changed: ' + path)
    source_post_entries(base_run)


@contextmanager
def configured_core():
    # A scoped DGP registry adapter reuses the untouched orchestration/estimator.
    names = ['CONFIG', 'SCENARIOS', 'TRUTH', 'generate', 'protocol']
    old = {name: getattr(core, name) for name in names}

    def extension_protocol(args, data, cols, controls):
        value = old['protocol'](args, data, cols, controls)
        original = json.loads((args.base_run / 'protocol.json').read_text(encoding='utf-8'))
        for key in ['source_hashes', 'input_hashes']:
            for path, expected in original[key].items():
                if value[key].get(path) != expected:
                    raise ValueError('Paired source/model/design protocol differs: ' + path)
        value.update(kind='PAIRED_ASYMMETRIC_CONDITIONAL_NULL', extension_version=1,
            paired_source=str(args.base_run.resolve()),
            pairing='POST/features unchanged from source nonlinear null; only PRE nonlinear term removed.',
            planned_jobs=dict(reused_post=10, new_pre=10, new_model_fits=20),
            interpretation='Developmental counterexample test; known POST reused, not independent confirmation or calibrated information test.',
            deferred=['Weak-signal sweep', 'Additional seeds', 'Existing conditional-importance comparator',
                      'Held-out generators', 'Independent animal validation'])
        here = Path(__file__).parent
        for name in ['calibration_dgp_asymmetric.py', 'run_asymmetric_null_calibration.py',
                     'run_asymmetric_null400.ps1']:
            path = here / name
            value['source_hashes'][str(path)] = core.digest(path)
        source_files = [args.base_run / name for name in ['protocol.json', 'verification.json',
            'synthetic_oof_manifest.csv', 'designs/sim_seed431_null_nonlinear.npz']]
        source_files += [Path(p) for p in source_post_entries(args.base_run).path]
        for path in source_files:
            value['input_hashes'][str(path)] = core.digest(path)
        return value

    try:
        core.CONFIG = dgp.CONFIG
        core.SCENARIOS = [dgp.SCENARIO]
        core.TRUTH = dgp.TRUTH
        core.generate = dgp.generate
        core.protocol = extension_protocol
        yield core
    finally:
        for name, value in old.items():
            setattr(core, name, value)


def load_manifest(args):
    return {(r['seed'], r['scenario']): r for r in json.loads(
        (args.output_dir / 'synthetic_design_manifest.json').read_text(encoding='utf-8'))}


def reuse_post_jobs(args, data, manifest):
    packet = core.read_packet(args, data, 431, dgp.SCENARIO, manifest)
    original_path = args.base_run / 'designs/sim_seed431_null_nonlinear.npz'
    with np.load(original_path, allow_pickle=False) as old:
        for name in ['call_ids', 'sim_feature_values', 'sim_mark', 'sim_h', 'sim_state',
                     'sim_p_post', 'sim_y_post']:
            if not np.array_equal(packet[name], old[name]):
                raise ValueError('Pairing differs, cannot reuse POST: ' + name)
    current = core.materialize(data, packet)
    (args.output_dir / 'jobs').mkdir(exist_ok=True)
    rows = []
    for item in source_post_entries(args.base_run).itertuples(index=False):
        test = current.loc[current.fold.eq(item.fold)]
        source = read(item.path)
        core.check_job(source, test, 431, dgp.SOURCE_SCENARIO, item.analysis_arm,
                       item.fold, 'post', len(current)-len(test))
        path = core.job_path(args, 431, dgp.SCENARIO, item.analysis_arm, item.fold, 'post')
        expected = source.copy()
        expected['scenario'] = dgp.SCENARIO
        expected['fit_origin'] = 'reused_original_post'
        expected['source_job_path'] = item.path
        expected['source_job_sha256'] = item.sha256
        if path.exists():
            saved = read(path)
            core.check_job(saved, test, 431, dgp.SCENARIO, item.analysis_arm,
                           item.fold, 'post', len(current)-len(test))
            aligned = saved.set_index('call_id').loc[expected.call_id]
            for col in ['m0_pred', 'm1_pred', 'generator_probability', 'y_sim']:
                if not np.array_equal(aligned[col], expected[col]):
                    raise ValueError('Existing reused POST differs: ' + col)
            if not (saved.fit_origin.eq('reused_original_post').all()
                    and saved.source_job_path.eq(item.path).all()
                    and saved.source_job_sha256.eq(item.sha256).all()):
                raise ValueError('Reused POST provenance differs.')
        else:
            temporary = path.with_suffix('.tmp')
            expected.to_csv(temporary, index=False)
            temporary.replace(path)
        rows.append(dict(source_path=item.path, source_sha256=item.sha256,
            destination_path=str(path), destination_sha256=core.digest(path),
            analysis_arm=item.analysis_arm, fold=item.fold, n_calls=len(test),
            model_fits=0, post_target_and_features_identical=True))
    pd.DataFrame(rows).to_csv(args.output_dir / 'post_reuse_manifest.csv', index=False)
    core.write_json(args.output_dir / 'pairing_check.json', dict(status='passed',
        synthetic=True, n_calls=len(data), n_clips=data['_clip'].nunique(),
        reused_post_jobs=len(rows), planned_new_pre_jobs=10, planned_new_model_fits=20,
        post_targets_probabilities_and_features_identical=True,
        source=str(args.base_run), new_post_evidence=False,
        warning='Known POST gains are reused; this is a developmental PRE contrast, not independent confirmation.'))


def summarize_extension(args, data, manifest):
    core.summarize(args, data, manifest)
    path = args.output_dir / 'verification.json'
    status = json.loads(path.read_text(encoding='utf-8'))
    status.update(reused_post_jobs=10, newly_fitted_pre_jobs_in_design=10,
        required_new_model_fits_in_design=20, paired_to_symmetric_seed431=True,
        new_post_evidence=False, source=str(args.base_run))
    core.write_json(path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root', 'shared-dir', 'controls-dir', 'base-run', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summary'], default='prepare')
    args = parser.parse_args()
    args.seeds, args.n_bootstrap, args.threads = [431], 1000, 4
    protected = {p.resolve() for p in [args.base_run, args.shared_dir, args.controls_dir]}
    if args.output_dir.resolve() in protected:
        raise ValueError('Never overwrite source results.')
    validate_original(args.base_run)
    data, cols, controls, shared = core.load_real_design(args)
    with threadpool_limits(limits=4), configured_core():
        core.freeze(args, data, cols, controls)
        if args.stage == 'prepare':
            core.prepare(args, data)
        manifest = load_manifest(args)
        reuse_post_jobs(args, data, manifest)
        if args.stage == 'prepare':
            print('Prepared one paired design and reused10 POST jobs; zero new model fits.')
            return
        if args.stage == 'fit':
            core.fit_all(args, data, cols, controls, shared, manifest)
        summarize_extension(args, data, manifest)


if __name__ == '__main__':
    main()
