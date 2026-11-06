"""Audit cached extraction configuration against sample metadata; never fit/read audio."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_baseline_capacity_b3 as b3
import run_nested_logistic_regularization as fixed

CONFIG = ['n_fft','hop_length','n_mels']
EXPRESSIONS = dict(n_fft='min(512, max(64, 2 ** int(math.floor(math.log2(max(n_samples, 64))))))',
    hop_length='max(16, n_fft // 4)',n_mels='min(32, max(8, n_fft // 4))')


def verify_source_formulas(source):
    module = ast.parse(source)
    function = next((node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='extract_features'),None)
    if function is None:
        raise ValueError('Missing current extract_features function.')
    assignments = {node.targets[0].id:node.value for node in function.body
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    for name,expression in {'n_samples':'int(audio.size)',**EXPRESSIONS}.items():
        if name not in assignments or ast.dump(assignments[name])!=ast.dump(ast.parse(expression,mode='eval').body):
            raise ValueError('Current extractor formula differs: '+name)
    return dict(n_samples='int(audio.size)',**EXPRESSIONS)


def expected_config(samples):
    n_fft = np.minimum(512,np.maximum(64,2.**np.floor(np.log2(np.maximum(np.asarray(samples,float),64)))))
    return dict(n_fft=n_fft,hop_length=np.maximum(16,n_fft//4),n_mels=np.minimum(32,np.maximum(8,n_fft//4)))


def check_feature_rows(raw):
    needed = ['call_id','n_samples','sample_rate','start_time_sec','end_time_sec','duration_sec_from_audio','rms']+CONFIG
    if any(c not in raw for c in needed) or raw.call_id.duplicated().any():
        raise ValueError('Missing fields or duplicated raw feature call IDs.')
    samples,rate = raw.n_samples.to_numpy(float),raw.sample_rate.to_numpy(float)
    if (not np.isfinite(samples).all() or not np.isfinite(rate).all() or (samples<0).any()
            or (rate<=0).any() or not np.equal(samples,np.rint(samples)).all()):
        raise ValueError('Invalid integer sample count/rate metadata.')
    out = raw[needed].copy()
    early = (samples<4) | (raw.rms.to_numpy(float)<=1e-10)
    out['extractor_early_return'] = early
    matches = []
    for name,expected in expected_config(samples).items():
        expected = np.where(early,np.nan,expected)
        observed = raw[name].to_numpy(float)
        matched = (observed==expected) | (np.isnan(observed)&np.isnan(expected))
        out['expected_'+name],out[name+'_match'] = expected,matched
        matches.append(matched)
    out['formula_match'] = np.logical_and.reduce(matches)
    boundary = np.rint(raw.end_time_sec.to_numpy(float)*rate)-np.rint(raw.start_time_sec.to_numpy(float)*rate)
    out['rounded_boundary_n_samples'] = boundary
    out['boundary_sample_difference'] = samples-boundary
    out['sample_count_boundary_match'] = np.abs(samples-boundary)<=1
    error = np.abs(raw.duration_sec_from_audio.to_numpy(float)-samples/rate)
    out['audio_duration_error_s'],out['audio_duration_match'] = error,error<=1e-12
    return out


def perform_audit(args,data):
    source = args.project_root/'scripts/extract_acoustic_features.py'
    feature_path = args.project_root/'features/acoustic_features_capped_3000.csv'
    expressions = verify_source_formulas(source.read_text(encoding='utf-8-sig'))
    all_features = b3.read(feature_path)
    raw = all_features.loc[all_features.amplitude_variant.eq('raw')].copy()
    if len(raw)==0:
        raise ValueError('No explicitly raw-amplitude feature rows.')
    checked = check_feature_rows(raw)
    if not set(data.call_id).issubset(set(raw.call_id)):
        raise ValueError('Missing eligible calls in raw feature metadata.')
    eligible = checked.set_index('call_id').loc[data.call_id].reset_index()
    feature_rows = raw.set_index('call_id').loc[data.call_id]
    equal = np.isclose(data[CONFIG].to_numpy(float),feature_rows[CONFIG].to_numpy(float),rtol=0,atol=0,equal_nan=True).all(axis=1)
    duration_match = np.isclose(data.duration_s.to_numpy(float),feature_rows.duration_sec.to_numpy(float),rtol=0,atol=1e-12)
    label_match = data.manual_label.to_numpy()==feature_rows.label.to_numpy()
    eligible['model_feature_match'],eligible['annotation_duration_match'],eligible['manual_label_match'] = equal,duration_match,label_match
    for name in ['_clip','fold','manual_label','duration_bin','egemaps_valid','NF_post','NF_pre']:
        eligible[name] = data[name].to_numpy()
    checks = dict(raw_formula_mismatches=int((~checked.formula_match).sum()),
        eligible_formula_mismatches=int((~eligible.formula_match).sum()),
        eligible_boundary_mismatches_gt1sample=int((~eligible.sample_count_boundary_match).sum()),
        eligible_audio_duration_mismatches=int((~eligible.audio_duration_match).sum()),
        eligible_model_feature_mismatches=int((~equal).sum()),eligible_label_mismatches=int((~label_match).sum()),
        eligible_annotation_duration_mismatches=int((~duration_match).sum()))
    passed = not any(checks.values())
    if args.output_dir.exists():
        raise ValueError('Use a fresh source-audit output directory.')
    args.output_dir.mkdir(parents=True)
    checked.to_csv(args.output_dir/'formula_by_sample.csv',index=False)
    eligible.to_csv(args.output_dir/'eligible_formula_by_sample.csv',index=False)
    for name,keys in [('cohort_config_by_label.csv',['manual_label']+CONFIG),
            ('cohort_config_by_fold.csv',['fold']+CONFIG),
            ('cohort_config_by_duration_bin.csv',['duration_bin']+CONFIG)]:
        counts = eligible.groupby(keys,dropna=False).agg(n=('call_id','size'),n_clips=('_clip','nunique'),
            duration_min=('n_samples','min'),duration_max=('n_samples','max'),
            post_positive_n=('NF_post','sum'),pre_positive_n=('NF_pre','sum')).reset_index()
        counts = counts.rename(columns={'duration_min':'n_samples_min','duration_max':'n_samples_max'})
        counts.to_csv(args.output_dir/name,index=False)
    paths = [feature_path,source,Path(__file__),args.shared_dir/'cohort_with_history.pkl',
        args.shared_dir/'clip_fold_mapping.csv',args.controls_dir/'cohort_controls.csv']
    hashes = {str(p):b3.shared.digest(p) for p in paths}
    b3.shared.write_json(args.output_dir/'source_hashes.json',hashes)
    result = dict(status='passed' if passed else 'failed',model_fits=0,waveform_reextraction=False,
        n_raw=len(raw),n_eligible=len(data),n_clips=data['_clip'].nunique(),cohort_ids_sha256=fixed.ids_hash(data),
        checks=checks,current_source_formulas=expressions,source_hashes=hashes,
        sample_rates=sorted(raw.sample_rate.unique().tolist()),
        config_unique_values={c:sorted(raw[c].dropna().unique().tolist()) for c in CONFIG},
        n_config_tuples_eligible=len(eligible[CONFIG].drop_duplicates()),
        early_return_raw_n=int(checked.extractor_early_return.sum()),
        raw_boundary_mismatches_gt1sample=int((~checked.sample_count_boundary_match).sum()),
        historical_source_identity='Not proven;cached values compared empirically to current formulas.',
        interpretation='Deterministic sample-length/extraction-resolution proxies,not automatically noise or gain evidence.')
    b3.shared.write_json(args.output_dir/'summary.json',result)
    print(json.dumps(result,indent=2))
    if not passed:
        raise ValueError('Source audit failed;do not fit or filter the cohort to bypass it.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4),fixed.zero_fit_guard():
        data,_,_,_ = b3.load_design(args)
        perform_audit(args,data)


if __name__=='__main__':
    main()
