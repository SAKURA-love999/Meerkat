"""Configuration-source attribution:reuse M0/M1,30 user-run M0+config fits."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import audit_acoustic_config_fields as audit
import run_pca_sensitivity as pca

b3 = pca.b3
FAMILIES,TARGETS,CONFIG = pca.FAMILIES,pca.TARGETS,audit.CONFIG
COMPARISONS = dict(total=('M0','M1'),config=('M0','Mconfig'),remaining=('Mconfig','M1'))
SETTINGS = {f+'__'+c:(f,c) for f in FAMILIES for c in COMPARISONS}


def make_model(family,numeric):
    return pca.make_model(family,'no_pca',numeric,CONFIG)


def validate_audit(folder,data):
    value = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if value['status']!='passed':
        raise ValueError('Source audit did not pass;do not fit/filter to bypass it.')
    if (value['n_eligible']!=len(data) or value['cohort_ids_sha256']!=pca.fixed.ids_hash(data)
            or any(value['checks'].values()) or value['sample_rates']!=[8000]):
        raise ValueError('Source audit/cohort/rate contract differs.')
    for path,sha in value['source_hashes'].items():
        if b3.shared.digest(Path(path))!=sha:
            raise ValueError('Audited source changed: '+path)
    return value


def reconstruct_from_controls(data,sample_rate):
    start = data.clip_start_margin_s.to_numpy(float)
    end = 10.-data.clip_end_margin_s.to_numpy(float)
    samples = np.rint(end*sample_rate)-np.rint(start*sample_rate)
    out = pd.DataFrame(dict(call_id=data.call_id.to_numpy(),n_samples_from_M0_controls=samples))
    expected = audit.expected_config(samples)
    for col in CONFIG:
        out['reconstructed_'+col] = expected[col]
    out['config_match'] = np.isclose(data[CONFIG].to_numpy(float),
        np.column_stack([expected[c] for c in CONFIG]),rtol=0,atol=0,equal_nan=True).all(axis=1)
    return out


def read_references(args,data,cols):
    protocol = json.loads((args.pca_run/'protocol.json').read_text(encoding='utf-8'))
    b3.verify_hashes(protocol)
    status = json.loads((args.pca_run/'verification.json').read_text(encoding='utf-8'))
    if status['status']!='passed' or status['new_M1_jobs']!=60 or protocol['acoustic_columns']!=cols:
        raise ValueError('Require completed fixed S1 PCA screen.')
    originals,old_manifest = pca.fixed.load_references(args,data)
    declared = b3.read(args.pca_run/'all_oof_manifest.csv').set_index('path',verify_integrity=True)
    original_plan = pca.make_plan(data,old_manifest)
    parts,manifest = [],[]
    source_args = argparse.Namespace(output_dir=args.pca_run)
    source_sha = b3.shared.digest(args.pca_run/'protocol.json')
    for spec in original_plan.loc[original_plan.pca_mode.eq('no_pca')].to_dict('records'):
        path = args.pca_run/spec['relative_path']
        record = declared.loc[str(path)]
        if (record.role!='new_M1_only' or b3.shared.digest(path)!=record.sha256
                or b3.shared.digest(path.with_suffix('.json'))!=record.metadata_sha256):
            raise ValueError('No-PCA source artifact hash/role differs.')
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(originals,spec,test)
        frame,_ = pca.load_job(source_args,spec,train,test,ref,cols,source_sha)
        frame = frame.copy()
        frame['model_family'] = spec['family']
        parts.append(frame)
        manifest.append(dict(model_family=spec['family'],fold=spec['fold'],target=spec['target'],
            path=str(path),sha256=record.sha256,metadata_path=str(path.with_suffix('.json')),
            metadata_sha256=record.metadata_sha256,role='existing_M0_M1_noPCA',n_rows=len(frame)))
    return pd.concat(parts,ignore_index=True),manifest


def make_plan(data,manifest):
    plan = pca.make_plan(data,manifest)
    plan = plan.loc[plan.pca_mode.eq('no_pca')].copy()
    plan['model_role'] = 'M0_plus_config'
    plan['relative_path'] = [f'jobs/{r.family}_Mconfig_fold{r.fold}_{r.target}.csv' for r in plan.itertuples()]
    return plan.reset_index(drop=True)


def capture_metadata(model,train,spec,sha):
    meta = pca.transform_metadata(model,train,CONFIG,spec,sha)
    branch = model.named_steps['preprocess'].named_transformers_['acoustic']
    meta.update(config_imputer_statistics=branch.named_steps['imputer'].statistics_.tolist(),
        config_scaler_mean=branch.named_steps['scaler'].mean_.tolist(),
        config_scaler_variance=branch.named_steps['scaler'].var_.tolist(),
        config_scaler_scale=branch.named_steps['scaler'].scale_.tolist())
    return meta


def check_metadata(meta,spec,train,sha):
    pca.check_metadata(meta,spec,train,CONFIG,sha)
    values = train[CONFIG].to_numpy(float)
    median = np.nanmedian(values,axis=0)
    retained = np.isfinite(median)
    filled = np.where(np.isnan(values[:,retained]),median[retained],values[:,retained])
    for name,expected in [('config_imputer_statistics',median),('config_scaler_mean',filled.mean(axis=0)),
            ('config_scaler_variance',filled.var(axis=0))]:
        if not np.allclose(meta[name],expected,rtol=1e-12,atol=1e-9,equal_nan=True):
            raise ValueError('Training-only config preprocessing differs: '+name)


def check_job(frame,meta,spec,train,test,ref,sha):
    b3.check_record(frame,test,spec['family'],spec['fold'],spec['target'],len(train),sha)
    for key,value in spec.items():
        if not frame[key].eq(value).all():
            raise ValueError('Configuration job contract differs: '+key)
    aligned = frame.set_index('call_id').loc[test.call_id]
    if (not np.array_equal(aligned.m0_pred.to_numpy(),ref.m0_pred.to_numpy())
            or not np.array_equal(aligned.m1_pred.to_numpy(),ref.m1_pred.to_numpy())):
        raise ValueError('Existing M0/M1 predictions changed.')
    pred = aligned.mconfig_pred.to_numpy(float)
    if not np.isfinite(pred).all() or not ((pred>=0)&(pred<=1)).all():
        raise ValueError('Invalid Mconfig probabilities.')
    check_metadata(meta,spec,train,sha)


def load_job(args,spec,train,test,ref,sha):
    path = args.output_dir/spec['relative_path']
    metadata = path.with_suffix('.json')
    if path.exists()!=metadata.exists():
        raise ValueError('Orphan prediction/metadata pair;preserve and inspect.')
    if not path.exists():
        return None,None
    frame = b3.read(path)
    meta = json.loads(metadata.read_text(encoding='utf-8'))
    if not frame.transform_metadata_sha256.eq(b3.shared.digest(metadata)).all():
        raise ValueError('Metadata hash changed.')
    check_job(frame,meta,spec,train,test,ref,sha)
    return frame,meta


def fit_all(args,data,controls,references,manifest,sha):
    counts = dict(created_jobs=0,reused_jobs=0,model_fits=0)
    for spec in make_plan(data,manifest).to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(references,spec,test)
        saved,_ = load_job(args,spec,train,test,ref,sha)
        if saved is not None:
            counts['reused_jobs'] += 1
            continue
        if set(train['_clip']) & set(test['_clip']) or train[spec['target']].nunique()!=2:
            raise ValueError('Clip overlap or single-class training partition.')
        numeric = b3.shared.model_numeric_columns(train,spec['target'])+controls
        inputs = ['manual_label']+numeric+CONFIG
        model = make_model(spec['family'],numeric)
        started,clock = datetime.now(timezone.utc).isoformat(),time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[inputs],train[spec['target']].to_numpy(int))
            pred = model.predict_proba(test[inputs])[:,1]
        counts['model_fits'] += 1
        meta = capture_metadata(model,train,spec,sha)
        clf = getattr(model,'named_steps',{}).get('clf')
        meta.update(fit_start_utc=started,fit_end_utc=datetime.now(timezone.utc).isoformat(),
            fit_seconds=time.monotonic()-clock,convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
            warning_messages=[str(w.message) for w in caught],n_iter_max=int(np.max(clf.n_iter_)) if hasattr(clf,'n_iter_') else -1)
        out = test[b3.FIELDS].copy()
        for key,value in spec.items():
            out[key] = value
        out['model_family'],out['synthetic'],out['protocol_sha256'] = spec['family'],False,sha
        out['y_true'],out['mconfig_pred'] = test[spec['target']].to_numpy(int),pred
        out['m0_pred'],out['m1_pred'] = ref.m0_pred.to_numpy(),ref.m1_pred.to_numpy()
        check_job(out,meta,spec,train,test,ref,sha)
        path = args.output_dir/spec['relative_path']
        path.parent.mkdir(parents=True,exist_ok=True)
        metadata = path.with_suffix('.json')
        temporary_meta,temporary_csv = metadata.with_suffix('.json.tmp'),path.with_suffix('.csv.tmp')
        b3.shared.write_json(temporary_meta,meta)
        out['transform_metadata_sha256'] = b3.shared.digest(temporary_meta)
        out.to_csv(temporary_csv,index=False)
        temporary_meta.replace(metadata)
        temporary_csv.replace(path)
        counts['created_jobs'] += 1
        print('completed Mconfig '+spec['relative_path']+' '+json.dumps(counts),flush=True)
    return counts


def make_pairs(raw):
    losses = {m:b3.binary_loss(raw.y_true,raw[p]) for m,p in
        [('M0','m0_pred'),('Mconfig','mconfig_pred'),('M1','m1_pred')]}
    error = np.max(np.abs((losses['M0']-losses['M1'])-
        ((losses['M0']-losses['Mconfig'])+(losses['Mconfig']-losses['M1']))))
    if error>1e-12:
        raise ValueError('Per-call loss decomposition failed.')
    columns = dict(M0='m0_pred',Mconfig='mconfig_pred',M1='m1_pred')
    parts = []
    for comparison,(left,right) in COMPARISONS.items():
        part = raw.copy()
        part['model_family'] = raw.model_family+'__'+comparison
        part['m0_pred'],part['m1_pred'] = raw[columns[left]],raw[columns[right]]
        parts.append(part)
    return pd.concat(parts,ignore_index=True),float(error)


def decorate(frame):
    out = frame.rename(columns={'model_family':'setting'}).copy()
    out['model_family'] = out.setting.map(lambda s:SETTINGS[s][0])
    out['comparison'] = out.setting.map(lambda s:SETTINGS[s][1])
    out['left_model'] = out.comparison.map(lambda c:COMPARISONS[c][0])
    out['right_model'] = out.comparison.map(lambda c:COMPARISONS[c][1])
    return out


def summarize(args,data,references,ref_manifest,sha,n_draws=1000):
    plan = make_plan(data,ref_manifest)
    jobs = args.output_dir/'jobs'
    expected = set(plan.relative_path)
    if ({p.relative_to(args.output_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
            or {p.relative_to(args.output_dir).with_suffix('.csv').as_posix() for p in jobs.glob('*.json')}!=expected):
        raise ValueError('Missing/stale config jobs;no partial summary.')
    parts,manifest,diagnostics = [],list(ref_manifest),[]
    for spec in plan.to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(references,spec,test)
        frame,meta = load_job(args,spec,train,test,ref,sha)
        parts.append(frame)
        path = args.output_dir/spec['relative_path']
        manifest.append(dict(model_family=spec['family'],fold=spec['fold'],target=spec['target'],
            path=str(path),sha256=b3.shared.digest(path),metadata_path=str(path.with_suffix('.json')),
            metadata_sha256=b3.shared.digest(path.with_suffix('.json')),role='new_M0_plus_config',n_rows=len(frame)))
        diagnostics.append({**{k:spec[k] for k in ['family','fold','target']},**{k:meta[k] for k in
            ['fit_start_utc','fit_end_utc','fit_seconds','convergence_warning','n_iter_max','n_after_imputer']}})
    raw = pd.concat(parts,ignore_index=True)
    if raw.duplicated(['model_family','target','call_id']).any():
        raise ValueError('Duplicated model/target OOF.')
    pairs,error = make_pairs(raw)
    table = b3.score_table(pairs)
    values = table[['call_id','_clip','manual_label','target','model_family','view']].copy()
    values['contrast'],values['value'] = 'gain',table.gain.to_numpy()
    intervals = b3.clip_intervals(values,n_draws,7452).drop(columns='contrast')
    point = b3.point_metrics(table).merge(intervals,on=['model_family','view','manual_label'],validate='one_to_one')
    long = decorate(point)
    role_names = {c:c.replace('m0_','left_',1) if c.startswith('m0_') else c.replace('m1_','right_',1)
        for c in long if c.startswith('m0_') or c.startswith('m1_')}
    long.rename(columns=role_names).to_csv(args.output_dir/'paired_metrics_and_ci.csv',index=False)
    wide = []
    for tags,part in long.groupby(['model_family','view','manual_label'],sort=True):
        lookup = part.set_index('comparison')
        total,cfg = lookup.loc['total'],lookup.loc['config']
        row = dict(zip(['model_family','view','manual_label'],tags),n=int(total.n),n_clips=int(total.n_clips),
            n_folds=int(total.n_folds),post_positive_n=int(total.post_positive_n),pre_positive_n=int(total.pre_positive_n))
        for side in ['post','pre']:
            for model,source,prefix in [('M0',total,'m0'),('Mconfig',cfg,'m1'),('M1',total,'m1')]:
                for metric in ['logloss','auc','ap']:
                    row[model+'_'+side+'_'+metric] = source[prefix+'_'+side+'_'+metric]
        for comparison in COMPARISONS:
            for col in ['delta_post','delta_pre','post_minus_pre']+[c for c in part if '_ci_' in c or c.endswith('_finite_draws')]:
                row[comparison+'_'+col] = lookup.loc[comparison,col]
        for metric in ['delta_post','delta_pre','post_minus_pre']:
            if abs(row['total_'+metric]-row['config_'+metric]-row['remaining_'+metric])>1e-12:
                raise ValueError('Aggregated decomposition failed.')
        wide.append(row)
    pd.DataFrame(wide).to_csv(args.output_dir/'three_model_metrics.csv',index=False)
    folds = []
    for tags,part in table.groupby(['model_family','view','fold'],sort=True):
        row = dict(zip(['model_family','view','fold'],tags))
        for target,side in zip(TARGETS,['post','pre']):
            group = part.loc[part.target.eq(target)]
            row['left_'+side+'_logloss'],row['right_'+side+'_logloss'] = group.m0_loss.mean(),group.m1_loss.mean()
            row['delta_'+side] = group.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        folds.append(row)
    decorate(pd.DataFrame(folds)).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir/'fit_diagnostics.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    b3.shared.write_json(args.output_dir/'verification.json',dict(status='passed',new_fitted_jobs=30,
        new_M0_fits=0,new_M1_fits=0,existing_M0_M1_reused=True,n_calls=len(data),n_clips=data['_clip'].nunique(),
        raw_oof_rows=len(raw),paired_metric_rows=len(point),per_call_decomposition_max_error=error,
        causal_decomposition=False,conditional_information_certified=False,independent_confirmation=False,
        convergence_warning_jobs=int(pd.DataFrame(diagnostics).convergence_warning.sum()),
        intervals='Unadjusted paired clips conditional on OOF,no training uncertainty.',
        limitation='Remaining78 features still depend on extraction resolution;not certified pure vocal detail.'))
    print('Three-model decomposition complete;predictive losses,not causal percentages.')


def freeze(args,data,cols,controls,references,manifest,audit_result):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plan = make_plan(data,manifest)
    reconstruction = reconstruct_from_controls(data,8000)
    if not reconstruction.config_match.all():
        raise ValueError('Configs not reconstructible from existing M0 boundaries;inspect provenance before fitting.')
    for name,frame in [('fit_plan.csv',plan),('configuration_from_M0_controls.csv',reconstruction)]:
        if args.stage!='prepare' and not (args.output_dir/name).exists():
            raise ValueError('Run Prepare first.')
        pca.fixed.freeze_csv(args.output_dir/name,frame)
    source_protocol = json.loads((args.pca_run/'protocol.json').read_text(encoding='utf-8'))
    numeric = b3.shared.model_numeric_columns(data,'NF_post')+controls
    for family in FAMILIES:
        if make_model(family,numeric).named_steps['clf'].get_params()!=source_protocol['learner_configs'][family]:
            raise ValueError('Mconfig classifier settings differ from the cached M1 family.')
    paths = [args.pca_run/n for n in ['protocol.json','verification.json','all_oof_manifest.csv']]+[
        args.audit_dir/n for n in ['summary.json','source_hashes.json','eligible_formula_by_sample.csv']]+[
        args.output_dir/n for n in ['fit_plan.csv','configuration_from_M0_controls.csv']]
    inputs = dict(source_protocol['input_hashes'])
    inputs.update(audit_result['source_hashes'])
    inputs.update({str(p):b3.shared.digest(p) for p in paths})
    for item in manifest:
        inputs[item['path']],inputs[item['metadata_path']] = item['sha256'],item['metadata_sha256']
    names = ['audit_acoustic_config_fields.py','run_config_source_attribution.py',
        'test_config_source_attribution.py','run_config_source_attribution400.ps1']
    sources = list(source_protocol['source_hashes'])+[str(Path(__file__).parent/n) for n in names]
    value = dict(kind='CONFIG_SOURCE_ATTRIBUTION_400',version=1,n_calls=len(data),n_clips=data['_clip'].nunique(),
        window_ms=400,n_valid=int(data.egemaps_valid.sum()),families=FAMILIES,config_fields=CONFIG,
        M1_fields=cols,remaining_fields=[c for c in cols if c not in CONFIG],new_Mconfig_fits=30,
        M0_refits=0,M1_refits=0,pca='none;literal3-field subset of fixed S1 M1',
        learner_configs=source_protocol['learner_configs'],measured_controls=controls,
        numeric_by_target={t:b3.shared.model_numeric_columns(data,t)+controls for t in TARGETS},
        config_reconstructible_from_M0_boundary_controls=True,constant_sample_rate=8000,
        preprocessing='Same frozen per-feature median/scaler algorithms and training calls;new Mconfig state saved,old M1 state arrays not fabricated.',
        comparisons={k:list(v) for k,v in COMPARISONS.items()},identity='L0-L1=(L0-Lcfg)+(Lcfg-L1),per call and target.',
        evaluation='All calls primary;original-valid evaluation-only;all labels,absolute losses and POST/PRE/D for all3 contrasts.',
        bootstrap=dict(draws=1000,seed=7452,scope='Shared paired clip weights,unadjusted,conditional on stored OOF.'),
        interpretation='Finite-model prediction attribution,not causal share/response/conditional information or feature purity.',
        limitation='Config is length/boundary-derived;remaining features also depend on these extraction settings.',
        exploratory=True,practical_margin=None,independent_confirmation=False,threads=4,
        source_hashes={p:b3.shared.digest(Path(p)) for p in sorted(set(sources))},input_hashes=inputs,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen config protocol differs;preserve results,use a new output directory.')
    else:
        b3.shared.write_json(path,value)
    return b3.shared.digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run','pca-run','audit-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    protected = [args.project_root,args.shared_dir,args.controls_dir,args.b1_run,args.b2_run,args.b3_run,args.pca_run,args.audit_dir]
    if any(args.output_dir.resolve()==p.resolve() or p.resolve() in args.output_dir.resolve().parents for p in protected):
        raise ValueError('Use a separate private output directory.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    with threadpool_limits(limits=4),ExitStack() as stack:
        if args.stage=='prepare':
            stack.enter_context(pca.fixed.zero_fit_guard())
        data,cols,controls,_ = b3.load_design(args)
        if len(cols)!=81 or any(c not in cols for c in CONFIG):
            raise ValueError('Require exact original81 fields including the three configurations.')
        audit_result = validate_audit(args.audit_dir,data)
        references,manifest = read_references(args,data,cols)
        sha = freeze(args,data,cols,controls,references,manifest,audit_result)
        if args.stage=='prepare':
            pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            b3.cohort_counts(data).to_csv(args.output_dir/'cohort_label_fold_counts.csv',index=False)
            report = dict(status='passed',model_fits=0,zero_fit_guard=True,n_calls=len(data),n_clips=data['_clip'].nunique(),
                n_valid=int(data.egemaps_valid.sum()),config_fields=3,remaining_fields=78,reference_jobs=30,
                new_Mconfig_fits=30,M0_refits=0,M1_refits=0,source_audit_passed=True,
                configurations_reconstructed_from_M0_controls=True,protocol_sha256=sha,scientific_prediction_results_created=False)
            b3.shared.write_json(args.output_dir/'prepare_check.json',report)
            print(json.dumps(report,indent=2))
            return
        if args.stage=='fit':
            print('INVOCATION_FIT_COUNTS '+json.dumps(fit_all(args,data,controls,references,manifest,sha)))
        summarize(args,data,references,manifest,sha)


if __name__=='__main__':
    main()
