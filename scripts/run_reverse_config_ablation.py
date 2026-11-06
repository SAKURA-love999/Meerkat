"""Complete the four-model source comparison with30 user-run M78-only fits."""
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

import run_config_source_attribution as cfg
from audit_nested_logistic_regularization import paired_points

pca,b3 = cfg.pca,cfg.b3
FAMILIES,TARGETS,CONFIG = cfg.FAMILIES,cfg.TARGETS,cfg.CONFIG
PREDICTIONS = dict(M0='m0_pred',Mconfig='mconfig_pred',M78='m78_pred',M1='m1_pred')
COMPARISONS = dict(total=('M0','M1'),config_alone=('M0','Mconfig'),
    remaining_given_config=('Mconfig','M1'),remaining_alone=('M0','M78'),
    config_given_remaining=('M78','M1'))
SETTINGS = {f+'__'+c:(f,c) for f in FAMILIES for c in COMPARISONS}


def remaining_columns(cols):
    if (len(cols)!=81 or len(set(cols))!=81 or not set(CONFIG).issubset(cols)
            or any(c in ['NF_post','NF_pre','call_id','_clip','fold','egemaps_valid']
                or c.startswith('sim_') for c in cols)):
        raise ValueError('Require exact original81 schema with configurations and no target/ID fields.')
    return [c for c in cols if c not in CONFIG]


def make_model(family,numeric,cols):
    if len(cols)!=78 or set(CONFIG)&set(cols):
        raise ValueError('M78 requires the ordered78 remaining fields,without configuration inputs.')
    return pca.make_model(family,'no_pca',numeric,cols)


def read_references(args,data,cols):
    path = args.config_run/'protocol.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    b3.verify_hashes(protocol)
    sha = b3.shared.digest(path)
    status = json.loads((args.config_run/'verification.json').read_text(encoding='utf-8'))
    checked = json.loads((args.config_analysis/'verification.json').read_text(encoding='utf-8'))
    if (status['status']!='passed' or status['new_fitted_jobs']!=30 or status['n_calls']!=len(data)
            or protocol['M1_fields']!=cols or checked['status']!='passed'
            or checked['source_protocol_sha256']!=sha or any(checked['max_abs_errors'].values())):
        raise ValueError('Require completed,audited,fixed configuration-source results.')
    originals,old_manifest = cfg.read_references(args,data,cols)
    declared = b3.read(args.config_run/'all_oof_manifest.csv')
    if len(declared)!=60 or declared.path.duplicated().any():
        raise ValueError('Require exact60 existing configuration-source manifest entries.')
    for item in declared.to_dict('records'):
        for p,h in [('path','sha256'),('metadata_path','metadata_sha256')]:
            if b3.shared.digest(Path(item[p]))!=item[h]:
                raise ValueError('Configuration reference artifact/metadata hash changed.')
    records = declared.set_index('path',verify_integrity=True)
    parts,manifest = [],[]
    source = argparse.Namespace(output_dir=args.config_run)
    for spec in cfg.make_plan(data,old_manifest).to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(originals,spec,test)
        frame,_ = cfg.load_job(source,spec,train,test,ref,sha)
        path = args.config_run/spec['relative_path']
        record = records.loc[str(path)]
        if record.role!='new_M0_plus_config':
            raise ValueError('Wrong reference prediction role.')
        parts.append(frame)
        manifest.append(dict(model_family=spec['family'],fold=spec['fold'],target=spec['target'],
            path=str(path),sha256=record.sha256,metadata_path=str(path.with_suffix('.json')),
            metadata_sha256=record.metadata_sha256,role='existing_M0_Mconfig_M1',n_rows=len(frame)))
    return pd.concat(parts,ignore_index=True),manifest


def make_plan(data,manifest):
    plan = cfg.make_plan(data,manifest)
    plan['model_role'] = 'M78_only'
    plan['relative_path'] = [f'jobs/{r.family}_M78_fold{r.fold}_{r.target}.csv' for r in plan.itertuples()]
    return plan


def capture_metadata(model,train,cols,spec,sha):
    meta = pca.transform_metadata(model,train,cols,spec,sha)
    branch = model.named_steps['preprocess'].named_transformers_['acoustic']
    meta.update(acoustic_imputer_statistics=branch.named_steps['imputer'].statistics_.tolist(),
        acoustic_scaler_mean=branch.named_steps['scaler'].mean_.tolist(),
        acoustic_scaler_variance=branch.named_steps['scaler'].var_.tolist(),
        acoustic_scaler_scale=branch.named_steps['scaler'].scale_.tolist())
    return meta


def check_metadata(meta,spec,train,cols,sha):
    pca.check_metadata(meta,spec,train,cols,sha)
    values = train[cols].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        medians = np.nanmedian(values,axis=0)
    retained = np.isfinite(medians)
    filled = np.where(np.isnan(values[:,retained]),medians[retained],values[:,retained])
    var = filled.var(axis=0)
    for name,expected in [('acoustic_imputer_statistics',medians),
            ('acoustic_scaler_mean',filled.mean(axis=0)),('acoustic_scaler_variance',var),
            ('acoustic_scaler_scale',np.where(var==0,1,np.sqrt(var)))]:
        actual = np.asarray(meta[name])
        if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-9,equal_nan=True):
            raise ValueError('Training-only preprocessing state differs: '+name)


def check_job(frame,meta,spec,train,test,reference,cols,sha):
    b3.check_record(frame,test,spec['family'],spec['fold'],spec['target'],len(train),sha)
    for key,value in spec.items():
        if not frame[key].eq(value).all():
            raise ValueError('M78 job contract differs: '+key)
    aligned = frame.set_index('call_id').loc[test.call_id]
    for col in ['m0_pred','mconfig_pred','m1_pred']:
        if not np.array_equal(aligned[col].to_numpy(),reference[col].to_numpy()):
            raise ValueError('Cached reference predictions changed: '+col)
    pred = aligned.m78_pred.to_numpy(float)
    if not np.isfinite(pred).all() or not ((pred>=0)&(pred<=1)).all():
        raise ValueError('Invalid M78 probabilities.')
    check_metadata(meta,spec,train,cols,sha)


def load_job(args,spec,train,test,ref,cols,sha):
    path = args.output_dir/spec['relative_path']
    meta_path = path.with_suffix('.json')
    if path.exists()!=meta_path.exists():
        raise ValueError('Orphan prediction/metadata pair;preserve and inspect.')
    if not path.exists():
        return None,None
    frame = b3.read(path)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if not frame.transform_metadata_sha256.eq(b3.shared.digest(meta_path)).all():
        raise ValueError('M78 metadata hash differs.')
    check_job(frame,meta,spec,train,test,ref,cols,sha)
    return frame,meta


def fit_all(args,data,cols,controls,references,manifest,sha):
    counts = dict(created_jobs=0,reused_jobs=0,model_fits=0)
    for spec in make_plan(data,manifest).to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(references,spec,test)
        saved,_ = load_job(args,spec,train,test,ref,cols,sha)
        if saved is not None:
            counts['reused_jobs'] += 1
            continue
        if set(train['_clip'])&set(test['_clip']) or train[spec['target']].nunique()!=2:
            raise ValueError('Clip leakage or single-class training partition.')
        numeric = b3.shared.model_numeric_columns(train,spec['target'])+controls
        inputs = ['manual_label']+numeric+cols
        model = make_model(spec['family'],numeric,cols)
        started,clock = datetime.now(timezone.utc).isoformat(),time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[inputs],train[spec['target']].to_numpy(int))
            pred = model.predict_proba(test[inputs])[:,1]
        counts['model_fits'] += 1
        meta = capture_metadata(model,train,cols,spec,sha)
        clf = getattr(model,'named_steps',{}).get('clf')
        meta.update(fit_start_utc=started,fit_end_utc=datetime.now(timezone.utc).isoformat(),
            fit_seconds=time.monotonic()-clock,warning_messages=[str(w.message) for w in caught],
            convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
            n_iter_max=int(np.max(clf.n_iter_)) if hasattr(clf,'n_iter_') else -1)
        out = test[b3.FIELDS].copy()
        for key,value in spec.items():
            out[key] = value
        out['model_family'],out['synthetic'],out['protocol_sha256'] = spec['family'],False,sha
        out['y_true'],out['m78_pred'] = test[spec['target']].to_numpy(int),pred
        for col in ['m0_pred','mconfig_pred','m1_pred']:
            out[col] = ref[col].to_numpy()
        check_job(out,meta,spec,train,test,ref,cols,sha)
        path = args.output_dir/spec['relative_path']
        path.parent.mkdir(parents=True,exist_ok=True)
        meta_path = path.with_suffix('.json')
        tmp_meta,tmp_csv = meta_path.with_suffix('.json.tmp'),path.with_suffix('.csv.tmp')
        b3.shared.write_json(tmp_meta,meta)
        out['transform_metadata_sha256'] = b3.shared.digest(tmp_meta)
        out.to_csv(tmp_csv,index=False)
        tmp_meta.replace(meta_path)
        tmp_csv.replace(path)
        counts['created_jobs'] += 1
        print('completed M78 '+spec['relative_path']+' '+json.dumps(counts),flush=True)
    return counts


def make_pairs(raw):
    for col in PREDICTIONS.values():
        pred = raw[col].to_numpy(float)
        if not np.isfinite(pred).all() or not ((pred>=0)&(pred<=1)).all():
            raise ValueError('Invalid four-model probabilities: '+col)
    losses = {m:b3.binary_loss(raw.y_true,raw[p]) for m,p in PREDICTIONS.items()}
    gains = {c:losses[left]-losses[right] for c,(left,right) in COMPARISONS.items()}
    interaction = gains['config_given_remaining']-gains['config_alone']
    checks = [gains['total']-gains['config_alone']-gains['remaining_given_config'],
        gains['total']-gains['remaining_alone']-gains['config_given_remaining'],
        interaction-(gains['remaining_given_config']-gains['remaining_alone'])]
    error = max(float(np.max(np.abs(x))) for x in checks)
    if error>1e-12:
        raise ValueError('Four-model per-call risk identities failed.')
    parts = []
    for comparison,(left,right) in COMPARISONS.items():
        part = raw.copy()
        part['model_family'] = raw.model_family+'__'+comparison
        part['m0_pred'],part['m1_pred'] = raw[PREDICTIONS[left]],raw[PREDICTIONS[right]]
        parts.append(part)
    values = raw[['call_id','_clip','manual_label','target','model_family','egemaps_valid']].copy()
    values['contrast'],values['value'] = 'predictive_interaction',interaction
    return pd.concat(parts,ignore_index=True),error,values


def decorate(frame):
    out = frame.rename(columns={'model_family':'setting'}).copy()
    out['model_family'] = out.setting.map(lambda s:SETTINGS[s][0])
    out['comparison'] = out.setting.map(lambda s:SETTINGS[s][1])
    out['left_model'] = out.comparison.map(lambda c:COMPARISONS[c][0])
    out['right_model'] = out.comparison.map(lambda c:COMPARISONS[c][1])
    return out


def summarize(args,data,cols,references,reference_manifest,sha,n_draws=1000):
    plan = make_plan(data,reference_manifest)
    expected = set(plan.relative_path)
    jobs = args.output_dir/'jobs'
    if ({p.relative_to(args.output_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
            or {p.relative_to(args.output_dir).with_suffix('.csv').as_posix() for p in jobs.glob('*.json')}!=expected):
        raise ValueError('Missing/stale jobs;no partial four-model summary.')
    parts,manifest,diagnostics = [],list(reference_manifest),[]
    for spec in plan.to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = pca.reference_for(references,spec,test)
        frame,meta = load_job(args,spec,train,test,ref,cols,sha)
        parts.append(frame)
        path = args.output_dir/spec['relative_path']
        manifest.append(dict(model_family=spec['family'],fold=spec['fold'],target=spec['target'],
            path=str(path),sha256=b3.shared.digest(path),metadata_path=str(path.with_suffix('.json')),
            metadata_sha256=b3.shared.digest(path.with_suffix('.json')),role='new_M78_only',n_rows=len(frame)))
        diagnostics.append({**{k:spec[k] for k in ['family','fold','target']},**{k:meta[k] for k in
            ['fit_start_utc','fit_end_utc','fit_seconds','convergence_warning','n_iter_max','n_after_imputer']}})
    raw = pd.concat(parts,ignore_index=True)
    if raw.duplicated(['model_family','target','call_id']).any():
        raise ValueError('Duplicate family/target/call OOF rows.')
    pairs,error,interaction = make_pairs(raw)
    table = b3.score_table(pairs)
    values = table[['call_id','_clip','manual_label','target','model_family','view']].copy()
    values['contrast'],values['value'] = 'gain',table.gain.to_numpy()
    intervals = b3.clip_intervals(values,n_draws,7452).drop(columns='contrast')
    point = b3.point_metrics(table).merge(intervals,on=['model_family','view','manual_label'],validate='one_to_one')
    long = decorate(point)
    roles = {c:c.replace('m0_','left_',1) if c.startswith('m0_') else c.replace('m1_','right_',1)
        for c in long if c.startswith('m0_') or c.startswith('m1_')}
    long.rename(columns=roles).to_csv(args.output_dir/'paired_metrics_and_ci.csv',index=False)
    iv = pd.concat([interaction.assign(view='all'),
        interaction.loc[interaction.egemaps_valid].assign(view='original_valid_rows')],ignore_index=True)
    interaction_ci = b3.clip_intervals(iv,n_draws,7452)
    interaction_point = paired_points(iv,interaction_ci)
    interaction_point.to_csv(args.output_dir/'predictive_interaction_and_ci.csv',index=False)
    iv_lookup = interaction_point.set_index(['model_family','view','manual_label'])
    ci_columns = [c for c in long if '_ci_' in c or c.endswith('_finite_draws')]
    metrics = ['delta_post','delta_pre','post_minus_pre']+ci_columns
    wide = []
    for tags,part in long.groupby(['model_family','view','manual_label'],sort=True):
        lookup = part.set_index('comparison')
        total,cfg_row,rest_row = lookup.loc['total'],lookup.loc['config_alone'],lookup.loc['remaining_alone']
        row = dict(zip(['model_family','view','manual_label'],tags))
        for col in ['n','n_clips','n_folds','post_positive_n','pre_positive_n']:
            row[col] = int(total[col])
        for side in ['post','pre']:
            for model,source,prefix in [('M0',total,'m0'),('Mconfig',cfg_row,'m1'),
                    ('M78',rest_row,'m1'),('M1',total,'m1')]:
                for metric in ['logloss','auc','ap']:
                    row[model+'_'+side+'_'+metric] = source[prefix+'_'+side+'_'+metric]
        for comparison in COMPARISONS:
            for col in metrics:
                row[comparison+'_'+col] = lookup.loc[comparison,col]
        for col in metrics:
            row['predictive_interaction_'+col] = iv_lookup.loc[tags,col]
        for metric in ['delta_post','delta_pre','post_minus_pre']:
            if (abs(row['total_'+metric]-row['remaining_alone_'+metric]-row['config_given_remaining_'+metric])>1e-12
                    or abs(row['predictive_interaction_'+metric]-row['config_given_remaining_'+metric]+row['config_alone_'+metric])>1e-12):
                raise ValueError('Aggregated risk identities failed.')
        wide.append(row)
    pd.DataFrame(wide).to_csv(args.output_dir/'four_model_metrics.csv',index=False)
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
    b3.shared.write_json(args.output_dir/'verification.json',dict(status='passed',new_M78_jobs=30,
        reference_model_refits=0,existing_three_models_reused=True,n_calls=len(data),n_clips=data['_clip'].nunique(),
        raw_oof_rows=len(raw),metric_rows=len(long),interaction_rows=len(interaction_point),
        per_call_identity_max_error=error,convergence_warning_jobs=int(pd.DataFrame(diagnostics).convergence_warning.sum()),
        causal_decomposition=False,conditional_information_certified=False,independent_confirmation=False,
        intervals='Unadjusted paired clips conditional on stored OOF;no training uncertainty.',
        limitation='M78 still contains duration/amplitude/resolution-dependent estimates;not pure vocal detail.'))
    print('Four-model risk comparison complete;predictive interaction is not biological synergy.')


def freeze(args,data,cols,controls,manifest):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plan,support = make_plan(data,manifest),pca.training_support(data,cols)
    for name,frame in [('fit_plan.csv',plan),('training_feature_support.csv',support)]:
        if args.stage!='prepare' and not (args.output_dir/name).exists():
            raise ValueError('Run Prepare first.')
        pca.fixed.freeze_csv(args.output_dir/name,frame)
    previous = json.loads((args.config_run/'protocol.json').read_text(encoding='utf-8'))
    configs = {}
    for family in FAMILIES:
        numeric = b3.shared.model_numeric_columns(data,'NF_post')+controls
        configs[family] = make_model(family,numeric,cols).named_steps['clf'].get_params()
        if configs[family]!=previous['learner_configs'][family]:
            raise ValueError('Original fixed learner parameters changed.')
    paths = [args.config_run/n for n in ['protocol.json','verification.json','all_oof_manifest.csv']]+[
        args.config_analysis/'verification.json']+[args.output_dir/n for n in ['fit_plan.csv','training_feature_support.csv']]
    inputs = dict(previous['input_hashes'])
    inputs.update({str(p):b3.shared.digest(p) for p in paths})
    for item in manifest:
        inputs[item['path']],inputs[item['metadata_path']] = item['sha256'],item['metadata_sha256']
    names = ['run_reverse_config_ablation.py','test_reverse_config_ablation.py','run_reverse_config_ablation400.ps1']
    sources = list(previous['source_hashes'])+[str(Path(__file__).parent/n) for n in names]+[
        str(Path(__file__).parent/'audit_nested_logistic_regularization.py')]
    value = dict(kind='REVERSE_CONFIG_ABLATION_400',version=1,n_calls=len(data),n_clips=data['_clip'].nunique(),
        n_valid=int(data.egemaps_valid.sum()),window_ms=400,families=FAMILIES,
        input_columns_M78=cols,removed_config_fields=CONFIG,new_M78_fits=30,reference_model_refits=0,
        pca='none',learner_configs=configs,measured_controls=controls,
        numeric_by_target={t:b3.shared.model_numeric_columns(data,t)+controls for t in TARGETS},
        comparisons={k:list(v) for k,v in COMPARISONS.items()},
        predictive_interaction='(L78-L1)-(L0-Lconfig)=(Lconfig-L1)-(L0-L78)',
        preprocessing='Same train-fold median/scaler,ordered78 subset;no sample filtering or re-extraction.',
        evaluation='All eligible calls primary;original-valid evaluation-only;all labels/PRE/POST/D/absolute losses.',
        bootstrap=dict(draws=1000,seed=7452,scope='Shared paired clips,unadjusted,conditional on stored OOF.'),
        exploratory=True,practical_margin=None,independent_confirmation=False,threads=4,
        interpretation='Finite-model risk attribution;not causal share,response,biological synergy or conditional information.',
        limitation='Removing3 config fields does not remove their influence on spectral estimates or isolate vocal shape.',
        source_hashes={p:b3.shared.digest(Path(p)) for p in sorted(set(sources))},input_hashes=inputs,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen reverse-ablation protocol changed;preserve results and inspect.')
    else:
        b3.shared.write_json(path,value)
    return b3.shared.digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run',
            'pca-run','audit-dir','config-run','config-analysis','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    protected = [v for k,v in vars(args).items() if isinstance(v,Path) and k!='output_dir']
    if any(args.output_dir.resolve()==p.resolve() or p.resolve() in args.output_dir.resolve().parents for p in protected):
        raise ValueError('Use a separate private output directory.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    with threadpool_limits(limits=4),ExitStack() as stack:
        if args.stage=='prepare':
            stack.enter_context(pca.fixed.zero_fit_guard())
        data,original_cols,controls,_ = b3.load_design(args)
        cols = remaining_columns(original_cols)
        cfg.validate_audit(args.audit_dir,data)
        references,manifest = read_references(args,data,original_cols)
        sha = freeze(args,data,cols,controls,manifest)
        if args.stage=='prepare':
            pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            b3.cohort_counts(data).to_csv(args.output_dir/'cohort_label_fold_counts.csv',index=False)
            result = dict(status='passed',model_fits=0,zero_fit_guard=True,n_calls=len(data),
                n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),
                M78_features=len(cols),config_inputs_removed=CONFIG,new_M78_fits=30,reference_model_refits=0,
                reference_jobs=len(manifest),protocol_sha256=sha,scientific_prediction_results_created=False)
            b3.shared.write_json(args.output_dir/'prepare_check.json',result)
            print(json.dumps(result,indent=2))
            return
        if args.stage=='fit':
            print('INVOCATION_FIT_COUNTS '+json.dumps(fit_all(args,data,cols,controls,references,manifest,sha)))
        summarize(args,data,cols,references,manifest,sha)


if __name__=='__main__':
    main()
