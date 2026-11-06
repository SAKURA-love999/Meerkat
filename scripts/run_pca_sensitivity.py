"""S1 fixed-B3 PCA screen; zero-fit Prepare and user-run60 M1 fits."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import run_baseline_capacity_b3 as b3
import run_nested_logistic_regularization as fixed

FAMILIES = ['logistic_linear','spline_rms','hist_gbdt']
TARGETS = ['NF_post','NF_pre']
NEW_MODES = ['pca_full','no_pca']
MODES = ['pca30']+NEW_MODES
SETTINGS = {f+'__'+m:(f,m) for f in FAMILIES for m in MODES}


def make_model(family,mode,numeric,acoustic):
    forbidden = [c for c in numeric+acoustic if c.startswith('sim_') or c in
        ['call_id','_clip','fold','NF_post','NF_pre','egemaps_valid']]
    if (family not in FAMILIES or mode not in NEW_MODES or not acoustic
            or models.RMS not in numeric or forbidden
            or len(set(numeric+acoustic))!=len(numeric+acoustic)):
        raise ValueError('Undeclared PCA/family,empty acoustic branch or forbidden inputs.')
    if family=='logistic_linear':
        model = models.base.make_pipeline(numeric,acoustic,True,30,0)
    else:
        model = models.make_model(family,numeric,acoustic)
    branch = next(t[1] for t in model.named_steps['preprocess'].transformers if t[0]=='acoustic')
    if mode=='pca_full':
        branch.set_params(pca=PCA(n_components=None,svd_solver='full',whiten=False,random_state=0))
    else:
        branch.steps = [(name,step) for name,step in branch.steps if name!='pca']
    return model


def make_plan(data,manifest):
    lookup = {(r['model_family'],int(r['fold']),r['target']):r for r in manifest}
    expected = {(f,k,t) for f in FAMILIES for k in range(1,6) for t in TARGETS}
    if len(manifest)!=30 or set(lookup)!=expected:
        raise ValueError('Require exactly30 fixed-B3 reference records.')
    rows = []
    for family in FAMILIES:
        for fold in range(1,6):
            train,test = data.loc[data.fold.ne(fold)],data.loc[data.fold.eq(fold)]
            for target in TARGETS:
                ref = lookup[(family,fold,target)]
                for mode in NEW_MODES:
                    rows.append(dict(family=family,pca_mode=mode,fold=fold,target=target,
                        n_train=len(train),n_test=len(test),train_ids_sha256=fixed.ids_hash(train),
                        test_ids_sha256=fixed.ids_hash(test),reference_path=ref['path'],
                        reference_sha256=ref['sha256'],relative_path=f'jobs/{family}_{mode}_fold{fold}_{target}.csv'))
    return pd.DataFrame(rows)


def reference_for(references,spec,test):
    frame = references.loc[references.model_family.eq(spec['family']) & references.fold.eq(spec['fold'])
        & references.target.eq(spec['target'])]
    if len(frame)!=len(test) or frame.call_id.duplicated().any():
        raise ValueError('Reference call coverage differs.')
    return frame.set_index('call_id').loc[test.call_id]


def transform_metadata(model,train,cols,spec,protocol_sha):
    branch = model.named_steps['preprocess'].named_transformers_['acoustic']
    imputer,scaler = branch.named_steps['imputer'],branch.named_steps['scaler']
    retained = imputer.get_feature_names_out(cols).tolist()
    samples = np.asarray(scaler.n_samples_seen_)
    pca = branch.named_steps.get('pca')
    full = pca is not None
    meta = dict(spec=spec,protocol_sha256=protocol_sha,input_columns=cols,
        retained_columns=retained,dropped_empty_columns=[c for c in cols if c not in retained],
        n_train=len(train),n_input=len(cols),n_after_imputer=len(retained),
        acoustic_output_dim=int(pca.n_components_) if full else len(retained),
        scaler_n_samples_min=int(samples.min()),scaler_n_samples_max=int(samples.max()),
        constant_scaled_columns=[c for c,var in zip(retained,scaler.var_) if var==0],
        pca_applied=full,pca_components=int(pca.n_components_) if full else 0,
        pca_n_samples=int(pca.n_samples_) if full else None,whiten=bool(pca.whiten) if full else None,
        pca_parameters=pca.get_params() if full else None,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist() if full else [],
        explained_variance=pca.explained_variance_.tolist() if full else [],
        singular_values=pca.singular_values_.tolist() if full else [],
        full_fit_cumulative_variance_first30=float(pca.explained_variance_ratio_[:30].sum()) if full else None,
        orthogonality_max_abs_error=float(np.max(np.abs(pca.components_@pca.components_.T-np.eye(pca.n_components_)))) if full else None,
        clf_config=model.named_steps['clf'].get_params())
    return meta


def check_metadata(meta,spec,train,cols,sha):
    if meta['spec']!=spec or meta['protocol_sha256']!=sha or meta['input_columns']!=cols:
        raise ValueError('Transform metadata contract differs.')
    retained = [c for c in cols if train[c].notna().any()]
    if (meta['retained_columns']!=retained or meta['dropped_empty_columns']!=[c for c in cols if c not in retained]
            or meta['n_train']!=len(train) or meta['n_input']!=len(cols)
            or meta['n_after_imputer']!=len(retained)
            or meta['scaler_n_samples_min']!=len(train) or meta['scaler_n_samples_max']!=len(train)):
        raise ValueError('Imputer/scaler support or training-only sample count differs.')
    full = spec['pca_mode']=='pca_full'
    if (meta['pca_applied']!=full or meta['acoustic_output_dim']!=len(retained)
            or meta['pca_components']!=(len(retained) if full else 0)):
        raise ValueError('Full PCA did not retain every available acoustic dimension.')
    if full and (meta['whiten'] is not False or meta['pca_n_samples']!=len(train)
            or len(meta['explained_variance_ratio'])!=len(retained)
            or not np.isfinite(meta['explained_variance_ratio']).all()):
        raise ValueError('Invalid full-PCA parameters/training support/spectrum.')
    config = meta['clf_config']
    if spec['family']=='hist_gbdt':
        if any(config[k]!=v for k,v in models.HGB.items()):
            raise ValueError('HGB settings changed.')
    elif (config['C']!=1. or config['penalty']!='l2' or config['solver']!='lbfgs'
            or config['max_iter']!=2000 or config['tol']!=1e-4 or config['class_weight'] is not None):
        raise ValueError('Original fixed C1 logistic settings changed.')


def check_job(frame,meta,spec,train,test,reference,cols,sha):
    setting = spec['family']+'__'+spec['pca_mode']
    b3.check_record(frame,test,setting,spec['fold'],spec['target'],len(train),sha)
    for key,value in spec.items():
        if not frame[key].eq(value).all():
            raise ValueError('PCA job contract differs: '+key)
    aligned = frame.set_index('call_id').loc[test.call_id]
    if not np.array_equal(aligned.m0_pred.to_numpy(),reference.m0_pred.to_numpy()):
        raise ValueError('M0 predictions changed: only M1 may be refitted.')
    check_metadata(meta,spec,train,cols,sha)


def load_job(args,spec,train,test,ref,cols,sha):
    path = args.output_dir/spec['relative_path']
    meta_path = path.with_suffix('.json')
    if path.exists()!=meta_path.exists():
        raise ValueError('Orphan prediction/metadata pair; preserve files and inspect before continuing.')
    if not path.exists():
        return None,None
    frame = b3.read(path)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if not frame.transform_metadata_sha256.eq(b3.shared.digest(meta_path)).all():
        raise ValueError('Transform metadata hash differs.')
    check_job(frame,meta,spec,train,test,ref,cols,sha)
    return frame,meta


def fit_all(args,data,cols,controls,references,manifest,sha):
    counts = dict(created_jobs=0,reused_jobs=0,model_fits=0)
    for spec in make_plan(data,manifest).to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = reference_for(references,spec,test)
        cached,_ = load_job(args,spec,train,test,ref,cols,sha)
        if cached is not None:
            counts['reused_jobs'] += 1
            continue
        if set(train['_clip']) & set(test['_clip']) or train[spec['target']].nunique()!=2:
            raise ValueError('Clip leakage or single-class training partition.')
        numeric = b3.shared.model_numeric_columns(train,spec['target'])+controls
        inputs = ['manual_label']+numeric+cols
        model = make_model(spec['family'],spec['pca_mode'],numeric,cols)
        started,clock = datetime.now(timezone.utc).isoformat(),time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[inputs],train[spec['target']].to_numpy(int))
            pred = model.predict_proba(test[inputs])[:,1]
        counts['model_fits'] += 1
        meta = transform_metadata(model,train,cols,spec,sha)
        clf = getattr(model,'named_steps',{}).get('clf')
        meta.update(fit_start_utc=started,fit_end_utc=datetime.now(timezone.utc).isoformat(),
            fit_seconds=time.monotonic()-clock,convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
            warning_messages=[str(w.message) for w in caught],n_iter_max=int(np.max(clf.n_iter_)) if hasattr(clf,'n_iter_') else -1)
        out = test[b3.FIELDS].copy()
        for key,value in spec.items():
            out[key] = value
        out['model_family'],out['synthetic'],out['protocol_sha256'] = spec['family']+'__'+spec['pca_mode'],False,sha
        out['y_true'],out['m0_pred'],out['m1_pred'] = test[spec['target']].to_numpy(int),ref.m0_pred.to_numpy(),pred
        check_job(out,meta,spec,train,test,ref,cols,sha)
        path = args.output_dir/spec['relative_path']
        path.parent.mkdir(parents=True,exist_ok=True)
        meta_path = path.with_suffix('.json')
        temporary_meta,temporary_csv = meta_path.with_suffix('.json.tmp'),path.with_suffix('.csv.tmp')
        b3.shared.write_json(temporary_meta,meta)
        out['transform_metadata_sha256'] = b3.shared.digest(temporary_meta)
        out.to_csv(temporary_csv,index=False)
        temporary_meta.replace(meta_path)
        temporary_csv.replace(path)
        counts['created_jobs'] += 1
        print('completed M1 '+spec['relative_path']+' '+json.dumps(counts),flush=True)
    return counts


def training_support(data,cols):
    rows = []
    for fold in range(1,6):
        train = data.loc[data.fold.ne(fold)]
        for col in cols:
            values = train[col]
            if np.isinf(values.to_numpy(float)).any():
                raise ValueError('Infinite original feature: '+col)
            rows.append(dict(fold=fold,feature=col,n_train=len(train),missing_n=int(values.isna().sum()),
                all_missing=bool(values.isna().all()),observed_unique_n=int(values.nunique()),
                constant_observed=bool(values.nunique()==1)))
    return pd.DataFrame(rows)


def freeze(args,data,cols,controls,b1,manifest):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plan,support = make_plan(data,manifest),training_support(data,cols)
    for fold in range(1,6):
        train = data.loc[data.fold.ne(fold)]
        active = int((~support.loc[support.fold.eq(fold),'all_missing']).sum())
        if active<30 or len(train)<active or any(train[t].nunique()!=2 for t in TARGETS):
            raise ValueError('Insufficient training support for declared PCA/comparisons.')
    for name,frame in [('fit_plan.csv',plan),('training_feature_support.csv',support)]:
        if args.stage!='prepare' and not (args.output_dir/name).exists():
            raise ValueError('Run Prepare before Fit/Summary.')
        fixed.freeze_csv(args.output_dir/name,frame)
    configs = {}
    for family in FAMILIES:
        numeric = b3.shared.model_numeric_columns(data,'NF_post')+controls
        model = make_model(family,'pca_full',numeric,cols)
        expected = b1['hgb'] if family=='hist_gbdt' else b1['logistic']
        if model.named_steps['clf'].get_params()!=expected:
            raise ValueError('S1 must preserve original B3 classifier configuration.')
        configs[family] = expected
    previous = json.loads((args.b3_run/'protocol.json').read_text(encoding='utf-8'))
    names = ['run_pca_sensitivity.py','test_pca_sensitivity.py','run_pca_sensitivity400.ps1',
        'run_nested_logistic_regularization.py']
    sources = list(previous['source_hashes'])+[str(Path(__file__).parent/n) for n in names]
    inputs = dict(previous['input_hashes'])
    inputs.update({r['path']:r['sha256'] for r in manifest})
    for p in [args.b3_run/n for n in ['protocol.json','verification.json','all_oof_manifest.csv']]+[
            args.output_dir/n for n in ['fit_plan.csv','training_feature_support.csv']]:
        inputs[str(p)] = b3.shared.digest(p)
    value = dict(kind='FIXED_B3_PCA_SENSITIVITY_S1',version=1,window_ms=400,
        n_calls=len(data),n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),
        families=FAMILIES,modes=MODES,new_modes=NEW_MODES,new_M1_fits=60,M0_refits=0,
        reference_files=30,acoustic_columns=cols,measured_controls=controls,learner_configs=configs,
        spline=models.SPLINE,hgb_requested=models.HGB,
        numeric_by_target={t:b3.shared.model_numeric_columns(data,t)+controls for t in TARGETS},
        full_pca=dict(n_components=None,svd_solver='full',whiten=False,random_state=0),
        no_pca='Remove only acoustic PCA;retain original imputer/scaler and all other branches.',
        pca30='Original B3 fixed-C1/HGB OOF reused;original fitted transform spectrum not recorded.',
        no_nested_C_inputs=True,selection='None;report all three PCA settings and all labels,never choose by outer gain.',
        preprocessing='Train-fold-only original median imputation/scaling;full PCA fitted only on same training calls.',
        evaluation='All18345 primary;original-valid14786 evaluation-only;all labels;absolute M0/M1 logloss plus DeltaPOST/PRE/D.',
        paired='Within each family,M0 identical;Delta change equals old-minus-new M1 loss. Compare full-vs30,none-vs30,none-vsfull.',
        bootstrap=dict(draws=1000,seed=7452,scope='Shared paired clip weights,conditional on stored OOF;unadjusted,no refits.'),
        coordinate_diagnostic='Report continuous full/no-PCA prediction/loss differences;no filtering threshold or winner.',
        target='Clip-local annotated nonfocal activity proxy,not verified response/conditional intensity/causality.',
        snr='Not controlled:legacy proxy reads before/after audio.',exploratory=True,
        independent_confirmation=False,practical_margin=None,threads=4,
        source_hashes={p:b3.shared.digest(Path(p)) for p in sorted(set(sources))},input_hashes=inputs,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen S1 protocol differs;preserve results,use a new directory.')
    else:
        b3.shared.write_json(path,value)
    return b3.shared.digest(path)


def decorate(frame):
    out = frame.rename(columns={'model_family':'setting'}).copy()
    out['model_family'] = out.setting.map(lambda x:SETTINGS[x][0])
    out['pca_mode'] = out.setting.map(lambda x:SETTINGS[x][1])
    if 'contrast' in out:
        out['reference_pca_mode'] = out.contrast.map({'vs_pca30':'pca30','vs_pca_full':'pca_full'})
    return out


def contrast_values(table):
    columns = ['call_id','_clip','manual_label','target','model_family','view']
    primary = table[columns].copy()
    primary['contrast'],primary['value'] = 'gain',table.gain.to_numpy()
    parts = [primary]
    for family in FAMILIES:
        for mode,reference_mode in [('pca_full','pca30'),('no_pca','pca30'),('no_pca','pca_full')]:
            selected = table.loc[table.model_family.eq(family+'__'+mode)]
            baseline = table.loc[table.model_family.eq(family+'__'+reference_mode)].set_index(['view','target','call_id'])
            aligned = baseline.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
            if not np.array_equal(selected.m0_pred.to_numpy(),aligned.m0_pred.to_numpy()):
                raise ValueError('M0 differs across PCA modes.')
            value = aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy()
            if not np.allclose(value,selected.gain.to_numpy()-aligned.gain.to_numpy(),rtol=0,atol=1e-14):
                raise ValueError('Gain/M1-loss change identity failed.')
            block = selected[columns].copy()
            block['contrast'],block['value'] = 'vs_'+reference_mode,value
            parts.append(block)
    return pd.concat(parts,ignore_index=True)


def summarize(args,data,cols,references,ref_manifest,sha,n_draws=1000):
    plan = make_plan(data,ref_manifest)
    expected = set(plan.relative_path)
    jobs = args.output_dir/'jobs'
    if ({p.relative_to(args.output_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
            or {p.relative_to(args.output_dir).with_suffix('.csv').as_posix() for p in jobs.glob('*.json')}!=expected):
        raise ValueError('Missing/stale PCA prediction+metadata jobs;no partial summary.')
    baseline = references.copy()
    baseline['model_family'] = baseline.model_family+'__pca30'
    parts,manifest,metadata = [baseline],list(ref_manifest),[]
    for spec in plan.to_dict('records'):
        train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
        ref = reference_for(references,spec,test)
        frame,meta = load_job(args,spec,train,test,ref,cols,sha)
        parts.append(frame)
        path = args.output_dir/spec['relative_path']
        manifest.append(dict(path=str(path),sha256=b3.shared.digest(path),model_family=spec['family'],
            pca_mode=spec['pca_mode'],fold=spec['fold'],target=spec['target'],n_rows=len(frame),
            metadata_path=str(path.with_suffix('.json')),metadata_sha256=b3.shared.digest(path.with_suffix('.json')),
            selector='all rows',role='new_M1_only'))
        keys = ['n_input','n_after_imputer','acoustic_output_dim','pca_components',
            'full_fit_cumulative_variance_first30','orthogonality_max_abs_error','fit_start_utc','fit_end_utc',
            'fit_seconds','convergence_warning','n_iter_max']
        metadata.append({**{k:spec[k] for k in ['family','pca_mode','fold','target']},
            **{k:meta[k] for k in keys},'dropped_empty_columns_json':json.dumps(meta['dropped_empty_columns'])})
    raw = pd.concat(parts,ignore_index=True)
    if raw.duplicated(['model_family','target','call_id']).any():
        raise ValueError('Duplicate PCA OOF records.')
    table = b3.score_table(raw)
    values = contrast_values(table)
    intervals = b3.clip_intervals(values,n_draws,7452)
    primary = intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast')
    metrics = b3.point_metrics(table).merge(primary,on=['model_family','view','manual_label'],validate='one_to_one')
    changes = []
    for tags,group in values.loc[values.contrast.ne('gain')].groupby(b3.KEYS[:-1],sort=True):
        for label,part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            post,pre = [part.loc[part.target.eq(t),'value'].mean() for t in TARGETS]
            changes.append(dict(zip(b3.KEYS,tags+(label,)),n=part.call_id.nunique(),
                delta_post=post,delta_pre=pre,post_minus_pre=post-pre,
                m1_post_loss_improvement=post,m1_pre_loss_improvement=pre))
    paired = pd.DataFrame(changes).merge(intervals.loc[intervals.contrast.ne('gain')],on=b3.KEYS,validate='one_to_one')
    folds = []
    for tags,part in table.groupby(['model_family','view','fold'],sort=True):
        row = dict(zip(['model_family','view','fold'],tags))
        for target,side in zip(TARGETS,['post','pre']):
            selected = part.loc[part.target.eq(target)]
            for model in ['m0','m1']:
                row[model+'_'+side+'_logloss'] = selected[model+'_loss'].mean()
            row['delta_'+side] = selected.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        folds.append(row)
    diagnostics = []
    for family in FAMILIES:
        for fold in range(1,6):
            for target in TARGETS:
                part = raw.loc[raw.fold.eq(fold) & raw.target.eq(target)]
                full = part.loc[part.model_family.eq(family+'__pca_full')].set_index('call_id')
                none = part.loc[part.model_family.eq(family+'__no_pca')].set_index('call_id').loc[full.index]
                diff = none.m1_pred.to_numpy()-full.m1_pred.to_numpy()
                diagnostics.append(dict(model_family=family,fold=fold,target=target,n=len(full),
                    max_abs_probability_difference=float(np.max(np.abs(diff))),
                    mean_abs_probability_difference=float(np.mean(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),
                    no_pca_minus_full_m1_logloss=float(np.mean(b3.binary_loss(none.y_true,none.m1_pred)-b3.binary_loss(full.y_true,full.m1_pred))),
                    interpretation='Approximate rotation-invariance diagnostic' if family!='hist_gbdt' else 'Coordinate-sensitive tree comparison',
                    automatic_filtering=False))
    decorate(metrics).to_csv(args.output_dir/'comparison_metrics_and_ci.csv',index=False)
    decorate(paired).to_csv(args.output_dir/'paired_changes_and_ci.csv',index=False)
    decorate(pd.DataFrame(folds)).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir/'full_vs_none_prediction_diagnostics.csv',index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir/'transform_metadata_summary.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    b3.shared.write_json(args.output_dir/'verification.json',dict(status='passed',scope='EXPLORATORY_FIXED_B3_PCA_SCREEN',
        n_calls=len(data),n_clips=data['_clip'].nunique(),outer_oof_rows=len(raw),metric_rows=len(metrics),
        new_M1_jobs=60,M0_refits=0,reference_jobs=30,m0_predictions_identical_across_modes=True,
        convergence_warning_jobs=int(pd.DataFrame(metadata).convergence_warning.sum()),
        conditional_information_certified=False,causal_effect_assessed=False,independent_confirmation=False,
        practical_equivalence_margin=None,pca30_original_transform_spectrum='Not recorded;not reconstructed/refitted.',
        intervals='Paired clips conditional on existing OOF,unadjusted,no training refits.'))
    print('S1 summary complete;all PCA modes/labels retained. No winner or information-absence certificate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    protected = [args.project_root,args.shared_dir,args.controls_dir,args.b1_run,args.b2_run,args.b3_run]
    if any(args.output_dir.resolve()==p.resolve() or p.resolve() in args.output_dir.resolve().parents for p in protected):
        raise ValueError('Use a separate private output,not source or old results.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    from contextlib import ExitStack
    with threadpool_limits(limits=4),ExitStack() as stack:
        if args.stage=='prepare':
            stack.enter_context(fixed.zero_fit_guard())
        data,cols,controls,b1 = b3.load_design(args)
        references,manifest = fixed.load_references(args,data)
        sha = freeze(args,data,cols,controls,b1,manifest)
        if args.stage=='prepare':
            pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            b3.cohort_counts(data).to_csv(args.output_dir/'cohort_label_fold_counts.csv',index=False)
            report = dict(status='passed',model_fits=0,zero_fit_guard=True,n_calls=len(data),
                n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),
                raw_acoustic_columns=len(cols),measured_controls=len(controls),reference_jobs=30,
                new_M1_fits=60,M0_refits=0,C_selection_used=False,protocol_sha256=sha,
                pca30_original_transform_spectrum='Not recorded;not refitted.',scientific_results_created=False)
            b3.shared.write_json(args.output_dir/'prepare_check.json',report)
            print(json.dumps(report,indent=2))
            return
        if args.stage=='fit':
            print('INVOCATION_FIT_COUNTS '+json.dumps(fit_all(args,data,cols,controls,references,manifest,sha)))
        summarize(args,data,cols,references,manifest,sha)


if __name__=='__main__':
    main()
