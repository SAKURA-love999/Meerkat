"""B3 real-call baseline-capacity sensitivity; formal fits are explicitly user-run."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import extend_shared_cohort_controls as ext
import run_control_block_attribution as attribution
import run_shared_cohort_selection_control as shared
from run_baseline_capacity import read
from run_baseline_capacity_b2 import verify_hashes

REFERENCE = '20_raw_shared_full'
FIELDS = ['call_id','_clip','fold','manual_label','duration_s','duration_bin','egemaps_valid','window_ms']
KEYS = ['model_family','contrast','view','manual_label']


def binary_loss(y,p):
    p = np.clip(np.asarray(p,float),1e-15,1-1e-15)
    y = np.asarray(y,float)
    return -y*np.log(p)-(1-y)*np.log1p(-p)


def check_record(frame,test,family,fold,target,n_train,protocol_sha=None):
    if len(frame)!=len(test) or frame.call_id.duplicated().any() or set(frame.call_id)!=set(test.call_id):
        raise ValueError('B3 OOF call coverage differs.')
    aligned = frame.set_index('call_id').loc[test.call_id]
    for name,value in [('target',target),('fold',fold),('window_ms',400),('n_train',n_train)]:
        if not aligned[name].eq(value).all():
            raise ValueError('B3 OOF key differs: '+name)
    if not np.array_equal(aligned.y_true.to_numpy(int),test[target].to_numpy(int)):
        raise ValueError('B3 actual target differs.')
    for name in ['_clip','manual_label','duration_bin','egemaps_valid']:
        if not np.array_equal(aligned[name],test[name]):
            raise ValueError('B3 OOF metadata differs: '+name)
    if not np.allclose(aligned.duration_s,test.duration_s,rtol=0,atol=1e-14):
        raise ValueError('B3 duration differs.')
    if 'synthetic' in aligned and not aligned.synthetic.eq(False).all():
        raise ValueError('Synthetic data must not enter B3.')
    p = aligned[['m0_pred','m1_pred']].to_numpy(float)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid B3 probabilities.')
    if protocol_sha is not None and not (aligned.model_family.eq(family).all()
            and aligned.protocol_sha256.eq(protocol_sha).all()):
        raise ValueError('B3 learner/protocol differs.')


def load_design(args):
    data,_,cols = attribution.load(args,ext,shared)
    if len(data)!=18345 or data['_clip'].nunique()!=9756 or any(c.startswith('sim_') for c in data):
        raise ValueError('Require the original real18345-call/9756-clip cohort.')
    if not data.window_ms.eq(400).all() or data.pre_censored.any() or data.post_censored.any():
        raise ValueError('B3 requires both sides observable at400ms.')
    if set(data.fold)!={1,2,3,4,5} or data.groupby('_clip').fold.nunique().max()!=1:
        raise ValueError('Original clip folds differ.')
    controls = attribution.AMPLITUDE+attribution.OVERLAP
    b1 = json.loads((args.b1_run/'protocol.json').read_text(encoding='utf-8'))
    b2 = json.loads((args.b2_run/'protocol.json').read_text(encoding='utf-8'))
    status = json.loads((args.b2_run/'verification.json').read_text(encoding='utf-8'))
    if status['status']!='passed' or status['scope']!='B2_SYNTHETIC_CONTROLS_ONLY':
        raise ValueError('Require completed B2 controls before B3.')
    verify_hashes(b2)
    factory = str(Path(__file__).parent/'baseline_capacity_models.py')
    numeric = shared.model_numeric_columns(data,'NF_post')+controls
    if (shared.digest(Path(factory))!=b1['source_hashes'][factory]
            or shared.digest(Path(factory))!=b2['source_hashes'][factory]
            or cols!=b1['acoustic_columns'] or cols!=b2['acoustic_columns']
            or controls!=b1['measured_controls'] or controls!=b2['measured_controls']
            or b1['spline']!=models.SPLINE or b2['spline']!=models.SPLINE
            or b1['hgb_requested']!=models.HGB or b2['hgb_requested']!=models.HGB
            or b1['logistic']!=models.make_model('spline_rms',numeric,[]).named_steps['clf'].get_params()
            or b1['hgb']!=models.make_model('hist_gbdt',numeric,[]).named_steps['clf'].get_params()):
        raise ValueError('B1/B2 model/feature/control contract differs.')
    return data,cols,controls,b1


def reference_frames(args,data):
    parts,manifest = [],[]
    for fold in range(1,6):
        test = data.loc[data.fold.eq(fold)]
        for target in ['NF_post','NF_pre']:
            path = args.controls_dir/'jobs'/f'{REFERENCE}_fold{fold}_{target}.csv'
            saved = read(path)
            attribution.assert_job(saved,test,REFERENCE,target,fold)
            frame = saved.loc[saved.condition.eq(REFERENCE)].copy()
            frame['model_family'],frame['synthetic'] = 'logistic_linear',False
            check_record(frame,test,'logistic_linear',fold,target,len(data)-len(test))
            parts.append(frame)
            manifest.append(dict(path=str(path),sha256=shared.digest(path),model_family='logistic_linear',
                fold=fold,target=target,n_rows=len(frame),source_rows=len(saved),
                selector='condition='+REFERENCE,role='existing_reference'))
    return pd.concat(parts,ignore_index=True),manifest


def freeze(args,data,cols,controls,b1,refs):
    here = Path(__file__).parent
    names = ['run_baseline_capacity_b3.py','run_baseline_capacity_b3_400.ps1',
        'baseline_capacity_models.py','run_baseline_capacity.py','run_baseline_capacity_b2.py',
        'run_control_block_attribution.py','extend_shared_cohort_controls.py',
        'run_shared_cohort_selection_control.py','audit_egemaps_selection_on_librosa.py',
        'fit_meerkat_m1_acoustic_residual.py']
    paths = [args.shared_dir/n for n in ['protocol.json','cohort_with_history.pkl','clip_fold_mapping.csv']]
    paths += [args.controls_dir/n for n in ['protocol.json','cohort_controls.csv']]
    paths += [args.b1_run/'protocol.json',args.b2_run/'protocol.json',args.b2_run/'verification.json',
              args.b2_run/'comparison_metrics_and_ci.csv',args.project_root/'features/acoustic_features_capped_3000.csv']
    inputs = {str(p):shared.digest(p) for p in paths}
    inputs.update({r['path']:r['sha256'] for r in refs})
    value = dict(kind='BASELINE_CAPACITY_B3_REAL_ONLY',version=1,window_ms=400,
        n_calls=len(data),n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),n_folds=5,
        acoustic_columns=cols,measured_controls=controls,new_families=models.FAMILIES,
        spline=models.SPLINE,hgb_requested=models.HGB,logistic=b1['logistic'],hgb=b1['hgb'],
        numeric_by_target={t:shared.model_numeric_columns(data,t)+controls for t in ['NF_post','NF_pre']},
        acoustic_pca_dim=30,new_jobs=20,new_pipeline_fits=40,reference_jobs_reused=10,
        reference_condition=REFERENCE,source_hashes={str(here/n):shared.digest(here/n) for n in names},
        input_hashes=inputs,threads=4,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        bootstrap=dict(draws=1000,seed=7452,scope='Paired clips within overall/label;shared weights for all models,targets,views,contrasts;no refits.'),
        primary='All eligible rows;M0/M1 absolute logloss,DeltaPOST/PRE,D;paired changes versus linear.',
        sensitivity='Original eGeMAPS-valid evaluation rows only;all-fold training unchanged;AUC/AP secondary.',
        preprocessing='B1/B2 factory,train-fold-only imputer/scaler/spline/PCA;no feature filtering or tuning.',
        per_label='All observed labels;exploratory unadjusted intervals;finite-draw/positive-count/clip coverage reported.',
        target='Real clip-local annotated nonfocal activity,not validated response or causal effect.',
        snr='Not controlled:legacy proxy reads before/after audio.',
        interpretation='Finite-model sensitivity;no conditional-information,causal,category-sufficiency or novelty certificate.',
        practical_margin=None,reference_fit_date='Historical evidence registry;not inferred from this preparation or mtime.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen B3 protocol differs;preserve old results,use a new directory.')
    else:
        shared.write_json(path,value)
    return shared.digest(path)


def job_path(args,family,fold,target):
    return args.output_dir/'jobs'/f'{family}_fold{fold}_{target}.csv'


def fit_all(args,data,cols,controls,protocol_sha):
    counts = dict(created_jobs=0,reused_jobs=0,model_fits=0)
    for family in models.FAMILIES:
        for fold in range(1,6):
            train,test = data.loc[data.fold.ne(fold)],data.loc[data.fold.eq(fold)]
            if set(train['_clip']) & set(test['_clip']):
                raise ValueError('Training/test clip overlap.')
            for target in ['NF_post','NF_pre']:
                path = job_path(args,family,fold,target)
                if path.exists():
                    check_record(read(path),test,family,fold,target,len(train),protocol_sha)
                    counts['reused_jobs'] += 1
                    continue
                if train[target].nunique()!=2:
                    raise ValueError('Single-class training cell;no silent fallback.')
                numeric = shared.model_numeric_columns(train,target)+controls
                predictions = []
                for acoustic in [[],cols]:
                    inputs = ['manual_label']+numeric+acoustic
                    model = models.make_model(family,numeric,acoustic)
                    model.fit(train[inputs],train[target].to_numpy(int))
                    predictions.append(model.predict_proba(test[inputs])[:,1])
                    counts['model_fits'] += 1
                out = test[FIELDS].copy()
                for key,value in dict(model_family=family,target=target,n_train=len(train),
                                      synthetic=False,protocol_sha256=protocol_sha).items():
                    out[key] = value
                out['y_true'],out['m0_pred'],out['m1_pred'] = test[target].to_numpy(int),*predictions
                check_record(out,test,family,fold,target,len(train),protocol_sha)
                temporary = path.with_suffix('.tmp')
                out.to_csv(temporary,index=False)
                temporary.replace(path)
                counts['created_jobs'] += 1
            print(f'completed real family={family} fold={fold}/5 '+json.dumps(counts),flush=True)
    return counts


def score_table(records):
    full = records.copy()
    full['view'] = 'all'
    valid = full.loc[full.egemaps_valid].copy()
    valid['view'] = 'original_valid_rows'
    table = pd.concat([full,valid],ignore_index=True)
    for model in ['m0','m1']:
        table[model+'_loss'] = binary_loss(table.y_true,table[model+'_pred'])
    table['gain'] = table.m0_loss-table.m1_loss
    return table


def contrast_values(table):
    columns = ['call_id','_clip','manual_label','target','model_family','view']
    primary = table[columns].copy()
    primary['contrast'],primary['value'] = 'gain',table.gain.to_numpy()
    parts = [primary]
    base = table.loc[table.model_family.eq('logistic_linear')].set_index(['view','target','call_id'])
    for family in models.FAMILIES:
        selected = table.loc[table.model_family.eq(family)]
        aligned = base.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
        values = dict(delta_change=selected.gain.to_numpy()-aligned.gain.to_numpy(),
            m0_loss_improvement=aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy(),
            m1_loss_improvement=aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy())
        for name,value in values.items():
            block = selected[columns].copy()
            block['contrast'],block['value'] = name,value
            parts.append(block)
    return pd.concat(parts,ignore_index=True)


def clip_intervals(table,n_draws=1000,seed=7452):
    rng,rows = np.random.default_rng(seed),[]
    keys = ['model_family','contrast','view','target']
    for label,part in [('overall',table)]+list(table.groupby('manual_label',sort=True)):
        clips = pd.Index(sorted(part['_clip'].unique()))
        cells,blocks = [],[]
        for tags,group in part.groupby(keys,sort=True):
            block = group.groupby('_clip').agg(total=('value','sum'),n=('call_id','size'))
            cells.append(tags)
            blocks.append(block.reindex(clips,fill_value=0).to_numpy())
        array = np.stack(blocks,axis=1)
        samples = []
        for start in range(0,n_draws,32):
            weights = rng.multinomial(len(clips),np.full(len(clips),1/len(clips)),size=min(32,n_draws-start))
            totals = (weights @ array.reshape(len(clips),-1)).reshape(len(weights),len(cells),2)
            samples.append(totals[:,:,0]/np.where(totals[:,:,1]>0,totals[:,:,1],np.nan))
        draws = np.concatenate(samples)
        for family,contrast,view in sorted({c[:3] for c in cells}):
            ip,ir = cells.index((family,contrast,view,'NF_post')),cells.index((family,contrast,view,'NF_pre'))
            row = dict(model_family=family,contrast=contrast,view=view,manual_label=label)
            for metric,values,short in [('delta_post',draws[:,ip],'post'),('delta_pre',draws[:,ir],'pre'),
                                       ('post_minus_pre',draws[:,ip]-draws[:,ir],'direction')]:
                finite = values[np.isfinite(values)]
                lo,hi = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                row[metric+'_ci_low'],row[metric+'_ci_high'] = lo,hi
                row[short+'_finite_draws'] = len(finite)
            rows.append(row)
    return pd.DataFrame(rows)


def point_metrics(table):
    rows = []
    for (family,view),group in table.groupby(['model_family','view'],sort=True):
        for label,part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            row = dict(model_family=family,view=view,manual_label=label,synthetic=False,
                window_ms=400,n=part.call_id.nunique(),n_clips=part['_clip'].nunique(),n_folds=part.fold.nunique())
            for target,side in [('NF_post','post'),('NF_pre','pre')]:
                selected = part.loc[part.target.eq(target)]
                if len(selected)!=row['n']:
                    raise ValueError('Real PRE/POST call sets differ.')
                y = selected.y_true.to_numpy(int)
                row[side+'_positive_n'],row[side+'_positive_rate'] = int(y.sum()),y.mean()
                for model in ['m0','m1']:
                    p = selected[model+'_pred'].to_numpy()
                    row[model+'_'+side+'_logloss'] = selected[model+'_loss'].mean()
                    row[model+'_'+side+'_auc'] = roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
                    row[model+'_'+side+'_ap'] = average_precision_score(y,p) if y.sum()>0 else np.nan
                row['delta_'+side] = selected.gain.mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            row['relative_post_loss_reduction_pct'] = 100*row['delta_post']/row['m0_post_logloss']
            row['sparse_positive_warning'] = min(row['post_positive_n'],row['pre_positive_n'])<20
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(args,data,references,ref_manifest,protocol_sha,n_draws=1000):
    expected = {job_path(args,f,k,t).name for f in models.FAMILIES for k in range(1,6) for t in ['NF_post','NF_pre']}
    if {p.name for p in (args.output_dir/'jobs').glob('*.csv')}!=expected:
        raise ValueError('Missing/stale B3 jobs;no partial summary.')
    parts,manifest = [references],list(ref_manifest)
    for family in models.FAMILIES:
        for fold in range(1,6):
            test = data.loc[data.fold.eq(fold)]
            for target in ['NF_post','NF_pre']:
                path = job_path(args,family,fold,target)
                frame = read(path)
                check_record(frame,test,family,fold,target,len(data)-len(test),protocol_sha)
                parts.append(frame)
                manifest.append(dict(path=str(path),sha256=shared.digest(path),model_family=family,
                    fold=fold,target=target,n_rows=len(frame),source_rows=len(frame),selector='all rows',role='new_capacity_fit'))
    records = pd.concat(parts,ignore_index=True)
    if records.duplicated(['model_family','target','call_id']).any():
        raise ValueError('Duplicate B3 OOF calls.')
    table = score_table(records)
    values = contrast_values(table)
    intervals = clip_intervals(values,n_draws,7452)
    primary_ci = intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast')
    metrics = point_metrics(table).merge(primary_ci,on=['model_family','view','manual_label'],validate='one_to_one')
    metrics.to_csv(args.output_dir/'comparison_metrics_and_ci.csv',index=False)
    changes = []
    for tags,group in values.loc[values.contrast.ne('gain')].groupby(KEYS[:-1],sort=True):
        for label,part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            row = dict(zip(KEYS,tags+(label,)),n=part.call_id.nunique())
            row['delta_post'] = part.loc[part.target.eq('NF_post'),'value'].mean()
            row['delta_pre'] = part.loc[part.target.eq('NF_pre'),'value'].mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            changes.append(row)
    pd.DataFrame(changes).merge(intervals.loc[intervals.contrast.ne('gain')],on=KEYS,
        validate='one_to_one').to_csv(args.output_dir/'paired_changes_vs_linear.csv',index=False)
    folds = []
    for tags,part in table.groupby(['model_family','view','fold'],sort=True):
        row = dict(zip(['model_family','view','fold'],tags))
        for target,side in [('NF_post','post'),('NF_pre','pre')]:
            selected = part.loc[part.target.eq(target)]
            for model in ['m0','m1']:
                row[model+'_'+side+'_logloss'] = selected[model+'_loss'].mean()
            row['delta_'+side] = selected.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        folds.append(row)
    pd.DataFrame(folds).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    shared.write_json(args.output_dir/'verification.json',dict(status='passed',scope='B3_REAL_PREDICTION_SENSITIVITY',
        new_jobs=20,new_model_fits_required=40,reference_jobs=10,n_calls=len(data),n_clips=data['_clip'].nunique(),
        n_oof_rows=len(records),metric_rows=len(metrics),fixed_clip_folds=True,real_outcomes_used=True,
        causal_effect_assessed=False,conditional_information_certified=False,practical_equivalence_margin=None,
        intervals='Exploratory overall/per-label paired clips conditional on OOF;unadjusted,no refits.'))
    print('B3 summary complete. Read absolute losses and paired changes;no causal/category certificate.')


def cohort_counts(data):
    parts = []
    for view,rows in [('all',data),('original_valid_rows',data.loc[data.egemaps_valid])]:
        counts = rows.groupby(['fold','manual_label'],sort=True).agg(n=('call_id','size'),n_clips=('_clip','nunique'),
            duration_min=('duration_s','min'),duration_median=('duration_s','median'),duration_max=('duration_s','max'),
            post_positive_n=('NF_post','sum'),pre_positive_n=('NF_pre','sum')).reset_index()
        counts['view'] = view
        counts['post_positive_rate'],counts['pre_positive_rate'] = counts.post_positive_n/counts.n,counts.pre_positive_n/counts.n
        parts.append(counts)
    return pd.concat(parts,ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    if args.output_dir.resolve() in {p.resolve() for p in
            [args.project_root,args.shared_dir,args.controls_dir,args.b1_run,args.b2_run]}:
        raise ValueError('Never overwrite source results.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    with threadpool_limits(limits=4):
        data,cols,controls,b1 = load_design(args)
        references,manifest = reference_frames(args,data)
        protocol_sha = freeze(args,data,cols,controls,b1,manifest)
        (args.output_dir/'jobs').mkdir(exist_ok=True)
        if args.stage=='prepare':
            pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            cohort_counts(data).to_csv(args.output_dir/'cohort_label_fold_counts.csv',index=False)
            shared.write_json(args.output_dir/'prepare_check.json',dict(status='passed',model_fits=0,
                n_calls=len(data),n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),
                acoustic_columns=len(cols),reference_jobs=10,new_jobs=20,planned_new_model_fits=40,B3_only=True))
            print('B3 prepared;40 formal CPU fits are user-run. No new fitted metrics.')
            return
        if args.stage=='fit':
            print('INVOCATION_FIT_COUNTS '+json.dumps(fit_all(args,data,cols,controls,protocol_sha)))
        summarize(args,data,references,manifest,protocol_sha)


if __name__=='__main__':
    main()
