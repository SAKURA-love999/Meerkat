"""S2: grouped nested C selection; Prepare is zero-fit, formal fits are user-run."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import run_baseline_capacity_b3 as b3
from run_baseline_capacity import read

shared = b3.shared
FAMILIES = ['logistic_linear','spline_rms']
TARGETS = ['NF_post','NF_pre']
CS = [1.,.1,10.]
CELL = ['family','outer_fold','target','model']


def ids_hash(frame):
    return hashlib.sha256(json.dumps(sorted(frame.call_id.tolist()),ensure_ascii=True,
        separators=(',',':')).encode('ascii')).hexdigest()


@contextmanager
def zero_fit_guard():
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, SplineTransformer
    def blocked(*args,**kwargs):
        raise RuntimeError('Prepare zero-fit guard blocked an attempted fit.')
    with ExitStack() as stack:
        for cls,method in [(Pipeline,'fit'),(ColumnTransformer,'fit_transform'),
                (LogisticRegression,'fit'),(HistGradientBoostingClassifier,'fit'),
                (PCA,'fit'),(PCA,'fit_transform'),(SimpleImputer,'fit'),
                (StandardScaler,'fit'),(SplineTransformer,'fit')]:
            stack.enter_context(patch.object(cls,method,blocked))
        yield


def make_model(family,numeric,acoustic,c):
    if family not in FAMILIES or c not in CS or models.RMS not in numeric:
        raise ValueError('Undeclared logistic family/C or missing RMS control.')
    forbidden = [x for x in numeric+acoustic if x.startswith('sim_') or x in
        ['call_id','_clip','fold','NF_post','NF_pre','egemaps_valid']]
    if forbidden or len(set(numeric+acoustic))!=len(numeric+acoustic):
        raise ValueError('Forbidden/duplicated predictive inputs: '+str(forbidden))
    if family=='logistic_linear':
        model = models.base.make_pipeline(numeric,acoustic,bool(acoustic),30 if acoustic else None,0)
    else:
        model = models.make_model(family,numeric,acoustic)
    return model.set_params(clf__C=c)


def make_inner_maps(data):
    if data.call_id.duplicated().any() or data.groupby('_clip').fold.nunique().max()!=1:
        raise ValueError('Call IDs must be unique and outer folds clip-disjoint.')
    rows = []
    for outer in range(1,6):
        train = data.loc[data.fold.ne(outer)]
        for inner,(_,validation) in enumerate(GroupKFold(3).split(train,groups=train['_clip']),1):
            for clip in sorted(train.iloc[validation]['_clip'].unique()):
                rows.append(dict(outer_fold=outer,_clip=clip,inner_fold=inner))
    return pd.DataFrame(rows).sort_values(['outer_fold','_clip']).reset_index(drop=True)


def split_inner(data,maps,outer,inner):
    train = data.loc[data.fold.ne(outer)]
    mapping = maps.loc[maps.outer_fold.eq(outer)].set_index('_clip').inner_fold
    if mapping.index.duplicated().any() or set(mapping.index)!=set(train['_clip']):
        raise ValueError('Inner map includes missing/outer-test clips.')
    selected = train['_clip'].map(mapping).eq(inner)
    fit,val = train.loc[~selected],train.loc[selected]
    if len(fit)==0 or len(val)==0 or set(fit['_clip']) & set(val['_clip']):
        raise ValueError('Invalid inner training/validation partition.')
    return fit,val


def job_spec(train,test,stage,family,outer,target,model,c,inner=0):
    tag = {1.:'1',.1:'0p1',10.:'10'}[c]
    name = f'{family}_outer{outer}_{target}_{model}_C{tag}_inner{inner}.csv'
    return dict(stage=stage,family=family,outer_fold=outer,target=target,model=model,C=c,
        inner_fold=inner,n_train=len(train),n_test=len(test),train_ids_sha256=ids_hash(train),
        validation_ids_sha256=ids_hash(test),relative_path='jobs/'+stage+'/'+name)


def make_fit_plan(data,maps):
    rows = []
    for outer in range(1,6):
        splits = {k:split_inner(data,maps,outer,k) for k in range(1,4)}
        for family in FAMILIES:
            for target in TARGETS:
                for model in ['m0','m1']:
                    for c in CS:
                        for inner,(train,val) in splits.items():
                            rows.append(job_spec(train,val,'inner',family,outer,target,model,c,inner))
    return pd.DataFrame(rows)


def check_job(frame,spec,test,protocol_sha):
    if len(frame)!=len(test) or frame.call_id.duplicated().any() or set(frame.call_id)!=set(test.call_id):
        raise ValueError('Nested prediction call coverage differs.')
    aligned = frame.set_index('call_id').loc[test.call_id]
    for name,value in spec.items():
        if not aligned[name].eq(value).all():
            raise ValueError('Nested job contract differs: '+name)
    for name in b3.FIELDS[1:]:
        if name=='duration_s':
            valid = np.allclose(aligned[name],test[name],rtol=0,atol=1e-14)
        else:
            valid = np.array_equal(aligned[name],test[name])
        if not valid:
            raise ValueError('Nested metadata differs: '+name)
    if (not aligned.protocol_sha256.eq(protocol_sha).all()
            or not np.array_equal(aligned.y_true.to_numpy(int),test[spec['target']].to_numpy(int))):
        raise ValueError('Nested protocol/actual outcome differs.')
    p = aligned.pred.to_numpy(float)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid nested probabilities.')


def execute_job(args,spec,train,test,cols,controls,protocol_sha):
    path = args.output_dir/spec['relative_path']
    if path.exists():
        check_job(read(path),spec,test,protocol_sha)
        return False
    if train[spec['target']].nunique()!=2:
        raise ValueError('Single-class training cell; no silent fallback.')
    if set(train['_clip']) & set(test['_clip']):
        raise ValueError('Fit/test clips overlap.')
    numeric = shared.model_numeric_columns(train,spec['target'])+controls
    acoustic = cols if spec['model']=='m1' else []
    inputs = ['manual_label']+numeric+acoustic
    model = make_model(spec['family'],numeric,acoustic,spec['C'])
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(train[inputs],train[spec['target']].to_numpy(int))
        probability = model.predict_proba(test[inputs])[:,1]
    out = test[b3.FIELDS].copy()
    for key,value in spec.items():
        out[key] = value
    out['protocol_sha256'],out['y_true'],out['pred'] = protocol_sha,test[spec['target']].to_numpy(int),probability
    out['fit_start_utc'],out['fit_end_utc'] = start,datetime.now(timezone.utc).isoformat()
    out['fit_seconds'] = time.monotonic()-clock
    out['convergence_warning'] = any(issubclass(w.category,ConvergenceWarning) for w in caught)
    out['warning_messages'] = json.dumps([str(w.message) for w in caught],ensure_ascii=True)
    clf = model.named_steps.get('clf')
    out['n_iter_max'] = int(np.max(clf.n_iter_)) if hasattr(clf,'n_iter_') else -1
    check_job(out,spec,test,protocol_sha)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    out.to_csv(temporary,index=False)
    temporary.replace(path)
    return True


def choose_candidates(scores):
    candidates,chosen = [],[]
    for tags,part in scores.groupby(CELL,sort=True):
        if len(part)!=9 or part.duplicated(['C','inner_fold']).any() or set(part.C)!=set(CS):
            raise ValueError('Require all nine inner scores per model/target, not a partial grid.')
        values = []
        for c in CS:
            block = part.loc[part.C.eq(c)]
            if set(block.inner_fold)!={1,2,3} or (block.n<=0).any():
                raise ValueError('Incomplete inner folds.')
            score = float(block.loss_sum.sum()/block.n.sum())
            if not np.isfinite(score):
                raise ValueError('Nonfinite inner selection score.')
            row = dict(zip(CELL,tags),C=c,pooled_logloss=score,n=int(block.n.sum()))
            candidates.append(row)
            values.append(row)
        best = min(v['pooled_logloss'] for v in values)
        ties = [v for v in values if v['pooled_logloss']==best]
        chosen.append(dict(zip(CELL,tags),selected_C=ties[0]['C'],pooled_logloss=best,
            tied_C_json=json.dumps([v['C'] for v in ties]),score_precision='binary64; exact equality; no tolerance'))
    return pd.DataFrame(candidates),pd.DataFrame(chosen)


def inner_scores(args,data,maps,protocol_sha):
    plan = make_fit_plan(data,maps)
    expected = set(plan.relative_path)
    observed = {p.relative_to(args.output_dir).as_posix() for p in (args.output_dir/'jobs/inner').glob('*.csv')}
    if observed!=expected:
        raise ValueError('Missing/stale inner jobs; refuse partial selection.')
    rows = []
    for spec in plan.to_dict('records'):
        _,val = split_inner(data,maps,spec['outer_fold'],spec['inner_fold'])
        frame = read(args.output_dir/spec['relative_path'])
        check_job(frame,spec,val,protocol_sha)
        rows.append({**{k:spec[k] for k in CELL+['C','inner_fold']},'n':len(frame),
            'loss_sum':float(b3.binary_loss(frame.y_true,frame.pred).sum()),
            'convergence_warning':bool(frame.convergence_warning.any())})
    scores = pd.DataFrame(rows)
    candidates,chosen = choose_candidates(scores)
    if len(chosen)!=40:
        raise ValueError('Expected 40 independently selected settings.')
    return scores,candidates,chosen


def outer_plan(data,chosen):
    rows = []
    for selected in chosen.to_dict('records'):
        outer = selected['outer_fold']
        train,test = data.loc[data.fold.ne(outer)],data.loc[data.fold.eq(outer)]
        rows.append(job_spec(train,test,'outer',selected['family'],outer,selected['target'],
            selected['model'],selected['selected_C']))
    return pd.DataFrame(rows)


def fit_all(args,data,cols,controls,maps,protocol_sha):
    counts = dict(created_inner_jobs=0,reused_inner_jobs=0,created_outer_jobs=0,reused_outer_jobs=0,model_fits=0)
    plan = make_fit_plan(data,maps)
    for tags,part in plan.groupby(['outer_fold','family','target'],sort=True):
        for spec in part.to_dict('records'):
            train,val = split_inner(data,maps,spec['outer_fold'],spec['inner_fold'])
            created = execute_job(args,spec,train,val,cols,controls,protocol_sha)
            counts['created_inner_jobs' if created else 'reused_inner_jobs'] += 1
            counts['model_fits'] += int(created)
        print('completed inner outer/family/target='+str(tags)+' '+json.dumps(counts),flush=True)
    scores,candidates,chosen = inner_scores(args,data,maps,protocol_sha)
    for name,frame in [('inner_fold_scores.csv',scores),('candidate_scores.csv',candidates),('selected_C.csv',chosen)]:
        frame.to_csv(args.output_dir/name,index=False)
    plan = outer_plan(data,chosen)
    plan.to_csv(args.output_dir/'selected_outer_fit_plan.csv',index=False)
    for tags,part in plan.groupby(['outer_fold','family','target'],sort=True):
        for spec in part.to_dict('records'):
            train,test = data.loc[data.fold.ne(spec['outer_fold'])],data.loc[data.fold.eq(spec['outer_fold'])]
            created = execute_job(args,spec,train,test,cols,controls,protocol_sha)
            counts['created_outer_jobs' if created else 'reused_outer_jobs'] += 1
            counts['model_fits'] += int(created)
        print('completed outer outer/family/target='+str(tags)+' '+json.dumps(counts),flush=True)
    return counts


def load_references(args,data):
    protocol = json.loads((args.b3_run/'protocol.json').read_text(encoding='utf-8'))
    b3.verify_hashes(protocol)
    status = json.loads((args.b3_run/'verification.json').read_text(encoding='utf-8'))
    if status['status']!='passed' or status['n_calls']!=18345:
        raise ValueError('Require completed original B3.')
    manifest = read(args.b3_run/'all_oof_manifest.csv')
    if len(manifest)!=30 or manifest.duplicated(['model_family','fold','target']).any():
        raise ValueError('B3 reference coverage differs.')
    parts = []
    for row in manifest.to_dict('records'):
        path = Path(row['path'])
        if shared.digest(path)!=row['sha256']:
            raise ValueError('Historical B3 OOF changed: '+str(path))
        frame = read(path)
        family,fold,target = row['model_family'],row['fold'],row['target']
        if family=='logistic_linear':
            frame = frame.loc[frame.condition.eq(b3.REFERENCE)].copy()
        elif family not in models.FAMILIES:
            raise ValueError('Unknown B3 reference family.')
        frame['model_family'],frame['synthetic'] = family,False
        test = data.loc[data.fold.eq(fold)]
        b3.check_record(frame,test,family,fold,target,len(data)-len(test),
            shared.digest(args.b3_run/'protocol.json') if family!='logistic_linear' else None)
        parts.append(frame)
    return pd.concat(parts,ignore_index=True),manifest.to_dict('records')


def freeze_csv(path,frame):
    if path.exists():
        saved = read(path)
        try:
            pd.testing.assert_frame_equal(saved,frame,check_dtype=False,check_exact=True)
        except AssertionError as error:
            raise ValueError('Frozen design artifact differs: '+str(path)) from error
    else:
        frame.to_csv(path,index=False)


def freeze(args,data,cols,controls,maps,manifest,b1):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths = ['inner_clip_folds.csv','fit_plan.csv']
    frames = [maps,make_fit_plan(data,maps)]
    for name,frame in zip(paths,frames):
        if args.stage!='prepare' and not (args.output_dir/name).exists():
            raise ValueError('Run Prepare first: missing frozen design.')
        freeze_csv(args.output_dir/name,frame)
    previous = json.loads((args.b3_run/'protocol.json').read_text(encoding='utf-8'))
    source_paths = list(previous['source_hashes'])+[str(Path(__file__).parent/n) for n in
        ['run_nested_logistic_regularization.py','test_nested_logistic_regularization.py',
         'run_nested_logistic_regularization400.ps1']]
    inputs = dict(previous['input_hashes'])
    for path in [args.b3_run/n for n in ['protocol.json','verification.json','all_oof_manifest.csv']]+[
            args.output_dir/n for n in paths]:
        inputs[str(path)] = shared.digest(path)
    inputs.update({r['path']:r['sha256'] for r in manifest})
    value = dict(kind='NESTED_LOGISTIC_C400_EXPLORATORY',version=1,n_calls=len(data),
        n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),window_ms=400,
        families=FAMILIES,C_candidates_in_exact_tie_order=CS,inner_folds=3,outer_folds=5,
        targets=TARGETS,models=['m0','m1'],acoustic_columns=cols,acoustic_pca_dim=30,
        measured_controls=controls,numeric_by_target={t:shared.model_numeric_columns(data,t)+controls for t in TARGETS},
        logistic_fixed=b1['logistic'],spline=models.SPLINE,threads=4,
        selection='Each family/fold/target/M0-or-M1 independently minimizes pooled inner OOF natural-log logloss; exact binary64 ties C1,0.1,10.',
        preprocessing='Fresh original pipeline; inner training only during selection; outer training only during final refit. No global imputation/PCA.',
        budget=dict(inner_fits=360,outer_refits=40,total_fits=400,old_outer_fit_reuse=False),
        reference='Original B3 three fixed families; never used for candidate selection; HGB not tuned.',
        evaluation='All eligible calls primary; same OOF original-valid rows sensitivity only; all labels; no selective reporting.',
        bootstrap=dict(draws=1000,seed=7452,scope='Paired clip resampling conditional on OOF; no training/tuning refits; unadjusted exploratory.'),
        interpretation='Finite-model sensitivity,not conditional-information/response/causal/category-sufficiency proof; no equivalence margin.',
        snr='Not controlled: legacy SNR accesses past-and-future audio.',
        exploratory=True,independent_confirmation=False,practical_margin=None,
        source_hashes={p:shared.digest(Path(p)) for p in sorted(set(source_paths))},input_hashes=inputs,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen nested protocol changed; preserve old results, use a new output directory.')
    else:
        shared.write_json(path,value)
    return shared.digest(path)


def tuned_records(args,data,maps,protocol_sha):
    scores,candidates,chosen = inner_scores(args,data,maps,protocol_sha)
    plan = outer_plan(data,chosen)
    observed = {p.relative_to(args.output_dir).as_posix() for p in (args.output_dir/'jobs/outer').glob('*.csv')}
    if observed!=set(plan.relative_path):
        raise ValueError('Missing/stale outer jobs; no partial summary.')
    parts,manifest,diagnostics = [],[],[]
    for spec in plan.to_dict('records'):
        path = args.output_dir/spec['relative_path']
        frame = read(path)
        test = data.loc[data.fold.eq(spec['outer_fold'])]
        check_job(frame,spec,test,protocol_sha)
        parts.append(frame)
    long = pd.concat(parts,ignore_index=True)
    records = []
    for (family,outer,target),part in long.groupby(['family','outer_fold','target'],sort=True):
        test = data.loc[data.fold.eq(outer)]
        frame = test[b3.FIELDS].copy()
        for model in ['m0','m1']:
            selected = part.loc[part.model.eq(model)].set_index('call_id').loc[test.call_id]
            frame[model+'_pred'] = selected.pred.to_numpy()
        frame['model_family'],frame['target'],frame['y_true'] = family+'_tuned',target,test[target].to_numpy(int)
        frame['n_train'],frame['synthetic'],frame['protocol_sha256'] = len(data)-len(test),False,protocol_sha
        b3.check_record(frame,test,family+'_tuned',outer,target,len(data)-len(test),protocol_sha)
        records.append(frame)
    for stage,paths in [('inner',make_fit_plan(data,maps)),('outer',plan)]:
        for spec in paths.to_dict('records'):
            path = args.output_dir/spec['relative_path']
            first = read(path).iloc[0]
            diagnostics.append({**spec,**{k:first[k] for k in ['fit_start_utc','fit_end_utc','fit_seconds',
                'convergence_warning','warning_messages','n_iter_max']}})
            manifest.append(dict(path=str(path),sha256=shared.digest(path),stage=stage,**{k:spec[k] for k in CELL+['C','inner_fold']}))
    for name,frame in [('inner_fold_scores.csv',scores),('candidate_scores.csv',candidates),
            ('selected_C.csv',chosen),('selected_outer_fit_plan.csv',plan)]:
        frame.to_csv(args.output_dir/name,index=False)
    return pd.concat(records,ignore_index=True),manifest,pd.DataFrame(diagnostics)


def contrast_values(table):
    columns = ['call_id','_clip','manual_label','target','model_family','view']
    primary = table[columns].copy()
    primary['contrast'],primary['value'] = 'gain',table.gain.to_numpy()
    parts = [primary]
    for family in FAMILIES:
        base = table.loc[table.model_family.eq(family)].set_index(['view','target','call_id'])
        selected = table.loc[table.model_family.eq(family+'_tuned')]
        aligned = base.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
        values = dict(delta_change=selected.gain.to_numpy()-aligned.gain.to_numpy(),
            m0_loss_improvement=aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy(),
            m1_loss_improvement=aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy())
        for name,value in values.items():
            block = selected[columns].copy()
            block['contrast'],block['value'] = name,value
            parts.append(block)
    return pd.concat(parts,ignore_index=True)


def summarize(args,data,maps,references,ref_manifest,protocol_sha,n_draws=1000):
    records,manifest,diagnostics = tuned_records(args,data,maps,protocol_sha)
    records = pd.concat([references,records],ignore_index=True)
    if records.duplicated(['model_family','target','call_id']).any():
        raise ValueError('Duplicate OOF rows.')
    table = b3.score_table(records)
    values = contrast_values(table)
    intervals = b3.clip_intervals(values,n_draws,7452)
    metrics = b3.point_metrics(table).merge(intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast'),
        on=['model_family','view','manual_label'],validate='one_to_one')
    changes = []
    for tags,group in values.loc[values.contrast.ne('gain')].groupby(b3.KEYS[:-1],sort=True):
        for label,part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            row = dict(zip(b3.KEYS,tags+(label,)),n=part.call_id.nunique())
            row['delta_post'] = part.loc[part.target.eq('NF_post'),'value'].mean()
            row['delta_pre'] = part.loc[part.target.eq('NF_pre'),'value'].mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            changes.append(row)
    paired = pd.DataFrame(changes).merge(intervals.loc[intervals.contrast.ne('gain')],on=b3.KEYS,validate='one_to_one')
    folds = []
    for tags,part in table.groupby(['model_family','view','fold'],sort=True):
        row = dict(zip(['model_family','view','fold'],tags))
        for target,side in zip(TARGETS,['post','pre']):
            block = part.loc[part.target.eq(target)]
            for model in ['m0','m1']:
                row[model+'_'+side+'_logloss'] = block[model+'_loss'].mean()
            row['delta_'+side] = block.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        folds.append(row)
    metrics.to_csv(args.output_dir/'comparison_metrics_and_ci.csv',index=False)
    paired.to_csv(args.output_dir/'paired_changes_vs_fixed.csv',index=False)
    pd.DataFrame(folds).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    diagnostics.to_csv(args.output_dir/'fit_diagnostics.csv',index=False)
    pd.DataFrame(ref_manifest+manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    shared.write_json(args.output_dir/'verification.json',dict(status='passed',scope='EXPLORATORY_NESTED_C_SENSITIVITY',
        n_calls=len(data),n_clips=data['_clip'].nunique(),n_oof_rows=len(records),metric_rows=len(metrics),
        inner_jobs=360,outer_jobs=40,convergence_warning_jobs=int(diagnostics.convergence_warning.sum()),
        fixed_clip_folds=True,conditional_information_certified=False,causal_effect_assessed=False,
        independent_confirmation=False,practical_equivalence_margin=None,
        intervals='Unadjusted paired clip resampling conditional on fitted OOF; not nested-training uncertainty.'))
    print('Nested summary complete; all labels/settings retained. No causal or absence certificate.')


def prepare(args,data,cols,controls,maps,manifest,b1):
    protocol_sha = freeze(args,data,cols,controls,maps,manifest,b1)
    pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
    b3.cohort_counts(data).to_csv(args.output_dir/'cohort_label_fold_counts.csv',index=False)
    rows = []
    for outer in range(1,6):
        for inner in range(1,4):
            train,val = split_inner(data,maps,outer,inner)
            for role,part in [('train',train),('validation',val)]:
                for label,block in [('overall',part)]+list(part.groupby('manual_label',sort=True)):
                    rows.append(dict(outer_fold=outer,inner_fold=inner,role=role,manual_label=label,
                        n=len(block),n_clips=block['_clip'].nunique(),post_positive_n=int(block.NF_post.sum()),
                        pre_positive_n=int(block.NF_pre.sum()),n_valid=int(block.egemaps_valid.sum())))
    pd.DataFrame(rows).to_csv(args.output_dir/'inner_fold_counts.csv',index=False)
    report = dict(status='passed',model_fits=0,zero_fit_guard=True,n_calls=len(data),
        n_clips=data['_clip'].nunique(),n_valid=int(data.egemaps_valid.sum()),acoustic_columns=len(cols),
        measured_controls=len(controls),original_outer_folds=5,shared_inner_folds=3,inner_map_rows=len(maps),
        inner_jobs=360,outer_jobs=40,planned_model_fits=400,reference_jobs=len(manifest),
        protocol_sha256=protocol_sha,scientific_results_created=False)
    shared.write_json(args.output_dir/'prepare_check.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    protected = [args.project_root,args.shared_dir,args.controls_dir,args.b1_run,args.b2_run,args.b3_run]
    if any(args.output_dir.resolve()==p.resolve() or p.resolve() in args.output_dir.resolve().parents for p in protected):
        raise ValueError('Keep nested outputs outside source/old result directories.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    with threadpool_limits(limits=4),ExitStack() as stack:
        if args.stage=='prepare':
            stack.enter_context(zero_fit_guard())
        data,cols,controls,b1 = b3.load_design(args)
        references,manifest = load_references(args,data)
        maps = make_inner_maps(data)
        if args.stage=='prepare':
            prepare(args,data,cols,controls,maps,manifest,b1)
            return
        protocol_sha = freeze(args,data,cols,controls,maps,manifest,b1)
        if args.stage=='fit':
            counts = fit_all(args,data,cols,controls,maps,protocol_sha)
            print('INVOCATION_FIT_COUNTS '+json.dumps(counts),flush=True)
        summarize(args,data,maps,references,manifest,protocol_sha)


if __name__=='__main__':
    main()
