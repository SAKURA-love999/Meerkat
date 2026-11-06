"""B1 only: paired capacity sensitivity on the frozen asymmetric null."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import run_asymmetric_null_calibration as asym
import run_semisynthetic_calibration as core

SOURCE_FAMILIES = {'standard':'logistic_linear','oracle_parent_reference':'known_h_reference'}
FIELDS = ['call_id','_clip','fold','manual_label','duration_s','egemaps_valid']


def read(path):
    return pd.read_csv(path,float_precision='round_trip',dtype={'call_id':str,'_clip':str})


def check_record(frame,test,family,fold,side,n_train,protocol_sha=None):
    arm = 'oracle_parent_reference' if family=='known_h_reference' else 'standard'
    core.check_job(frame,test,431,asym.dgp.SCENARIO,arm,fold,side,n_train)
    aligned = frame.set_index('call_id').loc[test.call_id]
    for name in ['_clip','manual_label','egemaps_valid']:
        if not np.array_equal(aligned[name],test[name]):
            raise ValueError('OOF metadata differs: '+name)
    if not np.allclose(aligned.duration_s,test.duration_s,rtol=0,atol=1e-14):
        raise ValueError('OOF duration differs.')
    if protocol_sha is not None and not (frame.model_family.eq(family).all()
            and frame.protocol_sha256.eq(protocol_sha).all()):
        raise ValueError('OOF learner/protocol differs.')


def load_design(args):
    original = json.loads((args.reference_run/'protocol.json').read_text(encoding='utf-8'))
    status = json.loads((args.reference_run/'verification.json').read_text(encoding='utf-8'))
    if status['status']!='passed' or status['expected_jobs']!=20:
        raise ValueError('Require completed asymmetric-null reference.')
    for section in ['source_hashes','input_hashes']:
        for path,expected in original[section].items():
            if core.digest(path)!=expected:
                raise ValueError('Frozen reference changed: '+path)
    data,cols,controls,shared = core.load_real_design(args)
    base_protocol = core.protocol(args,data,cols,controls)
    for section in ['source_hashes','input_hashes']:
        for path,expected in base_protocol[section].items():
            if original[section].get(path)!=expected:
                raise ValueError('Reference/helper/design protocol mismatch: '+path)
    entries = json.loads((args.reference_run/'synthetic_design_manifest.json').read_text(encoding='utf-8'))
    if len(entries)!=1 or core.digest(entries[0]['path'])!=entries[0]['sha256']:
        raise ValueError('Synthetic design manifest differs.')
    with np.load(entries[0]['path'],allow_pickle=False) as saved:
        packet = {key:saved[key] for key in saved.files}
    if not np.array_equal(packet['call_ids'],data.call_id.to_numpy()):
        raise ValueError('Reference call order differs.')
    current = core.materialize(data,packet)
    members = read(args.reference_run/'fixed_design_membership.csv').set_index('call_id')
    selected = current.set_index('call_id').loc[members.index,FIELDS[1:]]
    pd.testing.assert_frame_equal(selected,members[FIELDS[1:]],check_dtype=False)
    return current,cols,controls,shared,base_protocol


def reference_frames(args,current):
    manifest = pd.read_csv(args.reference_run/'synthetic_oof_manifest.csv')
    expected = {(arm,fold,side) for arm in SOURCE_FAMILIES for fold in range(1,6) for side in ['post','pre']}
    if len(manifest)!=20 or set(zip(manifest.analysis_arm,manifest.fold,manifest.target))!=expected:
        raise ValueError('Reference jobs differ.')
    frames,records = [],[]
    for item in manifest.itertuples(index=False):
        if core.digest(item.path)!=item.sha256:
            raise ValueError('Reference OOF hash differs.')
        family = SOURCE_FAMILIES[item.analysis_arm]
        frame = read(item.path)
        test = current.loc[current.fold.eq(item.fold)]
        check_record(frame,test,family,item.fold,item.target,len(current)-len(test))
        frame['model_family'] = family
        frames.append(frame)
        records.append(dict(path=item.path,sha256=item.sha256,model_family=family,
            fold=item.fold,target=item.target,n_rows=len(frame),role='existing_reference'))
    return pd.concat(frames,ignore_index=True),records


def freeze(args,data,cols,controls,shared,base_protocol,refs):
    here = Path(__file__).parent
    sources = dict(base_protocol['source_hashes'])
    original = json.loads((args.reference_run/'protocol.json').read_text(encoding='utf-8'))
    sources.update(original['source_hashes'])
    for name in ['baseline_capacity_models.py','run_baseline_capacity.py','run_baseline_capacity400.ps1']:
        path = here/name
        sources[str(path)] = core.digest(path)
    inputs = dict(base_protocol['input_hashes'])
    for name in ['protocol.json','verification.json','synthetic_design_manifest.json',
                 'synthetic_oof_manifest.csv','fixed_design_membership.csv']:
        path = args.reference_run/name
        inputs[str(path)] = core.digest(path)
    for item in refs:
        inputs[item['path']] = item['sha256']
    entry = json.loads((args.reference_run/'synthetic_design_manifest.json').read_text(encoding='utf-8'))[0]
    inputs[entry['path']] = entry['sha256']
    numeric = shared.model_numeric_columns(data,'NF_post')+controls
    value = dict(kind='BASELINE_CAPACITY_B1_ONLY',version=1,scenario=asym.dgp.SCENARIO,
        seed=431,n_calls=len(data),n_clips=data['_clip'].nunique(),n_folds=5,
        acoustic_columns=cols,measured_controls=controls,new_families=models.FAMILIES,
        spline=models.SPLINE,spline_replaces_linear_rms=True,spline_output_scaling='Training-fold StandardScaler',
        hgb=models.make_model('hist_gbdt',numeric,[]).named_steps['clf'].get_params(),
        hgb_requested=models.HGB,logistic=models.make_model('spline_rms',numeric,[]).named_steps['clf'].get_params(),
        acoustic_pca_dim=30,new_pipeline_fits=40,reference_jobs_reused=20,
        bootstrap=dict(draws=1000,seed=7452,scope='Paired clips conditional on fixed OOF,no refit uncertainty'),
        source_hashes=sources,input_hashes=inputs,threads=4,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        interpretation='Known-null sensitivity only;no real stable-information certificate or calibrated error control.',
        tuning='None;no event-random early stopping or outer-test selection.',
        deferred=['B2 signal/state controls','B3 real calls','Broader Z curves/interactions',
                  'Weak signals','Conditional-information comparator','Modern table model'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path = args.output_dir/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        raise ValueError('Frozen protocol/code/environment differs;use a new directory.')
    core.write_json(path,value)
    return core.digest(path)


def job_path(args,family,fold,side):
    return args.output_dir/'jobs'/f'{family}_fold{fold}_{side}.csv'


def fit_all(args,current,cols,controls,shared,protocol_sha):
    for family in models.FAMILIES:
        for fold in range(1,6):
            train,test = current.loc[current.fold.ne(fold)],current.loc[current.fold.eq(fold)]
            if set(train['_clip']) & set(test['_clip']):
                raise ValueError('Train/test clips overlap.')
            for side in ['post','pre']:
                path = job_path(args,family,fold,side)
                if path.exists():
                    check_record(read(path),test,family,fold,side,len(train),protocol_sha)
                    continue
                target = 'sim_y_'+side
                if train[target].nunique()!=2:
                    raise ValueError('Unfittable training target,no silent fallback.')
                numeric = shared.model_numeric_columns(train,'NF_'+side)+controls
                predictions = []
                for acoustic in [[],cols]:
                    inputs = ['manual_label']+numeric+acoustic
                    model = models.make_model(family,numeric,acoustic)
                    model.fit(train[inputs],train[target].to_numpy(int))
                    predictions.append(model.predict_proba(test[inputs])[:,1])
                record = test[FIELDS].copy()
                for key,value in dict(model_family=family,analysis_arm='standard',replicate_seed=431,
                    scenario=asym.dgp.SCENARIO,fold=fold,target='sim_'+side,synthetic=True,
                    n_train=len(train),protocol_sha256=protocol_sha).items():
                    record[key] = value
                record['y_sim'],record['generator_probability'] = test[target],test['sim_p_'+side]
                record['m0_pred'],record['m1_pred'] = predictions
                check_record(record,test,family,fold,side,len(train),protocol_sha)
                temporary = path.with_suffix('.tmp')
                record.to_csv(temporary,index=False)
                temporary.replace(path)
            print(f'completed family={family} fold={fold}/5',flush=True)


def statistics_table(records):
    full = records.copy()
    full['view'] = 'all'
    valid = full.loc[full.egemaps_valid].copy()
    valid['view'] = 'original_valid_rows'
    table = pd.concat([full,valid],ignore_index=True)
    # Reuse the existing grouped-statistics API; this key is grouping,not conditioning.
    table['analysis_arm'] = table.model_family
    for model in ['m0','m1']:
        table[model+'_loss'] = core.binary_loss(table.y_sim,table[model+'_pred'])
    table['gain'] = table.m0_loss-table.m1_loss
    return table


def paired_changes(table,n_draws=1000):
    base = table.loc[table.model_family.eq('logistic_linear')].set_index(['view','target','call_id'])
    blocks,points = [],[]
    for family in sorted(set(table.model_family)-{'logistic_linear'}):
        selected = table.loc[table.model_family.eq(family)].copy()
        aligned = base.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
        changes = {'delta_change':selected.gain.to_numpy()-aligned.gain.to_numpy(),
            'm0_loss_improvement':aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy(),
            'm1_loss_improvement':aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy()}
        for name,values in changes.items():
            part = selected.copy()
            part['scenario'],part['analysis_arm'],part['gain'] = family,name,values
            blocks.append(part)
            for view,group in part.groupby('view',sort=True):
                for label,scope in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
                    post = scope.loc[scope.target.eq('sim_post'),'gain'].mean()
                    pre = scope.loc[scope.target.eq('sim_pre'),'gain'].mean()
                    points.append(dict(scenario=family,analysis_arm=name,view=view,manual_label=label,
                        n=scope.call_id.nunique(),delta_post=post,delta_pre=pre,post_minus_pre=post-pre))
    intervals = core.bootstrap_overall(pd.concat(blocks,ignore_index=True),n_draws,7452)
    result = pd.DataFrame(points).merge(intervals,on=['scenario','analysis_arm','view','manual_label'],
                                        how='left',validate='one_to_one')
    return result.rename(columns={'scenario':'model_family','analysis_arm':'contrast'})


def add_secondary_metrics(metrics,table):
    for index,row in metrics.iterrows():
        selected = table.loc[table.model_family.eq(row.model_family) & table.view.eq(row.view)]
        if row.manual_label!='overall':
            selected = selected.loc[selected.manual_label.eq(row.manual_label)]
        for side in ['post','pre']:
            part = selected.loc[selected.target.eq('sim_'+side)]
            y = part.y_sim.to_numpy(int)
            for model in ['m0','m1']:
                p = part[model+'_pred'].to_numpy()
                metrics.loc[index,model+'_'+side+'_auc'] = roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
                metrics.loc[index,model+'_'+side+'_ap'] = average_precision_score(y,p) if y.sum()>0 else np.nan
    return metrics


def summarize(args,current,references,ref_manifest,protocol_sha,n_draws=1000):
    expected = {job_path(args,f,fld,s).name for f in models.FAMILIES for fld in range(1,6) for s in ['post','pre']}
    actual = {p.name for p in (args.output_dir/'jobs').glob('*.csv')}
    if actual!=expected:
        raise ValueError('Missing/stale new jobs;no partial summary.')
    parts,manifest = [references],list(ref_manifest)
    for family in models.FAMILIES:
        for fold in range(1,6):
            test = current.loc[current.fold.eq(fold)]
            for side in ['post','pre']:
                path = job_path(args,family,fold,side)
                frame = read(path)
                check_record(frame,test,family,fold,side,len(current)-len(test),protocol_sha)
                parts.append(frame)
                manifest.append(dict(path=str(path),sha256=core.digest(path),model_family=family,
                    fold=fold,target=side,n_rows=len(frame),role='new_capacity_fit'))
    records = pd.concat(parts,ignore_index=True)
    table = statistics_table(records)
    with asym.configured_core():
        metrics = core.point_metrics(table).merge(core.bootstrap_overall(table,n_draws,7452),
            on=['scenario','analysis_arm','view','manual_label'],how='left',validate='one_to_one')
    metrics = metrics.rename(columns={'analysis_arm':'model_family'})
    metrics = add_secondary_metrics(metrics,table)
    metrics['relative_post_loss_reduction_pct'] = 100*metrics.delta_post/metrics.m0_post_logloss
    metrics.to_csv(args.output_dir/'comparison_metrics_and_ci.csv',index=False)
    paired_changes(table,n_draws).to_csv(args.output_dir/'paired_changes_vs_linear.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    core.write_json(args.output_dir/'verification.json',dict(status='passed',scope='B1_SYNTHETIC_ONLY',
        new_jobs=20,new_model_fits_required=40,reference_jobs=20,n_calls=len(current),
        n_clips=current['_clip'].nunique(),n_oof_rows=len(records),metric_rows=len(metrics),
        fixed_clip_folds=True,real_outcomes_not_used=True,validated_error_control=False,
        real_stable_information_assessed=False,practical_equivalence_margin=None,
        intervals='Overall only,conditional paired clips;no refit uncertainty.'))
    print('B1 comparison complete;B2/B3 and real information status remain unresolved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','reference-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    args.seeds,args.n_bootstrap,args.threads = [431],1000,4
    if args.output_dir.resolve() in {p.resolve() for p in [args.project_root,args.shared_dir,args.controls_dir,args.reference_run]}:
        raise ValueError('Do not overwrite source data/results.')
    with threadpool_limits(limits=4):
        current,cols,controls,shared,base_protocol = load_design(args)
        references,ref_manifest = reference_frames(args,current)
        protocol_sha = freeze(args,current,cols,controls,shared,base_protocol,ref_manifest)
        (args.output_dir/'jobs').mkdir(exist_ok=True)
        if args.stage=='prepare':
            pd.DataFrame(ref_manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            core.write_json(args.output_dir/'prepare_check.json',dict(status='passed',model_fits=0,
                n_calls=len(current),n_clips=current['_clip'].nunique(),acoustic_columns=len(cols),
                reference_jobs=20,new_jobs=20,planned_new_model_fits=40,B1_only=True))
            print('Prepared B1;40 new fits are user-run. No new fitted metrics.')
            return
        if args.stage=='fit':
            fit_all(args,current,cols,controls,shared,protocol_sha)
        summarize(args,current,references,ref_manifest,protocol_sha)


if __name__=='__main__':
    main()
