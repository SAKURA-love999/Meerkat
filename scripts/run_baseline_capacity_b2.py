"""B2 only: matched positive-signal/shared-state capacity controls, user-run fits."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import run_baseline_capacity as b1
import run_semisynthetic_calibration as core

SCENARIOS = ['signal_post', 'state_bilateral']
FIELDS = b1.FIELDS
read = b1.read


def verify_hashes(protocol):
    for section in ['source_hashes', 'input_hashes']:
        for path, expected in protocol[section].items():
            if core.digest(path) != expected:
                raise ValueError('Frozen source/input changed: '+path)


def check_record(frame, test, scenario, family, fold, side, n_train, protocol_sha=None):
    core.check_job(frame,test,431,scenario,'standard',fold,side,n_train)
    aligned = frame.set_index('call_id').loc[test.call_id]
    for name in ['_clip', 'manual_label', 'egemaps_valid']:
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
    if status['status']!='passed' or status['expected_jobs']!=60 or original['seeds']!=[431]:
        raise ValueError('Require the completed original seed431 quick pilot.')
    verify_hashes(original)
    data,cols,controls,shared = core.load_real_design(args)
    base_protocol = core.protocol(args,data,cols,controls)
    if base_protocol != original:
        raise ValueError('Original reference/helper/data/environment protocol differs.')
    baseline = json.loads((args.b1_run/'protocol.json').read_text(encoding='utf-8'))
    done = json.loads((args.b1_run/'verification.json').read_text(encoding='utf-8'))
    if done['status']!='passed' or done['scope']!='B1_SYNTHETIC_ONLY':
        raise ValueError('Require completed B1 before B2.')
    factory = str(Path(__file__).parent/'baseline_capacity_models.py')
    numeric = shared.model_numeric_columns(data,'NF_post')+controls
    if (core.digest(factory)!=baseline['source_hashes'].get(factory)
            or baseline['acoustic_columns']!=cols or baseline['measured_controls']!=controls
            or baseline['spline']!=models.SPLINE or baseline['hgb_requested']!=models.HGB
            or baseline['new_families']!=models.FAMILIES
            or baseline['logistic']!=models.make_model('spline_rms',numeric,[]).named_steps['clf'].get_params()
            or baseline['hgb']!=models.make_model('hist_gbdt',numeric,[]).named_steps['clf'].get_params()):
        raise ValueError('B1 feature/model contract changed.')
    members = read(args.reference_run/'fixed_design_membership.csv').set_index('call_id')
    selected = data.set_index('call_id').loc[members.index,FIELDS[1:]]
    pd.testing.assert_frame_equal(selected,members[FIELDS[1:]],check_dtype=False)
    if len(members)!=len(data):
        raise ValueError('Original membership coverage differs.')
    entries = json.loads((args.reference_run/'synthetic_design_manifest.json').read_text(encoding='utf-8'))
    entries = [r for r in entries if r['scenario'] in SCENARIOS and r['seed']==431]
    if len(entries)!=2 or {r['scenario'] for r in entries}!=set(SCENARIOS):
        raise ValueError('Require exactly two original B2 design packets.')
    currents = {}
    for item in entries:
        if core.digest(item['path'])!=item['sha256']:
            raise ValueError('Original simulation packet changed.')
        with np.load(item['path'],allow_pickle=False) as saved:
            packet = {k:saved[k] for k in saved.files}
        if not np.array_equal(packet['call_ids'],data.call_id.to_numpy()):
            raise ValueError('Original packet call order differs.')
        currents[item['scenario']] = core.materialize(data,packet)
    return currents,cols,controls,shared,base_protocol,entries


def reference_frames(args,currents):
    manifest = pd.read_csv(args.reference_run/'synthetic_oof_manifest.csv')
    selected = manifest.loc[manifest.scenario.isin(SCENARIOS) & manifest.seed.eq(431)
                            & manifest.analysis_arm.eq('standard')]
    expected = {(s,f,t) for s in SCENARIOS for f in range(1,6) for t in ['post','pre']}
    if len(selected)!=20 or set(zip(selected.scenario,selected.fold,selected.target))!=expected:
        raise ValueError('Reference OOF coverage differs.')
    frames,records = [],[]
    for item in selected.itertuples(index=False):
        if core.digest(item.path)!=item.sha256:
            raise ValueError('Reference OOF hash differs.')
        current = currents[item.scenario]
        test = current.loc[current.fold.eq(item.fold)]
        frame = read(item.path)
        check_record(frame,test,item.scenario,'logistic_linear',item.fold,item.target,len(current)-len(test))
        if len(frame)!=item.n_rows:
            raise ValueError('Reference OOF row count differs.')
        frame['model_family'] = 'logistic_linear'
        frames.append(frame)
        records.append(dict(path=item.path,sha256=item.sha256,scenario=item.scenario,
            model_family='logistic_linear',fold=item.fold,target=item.target,
            n_rows=len(frame),role='existing_reference'))
    return pd.concat(frames,ignore_index=True),records


def freeze(args,currents,cols,controls,base_protocol,entries,refs):
    here = Path(__file__).parent
    sources = dict(base_protocol['source_hashes'])
    for name in ['baseline_capacity_models.py','run_baseline_capacity.py',
                 'run_baseline_capacity_b2.py','run_baseline_capacity_b2_400.ps1']:
        sources[str(here/name)] = core.digest(here/name)
    inputs = dict(base_protocol['input_hashes'])
    for folder,names in [(args.reference_run,['protocol.json','verification.json',
            'synthetic_design_manifest.json','synthetic_oof_manifest.csv','fixed_design_membership.csv']),
            (args.b1_run,['protocol.json','verification.json','comparison_metrics_and_ci.csv'])]:
        for name in names:
            inputs[str(folder/name)] = core.digest(folder/name)
    for item in entries+refs:
        inputs[item['path']] = item['sha256']
    data = currents[SCENARIOS[0]]
    value = dict(kind='BASELINE_CAPACITY_B2_ONLY',version=1,seed=431,scenarios=SCENARIOS,
        n_calls=len(data),n_clips=data['_clip'].nunique(),n_folds=5,acoustic_columns=cols,
        measured_controls=controls,new_families=models.FAMILIES,spline=models.SPLINE,
        hgb_requested=models.HGB,acoustic_pca_dim=30,new_pipeline_fits=80,new_jobs=40,
        reference_jobs_reused=20,arms='Standard observed-Z only;no new oracle-U fit.',
        source_hashes=sources,input_hashes=inputs,threads=4,
        environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        bootstrap=dict(draws=1000,seed=7452,scope='Overall paired clips,conditional on stored OOF,no refit uncertainty.'),
        primary='All rows;M0/M1 logloss,DeltaPOST,DeltaPRE,POST-PRE,direct paired changes versus linear.',
        secondary='Original-valid evaluation-only rows;AUC/AP;no refitting on selected rows.',
        truth={s:core.TRUTH[s] for s in SCENARIOS},
        model_contract='Identical B1 factory;both M0/M1 upgraded;train-fold-only imputer/scaler/basis/PCA.',
        tuning='None;fixed HGB with early_stopping=False;no outer-test choice.',
        scope='Chosen-strong-effect development controls,not animal behavior,causality,weak-signal power or error control.',
        deferred=['B3 real calls','Weak effects','New seeds','Held-out generators','Conditional-information comparator'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path = args.output_dir/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen B2 protocol differs;preserve old results,use a new directory.')
    else:
        core.write_json(path,value)
    return core.digest(path)


def job_path(args,scenario,family,fold,side):
    return args.output_dir/'jobs'/f'{scenario}_{family}_fold{fold}_{side}.csv'


def fit_all(args,currents,cols,controls,shared,protocol_sha):
    counts = dict(created_jobs=0,reused_jobs=0,model_fits=0)
    for scenario in SCENARIOS:
        current = currents[scenario]
        for family in models.FAMILIES:
            for fold in range(1,6):
                train,test = current.loc[current.fold.ne(fold)],current.loc[current.fold.eq(fold)]
                if set(train['_clip']) & set(test['_clip']):
                    raise ValueError('Train/test clips overlap.')
                for side in ['post','pre']:
                    path = job_path(args,scenario,family,fold,side)
                    if path.exists():
                        check_record(read(path),test,scenario,family,fold,side,len(train),protocol_sha)
                        counts['reused_jobs'] += 1
                        continue
                    target = 'sim_y_'+side
                    if train[target].nunique()!=2:
                        raise ValueError('Single-class training cell;no silent fallback.')
                    numeric = shared.model_numeric_columns(train,'NF_'+side)+controls
                    predictions = []
                    for acoustic in [[],cols]:
                        inputs = ['manual_label']+numeric+acoustic
                        model = models.make_model(family,numeric,acoustic)
                        model.fit(train[inputs],train[target].to_numpy(int))
                        predictions.append(model.predict_proba(test[inputs])[:,1])
                        counts['model_fits'] += 1
                    out = test[FIELDS].copy()
                    for key,value in dict(scenario=scenario,model_family=family,analysis_arm='standard',
                            replicate_seed=431,target='sim_'+side,synthetic=True,n_train=len(train),
                            protocol_sha256=protocol_sha).items():
                        out[key] = value
                    out['y_sim'],out['generator_probability'] = test[target],test['sim_p_'+side]
                    out['m0_pred'],out['m1_pred'] = predictions
                    check_record(out,test,scenario,family,fold,side,len(train),protocol_sha)
                    temporary = path.with_suffix('.tmp')
                    out.to_csv(temporary,index=False)
                    temporary.replace(path)
                    counts['created_jobs'] += 1
                print(f'completed scenario={scenario} family={family} fold={fold}/5 '+json.dumps(counts),flush=True)
    return counts


def secondary_metrics(metrics,table):
    result = metrics.copy()
    for index,row in result.iterrows():
        selected = table.loc[table.scenario.eq(row['scenario']) & table.model_family.eq(row['model_family'])
                             & table['view'].eq(row['view'])]
        if row['manual_label']!='overall':
            selected = selected.loc[selected.manual_label.eq(row['manual_label'])]
        for side in ['post','pre']:
            part = selected.loc[selected.target.eq('sim_'+side)]
            y = part.y_sim.to_numpy(int)
            for model in ['m0','m1']:
                p = part[model+'_pred'].to_numpy()
                result.loc[index,model+'_'+side+'_auc'] = roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
                result.loc[index,model+'_'+side+'_ap'] = average_precision_score(y,p) if y.sum()>0 else np.nan
    return result


def summarize(args,currents,references,ref_manifest,protocol_sha,n_draws=1000):
    expected = {job_path(args,s,f,k,t).name for s in SCENARIOS for f in models.FAMILIES
                for k in range(1,6) for t in ['post','pre']}
    if {p.name for p in (args.output_dir/'jobs').glob('*.csv')}!=expected:
        raise ValueError('Missing/stale B2 jobs;no partial summary.')
    parts,manifest = [references],list(ref_manifest)
    for scenario in SCENARIOS:
        current = currents[scenario]
        for family in models.FAMILIES:
            for fold in range(1,6):
                test = current.loc[current.fold.eq(fold)]
                for side in ['post','pre']:
                    path = job_path(args,scenario,family,fold,side)
                    frame = read(path)
                    check_record(frame,test,scenario,family,fold,side,len(current)-len(test),protocol_sha)
                    parts.append(frame)
                    manifest.append(dict(path=str(path),sha256=core.digest(path),scenario=scenario,
                        model_family=family,fold=fold,target=side,n_rows=len(frame),role='new_capacity_fit'))
    records = pd.concat(parts,ignore_index=True)
    if records.duplicated(['scenario','model_family','target','call_id']).any():
        raise ValueError('Duplicate B2 OOF calls.')
    table = b1.statistics_table(records)
    metrics = []
    for family,part in table.groupby('model_family',sort=True):
        # core's truth fields describe the standard observed-Z task,not family names.
        part = part.copy()
        part['analysis_arm'] = 'standard'
        block = core.point_metrics(part).merge(core.bootstrap_overall(part,n_draws,7452),
            on=['scenario','analysis_arm','view','manual_label'],how='left',validate='one_to_one')
        block['model_family'] = family
        metrics.append(block.drop(columns='analysis_arm'))
    metrics = secondary_metrics(pd.concat(metrics,ignore_index=True),table)
    metrics['relative_post_loss_reduction_pct'] = 100*metrics.delta_post/metrics.m0_post_logloss
    metrics.to_csv(args.output_dir/'comparison_metrics_and_ci.csv',index=False)
    changes = []
    for scenario,part in table.groupby('scenario',sort=True):
        block = b1.paired_changes(part,n_draws)
        block.insert(0,'scenario',scenario)
        changes.append(block)
    pd.concat(changes,ignore_index=True).to_csv(args.output_dir/'paired_changes_vs_linear.csv',index=False)
    folds = []
    for tags,part in table.groupby(['scenario','model_family','view','fold'],sort=True):
        row = dict(zip(['scenario','model_family','view','fold'],tags))
        for side in ['post','pre']:
            side_rows = part.loc[part.target.eq('sim_'+side)]
            for model in ['m0','m1']:
                row[model+'_'+side+'_logloss'] = side_rows[model+'_loss'].mean()
            row['delta_'+side] = side_rows.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        folds.append(row)
    pd.DataFrame(folds).to_csv(args.output_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir/'all_oof_manifest.csv',index=False)
    data = currents[SCENARIOS[0]]
    core.write_json(args.output_dir/'verification.json',dict(status='passed',scope='B2_SYNTHETIC_CONTROLS_ONLY',
        new_jobs=40,new_model_fits_required=80,reference_jobs=20,n_calls=len(data),n_clips=data['_clip'].nunique(),
        n_oof_rows=len(records),metric_rows=len(metrics),fixed_clip_folds=True,real_outcomes_not_used=True,
        validated_error_control=False,weak_signal_power_assessed=False,real_stable_information_assessed=False,
        practical_equivalence_margin=None,intervals='Overall only,conditional paired clips;no refit uncertainty.'))
    print('B2 comparison complete. Interpret both controls;B3 real data is not run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','reference-run','b1-run','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','summary'],default='prepare')
    args = parser.parse_args()
    args.seeds,args.n_bootstrap,args.threads = [431],1000,4
    if args.output_dir.resolve() in {p.resolve() for p in
            [args.project_root,args.shared_dir,args.controls_dir,args.reference_run,args.b1_run]}:
        raise ValueError('Do not overwrite source results.')
    if args.stage!='prepare' and not (args.output_dir/'protocol.json').exists():
        raise ValueError('Run Prepare before Fit/Summary.')
    with threadpool_limits(limits=4):
        currents,cols,controls,shared,base_protocol,entries = load_design(args)
        references,manifest = reference_frames(args,currents)
        protocol_sha = freeze(args,currents,cols,controls,base_protocol,entries,manifest)
        (args.output_dir/'jobs').mkdir(exist_ok=True)
        if args.stage=='prepare':
            pd.DataFrame(manifest).to_csv(args.output_dir/'reference_oof_manifest.csv',index=False)
            core.write_json(args.output_dir/'prepare_check.json',dict(status='passed',model_fits=0,
                n_calls=len(currents[SCENARIOS[0]]),n_clips=currents[SCENARIOS[0]]['_clip'].nunique(),
                scenarios=SCENARIOS,acoustic_columns=len(cols),reference_jobs=20,new_jobs=40,
                planned_new_model_fits=80,B2_only=True))
            print('Prepared B2;80 formal CPU fits are user-run. No new performance result.')
            return
        if args.stage=='fit':
            print('INVOCATION_FIT_COUNTS '+json.dumps(fit_all(args,currents,cols,controls,shared,protocol_sha)))
        summarize(args,currents,references,manifest,protocol_sha)


if __name__=='__main__':
    main()
