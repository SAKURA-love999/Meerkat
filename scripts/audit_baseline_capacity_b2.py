"""Verify completed B2 OOF, points and paired intervals without any model fit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from audit_baseline_capacity_b1 import loss
import run_baseline_capacity_b2 as b2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','reference-run','b1-run','run-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.seeds,args.n_bootstrap,args.threads = [431],1000,4
    if args.output_dir.exists():
        raise ValueError('Use a new audit directory;never overwrite source results.')
    protocol = json.loads((args.run_dir/'protocol.json').read_text(encoding='utf-8'))
    b2.verify_hashes(protocol)
    currents,_,_,_,_,_ = b2.load_design(args)
    manifest = pd.read_csv(args.run_dir/'all_oof_manifest.csv')
    families = ['logistic_linear','spline_rms','hist_gbdt']
    expected = {(s,f,k,t) for s in b2.SCENARIOS for f in families for k in range(1,6) for t in ['post','pre']}
    if len(manifest)!=60 or set(zip(manifest.scenario,manifest.model_family,manifest.fold,manifest.target))!=expected:
        raise ValueError('OOF job coverage differs.')
    protocol_sha = b2.core.digest(args.run_dir/'protocol.json')
    records = []
    for item in manifest.itertuples(index=False):
        expected_role = 'existing_reference' if item.model_family=='logistic_linear' else 'new_capacity_fit'
        if item.role!=expected_role or b2.core.digest(item.path)!=item.sha256:
            raise ValueError('OOF role/hash differs.')
        frame = b2.read(item.path)
        current = currents[item.scenario]
        test = current.loc[current.fold.eq(item.fold)]
        b2.check_record(frame,test,item.scenario,item.model_family,item.fold,item.target,
                        len(current)-len(test),protocol_sha if item.role=='new_capacity_fit' else None)
        if len(frame)!=item.n_rows:
            raise ValueError('OOF row count differs.')
        frame['model_family'] = item.model_family
        records.append(frame)
    raw = pd.concat(records,ignore_index=True)
    if len(raw)!=220140 or raw.duplicated(['scenario','model_family','target','call_id']).any():
        raise ValueError('OOF completeness/uniqueness differs.')
    meta = raw.groupby(['scenario','target','call_id'])[
        ['y_sim','_clip','fold','manual_label','duration_s','egemaps_valid']].nunique(dropna=False)
    if not meta.eq(1).all().all():
        raise ValueError('Cross-learner calls/targets/metadata do not match.')
    tables = []
    for view,part in [('all',raw),('original_valid_rows',raw.loc[raw.egemaps_valid])]:
        part = part.copy()
        part['view'] = view
        for model in ['m0','m1']:
            part[model+'_loss'] = loss(part.y_sim,part[model+'_pred'])
        part['gain'] = part.m0_loss-part.m1_loss
        tables.append(part)
    table = pd.concat(tables,ignore_index=True)
    reported = b2.read(args.run_dir/'comparison_metrics_and_ci.csv')
    keys = ['scenario','model_family','view','manual_label']
    indexed = reported.set_index(keys,verify_integrity=True)
    checks,folds = [],[]
    for tags,group in table.groupby(keys[:-1],sort=True):
        scenario,family,view = tags
        for label,scope in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            row = indexed.loc[tags+(label,)]
            truth = b2.core.TRUTH[scenario]
            values = dict(n=scope.call_id.nunique(),n_clips=scope['_clip'].nunique(),
                truth_extra_post=int(truth['extra_post']),truth_extra_pre=int(truth['extra_pre']),
                direct_post_mark_in_generator=int(truth['direct_post_mark']))
            for side in ['post','pre']:
                part = scope.loc[scope.target.eq('sim_'+side)]
                y = part.y_sim.to_numpy(int)
                values['delta_'+side] = part.gain.mean()
                values[side+'_positive_n'] = int(y.sum())
                values[side+'_positive_rate'] = y.mean()
                values['generator_parent_'+side+'_logloss'] = loss(y,part.generator_probability).mean()
                for model in ['m0','m1']:
                    p = part[model+'_pred'].to_numpy()
                    values[model+'_'+side+'_logloss'] = part[model+'_loss'].mean()
                    values[model+'_'+side+'_auc'] = roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
                    values[model+'_'+side+'_ap'] = average_precision_score(y,p) if y.sum() else np.nan
            values['post_minus_pre'] = values['delta_post']-values['delta_pre']
            values['relative_post_loss_reduction_pct'] = 100*values['delta_post']/values['m0_post_logloss']
            for metric,value in values.items():
                actual = row[metric]
                if pd.isna(value) and pd.isna(actual):
                    error = 0.
                else:
                    error = abs(float(value)-float(actual))
                if not np.isfinite(error) or error>1e-12:
                    raise ValueError('Metric mismatch: '+str(tags+(label,metric)))
                checks.append(dict(zip(keys,tags+(label,)),metric=metric,abs_error=error))
        for fold,scope in group.groupby('fold',sort=True):
            row = dict(zip(keys[:-1],tags),fold=fold)
            for side in ['post','pre']:
                part = scope.loc[scope.target.eq('sim_'+side)]
                for model in ['m0','m1']:
                    row[model+'_'+side+'_logloss'] = part[model+'_loss'].mean()
                row['delta_'+side] = part.gain.mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            folds.append(row)
    if len(indexed)!=108:
        raise ValueError('Unexpected metric coverage.')
    stats = b2.b1.statistics_table(raw)
    intervals,changes = [],[]
    for family,part in stats.groupby('model_family',sort=True):
        part = part.copy()
        part['analysis_arm'] = 'standard'
        block = b2.core.bootstrap_overall(part,1000,7452).drop(columns='analysis_arm')
        block['model_family'] = family
        intervals.append(block)
    merged = reported.merge(pd.concat(intervals,ignore_index=True),on=keys,suffixes=('_old','_new'))
    ci_cols = [c for c in merged if c.endswith('_ci_low_old') or c.endswith('_ci_high_old')]
    ci_error = max(float((merged[c]-merged[c[:-4]+'_new']).abs().max()) for c in ci_cols)
    for scenario,part in stats.groupby('scenario',sort=True):
        block = b2.b1.paired_changes(part,1000)
        block.insert(0,'scenario',scenario)
        changes.append(block)
    contrast_keys = ['scenario','model_family','contrast','view','manual_label']
    old = b2.read(args.run_dir/'paired_changes_vs_linear.csv').set_index(contrast_keys).sort_index()
    new = pd.concat(changes,ignore_index=True).set_index(contrast_keys).sort_index()
    pd.testing.assert_index_equal(old.index,new.index)
    change_error = float((old-new).abs().max().max())
    old_fold = b2.read(args.run_dir/'fold_metrics.csv').set_index(keys[:-1]+['fold']).sort_index()
    new_fold = pd.DataFrame(folds).set_index(keys[:-1]+['fold']).sort_index()
    pd.testing.assert_index_equal(old_fold.index,new_fold.index)
    fold_error = float((old_fold-new_fold).abs().max().max())
    if max(ci_error,change_error,fold_error)>1e-12:
        raise ValueError('Interval/contrast/fold discrepancy.')
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(args.output_dir/'metric_checks.csv',index=False)
    summary = dict(status='passed',model_fits=0,raw_oof_rows=len(raw),manifest_jobs=len(manifest),
        metric_rows=len(reported),point_scalars_checked=len(checks),
        point_max_abs_error=float(pd.DataFrame(checks).abs_error.max()),
        ci_max_abs_error=ci_error,paired_changes_max_abs_error=change_error,fold_max_abs_error=fold_error,
        source_protocol_sha256=protocol_sha,scope='B2 developmental controls,seed431 only;no real-data verdict',
        interval_scope='1000 paired clip draws conditional on stored OOF;no refit uncertainty',
        gate='Supports proceeding to B3 prediction sensitivity,not validated conditional-information inference')
    b2.core.write_json(args.output_dir/'verification.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
