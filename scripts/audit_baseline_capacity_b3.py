"""Read-only completed B3 audit and descriptive al attribution; never fit models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import run_baseline_capacity_b3 as b3


def loss(y,p):
    y = np.asarray(y,float)
    p = np.clip(np.asarray(p,float),1e-15,1-1e-15)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def compare_tables(old,new,keys,columns):
    old,new = old.set_index(keys).sort_index(),new.set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index,new.index)
    errors = []
    for name in columns:
        left,right = old[name].astype(float),new[name].astype(float)
        if not np.array_equal(left.isna(),right.isna()):
            raise ValueError('NaN pattern differs: '+name)
        value = float((left-right).abs().max())
        errors.append(0. if np.isnan(value) else value)
    if max(errors)>1e-12:
        raise ValueError('Metric discrepancy: '+str(max(errors)))
    return max(errors)


def al_attribution(table,output):
    part = table.loc[table['view'].eq('all') & table.manual_label.eq('al') & table.target.eq('NF_post')]
    base = part.loc[part.model_family.eq('logistic_linear')].set_index('call_id')
    folds,clips,summaries = [],[],[]
    for family in b3.models.FAMILIES:
        selected = part.loc[part.model_family.eq(family)].copy()
        aligned = base.loc[selected.call_id]
        selected['m0_improvement'] = aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy()
        selected['m1_improvement'] = aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy()
        selected['delta_change'] = selected.m1_improvement-selected.m0_improvement
        for fold,rows in selected.groupby('fold'):
            folds.append(dict(model_family=family,fold=fold,n=len(rows),
                post_positive_n=int(rows.y_true.sum()),m0_logloss=rows.m0_loss.mean(),m1_logloss=rows.m1_loss.mean(),
                delta_post=rows.gain.mean(),m0_improvement=rows.m0_improvement.mean(),
                m1_improvement=rows.m1_improvement.mean(),delta_change=rows.delta_change.mean()))
        group = selected.groupby('_clip').agg(n=('call_id','size'),post_positive_n=('y_true','sum'),
            m0_improvement_sum=('m0_improvement','sum'),m1_improvement_sum=('m1_improvement','sum'),
            delta_change_sum=('delta_change','sum')).reset_index()
        group['model_family'] = family
        clips.append(group)
        positive = group.m0_improvement_sum.clip(lower=0).sort_values(ascending=False)
        row = dict(model_family=family,n_clips=len(group),n_positive_m0_clips=int(group.m0_improvement_sum.gt(0).sum()),
            m0_net_improvement_sum=group.m0_improvement_sum.sum(),
            positive_m0_mass=positive.sum(),m0_positive_fold_n=sum(x['m0_improvement']>0 for x in folds if x['model_family']==family))
        for k in [1,5,10]:
            row['top'+str(k)+'_positive_mass_share'] = positive.head(k).sum()/positive.sum() if positive.sum()>0 else np.nan
        summaries.append(row)
    pd.DataFrame(folds).to_csv(output/'al_fold_attribution_post.csv',index=False)
    pd.concat(clips,ignore_index=True).to_csv(output/'al_clip_attribution_post.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'al_influence_summary.csv',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','run-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a fresh audit output;do not overwrite source results.')
    protocol = json.loads((args.run_dir/'protocol.json').read_text(encoding='utf-8'))
    b3.verify_hashes(protocol)
    data,_,_,_ = b3.load_design(args)
    manifest = pd.read_csv(args.run_dir/'all_oof_manifest.csv')
    families = ['logistic_linear']+b3.models.FAMILIES
    expected = {(f,k,t) for f in families for k in range(1,6) for t in ['NF_post','NF_pre']}
    if len(manifest)!=30 or set(zip(manifest.model_family,manifest.fold,manifest.target))!=expected:
        raise ValueError('B3 manifest coverage differs.')
    records = []
    protocol_sha = b3.shared.digest(args.run_dir/'protocol.json')
    for item in manifest.itertuples(index=False):
        if b3.shared.digest(Path(item.path))!=item.sha256:
            raise ValueError('OOF hash differs.')
        saved = b3.read(item.path)
        test = data.loc[data.fold.eq(item.fold)]
        if item.model_family=='logistic_linear':
            if item.role!='existing_reference' or item.selector!='condition='+b3.REFERENCE:
                raise ValueError('Wrong historical reference selector.')
            b3.attribution.assert_job(saved,test,b3.REFERENCE,item.target,item.fold)
            frame = saved.loc[saved.condition.eq(b3.REFERENCE)].copy()
            frame['model_family'],frame['synthetic'] = item.model_family,False
        else:
            if item.role!='new_capacity_fit' or item.selector!='all rows':
                raise ValueError('Wrong new job role.')
            frame = saved
        if len(saved)!=item.source_rows or len(frame)!=item.n_rows:
            raise ValueError('OOF source/selected row counts differ.')
        b3.check_record(frame,test,item.model_family,item.fold,item.target,len(data)-len(test),
                        protocol_sha if item.role=='new_capacity_fit' else None)
        records.append(frame)
    raw = pd.concat(records,ignore_index=True)
    if len(raw)!=110070 or raw.duplicated(['model_family','target','call_id']).any():
        raise ValueError('OOF unique coverage differs.')
    meta = raw.groupby(['target','call_id'])[
        ['y_true','_clip','fold','manual_label','duration_s','egemaps_valid']].nunique(dropna=False)
    if not meta.eq(1).all().all():
        raise ValueError('Cross-model paired metadata differ.')
    parts = []
    for view,rows in [('all',raw),('original_valid_rows',raw.loc[raw.egemaps_valid])]:
        part = rows.copy()
        part['view'] = view
        for model in ['m0','m1']:
            part[model+'_loss'] = loss(part.y_true,part[model+'_pred'])
        part['gain'] = part.m0_loss-part.m1_loss
        parts.append(part)
    table = pd.concat(parts,ignore_index=True)
    reported = b3.read(args.run_dir/'comparison_metrics_and_ci.csv')
    lookup = reported.set_index(['model_family','view','manual_label'],verify_integrity=True)
    checks,folds = [],[]
    for tags,group in table.groupby(['model_family','view'],sort=True):
        for label,rows in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            actual = lookup.loc[tags+(label,)]
            values = dict(n=rows.call_id.nunique(),n_clips=rows['_clip'].nunique(),n_folds=rows.fold.nunique())
            for target,side in [('NF_post','post'),('NF_pre','pre')]:
                selected = rows.loc[rows.target.eq(target)]
                y = selected.y_true.to_numpy(int)
                values[side+'_positive_n'],values[side+'_positive_rate'] = int(y.sum()),y.mean()
                for model in ['m0','m1']:
                    p = selected[model+'_pred'].to_numpy()
                    values[model+'_'+side+'_logloss'] = selected[model+'_loss'].mean()
                    values[model+'_'+side+'_auc'] = roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
                    values[model+'_'+side+'_ap'] = average_precision_score(y,p) if y.sum()>0 else np.nan
                values['delta_'+side] = selected.gain.mean()
            values['post_minus_pre'] = values['delta_post']-values['delta_pre']
            values['relative_post_loss_reduction_pct'] = 100*values['delta_post']/values['m0_post_logloss']
            values['sparse_positive_warning'] = min(values['post_positive_n'],values['pre_positive_n'])<20
            if bool(actual['synthetic']):
                raise ValueError('Wrong simulation flag in real metrics.')
            for metric,value in values.items():
                error = 0. if pd.isna(value) and pd.isna(actual[metric]) else abs(float(value)-float(actual[metric]))
                if not np.isfinite(error) or error>1e-12:
                    raise ValueError('Point metric mismatch: '+str(tags+(label,metric)))
                checks.append(dict(model_family=tags[0],view=tags[1],manual_label=label,metric=metric,abs_error=error))
        for fold,rows in group.groupby('fold'):
            row = dict(model_family=tags[0],view=tags[1],fold=fold)
            for target,side in [('NF_post','post'),('NF_pre','pre')]:
                selected = rows.loc[rows.target.eq(target)]
                for model in ['m0','m1']:
                    row[model+'_'+side+'_logloss'] = selected[model+'_loss'].mean()
                row['delta_'+side] = selected.gain.mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            folds.append(row)
    if len(lookup)!=54:
        raise ValueError('Metric row coverage differs.')
    values = b3.contrast_values(table)
    intervals = b3.clip_intervals(values,1000,7452)
    ci_columns = [c for c in intervals if '_ci_' in c or c.endswith('_finite_draws')]
    primary_ci = intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast')
    ci_error = compare_tables(reported,primary_ci,['model_family','view','manual_label'],ci_columns)
    points = []
    for tags,group in values.loc[values.contrast.ne('gain')].groupby(b3.KEYS[:-1],sort=True):
        for label,rows in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            post = rows.loc[rows.target.eq('NF_post'),'value'].mean()
            pre = rows.loc[rows.target.eq('NF_pre'),'value'].mean()
            points.append(dict(zip(b3.KEYS,tags+(label,)),n=rows.call_id.nunique(),
                               delta_post=post,delta_pre=pre,post_minus_pre=post-pre))
    contrast = pd.DataFrame(points).merge(intervals.loc[intervals.contrast.ne('gain')],on=b3.KEYS,validate='one_to_one')
    old = b3.read(args.run_dir/'paired_changes_vs_linear.csv')
    change_error = compare_tables(old,contrast,b3.KEYS,['n','delta_post','delta_pre','post_minus_pre']+ci_columns)
    old_fold = b3.read(args.run_dir/'fold_metrics.csv')
    fold_columns = [c for c in old_fold if c not in ['model_family','view','fold']]
    fold_error = compare_tables(old_fold,pd.DataFrame(folds),['model_family','view','fold'],fold_columns)
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(args.output_dir/'metric_checks.csv',index=False)
    al_attribution(table,args.output_dir)
    result = dict(status='passed',model_fits=0,manifest_jobs=len(manifest),raw_oof_rows=len(raw),metric_rows=len(reported),
        point_scalars_checked=len(checks),point_max_abs_error=float(pd.DataFrame(checks).abs_error.max()),
        ci_max_abs_error=ci_error,paired_changes_max_abs_error=change_error,fold_max_abs_error=fold_error,
        source_protocol_sha256=protocol_sha,scope='Real B3 descriptive/exploratory;no conditional-information or causal certificate',
        intervals='Recomputed with frozen bootstrap;unadjusted,conditional on stored fits,no refitting uncertainty',
        extra_diagnostic='Post-hoc full-row al POST clip/fold attribution,without sample deletion')
    b3.shared.write_json(args.output_dir/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
