"""Read-only nested-C result audit; no training, selection changes or source writes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_baseline_capacity_b3 as audit
import run_nested_logistic_regularization as runner


def paired_points(values,intervals):
    rows = []
    for tags,group in values.groupby(runner.b3.KEYS[:-1],sort=True):
        for label,part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            post = part.loc[part.target.eq('NF_post'),'value'].mean()
            pre = part.loc[part.target.eq('NF_pre'),'value'].mean()
            rows.append(dict(zip(runner.b3.KEYS,tags+(label,)),n=part.call_id.nunique(),
                delta_post=post,delta_pre=pre,post_minus_pre=post-pre))
    return pd.DataFrame(rows).merge(intervals,on=runner.b3.KEYS,validate='one_to_one')


def cross_family_values(table):
    columns = ['call_id','_clip','manual_label','target','model_family','view']
    base = table.loc[table.model_family.eq('logistic_linear_tuned')].set_index(['view','target','call_id'])
    selected = table.loc[table.model_family.eq('spline_rms_tuned')]
    aligned = base.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
    values = dict(delta_change=selected.gain.to_numpy()-aligned.gain.to_numpy(),
        m0_loss_improvement=aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy(),
        m1_loss_improvement=aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy())
    parts = []
    for name,value in values.items():
        block = selected[columns].copy()
        block['model_family'] = 'spline_tuned_vs_linear_tuned'
        block['contrast'],block['value'] = name,value
        parts.append(block)
    return pd.concat(parts,ignore_index=True)


def load_tuned(args,data,maps,sha):
    source = argparse.Namespace(**vars(args))
    source.output_dir = args.run_dir
    scores,candidates,chosen = runner.inner_scores(source,data,maps,sha)
    errors = {}
    for name,actual,keys,columns in [
        ('inner_fold_scores',scores,runner.CELL+['C','inner_fold'],['n','loss_sum']),
        ('candidate_scores',candidates,runner.CELL+['C'],['n','pooled_logloss']),
        ('selected_C',chosen,runner.CELL,['selected_C','pooled_logloss'])]:
        old = runner.read(args.run_dir/(name+'.csv'))
        errors[name] = audit.compare_tables(old,actual,keys,columns)
    saved = runner.read(args.run_dir/'selected_C.csv').set_index(runner.CELL).sort_index()
    new = chosen.set_index(runner.CELL).sort_index()
    if not saved.tied_C_json.equals(new.tied_C_json):
        raise ValueError('Exact-tie metadata differ.')
    plan = runner.outer_plan(data,chosen)
    expected = set(plan.relative_path)
    observed = {p.relative_to(args.run_dir).as_posix() for p in (args.run_dir/'jobs/outer').glob('*.csv')}
    if expected!=observed:
        raise ValueError('Outer cache coverage differs.')
    blocks = []
    for tags,part in plan.groupby(['family','outer_fold','target'],sort=True):
        family,fold,target = tags
        test = data.loc[data.fold.eq(fold)]
        out = test[runner.b3.FIELDS].copy()
        for spec in part.to_dict('records'):
            frame = runner.read(args.run_dir/spec['relative_path'])
            runner.check_job(frame,spec,test,sha)
            out[spec['model']+'_pred'] = frame.set_index('call_id').loc[test.call_id,'pred'].to_numpy()
        out['model_family'],out['target'],out['y_true'] = family+'_tuned',target,test[target].to_numpy(int)
        out['n_train'],out['synthetic'],out['protocol_sha256'] = len(data)-len(test),False,sha
        runner.b3.check_record(out,test,family+'_tuned',fold,target,len(data)-len(test),sha)
        blocks.append(out)
    return pd.concat(blocks,ignore_index=True),scores,candidates,chosen,errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run','run-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.run_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError('Use a fresh external audit directory, not the fitted source directory.')
    with threadpool_limits(limits=4),runner.zero_fit_guard():
        protocol = json.loads((args.run_dir/'protocol.json').read_text(encoding='utf-8'))
        runner.b3.verify_hashes(protocol)
        data,_,_,_ = runner.b3.load_design(args)
        references,_ = runner.load_references(args,data)
        maps = runner.make_inner_maps(data)
        pd.testing.assert_frame_equal(runner.read(args.run_dir/'inner_clip_folds.csv'),maps,check_dtype=False)
        manifest = runner.read(args.run_dir/'all_oof_manifest.csv')
        if len(manifest)!=430 or manifest.path.duplicated().any():
            raise ValueError('Expected 400 new and30 historical manifest files.')
        if manifest.stage.eq('inner').sum()!=360 or manifest.stage.eq('outer').sum()!=40:
            raise ValueError('Wrong manifest stages.')
        for item in manifest.itertuples(index=False):
            if runner.shared.digest(Path(item.path))!=item.sha256:
                raise ValueError('Prediction source hash differs: '+item.path)
        sha = runner.shared.digest(args.run_dir/'protocol.json')
        tuned,scores,candidates,chosen,errors = load_tuned(args,data,maps,sha)
        raw = pd.concat([references,tuned],ignore_index=True)
        if len(raw)!=183450 or raw.duplicated(['model_family','target','call_id']).any():
            raise ValueError('Outer OOF coverage differs.')
        meta = raw.groupby(['target','call_id'])[
            ['y_true','_clip','fold','manual_label','duration_s','egemaps_valid']].nunique(dropna=False)
        if not meta.eq(1).all().all():
            raise ValueError('Cross-model metadata differ.')
        table = runner.b3.score_table(raw)
        reported = runner.read(args.run_dir/'comparison_metrics_and_ci.csv')
        actual = runner.b3.point_metrics(table)
        columns = [c for c in actual if c not in ['model_family','view','manual_label']]
        errors['point_metrics'] = audit.compare_tables(reported,actual,['model_family','view','manual_label'],columns)
        values = runner.contrast_values(table)
        intervals = runner.b3.clip_intervals(values,1000,7452)
        ci_columns = [c for c in intervals if '_ci_' in c or c.endswith('_finite_draws')]
        primary = intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast')
        errors['intervals'] = audit.compare_tables(reported,primary,['model_family','view','manual_label'],ci_columns)
        changes = paired_points(values.loc[values.contrast.ne('gain')],intervals.loc[intervals.contrast.ne('gain')])
        errors['paired_changes'] = audit.compare_tables(runner.read(args.run_dir/'paired_changes_vs_fixed.csv'),
            changes,runner.b3.KEYS,['n','delta_post','delta_pre','post_minus_pre']+ci_columns)
        direct = cross_family_values(table)
        direct_ci = runner.b3.clip_intervals(direct,1000,7452)
        direct_summary = paired_points(direct,direct_ci)
        folds = []
        for tags,part in direct.groupby(['model_family','contrast','view','target'],sort=True):
            # Original outer-fold IDs come from the fixed cohort, not new splitting.
            part = part.merge(data[['call_id','fold']],on='call_id',validate='many_to_one')
            for fold,block in part.groupby('fold'):
                for label,rows in [('overall',block)]+list(block.groupby('manual_label',sort=True)):
                    folds.append(dict(zip(['model_family','contrast','view','target'],tags),fold=fold,
                        manual_label=label,n=rows.call_id.nunique(),value=rows.value.mean()))
        diagnostics = runner.read(args.run_dir/'fit_diagnostics.csv')
        if len(diagnostics)!=400 or diagnostics.convergence_warning.any():
            raise ValueError('Fit count/convergence report differs from completed run.')
        args.output_dir.mkdir(parents=True)
        direct_summary.to_csv(args.output_dir/'paired_spline_vs_linear_tuned.csv',index=False)
        pd.DataFrame(folds).to_csv(args.output_dir/'paired_spline_vs_linear_tuned_by_fold.csv',index=False)
        chosen.groupby(['family','target','model','selected_C']).size().rename('n_folds').reset_index().to_csv(
            args.output_dir/'selected_C_counts.csv',index=False)
        keep = reported.loc[reported['view'].eq('all') & reported.manual_label.eq('overall')]
        keep.to_csv(args.output_dir/'overall_full_cohort_metrics.csv',index=False)
        result = dict(status='passed',model_fits=0,zero_fit_guard=True,manifest_files=430,
            inner_jobs=360,outer_jobs=40,outer_oof_rows=len(raw),metric_rows=len(reported),
            selected_settings=len(chosen),candidate_rows=len(candidates),inner_score_rows=len(scores),
            max_abs_errors=errors,source_protocol_sha256=sha,
            new_diagnostic='Posthoc paired RMS-spline-tuned versus linear-tuned on all labels/views,without deletion.',
            intervals='Frozen1000 paired clip draws,unadjusted and conditional on stored fits;not independent statistical validation.',
            acceptance_status='Local executor/provisional;no fresh semantic review or novelty verdict.')
        runner.shared.write_json(args.output_dir/'verification.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
