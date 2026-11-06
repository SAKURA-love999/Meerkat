"""Read-only S1 OOF audit and all-mode cross-family paired diagnostics; zero fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_baseline_capacity_b3 as check
from audit_nested_logistic_regularization import paired_points
import run_pca_sensitivity as runner


def cross_family_values(table):
    columns = ['call_id','_clip','manual_label','target','model_family','view']
    parts = []
    for mode in runner.MODES:
        base = table.loc[table.model_family.eq('logistic_linear__'+mode)].set_index(['view','target','call_id'])
        for family in ['spline_rms','hist_gbdt']:
            selected = table.loc[table.model_family.eq(family+'__'+mode)]
            aligned = base.loc[pd.MultiIndex.from_frame(selected[['view','target','call_id']])]
            values = dict(delta_change=selected.gain.to_numpy()-aligned.gain.to_numpy(),
                m0_loss_improvement=aligned.m0_loss.to_numpy()-selected.m0_loss.to_numpy(),
                m1_loss_improvement=aligned.m1_loss.to_numpy()-selected.m1_loss.to_numpy())
            for name,value in values.items():
                block = selected[columns].copy()
                block['contrast'],block['value'] = name,value
                parts.append(block)
    return pd.concat(parts,ignore_index=True)


def fold_table(table):
    rows = []
    for tags,part in table.groupby(['model_family','view','fold'],sort=True):
        row = dict(zip(['model_family','view','fold'],tags))
        for target,side in zip(runner.TARGETS,['post','pre']):
            group = part.loc[part.target.eq(target)]
            for model in ['m0','m1']:
                row[model+'_'+side+'_logloss'] = group[model+'_loss'].mean()
            row['delta_'+side] = group.gain.mean()
        row['post_minus_pre'] = row['delta_post']-row['delta_pre']
        rows.append(row)
    return runner.decorate(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run','run-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.run_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError('Use a fresh audit output outside the fitted source directory.')
    with threadpool_limits(limits=4),runner.fixed.zero_fit_guard():
        protocol = json.loads((args.run_dir/'protocol.json').read_text(encoding='utf-8'))
        runner.b3.verify_hashes(protocol)
        data,cols,_,_ = runner.b3.load_design(args)
        references,reference_manifest = runner.fixed.load_references(args,data)
        plan = runner.make_plan(data,reference_manifest)
        pd.testing.assert_frame_equal(runner.b3.read(args.run_dir/'fit_plan.csv'),plan,check_dtype=False)
        manifest = runner.b3.read(args.run_dir/'all_oof_manifest.csv')
        if len(manifest)!=90 or manifest.path.duplicated().any():
            raise ValueError('Expected30 old and60 new OOF files.')
        for item in manifest.to_dict('records'):
            if runner.b3.shared.digest(Path(item['path']))!=item['sha256']:
                raise ValueError('Prediction source hash changed.')
            if pd.notna(item.get('metadata_path')):
                if runner.b3.shared.digest(Path(item['metadata_path']))!=item['metadata_sha256']:
                    raise ValueError('Transform source hash changed.')
        expected = set(plan.relative_path)
        jobs = args.run_dir/'jobs'
        if ({p.relative_to(args.run_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
                or {p.relative_to(args.run_dir).with_suffix('.csv').as_posix() for p in jobs.glob('*.json')}!=expected):
            raise ValueError('Unexpected/missing job coverage.')
        source_args = argparse.Namespace(output_dir=args.run_dir)
        sha = runner.b3.shared.digest(args.run_dir/'protocol.json')
        baseline = references.copy()
        baseline['model_family'] = baseline.model_family+'__pca30'
        parts,metadata = [baseline],[]
        for spec in plan.to_dict('records'):
            train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
            ref = runner.reference_for(references,spec,test)
            frame,meta = runner.load_job(source_args,spec,train,test,ref,cols,sha)
            parts.append(frame)
            metadata.append(meta)
        raw = pd.concat(parts,ignore_index=True)
        if len(raw)!=330210 or raw.duplicated(['model_family','target','call_id']).any():
            raise ValueError('Final paired OOF coverage differs.')
        paired_metadata = raw.groupby(['target','call_id'])[
            ['y_true','_clip','fold','manual_label','duration_s','egemaps_valid']].nunique(dropna=False)
        if not paired_metadata.eq(1).all().all():
            raise ValueError('Cross-setting metadata differs.')
        table = runner.b3.score_table(raw)
        actual_points = runner.decorate(runner.b3.point_metrics(table))
        reported = runner.b3.read(args.run_dir/'comparison_metrics_and_ci.csv')
        keys = ['model_family','pca_mode','view','manual_label']
        numeric = [c for c in actual_points if c not in keys+['setting']]
        errors = dict(point_metrics=check.compare_tables(reported,actual_points,keys,numeric))
        values = runner.contrast_values(table)
        intervals = runner.b3.clip_intervals(values,1000,7452)
        ci_cols = [c for c in intervals if '_ci_' in c or c.endswith('_finite_draws')]
        primary = runner.decorate(intervals.loc[intervals.contrast.eq('gain')].drop(columns='contrast'))
        errors['intervals'] = check.compare_tables(reported,primary,keys,ci_cols)
        changes = runner.decorate(paired_points(values.loc[values.contrast.ne('gain')],
            intervals.loc[intervals.contrast.ne('gain')]))
        old = runner.b3.read(args.run_dir/'paired_changes_and_ci.csv')
        errors['paired_changes'] = check.compare_tables(old,changes,keys+['contrast'],
            ['n','delta_post','delta_pre','post_minus_pre']+ci_cols)
        if (not np.array_equal(old.delta_post,old.m1_post_loss_improvement)
                or not np.array_equal(old.delta_pre,old.m1_pre_loss_improvement)):
            raise ValueError('Paired gain/M1-loss identity differs.')
        folds = fold_table(table)
        old_folds = runner.b3.read(args.run_dir/'fold_metrics.csv')
        errors['folds'] = check.compare_tables(old_folds,folds,['model_family','pca_mode','view','fold'],
            [c for c in folds if c.endswith('_logloss') or c in ['delta_post','delta_pre','post_minus_pre']])
        cross = cross_family_values(table)
        cross_ci = runner.b3.clip_intervals(cross,1000,7452)
        cross_points = runner.decorate(paired_points(cross,cross_ci)).drop(columns='reference_pca_mode')
        cross_points['reference_model_family'] = 'logistic_linear'
        cross_folds = []
        for tags,part in cross.groupby(['model_family','contrast','view','target'],sort=True):
            selected = part.merge(data[['call_id','fold']],on='call_id',validate='many_to_one')
            for fold,group in selected.groupby('fold'):
                for label,rows in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
                    cross_folds.append(dict(zip(['model_family','contrast','view','target'],tags),
                        fold=fold,manual_label=label,n=rows.call_id.nunique(),value=rows.value.mean()))
        full_meta = [m for m in metadata if m['pca_applied']]
        warnings_n = sum(bool(m['convergence_warning']) for m in metadata)
        if warnings_n!=0:
            raise ValueError('Convergence report differs from source completion.')
        args.output_dir.mkdir(parents=True)
        cross_points.to_csv(args.output_dir/'paired_families_vs_linear_by_pca.csv',index=False)
        runner.decorate(pd.DataFrame(cross_folds)).drop(columns='reference_pca_mode').to_csv(
            args.output_dir/'paired_families_vs_linear_by_pca_by_fold.csv',index=False)
        reported.loc[reported['view'].eq('all') & reported.manual_label.isin(['overall','al','cc','sn'])].to_csv(
            args.output_dir/'selected_full_cohort_summary.csv',index=False)
        result = dict(status='passed',model_fits=0,zero_fit_guard=True,source_oof_files=90,
            new_transform_files=60,outer_oof_rows=len(raw),metric_rows=len(reported),max_abs_errors=errors,
            source_protocol_sha256=sha,m0_identical_across_modes=True,
            new_full_pca_components=sorted({m['pca_components'] for m in full_meta}),
            new_full_fit_first30_variance_range=[min(m['full_fit_cumulative_variance_first30'] for m in full_meta),
                max(m['full_fit_cumulative_variance_first30'] for m in full_meta)],
            extra_diagnostic='Posthoc all-mode/all-label paired RMS-spline/HGB versus linear;no sample deletion.',
            intervals='Frozen1000 paired clips,unadjusted and conditional on existing OOF;not full-training uncertainty.',
            acceptance_status='Local executor/provisional;no new semantic review or novelty adjudication.')
        runner.b3.shared.write_json(args.output_dir/'verification.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
