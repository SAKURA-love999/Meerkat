"""Recompute completed four-model OOF summaries without any model fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_baseline_capacity_b3 as check
import run_reverse_config_ablation as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','shared-dir','controls-dir','b1-run','b2-run','b3-run',
            'pca-run','audit-dir','config-run','config-analysis','run-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.run_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError('Use a new audit output outside the fitted directory.')
    with threadpool_limits(limits=4),runner.pca.fixed.zero_fit_guard():
        protocol_path = args.run_dir/'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        runner.b3.verify_hashes(protocol)
        data,original_cols,_,_ = runner.b3.load_design(args)
        cols = runner.remaining_columns(original_cols)
        runner.cfg.validate_audit(args.audit_dir,data)
        references,reference_manifest = runner.read_references(args,data,original_cols)
        plan = runner.make_plan(data,reference_manifest)
        pd.testing.assert_frame_equal(runner.b3.read(args.run_dir/'fit_plan.csv'),plan,check_dtype=False)
        manifest = runner.b3.read(args.run_dir/'all_oof_manifest.csv')
        if (len(manifest)!=60 or manifest.path.duplicated().any()
                or manifest.role.value_counts().to_dict()!=dict(existing_M0_Mconfig_M1=30,new_M78_only=30)):
            raise ValueError('Wrong four-model OOF manifest/roles.')
        for record in manifest.to_dict('records'):
            for p,h in [('path','sha256'),('metadata_path','metadata_sha256')]:
                if runner.b3.shared.digest(Path(record[p]))!=record[h]:
                    raise ValueError('OOF/metadata hash changed.')
        expected = set(plan.relative_path)
        jobs = args.run_dir/'jobs'
        if ({p.relative_to(args.run_dir).as_posix() for p in jobs.glob('*.csv')}!=expected
                or {p.relative_to(args.run_dir).with_suffix('.csv').as_posix() for p in jobs.glob('*.json')}!=expected):
            raise ValueError('Incomplete/extra jobs.')
        parts,metadata,changes = [],[],[]
        source_args = argparse.Namespace(output_dir=args.run_dir)
        sha = runner.b3.shared.digest(protocol_path)
        for spec in plan.to_dict('records'):
            train,test = data.loc[data.fold.ne(spec['fold'])],data.loc[data.fold.eq(spec['fold'])]
            if set(train['_clip'])&set(test['_clip']):
                raise ValueError('Clip overlap.')
            ref = runner.pca.reference_for(references,spec,test)
            frame,meta = runner.load_job(source_args,spec,train,test,ref,cols,sha)
            parts.append(frame)
            metadata.append(meta)
            diff = frame.m78_pred.to_numpy()-frame.m1_pred.to_numpy()
            changes.append(dict(model_family=spec['family'],fold=spec['fold'],target=spec['target'],
                changed_predictions=int(np.count_nonzero(diff)),max_abs_prediction_change=float(np.abs(diff).max())))
        raw = pd.concat(parts,ignore_index=True)
        if len(raw)!=110070 or raw.duplicated(['model_family','target','call_id']).any():
            raise ValueError('Wrong OOF coverage.')
        ids = ['y_true','_clip','fold','manual_label','duration_s','egemaps_valid']
        if not raw.groupby(['target','call_id'])[ids].nunique(dropna=False).eq(1).all().all():
            raise ValueError('Cross-model metadata differs.')
        pairs,identity_error,interaction = runner.make_pairs(raw)
        table = runner.b3.score_table(pairs)
        points = runner.decorate(runner.b3.point_metrics(table))
        points.rename(columns={c:c.replace('m0_','left_',1) if c.startswith('m0_') else c.replace('m1_','right_',1)
            for c in points if c.startswith('m0_') or c.startswith('m1_')},inplace=True)
        reported = runner.b3.read(args.run_dir/'paired_metrics_and_ci.csv')
        keys = ['model_family','comparison','view','manual_label']
        numeric = [c for c in points if c not in keys+['setting','left_model','right_model']]
        errors = dict(points=check.compare_tables(reported,points,keys,numeric))
        values = table[['call_id','_clip','manual_label','target','model_family','view']].copy()
        values['contrast'],values['value'] = 'gain',table.gain.to_numpy()
        ci = runner.decorate(runner.b3.clip_intervals(values,1000,7452).drop(columns='contrast'))
        ci_cols = [c for c in ci if '_ci_' in c or c.endswith('_finite_draws')]
        errors['intervals'] = check.compare_tables(reported,ci,keys,ci_cols)
        long = points.merge(ci[keys+ci_cols],on=keys,validate='one_to_one')
        iv = pd.concat([interaction.assign(view='all'),interaction.loc[interaction.egemaps_valid]
            .assign(view='original_valid_rows')],ignore_index=True)
        interaction_point = runner.paired_points(iv,runner.b3.clip_intervals(iv,1000,7452))
        errors['interaction'] = check.compare_tables(runner.b3.read(args.run_dir/'predictive_interaction_and_ci.csv'),
            interaction_point,['model_family','contrast','view','manual_label'],
            ['n','delta_post','delta_pre','post_minus_pre']+ci_cols)
        iv_lookup = interaction_point.set_index(['model_family','view','manual_label'])
        rows = []
        for tags,group in long.groupby(['model_family','view','manual_label'],sort=True):
            lookup = group.set_index('comparison')
            total,cfg,rest = lookup.loc['total'],lookup.loc['config_alone'],lookup.loc['remaining_alone']
            row = dict(zip(['model_family','view','manual_label'],tags))
            for c in ['n','n_clips','n_folds','post_positive_n','pre_positive_n']:
                row[c] = total[c]
            for side in ['post','pre']:
                for model,record,prefix in [('M0',total,'left'),('Mconfig',cfg,'right'),('M78',rest,'right'),('M1',total,'right')]:
                    for metric in ['logloss','auc','ap']:
                        row[model+'_'+side+'_'+metric] = record[prefix+'_'+side+'_'+metric]
            for comp in runner.COMPARISONS:
                for metric in ['delta_post','delta_pre','post_minus_pre']+ci_cols:
                    row[comp+'_'+metric] = lookup.loc[comp,metric]
            for metric in ['delta_post','delta_pre','post_minus_pre']+ci_cols:
                row['predictive_interaction_'+metric] = iv_lookup.loc[tags,metric]
            rows.append(row)
        wide = pd.DataFrame(rows)
        errors['four_model'] = check.compare_tables(runner.b3.read(args.run_dir/'four_model_metrics.csv'),
            wide,['model_family','view','manual_label'],[c for c in wide if c not in ['model_family','view','manual_label']])
        folds = []
        for tags,group in table.groupby(['model_family','view','fold'],sort=True):
            row = dict(zip(['model_family','view','fold'],tags))
            for target,side in zip(runner.TARGETS,['post','pre']):
                part = group.loc[group.target.eq(target)]
                row['left_'+side+'_logloss'],row['right_'+side+'_logloss'] = part.m0_loss.mean(),part.m1_loss.mean()
                row['delta_'+side] = part.gain.mean()
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            folds.append(row)
        fold_table = runner.decorate(pd.DataFrame(folds))
        errors['folds'] = check.compare_tables(runner.b3.read(args.run_dir/'fold_metrics.csv'),fold_table,
            ['model_family','comparison','view','fold'],[c for c in fold_table if c.endswith('_logloss')
                or c in ['delta_post','delta_pre','post_minus_pre']])
        args.output_dir.mkdir(parents=True)
        long.to_csv(args.output_dir/'recomputed_metrics_and_ci.csv',index=False)
        interaction_point.to_csv(args.output_dir/'recomputed_interaction_and_ci.csv',index=False)
        pd.DataFrame(changes).to_csv(args.output_dir/'M78_vs_M1_prediction_changes.csv',index=False)
        result = dict(status='passed',model_fits=0,zero_fit_guard=True,source_oof_files=60,source_metadata_files=60,
            raw_oof_rows=len(raw),metric_rows=len(long),interaction_rows=len(interaction_point),
            max_abs_errors=errors,per_call_identity_max_error=identity_error,source_protocol_sha256=sha,
            fit_start_utc=min(m['fit_start_utc'] for m in metadata),fit_end_utc=max(m['fit_end_utc'] for m in metadata),
            convergence_warning_jobs=sum(bool(m['convergence_warning']) for m in metadata),
            acceptance_status='Deterministic local audit only;no new review/novelty adjudication.')
        runner.b3.shared.write_json(args.output_dir/'verification.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
