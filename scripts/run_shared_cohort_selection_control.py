#!/usr/bin/env python
"""Fixed-cohort PRE/POST selection audit using the existing librosa backbone.

No acoustic feature extraction or encoder training occurs. All output, including
protocol and OOF records, is written to the explicitly supplied private folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import fit_meerkat_m1_acoustic_residual as base
from audit_egemaps_selection_on_librosa import digest, retained_ids, model_numeric_columns


EDGES = [0, .03, .05, .075, .1, .15, .25, .5, 1, np.inf]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def bin_ids(data: pd.DataFrame) -> pd.Series:
    bins = pd.cut(data.duration_s, bins=EDGES, right=False, labels=False)
    if bins.isna().any():
        raise ValueError('Duration outside the frozen nonnegative bins.')
    return data.manual_label.astype(str) + '|d' + bins.astype(int).astype(str)


def fit_pair(train: pd.DataFrame, test: pd.DataFrame, target: str,
             cols: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if set(train['_clip']) & set(test['_clip']):
        raise ValueError('Train/test clip overlap.')
    numeric = model_numeric_columns(train, target)
    if train[target].nunique() < 2:
        raise ValueError('Training target contains one class; cell cannot be fitted.')
    result = []
    for audio in [False, True]:
        inputs = ['manual_label'] + numeric + (cols if audio else [])
        model = base.make_pipeline(numeric, cols if audio else [], audio,
            args.pca_dim if audio else None, args.random_state)
        model.fit(train[inputs], train[target].to_numpy(int))
        result.append(model.predict_proba(test[inputs])[:, 1])
    return result[0], result[1]


def predict_record(test: pd.DataFrame, target: str, condition: str, fit_arm: str,
                   p0: np.ndarray, p1: np.ndarray, n_train: int, seed: int = -1) -> pd.DataFrame:
    out = test[['call_id','_clip','manual_label','duration_s','duration_bin',
                'egemaps_valid','fold','window_ms']].copy()
    out['target'], out['condition'], out['fit_arm'], out['seed'] = target, condition, fit_arm, seed
    out['n_train'], out['y_true'], out['m0_pred'], out['m1_pred'] = n_train, test[target].to_numpy(int), p0, p1
    return out


def scores(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in table.groupby(['window_ms','condition'], sort=True):
        for label, part in [('overall',g)]+list(g.groupby('manual_label',sort=True)):
            row = dict(window_ms=key[0], condition=key[1], manual_label=label,
                n=part.call_id.nunique(), n_clips=part['_clip'].nunique(), n_folds=part.fold.nunique())
            for target, short in [('NF_post','post'),('NF_pre','pre')]:
                t = part.loc[part.target.eq(target)]
                row[f'{short}_positive_n'] = int(t.y_true.sum())
                row[f'{short}_positive_rate'] = float(t.y_true.mean())
                a = base.score_predictions(t.y_true.to_numpy(int),t.m0_pred.to_numpy())
                b = base.score_predictions(t.y_true.to_numpy(int),t.m1_pred.to_numpy())
                row[f'm0_{short}_logloss'], row[f'm1_{short}_logloss'] = a['log_loss'], b['log_loss']
                row[f'delta_{short}'] = a['log_loss']-b['log_loss']
                row[f'delta_{short}_auc'] = b['roc_auc']-a['roc_auc']
                row[f'delta_{short}_ap'] = b['average_precision']-a['average_precision']
            row['post_minus_pre'] = row['delta_post']-row['delta_pre']
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # Shared clip resampling across PRE/POST and all arms. Intervals condition
    # on the stored OOF predictions; train-refit variability is not included.
    rng = np.random.default_rng(args.bootstrap_seed)
    out = []
    for window, group in table.groupby('window_ms',sort=True):
        for label, part in [('overall',group)]+list(group.groupby('manual_label',sort=True)):
            part = part.copy()
            y = part.y_true.to_numpy(float)
            for model in ['m0','m1']:
                prob = np.clip(part[f'{model}_pred'].to_numpy(float),1e-15,1-1e-15)
                part[model+'_loss'] = -y*np.log(prob)-(1-y)*np.log1p(-prob)
            clips = pd.Index(part['_clip'].unique())
            cells = sorted(set(zip(part.condition,part.target)))
            arrays = []
            for condition,target in cells:
                g = part.loc[part.condition.eq(condition)&part.target.eq(target)]
                b = g.groupby('_clip').agg(m0=('m0_loss','sum'),m1=('m1_loss','sum'),n=('call_id','size'))
                arrays.append(b.reindex(clips,fill_value=0).to_numpy())
            block = np.stack(arrays,axis=1)  # clip, condition/target, loss/count
            sims = []
            for start in range(0,args.n_bootstrap,32):
                weights = rng.multinomial(len(clips),np.full(len(clips),1/len(clips)),
                    size=min(32,args.n_bootstrap-start))
                sums = (weights @ block.reshape(len(clips),-1)).reshape(len(weights),len(cells),3)
                denominator = np.where(sums[:,:,2]>0,sums[:,:,2],np.nan)
                sims.append((sums[:,:,0]-sums[:,:,1])/denominator)
            draws = np.concatenate(sims)
            for condition in sorted(part.condition.unique()):
                ip, ir = cells.index((condition,'NF_post')), cells.index((condition,'NF_pre'))
                row = dict(window_ms=window,condition=condition,manual_label=label)
                for name,values in [('delta_post',draws[:,ip]),('delta_pre',draws[:,ir]),
                                    ('post_minus_pre',draws[:,ip]-draws[:,ir])]:
                    finite = values[np.isfinite(values)]
                    row[name+'_ci_low'],row[name+'_ci_high'] = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                # Paired differences compare the same clip draws across arms.
                custom_references = getattr(args, 'reference_map', {})
                if condition in custom_references: reference=custom_references[condition]
                elif condition=='02_full_valid': reference='01_full_all'
                elif condition=='03_valid_valid': reference='02_full_valid'
                elif condition.startswith('04_randomN_'): reference='03_valid_valid'
                elif condition.startswith('13_matched_'): reference='12_support_valid'
                else: reference=None
                row['change_reference'] = reference
                if reference:
                    rp, rr = cells.index((reference,'NF_post')),cells.index((reference,'NF_pre'))
                    for name,values in [('delta_post_change',draws[:,ip]-draws[:,rp]),
                        ('delta_pre_change',draws[:,ir]-draws[:,rr]),
                        ('post_minus_pre_change',(draws[:,ip]-draws[:,ir])-(draws[:,rp]-draws[:,rr]))]:
                        finite = values[np.isfinite(values)]
                        row[name+'_ci_low'],row[name+'_ci_high'] = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                out.append(row)
    return pd.DataFrame(out)


def summarize_and_write(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    with threadpool_limits(limits=args.threads):
        result=scores(table).merge(bootstrap_summary(table,args),
            on=['window_ms','condition','manual_label'],validate='one_to_one')
    lookup=result.set_index(['window_ms','condition','manual_label'])
    for index,row in result.iterrows():
        reference=row.get('change_reference')
        if pd.isna(reference): continue
        ref=lookup.loc[(row.window_ms,reference,row.manual_label)]
        for name in ['delta_post','delta_pre','post_minus_pre']:
            result.at[index,name+'_change']=row[name]-ref[name]
    result.to_csv(args.output_dir/'paired_metrics_and_ci.csv',index=False)
    random=result.loc[result.condition.str.startswith(('04_randomN_','13_matched_'))].copy()
    random['sampling_rule']=np.where(random.condition.str.startswith('04'),'same_N','label_duration_matched')
    random.groupby(['window_ms','manual_label','sampling_rule']).agg(
        n_draws=('condition','nunique'),post_gain_mean=('delta_post','mean'),
        post_gain_min=('delta_post','min'),post_gain_max=('delta_post','max'),
        pre_gain_mean=('delta_pre','mean'),pre_gain_min=('delta_pre','min'),pre_gain_max=('delta_pre','max'),
        contrast_mean=('post_minus_pre','mean'),contrast_min=('post_minus_pre','min'),
        contrast_max=('post_minus_pre','max')).reset_index().to_csv(
        args.output_dir/'random_retention_seed_sensitivity.csv',index=False)
    write_json(args.output_dir/'summary_config.json',dict(n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,threads=args.threads,
        ci_scope='Conditional on fitted OOF; same clip draws across all arms/PRE/POST.',
        monte_carlo_scope='Individual retention seeds and their range, not a population confidence interval.',
        multiple_comparisons='Exploratory, unadjusted intervals; no equivalence claim.'))
    return result


def verify_outputs(args: argparse.Namespace) -> dict:
    cohort=pd.read_csv(args.output_dir/'cohort_membership.csv')
    folds=pd.read_csv(args.output_dir/'clip_fold_mapping.csv').set_index('_clip').fold
    pred=pd.read_csv(args.output_dir/'oof_predictions.csv')
    train=pd.read_csv(args.output_dir/'training_membership.csv')
    assert cohort.groupby('_clip').fold.nunique().max()==1
    assert (pred['_clip'].map(folds)==pred.fold).all()
    assert (train['_clip'].map(folds)!=train.fold).all()
    assert (pred.groupby(['window_ms','condition','call_id']).target.nunique()==2).all()
    keys=['window_ms','call_id','target']
    full=pred.loc[pred.condition.eq('01_full_all')].set_index(keys)
    selected=pred.loc[pred.condition.eq('02_full_valid')].set_index(keys)
    assert full.loc[selected.index,['m0_pred','m1_pred']].equals(selected[['m0_pred','m1_pred']])
    valid=pred.loc[pred.condition.eq('03_valid_valid')].set_index(keys)
    assert set(valid.index)==set(selected.index)
    for condition,g in pred.groupby('condition'):
        if condition.startswith('04_randomN_'): assert set(g.set_index(keys).index)==set(valid.index)
    for (window,fold),g in train.groupby(['window_ms','fold']):
        ns=len(g.loc[g.fit_arm.eq('valid')])
        target_counts=g.loc[g.fit_arm.eq('support_valid')].groupby('duration_bin').size()
        for arm,part in g.groupby('fit_arm'):
            assert not part.call_id.duplicated().any()
            if arm.startswith('randomN_'): assert len(part)==ns
            if arm.startswith('matched_'): assert part.groupby('duration_bin').size().equals(target_counts)
    loader=argparse.Namespace(feature_csv=args.project_root/'features/acoustic_features_capped_3000.csv',
        feature_npz=None,feature_call_id_col=None,drop_feature_regex=base.DEFAULT_DROP_FEATURE_REGEX,amplitude_variant='raw')
    original,cols=base.load_feature_table(loader)
    cached=pd.read_pickle(args.output_dir/'cohort_with_history.pkl').drop_duplicates('call_id')
    reference=original.set_index('call_id').loc[cached.call_id,cols].to_numpy()
    assert np.allclose(reference,cached[cols].to_numpy(),rtol=0,atol=0,equal_nan=True)
    protocol=json.loads((args.output_dir/'protocol.json').read_text(encoding='utf-8'))
    assert all(digest(Path(p))==h for p,h in protocol['input_hashes'].items())
    result=dict(status='passed',common_calls=len(cohort),valid_calls=int(cohort.egemaps_valid.sum()),
        shared_clip_folds=True,all_training_clips_exclude_heldout_fold=True,pre_post_same_calls_per_cell=True,
        condition_02_reuses_exact_01_predictions=True,condition_02_03_test_calls_identical=True,
        random_N_matches_valid_train=True,matched_label_duration_quotas_identical=True,
        original_librosa_values_unchanged=True,input_hashes_unchanged=True,
        current_runner_sha256=digest(Path(__file__)))
    write_json(args.output_dir/'verification.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(r'C:\学校\MeerKAT\repertoire_geometry'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--windows-ms',default='400')
    p.add_argument('--fold-map',type=Path,default=None,help='Reuse an existing primary cohort clip-fold CSV for sensitivity windows.')
    p.add_argument('--random-seeds',default='20261101,20261102,20261103')
    p.add_argument('--n-bootstrap',type=int,default=1000)
    p.add_argument('--bootstrap-seed',type=int,default=20261104)
    p.add_argument('--min-valid-train-per-stratum',type=int,default=5)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--run-models',action='store_true')
    p.add_argument('--summarize-only',action='store_true',help='Reuse stored OOF; no fitting or target reconstruction.')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.verify_only:
        print(json.dumps(verify_outputs(args),indent=2),flush=True)
        return
    if args.summarize_only:
        summarize_and_write(pd.read_csv(args.output_dir/'oof_predictions.csv'),args)
        return
    args.manifest=args.project_root/'data_manifest/call_manifest.csv'
    args.feature_csv=args.project_root/'features/acoustic_features_capped_3000.csv'
    args.egemaps_qc=args.project_root/'features/egemaps_feature_validity_QC_capped_3000.csv'
    args.egemaps_valid_features=args.project_root/'features/egemaps_features_capped_3000_raw_valid_only.csv'
    args.feature_npz=None
    args.feature_call_id_col=None
    args.amplitude_variant='raw'
    args.drop_feature_regex=base.DEFAULT_DROP_FEATURE_REGEX
    args.temporal_control_mode='symmetric_history'
    args.history_windows_ms='400,800,1500'
    args.pca_dim,args.n_splits,args.random_state,args.clip_duration_s=30,5,0,10.
    for k in ['clip_col','call_id_col','label_col','start_col','end_col','duration_col','focal_col']:
        setattr(args,k,None)
    valid_ids=retained_ids(args)
    feat,cols=base.load_feature_table(args)
    if len(cols)!=81 or feat.call_id.duplicated().any():
        raise ValueError('Expected original 81-feature unique-call librosa table.')
    meta=pd.read_csv(args.feature_csv,usecols=['call_id','source_id','amplitude_variant'],dtype=str)
    meta=meta.loc[meta.amplitude_variant.eq('raw')]
    manifest=pd.read_csv(args.manifest)
    manifest=manifest.loc[manifest.source_id.astype(str).isin(set(meta.source_id))].copy()
    cohort_path=args.output_dir/'cohort_with_history.pkl'
    hashes={str(x):digest(x) for x in [args.manifest,args.feature_csv,args.egemaps_qc,
        args.egemaps_valid_features,Path(base.__file__)]}
    if args.fold_map: hashes[str(args.fold_map)]=digest(args.fold_map)
    frozen_path=args.output_dir/'protocol.json'
    if frozen_path.exists():
        old=json.loads(frozen_path.read_text(encoding='utf-8'))
        if old['input_hashes']!=hashes or old['windows_ms']!=args.windows_ms:
            raise ValueError('Inputs/window changed: use another output directory.')
    if cohort_path.exists():
        data=pd.read_pickle(cohort_path)
    else:
        targets=base.build_targets(manifest,args)
        data=targets.merge(feat,on='call_id',how='inner',validate='many_to_one')
        # Map is always defined on the primary 400-ms common eligible cohort,
        # independent of later target, label, valid status or sensitivity window.
        eligible=(data['_start']>=.4)&(data['_end']<=9.6)
        data=data.loc[eligible].copy()
        primary=data.drop_duplicates('call_id').copy()
        if args.fold_map:
            existing=pd.read_csv(args.fold_map,dtype={'_clip':str})
            if existing['_clip'].duplicated().any() or set(existing['_clip'])!=set(primary['_clip']):
                raise ValueError('Supplied fold map does not exactly cover the common primary clips.')
            fold=primary['_clip'].map(existing.set_index('_clip').fold).to_numpy(int)
            if set(fold)!={1,2,3,4,5}: raise ValueError('Expected all five frozen folds.')
        else:
            fold=np.full(len(primary),-1,dtype=int)
            cv=GroupKFold(n_splits=5)
            for f,(_,te) in enumerate(cv.split(primary,groups=primary['_clip']),1): fold[te]=f
        mapping=pd.DataFrame({'call_id':primary.call_id.to_numpy(),'fold':fold})
        data=data.merge(mapping,on='call_id',validate='many_to_one')
        data['egemaps_valid']=data.call_id.isin(valid_ids)
        data['duration_bin']=bin_ids(data)
        data.to_pickle(cohort_path)
        data[['_clip','fold']].drop_duplicates().sort_values('_clip').to_csv(args.output_dir/'clip_fold_mapping.csv',index=False)
    if data.groupby('_clip').fold.nunique().max()!=1:
        raise ValueError('A clip has more than one shared fold.')
    seeds=[int(x) for x in args.random_seeds.split(',')]
    write_json(frozen_path,dict(input_hashes=hashes,windows_ms=args.windows_ms,primary_window_ms=400,
        eligible='start>=0.4 and end<=9.6; both sides observable',n_features=81,pca_dim=30,
        model='Original make_pipeline; train-fold imputer/scaler/PCA refit per arm',
        history_windows_ms=args.history_windows_ms,random_seeds=seeds,
        random_rule='uniform without replacement within each training fold, same total N as valid training',
        matched_rule='within label-duration bin, same N per bin as valid training; only bins with >=minimum valid train calls',
        duration_edges=['inf' if np.isinf(x) else x for x in EDGES],min_valid_train_per_stratum=args.min_valid_train_per_stratum,
        bootstrap='shared clip draws conditional on fitted OOF; seed ranges describe limited training deletion MC',
        status='exploratory follow-up; not preregistration',practical_margin=None,
        equivalence_claim_allowed=False,legacy_outcome='clip-local non-focal annotated activity proxy'))
    cohort=data.drop_duplicates('call_id')
    cohort[['call_id','_clip','fold','manual_label','duration_s','duration_bin','egemaps_valid']].to_csv(args.output_dir/'cohort_membership.csv',index=False)
    print(cohort.groupby(['manual_label','egemaps_valid']).size().to_string(),flush=True)
    if not args.run_models:
        print('Prepared and frozen. Add --run-models for CPU fits.',flush=True)
        return
    predictions=[]
    counts=[]
    support_tables=[]
    train_memberships=[]
    with threadpool_limits(limits=args.threads):
        for window,w in data.groupby('window_ms',sort=True):
            w=w.loc[~w.pre_censored&~w.post_censored].copy()
            for fold in range(1,6):
                tr=w.loc[w.fold.ne(fold)]
                te=w.loc[w.fold.eq(fold)]
                s=tr.loc[tr.egemaps_valid]
                v=te.loc[te.egemaps_valid]
                valid_counts=s.groupby('duration_bin').size()
                cells=set(valid_counts.index[valid_counts>=args.min_valid_train_per_stratum])
                support=tr.loc[tr.duration_bin.isin(cells)]
                ss=s.loc[s.duration_bin.isin(cells)]
                st=te.loc[te.duration_bin.isin(cells)]
                sv=st.loc[st.egemaps_valid]
                coverage=tr.groupby('duration_bin').size().rename('n_full_train').to_frame()
                coverage['n_valid_train']=valid_counts
                coverage['supported']=coverage.index.isin(cells)
                coverage['fold'],coverage['window_ms']=fold,window
                support_tables.append(coverage.reset_index())
                # Memberships are sampled once and reused for PRE and POST.
                plans=[('full',tr,te,-1),('valid',s,v,-1)]
                if len(sv): plans += [('support_full',support,st,-1),('support_valid',ss,sv,-1)]
                for seed in seeds:
                    rng=np.random.default_rng(seed+1009*fold+int(window))
                    r=tr.iloc[rng.choice(len(tr),size=len(s),replace=False)]
                    plans.append((f'randomN_{seed}',r,v,seed))
                    if len(sv):
                        selected=[]
                        for cell,n in ss.groupby('duration_bin').size().items():
                            pool=tr.loc[tr.duration_bin.eq(cell)]
                            selected.append(pool.iloc[rng.choice(len(pool),size=int(n),replace=False)])
                        matched=pd.concat(selected)
                        plans.append((f'matched_{seed}',matched,sv,seed))
                for arm,train,test,seed in plans:
                    if test.empty: continue
                    tm=train[['call_id','_clip','manual_label','duration_s','duration_bin']].copy()
                    tm['fold'],tm['fit_arm'],tm['window_ms']=fold,arm,window
                    train_memberships.append(tm)
                    for split,part in [('train',train),('test',test)]:
                        group=part.groupby(['manual_label','duration_bin']).agg(n=('call_id','size'),
                            n_clips=('_clip','nunique'),duration_min=('duration_s','min'),
                            duration_median=('duration_s','median'),duration_max=('duration_s','max'),
                            post_positive_n=('NF_post','sum'),pre_positive_n=('NF_pre','sum')).reset_index()
                        group['fold'],group['fit_arm'],group['split'],group['window_ms']=fold,arm,split,window
                        group['post_positive_rate']=group.post_positive_n/group.n
                        group['pre_positive_rate']=group.pre_positive_n/group.n
                        counts.append(group)
                    for target in ['NF_post','NF_pre']:
                        p0,p1=fit_pair(train,test,target,cols,args)
                        condition={'full':'01_full_all','valid':'03_valid_valid',
                            'support_full':'11_support_full','support_valid':'12_support_valid'}.get(arm,
                            '04_'+arm if arm.startswith('randomN') else '13_'+arm)
                        record=predict_record(test,target,condition,arm,p0,p1,len(train),seed)
                        predictions.append(record)
                        if arm in ['full','support_full']:
                            sub=record.loc[record.egemaps_valid].copy()
                            sub['condition']='02_full_valid' if arm=='full' else '11b_support_full_valid'
                            predictions.append(sub)
                print(f'completed window={window:g} fold={fold}/5 arms={len(plans)}',flush=True)
                pd.concat(predictions,ignore_index=True).to_csv(args.output_dir/'oof_predictions.csv',index=False)
        table=pd.concat(predictions,ignore_index=True)
        metrics=summarize_and_write(table,args)
    pd.concat(counts,ignore_index=True).to_csv(args.output_dir/'fold_label_duration_counts.csv',index=False)
    pd.concat(support_tables,ignore_index=True).to_csv(args.output_dir/'label_duration_support.csv',index=False)
    pd.concat(train_memberships,ignore_index=True).to_csv(args.output_dir/'training_membership.csv',index=False)
    print(metrics.loc[metrics.manual_label.isin(['overall','sn','cc']),
        ['window_ms','condition','manual_label','n','post_positive_n','delta_post','delta_pre','post_minus_pre']].to_string(index=False))


if __name__=='__main__':
    main()
