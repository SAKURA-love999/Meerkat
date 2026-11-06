#!/usr/bin/env python
"""Change only eGeMAPS-valid sample membership in the existing librosa M1.

Models, feature values/columns, PCA, history and labels come from the existing
backbone. Full-population GroupKFold assignments are frozen before selection.
No eGeMAPS feature value is used as a predictor. --run-models is explicit so
the default command is a quick membership/retention audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import fit_meerkat_m1_acoustic_residual as backbone


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def retained_ids(args: argparse.Namespace) -> set[str]:
    qc = pd.read_csv(args.egemaps_qc, dtype={'call_id': str})
    if qc.duplicated(['call_id', 'amplitude_variant']).any():
        raise ValueError('Duplicate call/variant entries in eGeMAPS QC.')
    keep = qc.amplitude_variant.astype(str).eq(args.amplitude_variant) & backbone.bool_series(qc.valid)
    ids = set(qc.loc[keep, 'call_id'])
    if args.egemaps_valid_features:
        valid = pd.read_csv(args.egemaps_valid_features, usecols=['call_id'], dtype={'call_id': str})
        if valid.call_id.duplicated().any() or set(valid.call_id) != ids:
            raise ValueError('QC valid IDs differ from the existing eGeMAPS valid-only table.')
    return ids


def model_numeric_columns(data: pd.DataFrame, target: str) -> list[str]:
    prefix = 'pre_' if target == 'NF_post' else 'post_'
    history = [c for c in data if c.startswith(prefix) and (
        c.endswith('_count') or c.endswith('_s') or c.endswith('_same_as_focal'))]
    return ['duration_s', 'log_duration_s', 'clip_start_margin_s',
            'clip_end_margin_s', 'clip_center_s'] + history


def score_views(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, target), group in predictions.groupby(['window_ms', 'target'], sort=True):
        for scope, part in [('overall', group)] + list(group.groupby('manual_label', sort=True)):
            for view, mask, prefix in [
                ('full_train_all_test', np.ones(len(part), dtype=bool), 'full'),
                ('full_train_valid_test', part.egemaps_valid.to_numpy(bool), 'full'),
                ('valid_train_valid_test', part.egemaps_valid.to_numpy(bool), 'valid'),
                ('full_train_rejected_test', ~part.egemaps_valid.to_numpy(bool), 'full'),
            ]:
                selected = part.loc[mask]
                if selected.empty:
                    continue
                y = selected.y_true.to_numpy(int)
                s0 = backbone.score_predictions(y, selected[f'{prefix}_m0'].to_numpy(float))
                s1 = backbone.score_predictions(y, selected[f'{prefix}_m1'].to_numpy(float))
                rows.append(dict(window_ms=window, target=target, manual_label=scope,
                    view=view, n=len(selected), positive_n=int(y.sum()),
                    n_clips=selected['_clip'].nunique(), positive_rate=float(y.mean()),
                    m0_log_loss=s0['log_loss'], m1_log_loss=s1['log_loss'],
                    delta_log_loss_reduction=s0['log_loss']-s1['log_loss'],
                    m0_roc_auc=s0['roc_auc'], m1_roc_auc=s1['roc_auc'],
                    delta_roc_auc=s1['roc_auc']-s0['roc_auc'],
                    delta_average_precision=s1['average_precision']-s0['average_precision']))
    return pd.DataFrame(rows)


def make_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ['window_ms', 'manual_label', 'view']
    post = metrics.loc[metrics.target.eq('NF_post')].set_index(keys)
    pre = metrics.loc[metrics.target.eq('NF_pre')].set_index(keys)
    rows = []
    for key in post.index.intersection(pre.index):
        a, b = post.loc[key], pre.loc[key]
        rows.append(dict(zip(keys, key), post_n=int(a.n), pre_n=int(b.n),
            post_positive_n=int(a.positive_n), pre_positive_n=int(b.positive_n),
            delta_post=float(a.delta_log_loss_reduction), delta_pre=float(b.delta_log_loss_reduction),
            post_minus_pre=float(a.delta_log_loss_reduction-b.delta_log_loss_reduction)))
    return pd.DataFrame(rows)


def run_models(data: pd.DataFrame, cols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    tables = []
    for window, group in data.groupby('window_ms', sort=True):
        for target in ['NF_post', 'NF_pre']:
            censor = 'post_censored' if target == 'NF_post' else 'pre_censored'
            eligible = group.loc[~group[censor]].reset_index(drop=True)
            numeric = model_numeric_columns(eligible, target)
            base_cols = ['manual_label'] + numeric
            y = eligible[target].to_numpy(int)
            valid = eligible.egemaps_valid.to_numpy(bool)
            fold_ids = np.full(len(eligible), -1, dtype=int)
            preds = {name: np.full(len(eligible), np.nan) for name in
                     ['full_m0', 'full_m1', 'valid_m0', 'valid_m1']}
            cv = GroupKFold(n_splits=args.n_splits)
            for fold, (train, test) in enumerate(cv.split(eligible, y, eligible['_clip']), 1):
                fold_ids[test] = fold
                for population, tr, te in [('full', train, test),
                        ('valid', train[valid[train]], test[valid[test]])]:
                    if not len(te):
                        continue
                    if len(np.unique(y[tr])) < 2:
                        raise ValueError(f'{window}/{target}/fold{fold}: selected training set has one class.')
                    for name, use_audio in [('m0', False), ('m1', True)]:
                        model = backbone.make_pipeline(numeric, cols if use_audio else [],
                            use_audio, args.pca_dim if use_audio else None, args.random_state)
                        inputs = base_cols + (cols if use_audio else [])
                        model.fit(eligible.iloc[tr][inputs], y[tr])
                        preds[f'{population}_{name}'][te] = model.predict_proba(eligible.iloc[te][inputs])[:, 1]
            if not np.isfinite(preds['full_m1']).all() or not np.isfinite(preds['valid_m1'][valid]).all():
                raise ValueError('Missing OOF predictions.')
            table = eligible[['call_id', '_clip', 'manual_label', 'window_ms', 'egemaps_valid']].copy()
            table['target'] = target
            table['y_true'] = y
            table['fold'] = fold_ids
            for key, value in preds.items():
                table[key] = value
            tables.append(table)
            pd.concat(tables, ignore_index=True).to_csv(args.output_dir / 'selection_oof_predictions.csv', index=False)
            print(f'finished window={window:g} target={target} n={len(eligible)} valid={valid.sum()}', flush=True)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data_manifest/call_manifest.csv'))
    p.add_argument('--feature-csv', type=Path, default=Path('features/acoustic_features_capped_3000.csv'))
    p.add_argument('--egemaps-qc', type=Path, default=Path('features/egemaps_feature_validity_QC_capped_3000.csv'))
    p.add_argument('--egemaps-valid-features', type=Path,
        default=Path('features/egemaps_features_capped_3000_raw_valid_only.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('reports/librosa_egemaps_selection_control'))
    p.add_argument('--windows-ms', default='400,800,1500')
    p.add_argument('--history-windows-ms', default='400,800,1500')
    p.add_argument('--pca-dim', type=int, default=30)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--random-state', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--amplitude-variant', default='raw')
    p.add_argument('--run-models', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_npz = None
    args.feature_call_id_col = None
    args.drop_feature_regex = backbone.DEFAULT_DROP_FEATURE_REGEX
    args.temporal_control_mode = 'symmetric_history'
    args.clip_duration_s = 10.0
    for field in ['clip_col','call_id_col','label_col','start_col','end_col','duration_col','focal_col']:
        setattr(args, field, None)

    ids = retained_ids(args)
    features, cols = backbone.load_feature_table(args)
    features['call_id'] = features.call_id.astype(str)
    if features.call_id.duplicated().any():
        raise ValueError('Feature call IDs are not unique after amplitude selection.')
    meta = pd.read_csv(args.feature_csv, dtype={'call_id': str, 'source_id': str})
    meta = meta.loc[meta.amplitude_variant.eq(args.amplitude_variant)].copy()
    meta['egemaps_valid'] = meta.call_id.isin(ids)
    if set(ids) - set(meta.call_id):
        raise ValueError('eGeMAPS valid table includes calls absent from librosa pilot.')
    meta[['call_id', 'source_id', 'label', 'egemaps_valid']].to_csv(
        args.output_dir / 'selection_membership.csv', index=False)
    retention = meta.groupby('label', sort=True).agg(n_full=('call_id', 'size'),
        n_valid=('egemaps_valid', 'sum')).reset_index()
    retention['retained_fraction'] = retention.n_valid / retention.n_full
    retention.to_csv(args.output_dir / 'selection_retention_by_label.csv', index=False)
    summaries = meta.groupby(['label','egemaps_valid']).agg(
        n=('call_id','size'),duration_median=('duration_sec','median'),
        duration_mean=('duration_sec','mean'),rms_dbfs_median=('rms_dbfs','median'))
    summaries.to_csv(args.output_dir / 'selection_covariates_by_label.csv')
    # Preserve original CSV cells so standalone reloading does not introduce
    # a second floating-point formatting round trip. Model fitting below uses
    # the original in-memory feature table in both populations.
    original_cells = pd.read_csv(args.feature_csv, dtype=str, keep_default_na=False)
    original_cells.loc[original_cells.amplitude_variant.eq(args.amplitude_variant)
        & original_cells.call_id.isin(ids)].to_csv(
        args.output_dir / 'librosa_raw_egemaps_valid_only.csv', index=False)
    config = dict(n_full=len(meta), n_valid=int(meta.egemaps_valid.sum()), n_features=len(cols),
        acoustic_columns=cols, pca_dim=args.pca_dim, n_splits=args.n_splits,
        windows_ms=args.windows_ms, history_windows_ms=args.history_windows_ms,
        random_state=args.random_state, amplitude_variant=args.amplitude_variant,
        temporal_control_mode=args.temporal_control_mode, python=platform.python_version(),
        sklearn=sklearn_version, numpy=np.__version__, pandas=pd.__version__,
        backbone_changed=False, egemaps_predictors_used=False,
        input_hashes={str(path):digest(path) for path in [args.manifest,args.feature_csv,
            args.egemaps_qc, args.egemaps_valid_features, Path(backbone.__file__)]},
        caveats=['Filtering is a selected-population sensitivity, not causal attribution.',
            'POST/PRE retain legacy side-specific censoring; difference is descriptive.',
            'Feature summaries can still encode amplitude, duration and background.',
            'GroupKFold is frozen separately per window/target, reused across filtering.'])
    with (args.output_dir / 'selection_control_config.json').open('w', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2)
    print(retention.to_string(index=False), flush=True)
    if not args.run_models:
        print('Membership prepared. Add --run-models to fit the frozen models.', flush=True)
        return

    manifest = pd.read_csv(args.manifest)
    # All events within selected clips remain in the event stream, including
    # rejected focal calls and non-focal annotations used to construct history.
    clips = set(meta.source_id.astype(str))
    manifest = manifest.loc[manifest.source_id.astype(str).isin(clips)].copy()
    with threadpool_limits(limits=args.threads):
        targets = backbone.build_targets(manifest, args)
        data = targets.merge(features, on='call_id', how='inner', validate='many_to_one')
        data['egemaps_valid'] = data.call_id.isin(ids)
        predictions = run_models(data, cols, args)
    metrics = score_views(predictions)
    contrasts = make_contrasts(metrics)
    metrics.to_csv(args.output_dir / 'selection_metrics.csv', index=False)
    contrasts.to_csv(args.output_dir / 'selection_post_pre_contrasts.csv', index=False)
    focus = contrasts.loc[contrasts.manual_label.isin(['overall','sn','cc'])]
    focus.to_csv(args.output_dir / 'selection_focus_overall_sn_cc.csv', index=False)
    print(focus.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
