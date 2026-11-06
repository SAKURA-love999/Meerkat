#!/usr/bin/env python
"""Describe calibration and clip influence in stored OOF; never fit a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONDITIONS = (
    '01_full_all,02_full_valid,03_valid_valid,20_raw_shared_full,'
    '20_raw_shared_full_validtest,21_normalized_shared_full,'
    '22_raw_shared_shape,23_normalized_shared_shape,24_normalized_original_full'
)
BIN_EDGES = np.array([0, .01, .02, .05, .1, .2, .4, .6, .8, .9, .95, .99, 1.])
KEYS = ['window_ms', 'condition', 'call_id', 'target']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def losses(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, int)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def validate(table):
    if table.empty or table.duplicated(KEYS).any():
        raise ValueError('Empty OOF or duplicate condition/call/target keys.')
    if not set(table.target).issubset({'NF_pre', 'NF_post'}):
        raise ValueError('Unknown target.')
    for col in ['m0_pred', 'm1_pred']:
        p = table[col].to_numpy(float)
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Invalid probabilities in ' + col)
    if not table.y_true.isin([0, 1]).all():
        raise ValueError('Nonbinary target.')
    if table.groupby('_clip').fold.nunique().max() != 1:
        raise ValueError('A clip has different held-out folds.')
    for _, g in table.groupby(['window_ms', 'condition']):
        post = g.loc[g.target.eq('NF_post')].set_index('call_id')
        pre = g.loc[g.target.eq('NF_pre')].set_index('call_id')
        if set(post.index) != set(pre.index):
            raise ValueError('PRE/POST evaluation calls differ.')
        cols = ['_clip', 'fold', 'manual_label']
        if not post[cols].sort_index().equals(pre[cols].sort_index()):
            raise ValueError('PRE/POST metadata differ.')


def calibration(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    ids = np.minimum(np.searchsorted(BIN_EDGES, p, side='right') - 1,
                     len(BIN_EDGES) - 2)
    rows = []
    for i in range(len(BIN_EDGES) - 1):
        use = ids == i
        n = int(use.sum())
        if n:
            rows.append(dict(bin_id=i, lower=BIN_EDGES[i], upper=BIN_EDGES[i + 1],
                n=n, positive_n=int(y[use].sum()), mean_probability=float(p[use].mean()),
                observed_rate=float(y[use].mean())))
    bins = pd.DataFrame(rows)
    ece = np.sum(bins.n * np.abs(bins.mean_probability - bins.observed_rate)) / len(y)
    ll = losses(y, p)
    metrics = dict(n=len(y), positive_n=int(y.sum()), positive_rate=float(y.mean()),
        mean_probability=float(p.mean()), calibration_bias=float(p.mean() - y.mean()),
        logloss=float(ll.mean()), brier=float(np.mean((p - y) ** 2)), ece=float(ece),
        positive_logloss=float(ll[y == 1].mean()) if y.sum() else np.nan,
        negative_logloss=float(ll[y == 0].mean()) if (y == 0).any() else np.nan)
    # A high-confidence error is a realized surprising outcome, not by itself
    # evidence that a probabilistic predictor is miscalibrated.
    wrong_confidence = np.where(y == 1, 1 - p, p)
    for threshold in [.9, .99]:
        use = wrong_confidence >= threshold
        name = str(threshold).replace('.', 'p')
        metrics['wrong_ge_' + name + '_n'] = int(use.sum())
        metrics['wrong_ge_' + name + '_loss_fraction'] = float(ll[use].sum() / ll.sum()) if ll.sum() else np.nan
    return metrics, bins


def clip_totals(part):
    p = part.copy()
    for model in ['m0', 'm1']:
        p[model + '_loss'] = losses(p.y_true, p[model + '_pred'])
    p['gain'] = p.m0_loss - p.m1_loss
    p['abs_event_gain'] = p.gain.abs()
    result = p.groupby(['_clip', 'fold'], sort=True).agg(
        n=('call_id', 'size'), positive_n=('y_true', 'sum'),
        m0_loss_sum=('m0_loss', 'sum'), m1_loss_sum=('m1_loss', 'sum'),
        gain_sum=('gain', 'sum'), abs_event_gain_sum=('abs_event_gain', 'sum')).reset_index()
    result['delta'] = result.gain_sum / result.n
    total_gain, total_n = result.gain_sum.sum(), result.n.sum()
    result['leave_one_clip_out_delta'] = (total_gain - result.gain_sum) / (total_n - result.n).replace(0, np.nan)
    mass = result.gain_sum.abs().sum()
    result['absolute_clip_gain_share'] = result.gain_sum.abs() / mass if mass else np.nan
    return result


def concentration(clips, fractions=(.01, .05, .1)):
    order = clips.assign(_mass=clips.gain_sum.abs()).sort_values(['_mass', '_clip'], ascending=[False, True])
    total, n = clips.gain_sum.sum(), clips.n.sum()
    mass = clips.gain_sum.abs().sum()
    rows = []
    for fraction in fractions:
        k = min(len(order), max(1, int(np.ceil(len(order) * fraction))))
        top, remaining = order.iloc[:k], order.iloc[k:]
        rows.append(dict(top_fraction=fraction, top_n_clips=k, n_clips=len(order), n=int(n),
            delta=float(total / n), total_signed_gain=float(total), absolute_clip_gain_mass=float(mass),
            positive_clip_gain_sum=float(clips.loc[clips.gain_sum.gt(0), 'gain_sum'].sum()),
            negative_clip_gain_sum=float(clips.loc[clips.gain_sum.lt(0), 'gain_sum'].sum()),
            top_absolute_gain_share=float(top.gain_sum.abs().sum() / mass) if mass else np.nan,
            top_signed_gain_sum=float(top.gain_sum.sum()), top_n_calls=int(top.n.sum()),
            delta_after_descriptive_removal=float(remaining.gain_sum.sum() / remaining.n.sum()) if len(remaining) else np.nan))
    return rows


def analyze(table, output):
    validate(table)
    metrics, bins, clips, errors = [], [], [], []
    for (window, condition, target), group in table.groupby(['window_ms', 'condition', 'target'], sort=True):
        for label, part in [('overall', group)] + list(group.groupby('manual_label', sort=True)):
            tags = dict(window_ms=window, condition=condition, target=target, manual_label=label)
            c = clip_totals(part)
            for key, val in tags.items():
                c[key] = val
            clips.append(c)
            for model in ['m0', 'm1']:
                m, b = calibration(part.y_true, part[model + '_pred'])
                metrics.append(dict(**tags, model=model, **m))
                for key, val in dict(**tags, model=model).items():
                    b[key] = val
                bins.append(b)
        # Write each error once, with its original label, not again as overall.
        for model in ['m0', 'm1']:
            confidence = np.where(group.y_true.eq(1), 1 - group[model + '_pred'], group[model + '_pred'])
            use = confidence >= .9
            e = group.loc[use, KEYS + ['_clip', 'fold', 'manual_label', 'duration_s', 'y_true', model + '_pred']].copy()
            e = e.rename(columns={model + '_pred': 'probability'})
            e['model'], e['wrong_confidence'] = model, confidence[use]
            e['logloss'] = losses(e.y_true, e.probability)
            errors.append(e)
    metric_table, clip_table = pd.DataFrame(metrics), pd.concat(clips, ignore_index=True)
    key = ['window_ms', 'condition', 'manual_label', '_clip', 'fold']
    post = clip_table.loc[clip_table.target.eq('NF_post')].set_index(key)
    pre = clip_table.loc[clip_table.target.eq('NF_pre')].set_index(key).reindex(post.index)
    if not np.array_equal(post.n, pre.n):
        raise ValueError('Clip-level PRE/POST counts differ.')
    direction = post[['n']].copy()
    direction['gain_sum'] = post.gain_sum - pre.gain_sum
    direction['delta'] = direction.gain_sum / direction.n
    direction = direction.reset_index()
    direction['target'] = 'POST_minus_PRE'
    for _, g in direction.groupby(['window_ms', 'condition', 'manual_label']):
        idx = g.index
        denominator = g.n.sum() - g.n
        direction.loc[idx, 'leave_one_clip_out_delta'] = (g.gain_sum.sum() - g.gain_sum) / denominator.replace(0, np.nan)
    rows = []
    grouping = ['window_ms', 'condition', 'manual_label', 'target']
    for source in [clip_table, direction]:
        for tags, g in source.groupby(grouping, sort=True):
            for row in concentration(g):
                rows.append(dict(zip(grouping, tags), **row))
    checks = []
    for tags, g in clip_table.groupby(grouping):
        w, condition, label, target = tags
        ref = metric_table.loc[metric_table.window_ms.eq(w) & metric_table.condition.eq(condition)
            & metric_table.manual_label.eq(label) & metric_table.target.eq(target)].set_index('model')
        expected = ref.loc['m0', 'logloss'] - ref.loc['m1', 'logloss']
        error = abs(g.gain_sum.sum() / g.n.sum() - expected)
        if error > 1e-12:
            raise ValueError('Clip contributions do not reconstruct gain.')
        checks.append(float(error))
    metric_table.to_csv(output / 'calibration_metrics.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(output / 'reliability_bins.csv', index=False)
    clip_table.to_csv(output / 'clip_loss_contributions.csv', index=False)
    direction.to_csv(output / 'clip_direction_contributions.csv', index=False)
    pd.DataFrame(rows).to_csv(output / 'concentration_sensitivity.csv', index=False)
    pd.concat(errors, ignore_index=True).to_csv(output / 'high_confidence_errors.csv', index=False)
    return dict(n_oof_rows=len(table), n_conditions=table.condition.nunique(),
                max_clip_reconstruction_error=max(checks), all_labels_reported=True)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = pd.read_csv(output / 'reliability_bins.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for i, label in enumerate(['overall', 'al']):
        for j, target in enumerate(['NF_post', 'NF_pre']):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)
            for condition, style in [('20_raw_shared_full', '-'), ('23_normalized_shared_shape', '--')]:
                for model, color in [('m0', '#346b9d'), ('m1', '#ba4c43')]:
                    g = bins.loc[bins.condition.eq(condition) & bins.manual_label.eq(label)
                        & bins.target.eq(target) & bins.model.eq(model)]
                    ax.plot(g.mean_probability, g.observed_rate, style, color=color,
                            marker='o', markersize=3, label=condition[:2] + ' ' + model)
            ax.set(title=label + ': ' + target, xlabel='Mean predicted probability',
                   ylabel='Observed fraction', xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=7)
    fig.savefig(output / 'calibration_primary.png', dpi=160)
    plt.close(fig)
    clips = pd.read_csv(output / 'clip_loss_contributions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, label in zip(axes, ['overall', 'al']):
        for target, color in [('NF_post', '#ba4c43'), ('NF_pre', '#346b9d')]:
            g = clips.loc[clips.condition.eq('20_raw_shared_full') & clips.manual_label.eq(label) & clips.target.eq(target)]
            mass = np.sort(g.gain_sum.abs().to_numpy())[::-1]
            if mass.sum():
                ax.plot(np.arange(1, len(mass) + 1) / len(mass), np.cumsum(mass) / mass.sum(), color=color, label=target)
        ax.set(title=label + ': absolute net clip gain', xlabel='Fraction of clips, ranked by absolute gain',
               ylabel='Cumulative fraction of absolute gain', xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.savefig(output / 'clip_gain_concentration.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-dir', type=Path, required=True)
    p.add_argument('--controls-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    sources = [args.shared_dir / 'oof_predictions.csv', args.controls_dir / 'oof_predictions_added.csv']
    requested = args.conditions.split(',')
    table = pd.concat([pd.read_csv(path, dtype={'_clip': str, 'call_id': str}) for path in sources], ignore_index=True)
    table = table.loc[table.window_ms.eq(400) & table.condition.isin(requested)].copy()
    if set(requested) != set(table.condition):
        raise ValueError('Missing requested conditions.')
    protocol = dict(input_hashes={str(path): digest(path) for path in sources},
        script_sha256=digest(__file__), conditions=requested, window_ms=400,
        probability_bins=BIN_EDGES.tolist(), wrong_confidence_thresholds=[.9, .99],
        clip_removal_fractions=[.01, .05, .1],
        interpretation='Exploratory descriptive influence, not deletion-adjusted primary performance or causality.',
        uncertainty='No new intervals. Existing conditional intervals remain in original result tables.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = args.output_dir / 'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Diagnostic input/protocol changed; use a new output directory.')
    frozen.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    summary = analyze(table, args.output_dir)
    # Check all recalculated losses against the original main result tables.
    original = pd.concat([pd.read_csv(args.shared_dir / 'paired_metrics_and_ci.csv'),
                          pd.read_csv(args.controls_dir / 'paired_metrics_and_ci.csv')], ignore_index=True)
    original = original.drop_duplicates(['window_ms', 'condition', 'manual_label'])
    metric = pd.read_csv(args.output_dir / 'calibration_metrics.csv')
    errors = []
    for row in metric.itertuples():
        ref = original.loc[original.window_ms.eq(row.window_ms) & original.condition.eq(row.condition)
                           & original.manual_label.eq(row.manual_label)].iloc[0]
        name = row.model + '_' + row.target.removeprefix('NF_') + '_logloss'
        errors.append(abs(row.logloss - ref[name]))
    summary['max_original_metric_error'] = max(errors)
    if max(errors) > 1e-12:
        raise ValueError('Recomputed OOF losses disagree with source metrics.')
    if args.plots:
        plot(args.output_dir)
    (args.output_dir / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
