#!/usr/bin/env python
"""Feature-level semi-synthetic assays, not simulated audio or animal responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, ndtr

MARK_COLUMNS = ['mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean']
SCENARIOS = ['null_linear', 'null_nonlinear', 'signal_post', 'state_bilateral']
CONFIG = dict(expected_rate=.09, nonlinear_beta=1., signal_beta=1., state_beta=1.,
              proxy_correlation=.85, channel_noise=.05, copula_clip_weight=.35,
              mark_columns=MARK_COLUMNS, linear_log_duration=.35, linear_rms=.25,
              linear_history_count=.30, linear_start_margin=.15)
TRUTH = {
    'null_linear': dict(extra_post=False, extra_pre=False, direct_post_mark=False, oracle_parent=None),
    'null_nonlinear': dict(extra_post=False, extra_pre=False, direct_post_mark=False, oracle_parent='sim_h'),
    'signal_post': dict(extra_post=True, extra_pre=False, direct_post_mark=True, oracle_parent=None),
    'state_bilateral': dict(extra_post=True, extra_pre=True, direct_post_mark=False, oracle_parent='sim_state'),
}


def zscore(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError('Generator requires finite selected baseline covariates.')
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.)


def linear_score(data, side, config=CONFIG):
    prefix = 'pre' if side == 'post' else 'post'
    labels = sorted(data.manual_label.astype(str).unique())
    offsets = dict(zip(labels, np.linspace(-.4, .4, len(labels))))
    return (config['linear_log_duration'] * zscore(data.log_duration_s)
        + config['linear_rms'] * zscore(data.control_raw_rms_dbfs)
        + config['linear_history_count'] * zscore(data[f'{prefix}_400ms_all_count'])
        + config['linear_start_margin'] * zscore(data.clip_start_margin_s)
        + data.manual_label.astype(str).map(offsets).to_numpy())


def scores(data, scenario, mark, state, h, config=CONFIG):
    post, pre = linear_score(data, 'post', config), linear_score(data, 'pre', config)
    if scenario == 'null_nonlinear':
        post = post + config['nonlinear_beta'] * h
        pre = pre + config['nonlinear_beta'] * h
    elif scenario == 'signal_post':
        post = post + config['signal_beta'] * mark
    elif scenario == 'state_bilateral':
        post = post + config['state_beta'] * state
        pre = pre + config['state_beta'] * state
    elif scenario != 'null_linear':
        raise ValueError('Unknown scenario: ' + scenario)
    return post, pre


def probability(score, rate):
    intercept = brentq(lambda b: expit(score + b).mean() - rate, -100., 100.)
    return expit(score + intercept), float(intercept)


def generate(data, seed, scenario, config=CONFIG):
    if scenario not in SCENARIOS or data.call_id.duplicated().any() or data.empty:
        raise ValueError('Require known scenario and nonempty unique-call design.')
    streams = [np.random.default_rng(s) for s in
               np.random.SeedSequence([int(seed), SCENARIOS.index(scenario)]).spawn(7)]
    clip_codes, clips = pd.factorize(data['_clip'], sort=True)
    state = streams[0].normal(size=len(clips))[clip_codes]
    h = zscore(np.clip(zscore(data.control_raw_rms_dbfs), -2.5, 2.5) ** 2)
    rho = config['proxy_correlation']
    marker = streams[1].normal(size=len(data))
    if scenario == 'null_nonlinear':
        marker = rho * h + np.sqrt(1 - rho ** 2) * marker
    elif scenario == 'state_bilateral':
        marker = rho * state + np.sqrt(1 - rho ** 2) * marker
    channels = marker[:, None] + config['channel_noise'] * streams[2].normal(size=(len(data), 4))
    mark = channels.mean(axis=1)
    # Global covariate moments define the DGP, not predictive preprocessing.
    # The actual model's imputer/scaler/PCA are still fitted within train folds.
    location = data[MARK_COLUMNS].mean().to_numpy(float)
    scale = data[MARK_COLUMNS].std(ddof=0).to_numpy(float)
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise ValueError('No finite generator moments for mark channels.')
    scale = np.where(scale > 1e-12, scale, 1.)
    feature_values = location + scale * channels
    post_score, pre_score = scores(data, scenario, mark, state, h, config)
    p_post, b_post = probability(post_score, config['expected_rate'])
    p_pre, b_pre = probability(pre_score, config['expected_rate'])
    weight = config['copula_clip_weight']
    common_noise = streams[3].normal(size=len(clips))[clip_codes]
    uniforms = [ndtr(weight * common_noise + np.sqrt(1 - weight ** 2) * streams[i].normal(size=len(data)))
                for i in [4, 5]]
    packet = dict(call_ids=data.call_id.to_numpy(dtype=str), sim_feature_values=feature_values,
        sim_mark=mark, sim_h=h, sim_state=state, sim_p_post=p_post, sim_p_pre=p_pre,
        sim_y_post=(uniforms[0] < p_post).astype(np.int8),
        sim_y_pre=(uniforms[1] < p_pre).astype(np.int8),
        sim_intercepts=np.array([b_post, b_pre]))
    summary = dict(scenario=scenario, seed=int(seed), n_calls=len(data), n_clips=len(clips),
        expected_post=float(p_post.mean()), expected_pre=float(p_pre.mean()),
        positive_post=int(packet['sim_y_post'].sum()), positive_pre=int(packet['sim_y_pre'].sum()),
        truth=TRUTH[scenario], synthetic=True,
        warning='Feature-level fixed-design assay; no waveform, point-process or MeerKAT response truth.')
    return packet, summary


def toy_design(n=320):
    rng = np.random.default_rng(8)
    data = pd.DataFrame(dict(call_id=['toy' + str(i) for i in range(n)],
        _clip=['toyclip' + str(i // 4) for i in range(n)],
        manual_label=np.resize(['cc', 'sn', 'al', 'oth'], n),
        log_duration_s=rng.uniform(.01, .6, n), control_raw_rms_dbfs=rng.normal(-25, 8, n),
        pre_400ms_all_count=rng.poisson(1., n), post_400ms_all_count=rng.poisson(1.2, n),
        clip_start_margin_s=rng.uniform(.4, 9., n)))
    for col in MARK_COLUMNS:
        data[col] = rng.normal(size=n)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy-smoke', action='store_true', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    summaries = [generate(toy_design(), 431, scenario)[1] for scenario in SCENARIOS]
    assert all(abs(s['expected_post'] - .09) < 1e-10 for s in summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='technical_smoke_passed', complete_calibration=False,
                  scientific_results=False, model_fits=0, summaries=summaries)
    (args.output_dir / 'SMOKE_ONLY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Generator toy smoke passed; no model or scientific result.')


if __name__ == '__main__':
    main()
