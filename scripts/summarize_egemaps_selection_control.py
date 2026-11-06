#!/usr/bin/env python
"""Paired clip-bootstrap sensitivity of existing selection-control OOF gains.

Intervals condition on the fitted out-of-fold predictions. They do not include
training or fold-selection uncertainty. POST/PRE are summarized separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def event_gain(y: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    p0, p1 = np.clip(p0, eps, 1-eps), np.clip(p1, eps, 1-eps)
    return -y*np.log(p0)-(1-y)*np.log1p(-p0)+y*np.log(p1)+(1-y)*np.log1p(-p1)


def scope_summary(part: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict:
    valid = part.egemaps_valid.to_numpy(bool)
    y = part.y_true.to_numpy(int)
    full_gain = event_gain(y, part.full_m0.to_numpy(), part.full_m1.to_numpy())
    selected_gain = np.zeros(len(part))
    selected_gain[valid] = event_gain(y[valid], part.loc[valid,'valid_m0'].to_numpy(),
        part.loc[valid,'valid_m1'].to_numpy())
    block = pd.DataFrame(dict(clip=part['_clip'].to_numpy(), full_sum=full_gain,
        valid_full_sum=np.where(valid, full_gain, 0), selected_sum=selected_gain,
        full_n=np.ones(len(part)), valid_n=valid.astype(float))).groupby('clip').sum().to_numpy()
    sims = []
    for start in range(0, n_boot, 64):
        indices = rng.integers(0, len(block), size=(min(64,n_boot-start), len(block)))
        sums = block[indices].sum(axis=1)
        good = sums[:,4] > 0
        sums = sums[good]
        sims.append(np.column_stack([sums[:,0]/sums[:,3],sums[:,1]/sums[:,4],sums[:,2]/sums[:,4]]))
    boot = np.concatenate(sims)
    sums = block.sum(axis=0)
    obs = np.array([sums[0]/sums[3], sums[1]/sums[4], sums[2]/sums[4]])
    bootstrap_values = np.column_stack([boot,boot[:,1]-boot[:,0],boot[:,2]-boot[:,1],boot[:,2]-boot[:,0]])
    obs = np.r_[obs,obs[1]-obs[0],obs[2]-obs[1],obs[2]-obs[0]]
    names = ['full_train_all_test','full_train_valid_test','valid_train_valid_test',
        'evaluation_population_change','retraining_change_on_valid','total_filtering_change']
    result = dict(n=len(part),valid_n=int(valid.sum()),valid_positive_n=int(y[valid].sum()),
        n_clips=len(block),bootstrap_completed=len(boot))
    for i, name in enumerate(names):
        result[name] = obs[i]
        result[name+'_ci_low'],result[name+'_ci_high'] = np.quantile(bootstrap_values[:,i],[.025,.975])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('reports/librosa_egemaps_selection_control'))
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261101)
    args = p.parse_args()
    d = pd.read_csv(args.input_dir/'selection_oof_predictions.csv')
    rng = np.random.default_rng(args.seed)
    rows = []
    for (window,target), g in d.groupby(['window_ms','target'],sort=True):
        for label, part in [('overall',g)]+list(g.groupby('manual_label',sort=True)):
            if not part.egemaps_valid.any():
                continue
            rows.append(dict(window_ms=window,target=target,manual_label=label,
                **scope_summary(part,rng,args.n_bootstrap)))
    result = pd.DataFrame(rows)
    result.to_csv(args.input_dir/'selection_gain_decomposition_ci.csv',index=False)
    focus = result.loc[result.manual_label.isin(['overall','sn','cc']) & result.target.eq('NF_post')]
    print(focus[['window_ms','manual_label','valid_n','valid_positive_n',
        'full_train_all_test','full_train_valid_test','valid_train_valid_test',
        'valid_train_valid_test_ci_low','valid_train_valid_test_ci_high']].to_string(index=False))
    metadata = dict(n_bootstrap=args.n_bootstrap, seed=args.seed,
        uncertainty='Conditional on fitted OOF predictions, shared clip draws across arms.',
        causality='Descriptive population/retraining sensitivity; not causal attribution.',
        temporal='PRE and POST use separate legacy eligible populations. No paired directional test.')
    (args.input_dir/'selection_bootstrap_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
