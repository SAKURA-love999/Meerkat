"""Paired POST-only nonlinear null; no additional conditional acoustic signal."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

import calibration_dgp as base

SCENARIO = 'null_nonlinear_post_only'
SOURCE_SCENARIO = 'null_nonlinear'
CONFIG = dict(base.CONFIG)
TRUTH = {SCENARIO: dict(extra_post=False, extra_pre=False,
                       direct_post_mark=False, oracle_parent='sim_h')}


def scores(data, scenario, mark, state, h, config=CONFIG):
    if scenario != SCENARIO:
        raise ValueError('Require the declared asymmetric-null scenario.')
    return (base.linear_score(data, 'post', config) + config['nonlinear_beta'] * h,
            base.linear_score(data, 'pre', config))


def generate(data, seed, scenario, config=CONFIG):
    if scenario != SCENARIO:
        raise ValueError('Require the declared asymmetric-null scenario.')
    packet, info = base.generate(data, seed, SOURCE_SCENARIO, config)
    _, pre_score = scores(data, scenario, packet['sim_mark'], packet['sim_state'],
                          packet['sim_h'], config)
    p_pre, intercept = base.probability(pre_score, config['expected_rate'])
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(
        [int(seed), base.SCENARIOS.index(SOURCE_SCENARIO)]).spawn(7)]
    clip_codes, clips = pd.factorize(data['_clip'], sort=True)
    weight = config['copula_clip_weight']
    common = streams[3].normal(size=len(clips))[clip_codes]
    uniform_pre = ndtr(weight * common + np.sqrt(1 - weight ** 2)
                       * streams[5].normal(size=len(data)))
    packet['sim_p_pre'] = p_pre
    packet['sim_y_pre'] = (uniform_pre < p_pre).astype(np.int8)
    packet['sim_intercepts'][1] = intercept
    info.update(scenario=scenario, truth=TRUTH[SCENARIO],
                expected_pre=float(p_pre.mean()), positive_pre=int(packet['sim_y_pre'].sum()),
                pairing='Same source nonlinear-null draws and POST; only PRE score/intercept/targets change.')
    return packet, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toy-smoke', action='store_true', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = base.toy_design()
    old, _ = base.generate(data, 431, SOURCE_SCENARIO)
    new, info = generate(data, 431, SCENARIO)
    for name in ['sim_feature_values', 'sim_mark', 'sim_h', 'sim_state', 'sim_p_post', 'sim_y_post']:
        np.testing.assert_array_equal(new[name], old[name])
    assert abs(info['expected_pre'] - .09) < 1e-10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='technical_generator_smoke_passed', model_fits=0,
                  scientific_results=False, post_unchanged=True, summary=info)
    (args.output_dir / 'SMOKE_ONLY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Paired asymmetric-null generator smoke passed; no model fitting.')


if __name__ == '__main__':
    main()
