"""Fixed B1 capacity alternatives; no changes to frozen model helpers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import fit_meerkat_m1_acoustic_residual as base

RMS = 'control_raw_rms_dbfs'
FAMILIES = ['spline_rms', 'hist_gbdt']
SPLINE = dict(n_knots=5, degree=3, knots='uniform', extrapolation='constant', include_bias=False)
HGB = dict(learning_rate=.05, max_iter=200, max_depth=3, max_leaf_nodes=7,
           min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=0)


def make_model(family, numeric, acoustic):
    if family not in FAMILIES or RMS not in numeric:
        raise ValueError('Require a declared learner and measured RMS control.')
    forbidden = [c for c in numeric + acoustic if c.startswith('sim_') or c in
                 ['call_id', '_clip', 'fold', 'NF_post', 'NF_pre', 'egemaps_valid']]
    if forbidden:
        raise ValueError('Undeclared outcome/oracle/ID input: ' + str(forbidden))
    selected = [c for c in numeric if c != RMS] if family == 'spline_rms' else list(numeric)
    model = base.make_pipeline(selected, acoustic, bool(acoustic), 30 if acoustic else None, 0)
    model.named_steps['preprocess'].set_params(sparse_threshold=0.)
    if family == 'spline_rms':
        spline = Pipeline([('imputer', SimpleImputer(strategy='median')),
            ('basis', SplineTransformer(**SPLINE)), ('scaler', StandardScaler())])
        model.named_steps['preprocess'].transformers.append(('rms_spline', spline, [RMS]))
    else:
        model.set_params(clf=HistGradientBoostingClassifier(**HGB))
    return model


def toy_smoke(folder):
    rng = np.random.default_rng(18)
    n = 120
    numeric = ['duration_s', RMS]
    acoustic = ['a'+str(i) for i in range(35)]
    data = pd.DataFrame(dict(manual_label=np.resize(['cc','sn'],n),
        duration_s=rng.uniform(.02,.5,n), control_raw_rms_dbfs=rng.normal(-25,8,n)))
    for col in acoustic:
        data[col] = rng.normal(size=n)
    target = np.resize([0,1,0,0,1],n)
    result = []
    with threadpool_limits(limits=4):
        for family in FAMILIES:
            for audio in [False,True]:
                cols = ['manual_label']+numeric+(acoustic if audio else [])
                model = make_model(family,numeric,acoustic if audio else [])
                model.fit(data.iloc[:96][cols],target[:96])
                pred = model.predict_proba(data.iloc[96:][cols])[:,1]
                if not np.isfinite(pred).all() or not ((pred>=0)&(pred<=1)).all():
                    raise ValueError('Invalid toy probabilities.')
                result.append(dict(family=family,include_acoustic=audio,n_train=96,n_test=24))
    folder.mkdir(parents=True,exist_ok=True)
    value = dict(status='technical_smoke_passed',scientific_results=False,model_fits=4,
                 artificial_toy_only=True,checks=result)
    (folder/'SMOKE_ONLY.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Two capacity factories passed four tiny toy fits; no scientific result.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toy-smoke',action='store_true',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    toy_smoke(args.output_dir)
