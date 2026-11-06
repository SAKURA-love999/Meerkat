"""Reverse-ablation fixtures; fake classifiers and preprocessing-only fits."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_reverse_config_ablation as runner
from test_config_source_attribution import model_fixture
from test_pca_sensitivity import fake_metadata


def fixture(folder):
    args,data,cols,controls,refs,manifest = model_fixture(folder)
    refs['mconfig_pred'] = .13
    return args,data,cols,controls,refs,manifest


def fake_meta(model,train,cols,spec,sha):
    meta = fake_metadata(model,train,cols,spec,sha)
    x = train[cols].to_numpy(float)
    med = np.nanmedian(x,axis=0)
    x = np.where(np.isnan(x),med,x)
    var = x.var(axis=0)
    meta.update(acoustic_imputer_statistics=med.tolist(),acoustic_scaler_mean=x.mean(axis=0).tolist(),
        acoustic_scaler_variance=var.tolist(),acoustic_scaler_scale=np.where(var==0,1,np.sqrt(var)).tolist())
    return meta


class ReverseTests(unittest.TestCase):
    def test_ordered_78_subset_and_invalid_schemas(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,cols,_,_,_ = fixture(folder)
            self.assertEqual(runner.remaining_columns(cols),cols[:78])
            for bad in [cols[:-1],cols+[cols[0]],['NF_post']+cols[1:],
                    [c for c in cols if c!='n_fft']+['wrong']]:
                with self.assertRaises(ValueError):
                    runner.remaining_columns(bad)

    def test_exact_preprocessing_subset_and_original_classifier(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,cols,controls,_,_ = fixture(folder)
            rest = runner.remaining_columns(cols)
            train,test = data.loc[data.fold.ne(1)].copy(),data.loc[data.fold.eq(1)].copy()
            train.loc[train.index[:4],rest[0]] = np.nan
            train[rest[-1]],test[rest[-1]] = np.nan,999.
            train[rest[-2]],test[rest[-2]] = 3.,300.
            for family in runner.FAMILIES:
                numeric = ['log_duration_s']+controls
                model = runner.make_model(family,numeric,rest)
                full = runner.pca.make_model(family,'no_pca',numeric,cols)
                for fitted,features in [(model,rest),(full,cols)]:
                    fitted.named_steps['preprocess'].fit_transform(train[['manual_label']+numeric+features])
                    self.assertFalse(hasattr(fitted.named_steps['clf'],'classes_'))
                a = model.named_steps['preprocess'].named_transformers_['acoustic']
                b = full.named_steps['preprocess'].named_transformers_['acoustic']
                active = b.named_steps['imputer'].get_feature_names_out(cols).tolist()
                select = [active.index(c) for c in rest if c in active]
                np.testing.assert_allclose(a.transform(test[rest]),b.transform(test[cols])[:,select],rtol=0,atol=0)
                self.assertNotIn('pca',a.named_steps)
                self.assertEqual(model.named_steps['clf'].get_params(),full.named_steps['clf'].get_params())
                spec = dict(family=family,pca_mode='no_pca')
                meta = runner.capture_metadata(model,train,rest,spec,'fixture')
                runner.check_metadata(meta,spec,train,rest,'fixture')
                self.assertEqual(meta['dropped_empty_columns'],[rest[-1]])
                with self.assertRaises(ValueError):
                    runner.check_metadata(dict(meta,acoustic_scaler_mean=[0.]*77),spec,train,rest,'fixture')

    def test_budget_and_zero_fit_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,cols,controls,_,manifest = fixture(folder)
            plan = runner.make_plan(data,manifest)
            self.assertEqual(len(plan),30)
            self.assertEqual(set(plan.model_role),{'M78_only'})
            self.assertEqual(len(set(plan.relative_path)),30)
            with runner.pca.fixed.zero_fit_guard(),self.assertRaises(RuntimeError):
                runner.make_model('logistic_linear',controls,runner.remaining_columns(cols)).fit(
                    data[['manual_label']+controls+cols[:78]],data.NF_post)

    def test_five_risk_pairs_and_both_interaction_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,_,raw,_ = fixture(folder)
            raw['m78_pred'] = .11
            pairs,error,values = runner.make_pairs(raw)
            self.assertEqual(len(pairs),5*len(raw))
            self.assertLess(error,1e-14)
            self.assertEqual(set(values.contrast),{'predictive_interaction'})
            loss = {m:runner.b3.binary_loss(raw.y_true,raw[p]) for m,p in runner.PREDICTIONS.items()}
            expected = loss['M78']-loss['M1']-loss['M0']+loss['Mconfig']
            np.testing.assert_allclose(values.value,expected,rtol=0,atol=1e-14)
            bad = raw.copy()
            bad.loc[0,'m78_pred'] = np.nan
            with self.assertRaises(ValueError):
                runner.make_pairs(bad)

    def test_thirty_mock_fits_resume_summary_and_reference_tamper(self):
        calls = []
        class Fake:
            named_steps = {}
            def fit(self,x,y):
                if set(runner.CONFIG)&set(x) or any(c in x for c in ['NF_post','NF_pre','call_id','_clip','fold','egemaps_valid']):
                    raise AssertionError('Configuration/outcome/ID in M78 input.')
                self.clips = {i//2 for i in x.index}
                calls.append(self)
                return self
            def predict_proba(self,x):
                if self.clips & {i//2 for i in x.index}:
                    raise AssertionError('Clip leakage.')
                return np.tile([.88,.12],(len(x),1))
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls,refs,manifest = fixture(folder)
            rest = runner.remaining_columns(cols)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=lambda *a:Fake()), \
                    patch.object(runner,'capture_metadata',side_effect=fake_meta), \
                    patch.object(runner.b3.shared,'model_numeric_columns',return_value=['log_duration_s']):
                counts = runner.fit_all(args,data,rest,controls,refs,manifest,'fixture')
            self.assertEqual(counts,dict(created_jobs=30,reused_jobs=0,model_fits=30))
            self.assertEqual(len(calls),30)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=AssertionError('No resume fit.')):
                again = runner.fit_all(args,data,rest,controls,refs,manifest,'fixture')
                runner.summarize(args,data,rest,refs,manifest,'fixture',n_draws=8)
            self.assertEqual(again,dict(created_jobs=0,reused_jobs=30,model_fits=0))
            metrics = runner.b3.read(args.output_dir/'paired_metrics_and_ci.csv')
            self.assertEqual(len(metrics),3*5*2*4)
            interaction = runner.b3.read(args.output_dir/'predictive_interaction_and_ci.csv')
            self.assertEqual(len(interaction),3*2*4)
            wide = runner.b3.read(args.output_dir/'four_model_metrics.csv')
            self.assertEqual(len(wide),3*2*4)
            for side in ['post','pre']:
                np.testing.assert_allclose(wide['total_delta_'+side],wide['remaining_alone_delta_'+side]+
                    wide['config_given_remaining_delta_'+side],rtol=0,atol=1e-14)
            result = json.loads((args.output_dir/'verification.json').read_text())
            self.assertFalse(result['causal_decomposition'])
            self.assertEqual(result['new_M78_jobs'],30)
            spec = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/spec['relative_path']
            frame = runner.b3.read(path)
            frame.loc[0,'mconfig_pred'] += .01
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,data,rest,refs,manifest,'fixture',n_draws=8)

    def test_missing_orphan_extra_jobs_are_not_summarized(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls,refs,manifest = fixture(folder)
            spec = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/spec['relative_path']
            path.parent.mkdir(parents=True)
            path.with_suffix('.json').write_text('{}')
            rest = runner.remaining_columns(cols)
            with self.assertRaisesRegex(ValueError,'Orphan'):
                runner.fit_all(args,data,rest,controls,refs,manifest,'fixture')
            with self.assertRaises(ValueError):
                runner.summarize(args,data,rest,refs,manifest,'fixture',n_draws=8)


if __name__=='__main__':
    unittest.main(verbosity=2)
