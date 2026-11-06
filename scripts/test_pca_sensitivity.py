"""S1 technical fixtures: fake classifiers and tiny preprocessing-only fits."""
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

import run_pca_sensitivity as runner


def fixture(folder):
    n = 180
    i = np.arange(n)
    data = pd.DataFrame(dict(call_id=['c'+str(j) for j in i],
        _clip=['clip'+str(j//2) for j in i],fold=(i//2)%5+1,
        manual_label=np.resize(['cc','sn','al'],n),duration_s=.1+i/10000,
        duration_bin='fixture',egemaps_valid=i%4!=0,window_ms=400,
        NF_post=(i%7==0).astype(int),NF_pre=(i%9==0).astype(int),
        log_duration_s=np.log1p(.1+i/10000),control_raw_rms_dbfs=-20+i/100))
    cols = ['audio'+str(k) for k in range(35)]
    rng = np.random.default_rng(44)
    for col in cols:
        data[col] = rng.normal(size=n)
    parts,manifest = [],[]
    for family in runner.FAMILIES:
        for fold in range(1,6):
            test = data.loc[data.fold.eq(fold)]
            for target in runner.TARGETS:
                out = test[runner.b3.FIELDS].copy()
                out['model_family'],out['target'],out['y_true'] = family,target,test[target]
                out['n_train'],out['m0_pred'],out['m1_pred'],out['synthetic'] = 144,.15,.14,False
                parts.append(out)
                manifest.append(dict(model_family=family,fold=fold,target=target,
                    path=f'FIXTURE_{family}_{fold}_{target}.csv',sha256='FIXTURE_REFERENCE_SHA'))
    args = Namespace(output_dir=Path(folder))
    return args,data,cols,[runner.models.RMS],pd.concat(parts,ignore_index=True),manifest


def fake_metadata(model,train,cols,spec,protocol_sha):
    full = spec['pca_mode']=='pca_full'
    return dict(spec=spec,protocol_sha256=protocol_sha,input_columns=cols,
        retained_columns=cols,dropped_empty_columns=[],n_train=len(train),
        n_input=len(cols),n_after_imputer=len(cols),acoustic_output_dim=len(cols),
        scaler_n_samples_min=len(train),scaler_n_samples_max=len(train),
        pca_applied=full,pca_components=len(cols) if full else 0,
        pca_n_samples=len(train) if full else None,whiten=False if full else None,
        explained_variance_ratio=[1/len(cols)]*len(cols) if full else [],
        explained_variance=[1.]*len(cols) if full else [],singular_values=[1.]*len(cols) if full else [],
        full_fit_cumulative_variance_first30=min(30,len(cols))/len(cols) if full else None,
        orthogonality_max_abs_error=0. if full else None,constant_scaled_columns=[],
        clf_config=runner.make_model(spec['family'],spec['pca_mode'],[runner.models.RMS],cols).named_steps['clf'].get_params(),
        fit_start_utc='2026-01-01T00:00:00+00:00',fit_end_utc='2026-01-01T00:00:01+00:00',
        fit_seconds=1.,convergence_warning=False,warning_messages=[],n_iter_max=1)


class PCATests(unittest.TestCase):
    def test_factory_modes_and_fixed_classifier_parameters(self):
        for family in runner.FAMILIES:
            for mode in runner.NEW_MODES:
                model = runner.make_model(family,mode,['duration_s',runner.models.RMS],['a'+str(i) for i in range(35)])
                acoustic = next(t[1] for t in model.named_steps['preprocess'].transformers if t[0]=='acoustic')
                self.assertEqual(list(acoustic.named_steps)[:2],['imputer','scaler'])
                if mode=='pca_full':
                    pca = acoustic.named_steps['pca']
                    self.assertIsNone(pca.n_components)
                    self.assertFalse(pca.whiten)
                    self.assertEqual(pca.svd_solver,'full')
                else:
                    self.assertNotIn('pca',acoustic.named_steps)
                clf = model.named_steps['clf']
                if family!='hist_gbdt':
                    self.assertEqual(clf.C,1.)
                    self.assertEqual(clf.penalty,'l2')
                    self.assertEqual(clf.max_iter,2000)
                else:
                    for name,value in runner.models.HGB.items():
                        self.assertEqual(clf.get_params()[name],value)
        for bad in ['NF_post','NF_pre','fold','call_id','sim_x','egemaps_valid']:
            with self.assertRaises(ValueError):
                runner.make_model('logistic_linear','no_pca',[runner.models.RMS,bad],['a'])

    def test_preprocessing_only_train_support_metadata_and_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,cols,controls,_,manifest = fixture(folder)
            train,test = data.loc[data.fold.ne(1)].copy(),data.loc[data.fold.eq(1)].copy()
            train[cols[-1]],test[cols[-1]] = np.nan,999.
            train[cols[-2]],test[cols[-2]] = 3.,300.
            train.loc[train.index[:4],cols[0]] = np.nan
            for family in runner.FAMILIES:
                transformed = {}
                for mode in runner.NEW_MODES:
                    spec = runner.make_plan(data,manifest).query('family == @family and fold == 1 and target == "NF_post" and pca_mode == @mode').to_dict('records')[0]
                    model = runner.make_model(family,mode,['log_duration_s']+controls,cols)
                    inputs = ['manual_label','log_duration_s']+controls+cols
                    model.named_steps['preprocess'].fit_transform(train[inputs])
                    self.assertFalse(hasattr(model.named_steps['clf'],'classes_'))
                    meta = runner.transform_metadata(model,train,cols,spec,'fixture')
                    runner.check_metadata(meta,spec,train,cols,'fixture')
                    self.assertEqual(meta['dropped_empty_columns'],[cols[-1]])
                    self.assertEqual(meta['n_after_imputer'],34)
                    self.assertEqual(meta['acoustic_output_dim'],34)
                    with self.assertRaises(ValueError):
                        runner.check_metadata(dict(meta,scaler_n_samples_max=len(data)),spec,train,cols,'fixture')
                    if mode=='pca_full':
                        with self.assertRaises(ValueError):
                            runner.check_metadata(dict(meta,pca_components=30),spec,train,cols,'fixture')
                    acoustic = model.named_steps['preprocess'].named_transformers_['acoustic']
                    self.assertAlmostEqual(acoustic.named_steps['scaler'].mean_[0],
                        train[cols[0]].fillna(train[cols[0]].median()).mean())
                    self.assertIn(cols[-2],meta['constant_scaled_columns'])
                    transformed[mode] = acoustic.transform(test[cols])
                distances = {k:np.linalg.norm(v[0]-v[1]) for k,v in transformed.items()}
                self.assertAlmostEqual(distances['pca_full'],distances['no_pca'],places=9)

    def test_plan_budget_and_zero_fit_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,_,_,_,manifest = fixture(folder)
            plan = runner.make_plan(data,manifest)
            self.assertEqual(len(plan),60)
            self.assertTrue(plan.relative_path.is_unique)
            self.assertEqual(set(plan.pca_mode),{'pca_full','no_pca'})
            with self.assertRaisesRegex(RuntimeError,'zero-fit'):
                with runner.fixed.zero_fit_guard():
                    from sklearn.decomposition import PCA
                    PCA().fit([[0,1],[1,0]])

    def test_fake_60_fit_end_to_end_resume_and_summary(self):
        objects = []
        class FakeModel:
            def __init__(self,mode):
                self.mode = mode
            def fit(self,x,y):
                self.rows = set(x.index)
                self.clips = {i//2 for i in x.index}
                self.p = .11 if self.mode=='pca_full' else .12
                if any(c in x for c in ['call_id','_clip','fold','NF_post','NF_pre','egemaps_valid']):
                    raise AssertionError('Forbidden model input.')
                objects.append(self)
                return self
            def predict_proba(self,x):
                if self.rows & set(x.index) or self.clips & {i//2 for i in x.index}:
                    raise AssertionError('Split leakage.')
                return np.tile([1-self.p,self.p],(len(x),1))
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls,refs,manifest = fixture(folder)
            real_factory = runner.make_model
            def metadata(model,train,cols,spec,sha):
                with patch.object(runner,'make_model',real_factory):
                    return fake_metadata(model,train,cols,spec,sha)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=lambda f,m,*a:FakeModel(m)), \
                    patch.object(runner,'transform_metadata',side_effect=metadata), \
                    patch.object(runner.b3.shared,'model_numeric_columns',return_value=['log_duration_s']):
                counts = runner.fit_all(args,data,cols,controls,refs,manifest,'fixture')
            self.assertEqual(counts,dict(created_jobs=60,reused_jobs=0,model_fits=60))
            self.assertEqual(len(objects),60)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=AssertionError('No resume fits.')):
                # Metadata checking is structural; it must not construct/refit models on resume.
                again = runner.fit_all(args,data,cols,controls,refs,manifest,'fixture')
                runner.summarize(args,data,cols,refs,manifest,'fixture',n_draws=8)
            self.assertEqual(again,dict(created_jobs=0,reused_jobs=60,model_fits=0))
            metrics = runner.b3.read(args.output_dir/'comparison_metrics_and_ci.csv')
            self.assertEqual(len(metrics),3*3*2*4)
            self.assertEqual(set(metrics.pca_mode),{'pca30','pca_full','no_pca'})
            pairs = runner.b3.read(args.output_dir/'paired_changes_and_ci.csv')
            self.assertEqual(len(pairs),3*3*2*4)
            self.assertTrue(np.array_equal(pairs.delta_post,pairs.m1_post_loss_improvement))
            status = json.loads((args.output_dir/'verification.json').read_text())
            self.assertTrue(status['m0_predictions_identical_across_modes'])
            self.assertEqual(status['outer_oof_rows'],180*2*3*3)
            self.assertFalse(status['conditional_information_certified'])
            first = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/first['relative_path']
            frame = runner.b3.read(path)
            original = frame.copy()
            frame.loc[0,'m0_pred'] += .01
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,data,cols,refs,manifest,'fixture',n_draws=8)
            original.to_csv(path,index=False)
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text())
            meta['protocol_sha256'] = 'STALE'
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                runner.fit_all(args,data,cols,controls,refs,manifest,'fixture')

    def test_orphan_pair_and_partial_summary_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls,refs,manifest = fixture(folder)
            spec = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/spec['relative_path']
            path.parent.mkdir(parents=True)
            path.with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Orphan'):
                runner.fit_all(args,data,cols,controls,refs,manifest,'fixture')
            with self.assertRaises(ValueError):
                runner.summarize(args,data,cols,refs,manifest,'fixture',n_draws=8)


if __name__=='__main__':
    unittest.main(verbosity=2)
