"""Configuration provenance and attribution technical fixtures; no real training."""
from pathlib import Path
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import audit_acoustic_config_fields as audit
import run_config_source_attribution as runner


def sample_features():
    samples = np.array([64,127,128,255,256,511,512,800,1600])
    out = pd.DataFrame(dict(call_id=['c'+str(i) for i in range(len(samples))],
        source_id=['clip'+str(i) for i in range(len(samples))],label='sn',amplitude_variant='raw',
        n_samples=samples,sample_rate=8000,start_time_sec=0.,end_time_sec=samples/8000.,
        duration_sec=samples/8000.,duration_sec_from_audio=samples/8000.,rms=.1))
    for name,values in audit.expected_config(samples).items():
        out[name] = values
    return out


class ConfigAuditTests(unittest.TestCase):
    def test_thresholds_and_redundancy(self):
        cfg = audit.expected_config(np.array([4,63,64,127,128,255,256,511,512,10000]))
        np.testing.assert_array_equal(cfg['n_fft'],[64,64,64,64,128,128,256,256,512,512])
        np.testing.assert_array_equal(cfg['hop_length'],cfg['n_fft']/4)
        np.testing.assert_array_equal(cfg['n_mels'],[16,16,16,16,32,32,32,32,32,32])

    def test_cached_formula_boundary_and_early_return(self):
        raw = sample_features()
        out = audit.check_feature_rows(raw)
        self.assertTrue(out.formula_match.all())
        self.assertTrue(out.sample_count_boundary_match.all())
        raw.loc[0,'rms'] = 0.
        raw.loc[0,audit.CONFIG] = np.nan
        out = audit.check_feature_rows(raw)
        self.assertTrue(out.formula_match.all())
        self.assertTrue(out.iloc[0].extractor_early_return)
        raw.loc[2,'n_fft'] = 512.
        self.assertFalse(audit.check_feature_rows(raw).iloc[2].formula_match)

    def test_AST_matches_formula_and_detects_changed_source(self):
        values = {'n_samples':'int(audio.size)',**audit.EXPRESSIONS}
        lines = ['def extract_features(audio,sample_rate):']+['    '+k+' = '+v for k,v in values.items()]
        audit.verify_source_formulas('\n'.join(lines))
        with self.assertRaises(ValueError):
            audit.verify_source_formulas('\n'.join(lines).replace('min(512','min(1024'))

    def test_duplicate_and_noninteger_sample_metadata_refused(self):
        raw = sample_features()
        with self.assertRaises(ValueError):
            audit.check_feature_rows(pd.concat([raw,raw.iloc[:1]],ignore_index=True))
        raw.loc[0,'n_samples'] = 64.5
        with self.assertRaises(ValueError):
            audit.check_feature_rows(raw)


def model_fixture(folder):
    n = 180
    i = np.arange(n)
    ns = np.resize([80,160,320,800,1600],n)
    data = pd.DataFrame(dict(call_id=['c'+str(j) for j in i],_clip=['clip'+str(j//2) for j in i],
        fold=(i//2)%5+1,manual_label=np.resize(['cc','sn','al'],n),duration_s=ns/8000.,
        log_duration_s=np.log1p(ns/8000.),duration_bin='fixture',egemaps_valid=i%4!=0,
        window_ms=400,NF_post=(i%7==0).astype(int),NF_pre=(i%9==0).astype(int),
        clip_start_margin_s=.5,clip_end_margin_s=10-(.5+ns/8000.),control_raw_rms_dbfs=-20+i/100))
    cols = ['audio'+str(k) for k in range(78)]+audit.CONFIG
    rng = np.random.default_rng(98)
    for col in cols[:78]:
        data[col] = rng.normal(size=n)
    for col,value in audit.expected_config(ns).items():
        data[col] = value
    parts,manifest = [],[]
    for family in runner.FAMILIES:
        for fold in range(1,6):
            test = data.loc[data.fold.eq(fold)]
            for target in runner.TARGETS:
                frame = test[runner.b3.FIELDS].copy()
                frame['model_family'],frame['target'],frame['y_true'] = family,target,test[target]
                frame['n_train'],frame['m0_pred'],frame['m1_pred'],frame['synthetic'] = 144,.15,.14,False
                parts.append(frame)
                manifest.append(dict(model_family=family,fold=fold,target=target,
                    path=f'FIXTURE_{family}_{fold}_{target}.csv',sha256='FIXTURE_SOURCE'))
    return Namespace(output_dir=Path(folder)),data,cols,[runner.pca.models.RMS],pd.concat(parts,ignore_index=True),manifest


def fake_meta(model,train,spec,sha):
    from test_pca_sensitivity import fake_metadata
    meta = fake_metadata(model,train,audit.CONFIG,spec,sha)
    values = train[audit.CONFIG].to_numpy(float)
    median = np.nanmedian(values,axis=0)
    values = np.where(np.isnan(values),median,values)
    meta.update(config_imputer_statistics=median.tolist(),config_scaler_mean=values.mean(axis=0).tolist(),
        config_scaler_variance=values.var(axis=0).tolist(),config_scaler_scale=np.sqrt(values.var(axis=0)).tolist())
    return meta


class AttributionTests(unittest.TestCase):
    def test_literal_subset_preprocessing_and_fixed_classifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,cols,controls,_,_ = model_fixture(folder)
            train,test = data.loc[data.fold.ne(1)],data.loc[data.fold.eq(1)]
            for family in runner.FAMILIES:
                numeric = ['log_duration_s']+controls
                model = runner.make_model(family,numeric)
                full = runner.pca.make_model(family,'no_pca',numeric,cols)
                model.named_steps['preprocess'].fit_transform(train[['manual_label']+numeric+audit.CONFIG])
                full.named_steps['preprocess'].fit_transform(train[['manual_label']+numeric+cols])
                branch = model.named_steps['preprocess'].named_transformers_['acoustic']
                original = full.named_steps['preprocess'].named_transformers_['acoustic']
                x = branch.transform(test[audit.CONFIG])
                y = original.transform(test[cols])[:,[cols.index(c) for c in audit.CONFIG]]
                np.testing.assert_allclose(x,y,rtol=0,atol=0)
                self.assertNotIn('pca',branch.named_steps)
                self.assertFalse(hasattr(model.named_steps['clf'],'classes_'))
                self.assertEqual(model.named_steps['clf'].get_params(),full.named_steps['clf'].get_params())

    def test_boundary_controls_reconstruct_configuration_and_audit_failure_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data,_,_,_,_ = model_fixture(folder)
            reconstructed = runner.reconstruct_from_controls(data,8000)
            self.assertTrue(reconstructed.config_match.all())
            (args.output_dir/'summary.json').write_text(json.dumps(dict(status='failed')))
            with self.assertRaises(ValueError):
                runner.validate_audit(args.output_dir,data)

    def test_thirty_fake_fits_resume_and_exact_decomposition(self):
        fitted = []
        class FakeModel:
            named_steps = {}
            def fit(self,x,y):
                if any(c.startswith('audio') or c in ['NF_post','NF_pre','fold','call_id','_clip','egemaps_valid'] for c in x):
                    raise AssertionError('Invalid Mconfig inputs.')
                self.rows = set(x.index)
                self.clips = {i//2 for i in x.index}
                fitted.append(self)
                return self
            def predict_proba(self,x):
                if self.rows & set(x.index) or self.clips & {i//2 for i in x.index}:
                    raise AssertionError('Clip leakage.')
                return np.tile([.88,.12],(len(x),1))
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls,refs,manifest = model_fixture(folder)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=lambda *a:FakeModel()), \
                    patch.object(runner,'capture_metadata',side_effect=fake_meta), \
                    patch.object(runner.b3.shared,'model_numeric_columns',return_value=['log_duration_s']):
                counts = runner.fit_all(args,data,controls,refs,manifest,'fixture')
            self.assertEqual(counts,dict(created_jobs=30,reused_jobs=0,model_fits=30))
            self.assertEqual(len(fitted),30)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=AssertionError('Resume cannot fit.')):
                again = runner.fit_all(args,data,controls,refs,manifest,'fixture')
                runner.summarize(args,data,refs,manifest,'fixture',n_draws=8)
            self.assertEqual(again,dict(created_jobs=0,reused_jobs=30,model_fits=0))
            metrics = runner.b3.read(args.output_dir/'paired_metrics_and_ci.csv')
            self.assertEqual(len(metrics),3*3*2*4)
            self.assertEqual(set(metrics.comparison),{'total','config','remaining'})
            wide = runner.b3.read(args.output_dir/'three_model_metrics.csv')
            self.assertEqual(len(wide),3*2*4)
            np.testing.assert_allclose(wide.total_delta_post,wide.config_delta_post+wide.remaining_delta_post,rtol=0,atol=1e-14)
            verification = json.loads((args.output_dir/'verification.json').read_text())
            self.assertEqual(verification['new_fitted_jobs'],30)
            self.assertFalse(verification['causal_decomposition'])
            spec = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/spec['relative_path']
            frame = runner.b3.read(path)
            frame.loc[0,'m1_pred'] += .01
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,data,refs,manifest,'fixture',n_draws=8)

    def test_orphan_and_missing_job_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data,_,controls,refs,manifest = model_fixture(folder)
            spec = runner.make_plan(data,manifest).iloc[0].to_dict()
            path = args.output_dir/spec['relative_path']
            path.parent.mkdir(parents=True)
            path.with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Orphan'):
                runner.fit_all(args,data,controls,refs,manifest,'fixture')
            with self.assertRaises(ValueError):
                runner.summarize(args,data,refs,manifest,'fixture',n_draws=8)


if __name__=='__main__':
    unittest.main(verbosity=2)
