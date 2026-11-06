"""B2 temporary-fixture tests; no real classifier training or research metrics."""
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

import baseline_capacity_models as models
import calibration_dgp as dgp
import run_baseline_capacity_b2 as runner
import run_semisynthetic_calibration as core


def fixture(folder):
    data = dgp.toy_design(200)
    data['fold'] = (np.arange(len(data))//4)%5+1
    data['duration_s'] = np.expm1(data.log_duration_s)
    data['egemaps_valid'] = np.arange(len(data))%3 != 0
    currents = {name:core.materialize(data,dgp.generate(data,431,name)[0])
                for name in runner.SCENARIOS}
    args = Namespace(output_dir=Path(folder)/'new',reference_run=Path(folder)/'reference')
    (args.reference_run/'jobs').mkdir(parents=True)
    (args.output_dir/'jobs').mkdir(parents=True)
    manifest = []
    for name,current in currents.items():
        for fold in range(1,6):
            test = current.loc[current.fold.eq(fold)]
            for side in ['post','pre']:
                path = args.reference_run/'jobs'/f'{name}_{fold}_{side}.csv'
                record(current,test,name,fold,side,.12,.13).to_csv(path,index=False)
                manifest.append(dict(path=str(path),sha256=core.digest(path),seed=431,
                    scenario=name,analysis_arm='standard',fold=fold,target=side,n_rows=len(test)))
    pd.DataFrame(manifest).to_csv(args.reference_run/'synthetic_oof_manifest.csv',index=False)
    return args,currents


def record(current,test,name,fold,side,p0,p1,family=None):
    out = test[runner.FIELDS].copy()
    for key,value in dict(analysis_arm='standard',scenario=name,replicate_seed=431,
                          target='sim_'+side,synthetic=True,n_train=len(current)-len(test)).items():
        out[key] = value
    out['y_sim'],out['generator_probability'] = test['sim_y_'+side],test['sim_p_'+side]
    out['m0_pred'],out['m1_pred'] = p0,p1
    if family:
        out['model_family'],out['protocol_sha256'] = family,'FIXTURE_ONLY'
    return out


def put_new(args,currents):
    for name,current in currents.items():
        for family in models.FAMILIES:
            for fold in range(1,6):
                test = current.loc[current.fold.eq(fold)]
                for side in ['post','pre']:
                    record(current,test,name,fold,side,.10,.11,family).to_csv(
                        runner.job_path(args,name,family,fold,side),index=False)


class B2Tests(unittest.TestCase):
    def test_summary_truth_scenario_safe_pairing_and_secondary_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            args,currents = fixture(folder)
            references,manifest = runner.reference_frames(args,currents)
            put_new(args,currents)
            with redirect_stdout(io.StringIO()):
                runner.summarize(args,currents,references,manifest,'FIXTURE_ONLY',20)
            metrics = pd.read_csv(args.output_dir/'comparison_metrics_and_ci.csv')
            self.assertEqual(len(metrics),2*3*2*5)
            overall = metrics.loc[metrics.manual_label.eq('overall') & metrics['view'].eq('all')]
            self.assertTrue(overall.truth_extra_post.all())
            self.assertFalse(overall.loc[overall.scenario.eq('signal_post'),'truth_extra_pre'].any())
            self.assertTrue(overall.loc[overall.scenario.eq('state_bilateral'),'truth_extra_pre'].all())
            self.assertFalse(overall.loc[overall.scenario.eq('state_bilateral'),'direct_post_mark_in_generator'].any())
            self.assertTrue(np.isfinite(overall.filter(regex='_(auc|ap)$').to_numpy()).all())
            changes = pd.read_csv(args.output_dir/'paired_changes_vs_linear.csv')
            for scenario in runner.SCENARIOS:
                cell = overall.loc[overall.scenario.eq(scenario)].set_index('model_family')
                contrast = changes.loc[changes.scenario.eq(scenario) & changes.model_family.eq('spline_rms')
                    & changes.contrast.eq('delta_change') & changes.manual_label.eq('overall')
                    & changes['view'].eq('all')].iloc[0]
                self.assertAlmostEqual(contrast.delta_post,
                    cell.loc['spline_rms','delta_post']-cell.loc['logistic_linear','delta_post'])
            verified = json.loads((args.output_dir/'verification.json').read_text())
            self.assertEqual(verified['new_jobs'],40)
            self.assertEqual(verified['reference_jobs'],20)
            self.assertFalse(verified['real_stable_information_assessed'])

    def test_exact_resume_has_zero_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            args,currents = fixture(folder)
            put_new(args,currents)
            with redirect_stdout(io.StringIO()),patch.object(models,'make_model',side_effect=AssertionError('No refit.')):
                result = runner.fit_all(args,currents,[],[],None,'FIXTURE_ONLY')
            self.assertEqual(result,dict(created_jobs=0,reused_jobs=40,model_fits=0))

    def test_stale_missing_extra_and_wrong_target_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args,currents = fixture(folder)
            references,manifest = runner.reference_frames(args,currents)
            put_new(args,currents)
            first = runner.job_path(args,'signal_post','spline_rms',1,'post')
            original = runner.read(first)
            stale = original.copy()
            stale['protocol_sha256'] = 'WRONG'
            stale.to_csv(first,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,currents,references,manifest,'FIXTURE_ONLY',20)
            stale = original.copy()
            stale.loc[0,'y_sim'] = 1-int(stale.loc[0,'y_sim'])
            stale.to_csv(first,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,currents,references,manifest,'FIXTURE_ONLY',20)
            original.to_csv(first,index=False)
            extra = args.output_dir/'jobs'/'extra.csv'
            original.to_csv(extra,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,currents,references,manifest,'FIXTURE_ONLY',20)
            extra.unlink()
            first.unlink()
            with self.assertRaises(ValueError):
                runner.summarize(args,currents,references,manifest,'FIXTURE_ONLY',20)

    def test_reference_cross_scenario_or_metadata_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args,currents = fixture(folder)
            frame = record(currents['signal_post'],currents['signal_post'].query('fold == 1'),
                           'signal_post',1,'post',.1,.1)
            frame['manual_label'] = 'BAD'
            with self.assertRaises(ValueError):
                runner.check_record(frame,currents['signal_post'].query('fold == 1'),
                    'signal_post','logistic_linear',1,'post',160)
            with self.assertRaises(ValueError):
                runner.check_record(frame,currents['state_bilateral'].query('fold == 1'),
                    'state_bilateral','logistic_linear',1,'post',160)

    def test_eighty_mock_fits_use_only_training_inputs(self):
        calls = []
        class FakeModel:
            def fit(self,x,y):
                self.columns = list(x)
                self.ids = set(x.index)
                self.assertion = len(x) == 160 and len(y) == 160
                if not self.assertion or any(c.startswith('sim_') or c in ['call_id','fold','_clip','NF_post','NF_pre'] for c in x):
                    raise AssertionError('Training contract broken.')
                calls.append(list(x))
                return self
            def predict_proba(self,x):
                if self.ids & set(x.index):
                    raise AssertionError('Test rows in training.')
                return np.tile([.9,.1],(len(x),1))
        class Shared:
            @staticmethod
            def model_numeric_columns(train,target):
                side = 'pre' if target=='NF_post' else 'post'
                return ['log_duration_s',models.RMS,side+'_400ms_all_count']
        with tempfile.TemporaryDirectory() as folder:
            args,currents = fixture(folder)
            with redirect_stdout(io.StringIO()),patch.object(models,'make_model',side_effect=lambda *a:FakeModel()):
                result = runner.fit_all(args,currents,dgp.MARK_COLUMNS,[],Shared,'FIXTURE_ONLY')
            self.assertEqual(result,dict(created_jobs=40,reused_jobs=0,model_fits=80))
            self.assertEqual(len(calls),80)
            self.assertEqual(sum(set(dgp.MARK_COLUMNS)<=set(cols) for cols in calls),40)


if __name__=='__main__':
    unittest.main(verbosity=2)
