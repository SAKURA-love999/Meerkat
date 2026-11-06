"""B3 temporary real-target plumbing fixtures; no scientific model fitting."""
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
import calibration_dgp as toy
import run_baseline_capacity_b3 as runner


def fixture(folder):
    data = toy.toy_design(200)
    data['fold'] = (np.arange(len(data))//4)%5+1
    data['duration_s'] = np.expm1(data.log_duration_s)
    data['duration_bin'] = 'FIXTURE_BIN'
    data['egemaps_valid'] = np.arange(len(data))%3 != 0
    data['window_ms'] = 400
    data['NF_post'] = np.arange(len(data))%7 == 0
    data['NF_pre'] = np.arange(len(data))%9 == 0
    args = Namespace(output_dir=Path(folder)/'new',controls_dir=Path(folder)/'reference')
    (args.output_dir/'jobs').mkdir(parents=True)
    (args.controls_dir/'jobs').mkdir(parents=True)
    for fold in range(1,6):
        test = data.loc[data.fold.eq(fold)]
        for target in ['NF_post','NF_pre']:
            out = record(data,test,target,.12,.13)
            out['condition'] = runner.REFERENCE
            valid = out.loc[out.egemaps_valid].copy()
            valid['condition'] += '_validtest'
            pd.concat([out,valid],ignore_index=True).to_csv(
                args.controls_dir/'jobs'/f'{runner.REFERENCE}_fold{fold}_{target}.csv',index=False)
    return args,data


def record(data,test,target,p0,p1,family=None):
    out = test[runner.FIELDS].copy()
    out['target'],out['y_true'],out['n_train'] = target,test[target].astype(int),len(data)-len(test)
    out['m0_pred'],out['m1_pred'] = p0,p1
    if family:
        out['model_family'],out['protocol_sha256'],out['synthetic'] = family,'FIXTURE_ONLY',False
    return out


def put_new(args,data):
    for family in models.FAMILIES:
        for fold in range(1,6):
            test = data.loc[data.fold.eq(fold)]
            for target in ['NF_post','NF_pre']:
                record(data,test,target,.10,.11,family).to_csv(
                    runner.job_path(args,family,fold,target),index=False)


class B3Tests(unittest.TestCase):
    def test_reference_full_valid_rows_and_real_target_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data = fixture(folder)
            refs,manifest = runner.reference_frames(args,data)
            self.assertEqual(len(manifest),10)
            self.assertEqual(len(refs),2*len(data))
            self.assertFalse(refs.synthetic.any())
            self.assertEqual(set(refs.target),{'NF_post','NF_pre'})
            first = refs.loc[refs.fold.eq(1) & refs.target.eq('NF_post')].copy()
            first.loc[first.index[0],'y_true'] = 1-int(first.iloc[0].y_true)
            with self.assertRaises(ValueError):
                runner.check_record(first,data.query('fold == 1'),'logistic_linear',1,'NF_post',160)

    def test_exact_resume_and_stale_missing_extra_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data = fixture(folder)
            refs,manifest = runner.reference_frames(args,data)
            put_new(args,data)
            with redirect_stdout(io.StringIO()),patch.object(models,'make_model',side_effect=AssertionError('No resume fit.')):
                counts = runner.fit_all(args,data,[],[],'FIXTURE_ONLY')
            self.assertEqual(counts,dict(created_jobs=0,reused_jobs=20,model_fits=0))
            path = runner.job_path(args,'spline_rms',1,'NF_post')
            original = runner.read(path)
            changed = original.copy()
            changed['protocol_sha256'] = 'WRONG'
            changed.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,data,refs,manifest,'FIXTURE_ONLY',20)
            original.to_csv(path,index=False)
            extra = args.output_dir/'jobs'/'extra.csv'
            original.to_csv(extra,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,data,refs,manifest,'FIXTURE_ONLY',20)
            extra.unlink()
            path.unlink()
            with self.assertRaises(ValueError):
                runner.summarize(args,data,refs,manifest,'FIXTURE_ONLY',20)

    def test_summary_all_labels_pairing_and_no_simulated_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data = fixture(folder)
            refs,manifest = runner.reference_frames(args,data)
            put_new(args,data)
            with redirect_stdout(io.StringIO()):
                runner.summarize(args,data,refs,manifest,'FIXTURE_ONLY',20)
            metrics = pd.read_csv(args.output_dir/'comparison_metrics_and_ci.csv')
            self.assertEqual(len(metrics),3*2*5)
            self.assertFalse(any(c.startswith('truth_') or c.startswith('sim_') for c in metrics))
            self.assertFalse(metrics.synthetic.any())
            supported = metrics.loc[metrics.post_positive_n.gt(0) & metrics.pre_positive_n.gt(0)]
            self.assertTrue(np.isfinite(supported.filter(regex='_(auc|ap)$').to_numpy()).all())
            zero_pre = metrics.loc[metrics.pre_positive_n.eq(0)]
            self.assertGreater(len(zero_pre),0)
            self.assertTrue(zero_pre[['m0_pre_auc','m1_pre_auc','m0_pre_ap','m1_pre_ap']].isna().all().all())
            self.assertTrue(zero_pre.sparse_positive_warning.all())
            overall = metrics.loc[metrics.manual_label.eq('overall') & metrics['view'].eq('all')].set_index('model_family')
            changes = pd.read_csv(args.output_dir/'paired_changes_vs_linear.csv')
            selected = changes.loc[changes.model_family.eq('spline_rms') & changes.contrast.eq('delta_change')
                & changes.manual_label.eq('overall') & changes['view'].eq('all')].iloc[0]
            self.assertAlmostEqual(selected.delta_post,
                overall.loc['spline_rms','delta_post']-overall.loc['logistic_linear','delta_post'])
            per_label = metrics.loc[metrics.manual_label.ne('overall')]
            self.assertTrue(per_label.delta_post_ci_low.notna().all())
            status = json.loads((args.output_dir/'verification.json').read_text())
            self.assertEqual(status['new_jobs'],20)
            self.assertFalse(status['causal_effect_assessed'])

    def test_shared_clip_intervals_constant_values(self):
        rows = []
        for clip in range(12):
            for target,value in [('NF_post',.2),('NF_pre',.05)]:
                rows.append(dict(call_id=str(clip),_clip=str(clip),manual_label='cc',target=target,
                    model_family='spline_rms',contrast='gain',view='all',value=value))
        intervals = runner.clip_intervals(pd.DataFrame(rows),20,7452)
        for row in intervals.itertuples():
            self.assertAlmostEqual(row.delta_post_ci_low,.2)
            self.assertAlmostEqual(row.delta_pre_ci_high,.05)
            self.assertAlmostEqual(row.post_minus_pre_ci_low,.15)
            self.assertEqual(row.post_finite_draws,20)

    def test_forty_fake_fits_training_only_and_real_targets(self):
        calls = []
        class FakeModel:
            def fit(self,x,y):
                if len(x)!=160 or any(c.startswith('sim_') or c in ['NF_post','NF_pre','call_id','fold','_clip'] for c in x):
                    raise AssertionError('Invalid predictive input.')
                self.train = set(x.index)
                calls.append((list(x),np.asarray(y)))
                return self
            def predict_proba(self,x):
                if self.train & set(x.index):
                    raise AssertionError('Training/test rows overlap.')
                return np.tile([.9,.1],(len(x),1))
        with tempfile.TemporaryDirectory() as folder:
            args,data = fixture(folder)
            with redirect_stdout(io.StringIO()),patch.object(models,'make_model',side_effect=lambda *a:FakeModel()), \
                    patch.object(runner.shared,'model_numeric_columns',return_value=['log_duration_s']):
                counts = runner.fit_all(args,data,toy.MARK_COLUMNS,[models.RMS],'FIXTURE_ONLY')
            self.assertEqual(counts,dict(created_jobs=20,reused_jobs=0,model_fits=40))
            self.assertEqual(len(calls),40)
            self.assertEqual(sum(set(toy.MARK_COLUMNS)<=set(cols) for cols,_ in calls),20)


if __name__=='__main__':
    unittest.main(verbosity=2)
