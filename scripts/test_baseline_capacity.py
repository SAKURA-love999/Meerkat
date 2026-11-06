"""B1 plumbing and train-only transforms;no real-data classifier fits."""
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
from threadpoolctl import threadpool_limits

import baseline_capacity_models as models
import calibration_dgp as base
import calibration_dgp_asymmetric as dgp
import run_baseline_capacity as runner
import run_semisynthetic_calibration as core


class CapacityTests(unittest.TestCase):
    def test_factory_contracts(self):
        for family in models.FAMILIES:
            model = models.make_model(family,['duration_s',models.RMS],[])
            branches = {name:cols for name,_,cols in model.named_steps['preprocess'].transformers}
            if family=='spline_rms':
                self.assertNotIn(models.RMS,branches['num'])
                self.assertEqual(branches['rms_spline'],[models.RMS])
                self.assertEqual(model.named_steps['clf'].max_iter,2000)
            else:
                self.assertIn(models.RMS,branches['num'])
                self.assertFalse(model.named_steps['clf'].early_stopping)
                self.assertEqual(model.named_steps['clf'].max_leaf_nodes,7)

    def test_forbidden_inputs(self):
        for family in models.FAMILIES:
            for forbidden in ['sim_h','sim_state','NF_post','call_id','fold']:
                with self.assertRaises(ValueError):
                    models.make_model(family,[models.RMS,forbidden],[])

    def test_preprocessing_train_only(self):
        rng = np.random.default_rng(2)
        data = pd.DataFrame({'manual_label':np.resize(['cc','sn'],120),
            'duration_s':rng.uniform(.01,.5,120),models.RMS:np.linspace(-50,-10,120)})
        acoustic = ['a'+str(i) for i in range(35)]
        for col in acoustic:
            data[col] = rng.normal(size=120)
        pre = models.make_model('spline_rms',['duration_s',models.RMS],acoustic).named_steps['preprocess']
        with threadpool_limits(limits=4):
            pre.fit_transform(data.iloc[:96])
            basis = pre.named_transformers_['rms_spline'].named_steps['basis']
            knots = basis.bsplines_[0].t.copy()
            self.assertAlmostEqual(knots[3],data.iloc[:96][models.RMS].min())
            self.assertEqual(pre.named_transformers_['acoustic'].named_steps['pca'].n_samples_,96)
            extreme = data.iloc[96:].copy()
            extreme[models.RMS] = 10000.
            result = pre.transform(extreme)
        np.testing.assert_array_equal(knots,basis.bsplines_[0].t)
        self.assertTrue(np.isfinite(result).all())

    def test_full_summary_resume_and_stale_fixture(self):
        data = base.toy_design(200)
        data['fold'] = (np.arange(len(data))//4)%5+1
        data['duration_s'] = np.expm1(data.log_duration_s)
        data['egemaps_valid'] = np.arange(len(data))%3!=0
        packet,_ = dgp.generate(data,431,dgp.SCENARIO)
        current = core.materialize(data,packet)
        with tempfile.TemporaryDirectory() as folder:
            args = Namespace(output_dir=Path(folder)/'new',reference_run=Path(folder)/'reference')
            (args.reference_run/'jobs').mkdir(parents=True)
            (args.output_dir/'jobs').mkdir(parents=True)
            records = []

            def record(test,arm,m0,m1):
                out = test[runner.FIELDS].copy()
                out['analysis_arm'],out['scenario'],out['replicate_seed'] = arm,dgp.SCENARIO,431
                out['synthetic'],out['n_train'] = True,len(current)-len(test)
                out['target'],out['y_sim'] = 'sim_'+side,test['sim_y_'+side]
                out['generator_probability'] = test['sim_p_'+side]
                # Mock probabilities are temporary fixture data,not fitted results.
                out['m0_pred'],out['m1_pred'] = m0,m1
                return out

            for arm in runner.SOURCE_FAMILIES:
                for fold in range(1,6):
                    test = current.loc[current.fold.eq(fold)]
                    for side in ['post','pre']:
                        path = args.reference_run/'jobs'/f'{arm}_{fold}_{side}.csv'
                        record(test,arm,.12,.13).to_csv(path,index=False)
                        records.append(dict(path=str(path),sha256=core.digest(path),analysis_arm=arm,
                            fold=fold,target=side,n_rows=len(test)))
            pd.DataFrame(records).to_csv(args.reference_run/'synthetic_oof_manifest.csv',index=False)
            references,manifest = runner.reference_frames(args,current)
            for family in models.FAMILIES:
                for fold in range(1,6):
                    test = current.loc[current.fold.eq(fold)]
                    for side in ['post','pre']:
                        out = record(test,'standard',.10,.11)
                        out['model_family'],out['protocol_sha256'] = family,'FIXTURE_ONLY'
                        out.to_csv(runner.job_path(args,family,fold,side),index=False)
            with redirect_stdout(io.StringIO()),patch.object(models,'make_model',side_effect=AssertionError('No resume refits.')):
                runner.fit_all(args,current,[],[],None,'FIXTURE_ONLY')
                runner.summarize(args,current,references,manifest,'FIXTURE_ONLY',20)
            metrics = pd.read_csv(args.output_dir/'comparison_metrics_and_ci.csv')
            self.assertEqual(len(metrics),(data.manual_label.nunique()+1)*2*4)
            changes = pd.read_csv(args.output_dir/'paired_changes_vs_linear.csv')
            overall = metrics.loc[metrics.manual_label.eq('overall') & metrics.view.eq('all')].set_index('model_family')
            point = changes.loc[changes.model_family.eq('spline_rms') & changes.contrast.eq('delta_change')
                                & changes.manual_label.eq('overall') & changes.view.eq('all')].iloc[0]
            self.assertAlmostEqual(point.delta_post,overall.loc['spline_rms','delta_post']-overall.loc['logistic_linear','delta_post'])
            self.assertFalse(json.loads((args.output_dir/'verification.json').read_text())['real_stable_information_assessed'])
            first = runner.job_path(args,'spline_rms',1,'post')
            stale = runner.read(first)
            stale['protocol_sha256'] = 'WRONG'
            stale.to_csv(first,index=False)
            with self.assertRaises(ValueError):
                runner.summarize(args,current,references,manifest,'FIXTURE_ONLY',20)
            first.unlink()
            with self.assertRaises(ValueError):
                runner.summarize(args,current,references,manifest,'FIXTURE_ONLY',20)


if __name__=='__main__':
    unittest.main(verbosity=2)
