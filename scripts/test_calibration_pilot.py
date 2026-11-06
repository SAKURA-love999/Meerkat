"""Small DGP invariants; synthetic values are not real-data results."""
import unittest
import tempfile
from pathlib import Path
from argparse import Namespace
from contextlib import redirect_stdout
import io

import numpy as np

from calibration_dgp import CONFIG, SCENARIOS, generate, scores, toy_design


class GeneratorTests(unittest.TestCase):
    def test_deterministic_and_finite(self):
        data = toy_design()
        for scenario in SCENARIOS:
            a, _ = generate(data, 431, scenario)
            b, _ = generate(data, 431, scenario)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertTrue(np.isfinite(a['sim_feature_values']).all())
            self.assertEqual(a['sim_feature_values'].shape, (len(data), 4))

    def test_expected_rate_and_targets(self):
        for scenario in SCENARIOS:
            packet, info = generate(toy_design(), 431, scenario)
            self.assertAlmostEqual(info['expected_post'], CONFIG['expected_rate'], places=10)
            self.assertAlmostEqual(info['expected_pre'], CONFIG['expected_rate'], places=10)
            self.assertTrue(np.isin(packet['sim_y_post'], [0, 1]).all())

    def test_no_direct_mark_term_in_null_or_state(self):
        data = toy_design()
        for scenario in ['null_linear', 'null_nonlinear', 'state_bilateral']:
            a, _ = generate(data, 431, scenario)
            p0, r0 = scores(data, scenario, a['sim_mark'], a['sim_state'], a['sim_h'])
            p1, r1 = scores(data, scenario, a['sim_mark'] + 10., a['sim_state'], a['sim_h'])
            np.testing.assert_array_equal(p0, p1)
            np.testing.assert_array_equal(r0, r1)

    def test_post_signal_only(self):
        data = toy_design()
        a, _ = generate(data, 431, 'signal_post')
        p0, r0 = scores(data, 'signal_post', a['sim_mark'], a['sim_state'], a['sim_h'])
        p1, r1 = scores(data, 'signal_post', a['sim_mark'] + 1., a['sim_state'], a['sim_h'])
        np.testing.assert_allclose(p1 - p0, 1.)
        np.testing.assert_array_equal(r0, r1)

    def test_state_constant_within_clip(self):
        data = toy_design()
        a, _ = generate(data, 431, 'state_bilateral')
        data['sim_state'] = a['sim_state']
        self.assertEqual(data.groupby('_clip').sim_state.nunique().max(), 1)

    def test_npz_safe_round_trip(self):
        packet, _ = generate(toy_design(),431,'signal_post')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.npz'
            np.savez_compressed(path,**packet)
            with np.load(path,allow_pickle=False) as saved:
                for key,value in packet.items():
                    np.testing.assert_array_equal(saved[key],value)

    def test_clip_bootstrap_constant_difference(self):
        import pandas as pd
        from run_semisynthetic_calibration import bootstrap_overall, wilson
        rows = []
        for arm,gain in [('standard',.2),('oracle_parent_reference',.05)]:
            for side in ['post','pre']:
                for i in range(4):
                    rows.append(dict(scenario='null_nonlinear',analysis_arm=arm,view='all',
                        target='sim_'+side,_clip='c'+str(i//2),call_id=str(i),gain=gain))
        result = bootstrap_overall(pd.DataFrame(rows),40,3).set_index('analysis_arm')
        self.assertAlmostEqual(result.loc['standard','delta_post_ci_low'],.2)
        self.assertAlmostEqual(result.loc['oracle_parent_reference','delta_post_change_vs_standard_ci_high'],-.15)
        self.assertAlmostEqual(result.loc['standard','post_minus_pre_ci_high'],0.)
        lo,hi = wilson(0,5)
        self.assertLessEqual(lo,1e-10)
        self.assertGreater(hi,.4)

    def test_full_summary_fixture_and_resume(self):
        import json
        import pandas as pd
        from run_semisynthetic_calibration import prepare,materialize,summarize,arms,job_path,digest
        data = toy_design(400)
        data['fold'] = (np.arange(len(data))//4)%5+1
        data['duration_s'] = np.expm1(data.log_duration_s)
        data['egemaps_valid'] = np.arange(len(data))%3 != 0
        with tempfile.TemporaryDirectory() as folder:
            args = Namespace(output_dir=Path(folder),seeds=[431],n_bootstrap=20)
            with redirect_stdout(io.StringIO()):
                prepare(args,data)
            manifest = {(r['seed'],r['scenario']):r for r in json.loads(
                (Path(folder)/'synthetic_design_manifest.json').read_text())}
            hashes = [r['sha256'] for r in manifest.values()]
            with redirect_stdout(io.StringIO()):
                prepare(args,data)
            self.assertEqual(hashes,[digest(Path(r['path'])) for r in manifest.values()])
            (Path(folder)/'jobs').mkdir()
            for scenario in SCENARIOS:
                packet,_ = generate(data,431,scenario)
                current = materialize(data,packet)
                for arm in arms(scenario):
                    for fold in range(1,6):
                        test = current.loc[current.fold.eq(fold)]
                        for side in ['post','pre']:
                            record = test[['call_id','_clip','fold','manual_label','duration_s','egemaps_valid']].copy()
                            record['replicate_seed'],record['scenario'],record['analysis_arm'] = 431,scenario,arm
                            record['target'],record['synthetic'] = 'sim_'+side,True
                            record['n_train'] = len(current)-len(test)
                            record['y_sim'] = test['sim_y_'+side]
                            record['generator_probability'] = test['sim_p_'+side]
                            # Fixed artificial probabilities test plumbing only; no model or result claim.
                            record['m0_pred'],record['m1_pred'] = .1,.12
                            record.to_csv(job_path(args,431,scenario,arm,fold,side),index=False)
            with redirect_stdout(io.StringIO()):
                summarize(args,data,manifest)
            metrics = pd.read_csv(Path(folder)/'synthetic_metrics.csv')
            rates = pd.read_csv(Path(folder)/'replicate_diagnostic_rates.csv')
            self.assertTrue(metrics.synthetic.all())
            self.assertTrue(rates.n_replicates.eq(1).all())
            self.assertEqual(len(list((Path(folder)/'jobs').glob('*.csv'))),60)
            next((Path(folder)/'jobs').glob('*.csv')).unlink()
            with self.assertRaises(ValueError):
                summarize(args,data,manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
