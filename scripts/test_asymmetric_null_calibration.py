"""Paired asymmetric-null tests; mock probabilities stay in temporary fixtures."""
import unittest

import numpy as np

import calibration_dgp as base
import calibration_dgp_asymmetric as dgp


class AsymmetricTests(unittest.TestCase):
    def test_paired_post_and_features(self):
        data = base.toy_design()
        old, _ = base.generate(data, 431, dgp.SOURCE_SCENARIO)
        new, _ = dgp.generate(data, 431, dgp.SCENARIO)
        for key in ['call_ids', 'sim_feature_values', 'sim_mark', 'sim_h', 'sim_state',
                    'sim_p_post', 'sim_y_post']:
            np.testing.assert_array_equal(old[key], new[key])
        self.assertFalse(np.array_equal(old['sim_p_pre'], new['sim_p_pre']))

    def test_pre_is_linear_and_rate_preserved(self):
        data = base.toy_design()
        packet, info = dgp.generate(data, 431, dgp.SCENARIO)
        expected, _ = base.probability(base.linear_score(data, 'pre'), .09)
        np.testing.assert_array_equal(packet['sim_p_pre'], expected)
        self.assertAlmostEqual(info['expected_pre'], .09, places=10)
        self.assertAlmostEqual(info['expected_post'], .09, places=10)

    def test_mark_has_no_direct_term_and_h_only_affects_post(self):
        data = base.toy_design()
        packet, _ = dgp.generate(data, 431, dgp.SCENARIO)
        mark, state, h = (packet[k] for k in ['sim_mark', 'sim_state', 'sim_h'])
        p0, r0 = dgp.scores(data, dgp.SCENARIO, mark, state, h)
        p1, r1 = dgp.scores(data, dgp.SCENARIO, mark + 5., state + 5., h)
        np.testing.assert_array_equal(p0, p1)
        np.testing.assert_array_equal(r0, r1)
        p2, r2 = dgp.scores(data, dgp.SCENARIO, mark, state, h + 1.)
        np.testing.assert_allclose(p2 - p0, 1.)
        np.testing.assert_array_equal(r0, r2)

    def test_determinism_and_original_registry_preserved(self):
        data = base.toy_design()
        before = list(base.SCENARIOS)
        a, _ = dgp.generate(data, 431, dgp.SCENARIO)
        b, _ = dgp.generate(data, 431, dgp.SCENARIO)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(before, base.SCENARIOS)
        self.assertNotIn(dgp.SCENARIO, base.TRUTH)
        self.assertFalse(dgp.TRUTH[dgp.SCENARIO]['extra_post'])
        self.assertTrue(np.isfinite(a['sim_feature_values']).all())

    def test_scoped_adapter_restores_original_core(self):
        import run_asymmetric_null_calibration as extension
        import run_semisynthetic_calibration as core
        names = ['CONFIG', 'SCENARIOS', 'TRUTH', 'generate', 'protocol']
        before = {name: getattr(core, name) for name in names}
        with extension.configured_core():
            self.assertEqual(core.SCENARIOS, [dgp.SCENARIO])
            self.assertIs(core.generate, dgp.generate)
            self.assertNotIn(dgp.SCENARIO, base.SCENARIOS)
        for name in names:
            self.assertIs(getattr(core, name), before[name])
        with self.assertRaises(RuntimeError):
            with extension.configured_core():
                raise RuntimeError('Fixture exception tests restoration.')
        self.assertIs(core.generate, before['generate'])

    def test_paired_reuse_and_summary_fixture(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        import json
        from pathlib import Path
        import tempfile
        import pandas as pd
        import run_asymmetric_null_calibration as extension
        import run_semisynthetic_calibration as core
        data = base.toy_design(400)
        data['fold'] = (np.arange(len(data)) // 4) % 5 + 1
        data['duration_s'] = np.expm1(data.log_duration_s)
        data['egemaps_valid'] = np.arange(len(data)) % 3 != 0
        with tempfile.TemporaryDirectory() as folder:
            old_args = Namespace(output_dir=Path(folder)/'old', seeds=[431], n_bootstrap=20)
            old_args.output_dir.mkdir()
            with redirect_stdout(io.StringIO()):
                core.prepare(old_args, data)
            packet, _ = base.generate(data, 431, dgp.SOURCE_SCENARIO)
            current = core.materialize(data, packet)
            (old_args.output_dir/'jobs').mkdir()
            manifest_rows = []

            def record(test, scenario, arm, side):
                result = test[['call_id','_clip','fold','manual_label','duration_s','egemaps_valid']].copy()
                result['replicate_seed'], result['scenario'], result['analysis_arm'] = 431, scenario, arm
                result['target'], result['synthetic'] = 'sim_'+side, True
                result['n_train'] = len(data)-len(test)
                result['y_sim'] = test['sim_y_'+side]
                result['generator_probability'] = test['sim_p_'+side]
                # Mock probabilities test file plumbing, never research performance.
                result['m0_pred'], result['m1_pred'] = .1, .12
                return result

            for arm in ['standard', 'oracle_parent_reference']:
                for fold in range(1, 6):
                    test = current.loc[current.fold.eq(fold)]
                    path = core.job_path(old_args,431,dgp.SOURCE_SCENARIO,arm,fold,'post')
                    record(test,dgp.SOURCE_SCENARIO,arm,'post').to_csv(path,index=False)
                    manifest_rows.append(dict(path=str(path),sha256=core.digest(path),seed=431,
                        scenario=dgp.SOURCE_SCENARIO,analysis_arm=arm,fold=fold,target='post',n_rows=len(test)))
            pd.DataFrame(manifest_rows).to_csv(old_args.output_dir/'synthetic_oof_manifest.csv',index=False)
            args = Namespace(output_dir=Path(folder)/'new',base_run=old_args.output_dir,seeds=[431],n_bootstrap=20)
            args.output_dir.mkdir()
            with extension.configured_core(), redirect_stdout(io.StringIO()):
                core.prepare(args,data)
                manifest = extension.load_manifest(args)
                extension.reuse_post_jobs(args,data,manifest)
                self.assertEqual(len(list((args.output_dir/'jobs').glob('*.csv'))),10)
                first = sorted((args.output_dir/'jobs').glob('*.csv'))[0]
                before_hash = core.digest(first)
                extension.reuse_post_jobs(args,data,manifest)
                self.assertEqual(before_hash,core.digest(first))
                new_packet,_ = dgp.generate(data,431,dgp.SCENARIO)
                new_current = core.materialize(data,new_packet)
                for arm in ['standard','oracle_parent_reference']:
                    for fold in range(1,6):
                        test = new_current.loc[new_current.fold.eq(fold)]
                        path = core.job_path(args,431,dgp.SCENARIO,arm,fold,'pre')
                        record(test,dgp.SCENARIO,arm,'pre').to_csv(path,index=False)
                extension.summarize_extension(args,data,manifest)
                check = json.loads((args.output_dir/'verification.json').read_text())
                self.assertEqual(check['expected_jobs'],20)
                self.assertEqual(check['reused_post_jobs'],10)
                self.assertFalse(check['new_post_evidence'])
                self.assertEqual(len(pd.read_csv(args.output_dir/'synthetic_metrics.csv')),
                                 (data.manual_label.nunique()+1)*2*2)
                stale = extension.read(first)
                stale.loc[0,'m0_pred'] = .23
                stale.to_csv(first,index=False)
                with self.assertRaises(ValueError):
                    extension.reuse_post_jobs(args,data,manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
