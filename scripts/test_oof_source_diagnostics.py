"""Small synthetic invariants; no real-data model fitting."""
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from contextlib import redirect_stdout
import io
import warnings

import numpy as np
import pandas as pd

from diagnose_shared_oof import calibration, clip_totals, concentration, losses, validate, analyze
from run_control_block_attribution import AMPLITUDE, OVERLAP, PLANS, assert_job
from audit_confirmation_clip_pool import identify


class DiagnosticTests(unittest.TestCase):
    def example(self):
        rows = []
        for target in ['NF_pre', 'NF_post']:
            for i, y in enumerate([0, 1, 0, 1]):
                rows.append(dict(window_ms=400, condition='test', target=target,
                    call_id=str(i), _clip='a' if i < 2 else 'b', fold=1 if i < 2 else 2,
                    manual_label='cc' if i % 2 else 'sn', duration_s=.1,
                    y_true=y, m0_pred=.5, m1_pred=.3 if y == 0 else .7))
        return pd.DataFrame(rows)

    def test_loss_and_calibration(self):
        m, bins = calibration([0, 1], [.5, .5])
        self.assertAlmostEqual(m['logloss'], np.log(2))
        self.assertAlmostEqual(m['brier'], .25)
        self.assertAlmostEqual(m['ece'], 0)
        self.assertEqual(bins.n.sum(), 2)
        self.assertTrue(np.isfinite(losses([0, 1], [1, 0])).all())

    def test_clip_reconstruction(self):
        data = self.example()
        validate(data)
        part = data.loc[data.target.eq('NF_post')]
        c = clip_totals(part)
        self.assertAlmostEqual(c.gain_sum.sum() / c.n.sum(), np.log(2) + np.log(.7))

    def test_zero_and_cancelling_contributions(self):
        c = pd.DataFrame(dict(_clip=['a', 'b'], n=[1, 1], gain_sum=[1., -1.]))
        row = concentration(c, [.5])[0]
        self.assertEqual(row['delta'], 0)
        self.assertEqual(row['top_absolute_gain_share'], .5)
        c['gain_sum'] = 0.
        self.assertTrue(np.isnan(concentration(c)[0]['top_absolute_gain_share']))

    def test_duplicate_or_misaligned_oof_rejected(self):
        data = self.example()
        with self.assertRaises(ValueError):
            validate(pd.concat([data, data.iloc[:1]]))
        with self.assertRaises(ValueError):
            validate(data.iloc[1:])

    def test_full_output_spine(self):
        with tempfile.TemporaryDirectory() as folder:
            result = analyze(self.example(), Path(folder))
            self.assertLess(result['max_clip_reconstruction_error'], 1e-12)
            for name in ['calibration_metrics.csv', 'reliability_bins.csv',
                         'clip_loss_contributions.csv', 'clip_direction_contributions.csv',
                         'concentration_sensitivity.csv', 'high_confidence_errors.csv']:
                self.assertTrue((Path(folder) / name).exists())

    def test_control_blocks_disjoint(self):
        self.assertFalse(set(AMPLITUDE) & set(OVERLAP))
        self.assertEqual(len(AMPLITUDE), 4)
        self.assertEqual(len(OVERLAP), 3)
        self.assertEqual(len(PLANS), 4)

    def test_job_resume_checks(self):
        test = pd.DataFrame(dict(call_id=['a', 'b'], egemaps_valid=[True, False], NF_post=[1, 0]))
        saved = pd.DataFrame(dict(call_id=['a', 'b', 'a'], condition=['arm', 'arm', 'arm_validtest'],
            target=['NF_post'] * 3, fold=[1] * 3, y_true=[1, 0, 1],
            m0_pred=[.2, .1, .2], m1_pred=[.3, .2, .3]))
        assert_job(saved, test, 'arm', 'NF_post', 1)
        saved.loc[2, 'm1_pred'] = .8
        with self.assertRaises(ValueError):
            assert_job(saved, test, 'arm', 'NF_post', 1)

    def test_confirmation_identifier_resolution(self):
        ids = pd.Series(['a', 'a__000', r'C:\wav\b.wav', 'unrelated'])
        self.assertEqual(identify(ids, {'a', 'b'}, {'a__000': 'a'}), {'a', 'b'})

    def test_round_trip_control_reload(self):
        original = pd.DataFrame(dict(control=[np.nextafter(.1, 1.), np.pi, -np.pi,
            1. / 3., np.nextafter(-28., 0.)]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'controls.csv'
            original.to_csv(path, index=False)
            restored = pd.read_csv(path, float_precision='round_trip')
            np.testing.assert_array_equal(original.control, restored.control)

    def test_summary_aliases_on_synthetic_oof(self):
        import run_shared_cohort_selection_control as shared
        from run_control_block_attribution import summarize
        data = pd.DataFrame([dict(call_id=str(i), _clip='clip' + str(i // 2),
            fold=i // 2 + 1, manual_label='cc' if i % 2 else 'al', duration_s=.1,
            duration_bin='test', egemaps_valid=bool(i % 2), window_ms=400,
            NF_post=i % 2, NF_pre=(i + 1) % 2) for i in range(10)])

        def records(test, condition, target, include_valid=True):
            y = test[target].to_numpy()
            r = shared.predict_record(test, target, condition, condition,
                np.full(len(test), .5), np.where(y, .6, .4), 8)
            if include_valid:
                selected = r.loc[r.egemaps_valid].copy()
                selected['condition'] += '_validtest'
                r = pd.concat([r, selected], ignore_index=True)
            return r

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = Namespace(shared_dir=root / 'shared', controls_dir=root / 'old',
                output_dir=root / 'new', threads=1, n_bootstrap=20, bootstrap_seed=7)
            for p in [args.shared_dir, args.controls_dir, args.output_dir / 'jobs']:
                p.mkdir(parents=True)
            old, joint = [], []
            for target in ['NF_post', 'NF_pre']:
                old.append(records(data, '01_full_all', target, False))
                old.append(records(data.loc[data.egemaps_valid], '02_full_valid', target, False))
                for condition in ['20_raw_shared_full', '21_normalized_shared_full', '24_normalized_original_full']:
                    joint.append(records(data, condition, target))
            pd.concat(old).to_csv(args.shared_dir / 'oof_predictions.csv', index=False)
            pd.concat(joint).to_csv(args.controls_dir / 'oof_predictions_added.csv', index=False)
            for fold in range(1, 6):
                test = data.loc[data.fold.eq(fold)]
                for condition, _, _ in PLANS:
                    for target in ['NF_post', 'NF_pre']:
                        records(test, condition, target).to_csv(args.output_dir / 'jobs' /
                            f'{condition}_fold{fold}_{target}.csv', index=False)
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.filterwarnings('ignore', message='No positive class found in y_true')
                summarize(args, data, shared)
            result = pd.read_csv(args.output_dir / 'paired_metrics_and_ci.csv')
            overall = result.loc[result.manual_label.eq('overall')].set_index('condition')
            self.assertEqual(overall.loc['20_raw_shared_full', 'change_reference'], '30_raw_amplitude')
            self.assertEqual(overall.loc['20_raw_shared_full__vs_overlap', 'change_reference'], '32_raw_overlap')
            self.assertEqual(overall.loc['21_normalized_shared_full__vs_none', 'change_reference'], '24_normalized_original_full')
            self.assertAlmostEqual(overall.loc['20_raw_shared_full__vs_overlap', 'delta_post_change'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
