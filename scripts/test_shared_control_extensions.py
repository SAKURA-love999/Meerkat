"""Small CPU-only invariants; no real encoder/data extraction."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from extend_shared_cohort_controls import sample_label_quotas, union_duration, make_overlap_table
from audit_causal_animal2vec_inputs import support_plan, masked_normalize, read_allowed_window, MODES


class ControlTests(unittest.TestCase):
    def test_exact_label_quotas(self):
        train = pd.DataFrame({'call_id':[str(i) for i in range(15)],
                              'manual_label':['sn']*10+['cc']*5})
        valid = train.iloc[[0,1,10,11,12]]
        sample = sample_label_quotas(train,valid,19)
        self.assertEqual(sample.manual_label.value_counts().to_dict(),{'cc':3,'sn':2})
        self.assertEqual(len(sample),5)
        self.assertTrue(sample.equals(sample_label_quotas(train,valid,19)))

    def test_overlap_union_not_sum(self):
        self.assertAlmostEqual(union_duration([(1,2),(1.5,2.5),(3,4)]),2.5)
        calls = pd.DataFrame([dict(call_id='a',_clip='clip',_start=1.,_end=2.)])
        events = pd.DataFrame([dict(call_id='a',source_id='clip',label='cc',
            start_time_sec=1.,end_time_sec=2.,focal=True),
            dict(call_id='b',source_id='clip',label='sn',start_time_sec=1.5,end_time_sec=9.,focal=False),
            dict(call_id='noise',source_id='clip',label='beep',start_time_sec=1.,end_time_sec=2.,focal=False)])
        out = make_overlap_table(calls,events).iloc[0]
        self.assertAlmostEqual(out.control_overlap_fraction,.5)
        self.assertAlmostEqual(out.control_nonfocal_overlap_fraction,.5)
        self.assertEqual(out.control_overlap_count,1)

    def test_support_full_focal_no_future(self):
        for start,end in [(.01,.05),(1.,1.04),(1.,2.)]:
            plan = support_plan(start,end,8000,80000,500)
            self.assertEqual(plan['source_end_sample'],round(end*8000))
            self.assertEqual(plan['right_pad_samples'],0)
            self.assertEqual(plan['focal_end_sample']-plan['focal_start_sample'],round((end-start)*8000))
            self.assertGreaterEqual(plan['n_samples'],4000)

    def test_mask_before_normalization(self):
        x = np.arange(20,dtype=np.float32)
        changed = x.copy()
        changed[15:] += 10000
        a,_ = masked_normalize(x,15,20,2,'context_only')
        b,_ = masked_normalize(changed,15,20,2,'context_only')
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(a[15:]==0))
        x2 = x.copy()
        x2[:15] = -100000
        c,_ = masked_normalize(x,15,20,2,'call_only')
        d,_ = masked_normalize(x2,15,20,2,'call_only')
        np.testing.assert_array_equal(c,d)
        for mode in MODES:
            out,_ = masked_normalize(x,15,20,2,mode)
            self.assertTrue(np.all(out[:2]==0))

    def test_context_only_all_masked_is_flagged(self):
        out,stats = masked_normalize(np.arange(20,dtype=np.float32),0,20,0,'context_only')
        self.assertFalse(stats['informative_input'])
        self.assertEqual(stats['normalization_n_samples'],0)
        self.assertTrue(np.all(out==0))

    def test_future_waveform_changes_do_not_change_input(self):
        rate = 8000
        x = np.random.default_rng(7).normal(size=rate).astype(np.float32)
        plan = support_plan(.3,.4,rate,rate,500)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.wav'
            sf.write(path,x,rate,subtype='FLOAT')
            before = read_allowed_window(path,plan)
            x[3200:] *= -100
            sf.write(path,x,rate,subtype='FLOAT')
            after = read_allowed_window(path,plan)
            np.testing.assert_array_equal(before,after)
            for mode in MODES:
                a,_ = masked_normalize(before,plan['focal_start_sample'],plan['focal_end_sample'],
                                      plan['left_pad_samples'],mode)
                b,_ = masked_normalize(after,plan['focal_start_sample'],plan['focal_end_sample'],
                                      plan['left_pad_samples'],mode)
                np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main(verbosity=2)
