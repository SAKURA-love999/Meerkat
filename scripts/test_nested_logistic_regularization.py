"""Technical fixtures only: no actual estimator fits or research results."""
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

import run_nested_logistic_regularization as runner


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
    for k,col in enumerate(cols):
        data[col] = np.sin(i+k)
    args = Namespace(output_dir=Path(folder))
    for stage in ['inner','outer']:
        (args.output_dir/'jobs'/stage).mkdir(parents=True)
    return args,data,cols,[runner.models.RMS]


def references(data):
    parts = []
    for family in ['logistic_linear','spline_rms','hist_gbdt']:
        for target in runner.TARGETS:
            frame = data[runner.b3.FIELDS].copy()
            frame['model_family'],frame['target'] = family,target
            frame['y_true'],frame['m0_pred'],frame['m1_pred'] = data[target],.15,.14
            frame['n_train'],frame['synthetic'] = 144,False
            parts.append(frame)
    return pd.concat(parts,ignore_index=True)


class NestedTests(unittest.TestCase):
    def test_maps_are_shared_deterministic_and_outer_test_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,_,_ = fixture(folder)
            maps = runner.make_inner_maps(data)
            pd.testing.assert_frame_equal(maps,runner.make_inner_maps(data))
            self.assertEqual(len(maps),90*4)
            for fold in range(1,6):
                block = maps.loc[maps.outer_fold.eq(fold)]
                self.assertEqual(set(block.inner_fold),{1,2,3})
                self.assertFalse(set(block['_clip']) & set(data.loc[data.fold.eq(fold),'_clip']))
                for inner in range(1,4):
                    train,val = runner.split_inner(data,maps,fold,inner)
                    self.assertFalse(set(train['_clip']) & set(val['_clip']))
                    self.assertEqual(set(train.index)|set(val.index),set(data.index[data.fold.ne(fold)]))
            changed = data.copy()
            changed['NF_post'] = 1-changed.NF_post
            changed['manual_label'] = 'not_used_for_split'
            pd.testing.assert_frame_equal(maps,runner.make_inner_maps(changed))

    def test_pooled_loss_not_unweighted_fold_loss_and_exact_ties(self):
        rows = []
        for c,losses in [(1.,[.5,.1,.1]),(.1,[.4,.9,.9]),(10.,[.4,.9,.9])]:
            for inner,(n,loss) in enumerate(zip([100,1,1],losses),1):
                rows.append(dict(family='logistic_linear',outer_fold=1,target='NF_post',
                    model='m0',C=c,inner_fold=inner,n=n,loss_sum=n*loss))
        candidates,chosen = runner.choose_candidates(pd.DataFrame(rows))
        self.assertEqual(chosen.iloc[0].selected_C,.1)
        self.assertEqual(chosen.iloc[0].tied_C_json,'[0.1, 10.0]')
        rows = pd.DataFrame(rows)
        rows['loss_sum'] = rows.n*.2
        _,chosen = runner.choose_candidates(rows)
        self.assertEqual(chosen.iloc[0].selected_C,1.)
        with self.assertRaises(ValueError):
            runner.choose_candidates(rows.iloc[:-1])

    def test_factory_freezes_everything_except_C_and_blocks_forbidden_inputs(self):
        for family in runner.FAMILIES:
            model = runner.make_model(family,['duration_s',runner.models.RMS],['a'+str(i) for i in range(35)],.1)
            clf = model.named_steps['clf']
            self.assertEqual(clf.C,.1)
            self.assertEqual(clf.penalty,'l2')
            self.assertEqual(clf.max_iter,2000)
            self.assertEqual(clf.tol,1e-4)
            self.assertEqual(clf.solver,'lbfgs')
            self.assertIsNone(clf.class_weight)
            self.assertFalse(hasattr(clf,'coef_'))
            acoustic = next(t for t in model.named_steps['preprocess'].transformers if t[0]=='acoustic')
            self.assertEqual(acoustic[1].named_steps['pca'].n_components,30)
        for bad in ['NF_post','fold','call_id','sim_mark','egemaps_valid']:
            with self.assertRaises(ValueError):
                runner.make_model('logistic_linear',[runner.models.RMS,bad],[],1.)

    def test_M0_M1_and_PRE_POST_choose_independently(self):
        rows = []
        expected = {('NF_post','m0'):.1,('NF_post','m1'):10.,('NF_pre','m0'):1.,('NF_pre','m1'):.1}
        for (target,model),best in expected.items():
            for c in runner.CS:
                for inner in range(1,4):
                    rows.append(dict(family='spline_rms',outer_fold=1,target=target,model=model,
                        C=c,inner_fold=inner,n=20,loss_sum=20*(.1 if c==best else .2)))
        _,chosen = runner.choose_candidates(pd.DataFrame(rows))
        for row in chosen.itertuples():
            self.assertEqual(row.selected_C,expected[(row.target,row.model)])

    def test_frozen_map_refuses_changed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,_,_ = fixture(folder)
            maps = runner.make_inner_maps(data)
            path = Path(folder)/'map.csv'
            runner.freeze_csv(path,maps)
            runner.freeze_csv(path,maps)
            maps.loc[0,'inner_fold'] = 99
            with self.assertRaises(ValueError):
                runner.freeze_csv(path,maps)

    def test_plan_and_zero_fit_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            _,data,_,_ = fixture(folder)
            plan = runner.make_fit_plan(data,runner.make_inner_maps(data))
            self.assertEqual(len(plan),360)
            self.assertTrue(plan.relative_path.is_unique)
            with self.assertRaisesRegex(RuntimeError,'zero-fit'):
                with runner.zero_fit_guard():
                    from sklearn.linear_model import LogisticRegression
                    LogisticRegression().fit([[0],[1]],[0,1])

    def test_fake_end_to_end_400_fits_resume_summary_and_tamper_detection(self):
        calls = []
        class FakeModel:
            named_steps = {}
            def fit(self,x,y):
                self.train = set(x.index)
                self.clips = {int(i)//2 for i in x.index}
                self.p = float(np.mean(y))
                calls.append((len(x),tuple(x.columns)))
                if any(c in x for c in ['NF_post','NF_pre','call_id','_clip','fold','egemaps_valid']):
                    raise AssertionError('Invalid predictive input.')
                return self
            def predict_proba(self,x):
                if self.train & set(x.index) or self.clips & {int(i)//2 for i in x.index}:
                    raise AssertionError('Leakage across row/clip split.')
                return np.tile([1-self.p,self.p],(len(x),1))
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls = fixture(folder)
            maps = runner.make_inner_maps(data)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=lambda *a:FakeModel()), \
                    patch.object(runner.shared,'model_numeric_columns',return_value=['log_duration_s']):
                counts = runner.fit_all(args,data,cols,controls,maps,'fixture')
            self.assertEqual(counts['model_fits'],400)
            self.assertEqual(counts['created_inner_jobs'],360)
            self.assertEqual(counts['created_outer_jobs'],40)
            self.assertEqual(len(calls),400)
            self.assertEqual(sum(set(cols)<=set(names) for _,names in calls),200)
            with redirect_stdout(io.StringIO()),patch.object(runner,'make_model',side_effect=AssertionError('Resume must not fit')):
                again = runner.fit_all(args,data,cols,controls,maps,'fixture')
                runner.summarize(args,data,maps,references(data),[],'fixture',n_draws=8)
            self.assertEqual(again['model_fits'],0)
            self.assertEqual(again['reused_inner_jobs'],360)
            self.assertEqual(again['reused_outer_jobs'],40)
            metrics = runner.read(args.output_dir/'comparison_metrics_and_ci.csv')
            self.assertEqual(len(metrics),5*2*4)
            self.assertEqual(set(metrics.model_family),{'logistic_linear','spline_rms','hist_gbdt',
                'logistic_linear_tuned','spline_rms_tuned'})
            self.assertEqual(len(runner.read(args.output_dir/'selected_C.csv')),40)
            self.assertEqual(len(runner.read(args.output_dir/'candidate_scores.csv')),120)
            changes = runner.read(args.output_dir/'paired_changes_vs_fixed.csv')
            self.assertEqual(len(changes),2*3*2*4)
            verified = json.loads((args.output_dir/'verification.json').read_text())
            self.assertFalse(verified['conditional_information_certified'])
            self.assertEqual(verified['n_oof_rows'],180*2*5)
            inner = next((args.output_dir/'jobs'/'inner').glob('*.csv'))
            frame = runner.read(inner)
            frame['train_ids_sha256'] = 'tampered'
            frame.to_csv(inner,index=False)
            with self.assertRaises(ValueError):
                runner.inner_scores(args,data,maps,'fixture')
            inner.unlink()
            with self.assertRaises(ValueError):
                runner.inner_scores(args,data,maps,'fixture')

    def test_preprocessing_is_built_fresh_at_fit_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            args,data,cols,controls = fixture(folder)
            train,val = runner.split_inner(data,runner.make_inner_maps(data),1,1)
            row = runner.job_spec(train,val,'inner','logistic_linear',1,'NF_post','m1',1.,1)
            objects = []
            class Spy:
                named_steps = {}
                def fit(self,x,y):
                    self.mean = x[cols[0]].mean()
                    self.rows = set(x.index)
                    return self
                def predict_proba(self,x):
                    self.assertions = self.rows.isdisjoint(x.index)
                    return np.tile([.8,.2],(len(x),1))
            def fresh(*unused):
                obj = Spy()
                objects.append(obj)
                return obj
            with patch.object(runner,'make_model',side_effect=fresh), \
                    patch.object(runner.shared,'model_numeric_columns',return_value=['log_duration_s']):
                runner.execute_job(args,row,train,val,cols,controls,'fixture')
                row = dict(row,C=.1,relative_path='jobs/inner/separate.csv')
                runner.execute_job(args,row,train,val,cols,controls,'fixture')
            self.assertEqual(len(objects),2)
            self.assertIsNot(objects[0],objects[1])
            self.assertEqual(objects[0].mean,train[cols[0]].mean())
            self.assertTrue(all(o.assertions for o in objects))


if __name__=='__main__':
    unittest.main(verbosity=2)
