"""SYNTHETIC TEST FIXTURES ONLY. No manuscript performance data are generated."""
from pathlib import Path
import copy
import importlib.util
import json
import subprocess
import sys
import unittest
import tempfile
import atexit

_TEMP = tempfile.TemporaryDirectory(prefix='rc-statistics-test-')
atexit.register(_TEMP.cleanup)
HERE = Path(_TEMP.name)
import numpy as np
import pandas as pd
import yaml
from rc_shapleynet import statistics as ps


def synthetic_member(mid='SYN_PAIR', condition='SYN_CONDITION'):
    return dict(id=mid, kind='paired', subject='SYN_SUBJECT', baseline='SYN_BASELINE',
                direction='higher', minimum_effect=0.0,
                context=dict(experiment_id='SYN_EXPERIMENT', condition_id=condition,
                             domain='SYN_DOMAIN', profile='SYN_PROFILE', metric='SYN_METRIC'))


def synthetic_config():
    return dict(schema_version=1, data_kind='synthetic_fixture', pair_keys=list(ps.PAIR_KEYS),
                selected_families=['SYN_PAIRED'], defaults=dict(subject='SYN_SUBJECT',
                expected_scenarios=3, expected_training_seeds=2, bootstrap_resamples=400,
                permutations=600, confidence=.95, alpha=.05, random_seed=123,
                difference_decimals=None, wilcoxon_method='approx'),
                families=[dict(id='SYN_PAIRED', expected_size=1, members=[synthetic_member()])])


def synthetic_rows():
    rows=[]
    for i in range(3):
        for j in range(2):
            for name, value in [('SYN_SUBJECT', 2+i+j), ('SYN_BASELINE', 1+i+j)]:
                rows.append(dict(**synthetic_member()['context'], scenario_id='SYN_S'+str(i),
                                 simulation_seed='SYN_SIM'+str(i), training_seed='SYN_T'+str(j),
                                 method=name, value=float(value)))
    return pd.DataFrame(rows)


class StatisticsTests(unittest.TestCase):
    def test_100_scenarios_and_5_training_repeats_not_500_units(self):
        rows=[]
        for i in range(100):
            for j in range(5):
                for method,value in [('SYN_SUBJECT',2+i+j),('SYN_BASELINE',1+i+j)]:
                    rows.append(dict(**synthetic_member()['context'],scenario_id='SYN_S'+str(i),
                                     simulation_seed='SYN_SIM'+str(i),training_seed='SYN_T'+str(j),
                                     method=method,value=float(value)))
        c=synthetic_config()
        c['defaults'].update(expected_scenarios=100,expected_training_seeds=5,
                             bootstrap_resamples=100,permutations=200)
        row=ps.analyze(pd.DataFrame(rows),c)['results'][0]
        self.assertEqual(row['n_matched_episode_summaries'],500)
        self.assertEqual(row['n_scenarios'],100)
        self.assertEqual(row['n_nonzero_scenario_differences'],100)
        self.assertEqual(row['mean_crossed_bootstrap_ci'],[1,1])

    def test_three_key_pair_and_repeated_measure_unit(self):
        result=ps.analyze(synthetic_rows(), synthetic_config())['results'][0]
        self.assertEqual(result['n_scenarios'],3)
        self.assertEqual(result['n_training_seeds'],2)
        self.assertEqual(result['n_matched_episode_summaries'],6)
        self.assertEqual(result['mean_difference'],1)
        self.assertEqual(result['median_difference'],1)
        self.assertEqual(result['mean_crossed_bootstrap_ci'],[1,1])
        self.assertEqual(result['median_crossed_bootstrap_ci'],[1,1])
        self.assertEqual(result['rank_biserial'],1)
        self.assertEqual(result['profile'],'SYN_PROFILE')

    def test_simulation_seed_mismatch_rejected(self):
        d=synthetic_rows(); d.loc[1,'simulation_seed']='SYN_WRONG'
        with self.assertRaisesRegex(ValueError,'Simulation-seed mismatch'):
            ps.analyze(d,synthetic_config())

    def test_duplicate_training_seed_rejected(self):
        d=synthetic_rows(); d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'Duplicate training-seed'):
            ps.analyze(d,synthetic_config())

    def test_unmatched_training_identity_rejected(self):
        d=synthetic_rows()
        d.loc[(d.method=='SYN_BASELINE') & (d.training_seed=='SYN_T1'),'training_seed']='SYN_T9'
        with self.assertRaisesRegex(ValueError,'unmatched scenario/simulation/training'):
            ps.analyze(d,synthetic_config())

    def test_symmetric_missing_cell_rejected(self):
        d=synthetic_rows()
        d=d.loc[~((d.scenario_id=='SYN_S0') & (d.training_seed=='SYN_T0'))]
        with self.assertRaisesRegex(ValueError,'incomplete crossed'):
            ps.analyze(d,synthetic_config())

    def test_default_scenario_count_enforced(self):
        c=synthetic_config(); c['defaults']['expected_scenarios']=100
        with self.assertRaisesRegex(ValueError,'expected 100 independent scenarios'):
            ps.analyze(synthetic_rows(),c)

    def test_zero_differences(self):
        d=synthetic_rows(); d.loc[d.method=='SYN_SUBJECT','value']-=1
        row=ps.analyze(d,synthetic_config())['results'][0]
        self.assertEqual((row['wilcoxon_W'],row['wilcoxon_raw_p'],row['rank_biserial']),(0,1,0))
        self.assertEqual(row['permutation_raw_p'],1)
        self.assertEqual(row['median_crossed_bootstrap_ci'],[0,0])
        self.assertEqual(row['mean_crossed_bootstrap_ci'],[0,0])
        self.assertFalse(row['original_primary_dagger_eligible'])

    def test_optional_rounding_has_one_consistent_estimand(self):
        rows=[]
        for i in range(3):
            for j,gain in enumerate([0,0,0,1,1]):
                for method,value in [('SYN_SUBJECT',gain),('SYN_BASELINE',0)]:
                    rows.append(dict(**synthetic_member()['context'],scenario_id='SYN_S'+str(i),
                                     simulation_seed='SYN_SIM'+str(i),training_seed='SYN_T'+str(j),
                                     method=method,value=float(value)))
        c=synthetic_config();c['defaults'].update(expected_training_seeds=5,difference_decimals=0)
        row=ps.analyze(pd.DataFrame(rows),c)['results'][0]
        self.assertAlmostEqual(row['mean_difference'],.4)
        self.assertAlmostEqual(row['median_difference'],.4)
        self.assertEqual(row['n_nonzero_scenario_differences'],3)
        self.assertLess(row['wilcoxon_raw_p'],1)
        self.assertLessEqual(row['median_crossed_bootstrap_ci'][0],.4)
        self.assertGreaterEqual(row['median_crossed_bootstrap_ci'][1],.4)

    def test_missing_effect_threshold_rejected(self):
        c=synthetic_config(); del c['families'][0]['members'][0]['minimum_effect']
        with self.assertRaisesRegex(ValueError,'explicit minimum_effect'):
            ps.analyze(synthetic_rows(),c)

    def test_fraction_rate_zero_threshold_rejected(self):
        c=synthetic_config();c['families'][0]['members'][0]['effect_unit']='fraction_rate'
        with self.assertRaisesRegex(ValueError,'1.0 percentage point'):
            ps.analyze(synthetic_rows(),c)

    def test_sub_one_pp_effect_cannot_receive_original_dagger(self):
        records=[dict(wilcoxon_raw_p=.001,permutation_raw_p=.001,
                      mean_crossed_bootstrap_ci=[.004,.006],mean_difference=.005,
                      minimum_mean_effect=.01,median_crossed_bootstrap_ci=[.004,.006],
                      median_difference=.005,minimum_median_effect=.01)]
        row=ps.adjust_family(records,'SYN_RATE',1,.05)[0]
        self.assertFalse(row['original_primary_dagger_eligible'])
        self.assertFalse(row['wilcoxon_dagger_eligible'])

    def test_wilcoxon_median_gate_is_separate_from_original_mean_gate(self):
        base=dict(wilcoxon_raw_p=.001,permutation_raw_p=.001,
                  mean_crossed_bootstrap_ci=[.01,.03],mean_difference=.02,minimum_mean_effect=.01,
                  median_crossed_bootstrap_ci=[.004,.007],median_difference=.005,minimum_median_effect=.01)
        row=ps.adjust_family([base],'SYN_RATE',1,.05)[0]
        self.assertTrue(row['original_primary_dagger_eligible'])
        self.assertTrue(row['dagger_eligible'])
        self.assertFalse(row['wilcoxon_dagger_eligible'])
        base.update(permutation_raw_p=.2,mean_crossed_bootstrap_ci=[-.001,.02],mean_difference=.009,
                    median_crossed_bootstrap_ci=[.01,.03],median_difference=.02)
        row=ps.adjust_family([base],'SYN_RATE',1,.05)[0]
        self.assertFalse(row['dagger_eligible'])
        self.assertTrue(row['wilcoxon_dagger_eligible'])

    def test_rbc_hand_calculation_with_zero_and_ties(self):
        # Nonzero absolute ranks: 1.5,1.5,4,4,4. Positive=9.5, negative=5.5.
        r=ps.matched_rank_biserial([0,1,-1,2,-2,2])
        self.assertEqual((r['W_plus'],r['W_minus'],r['W']),(9.5,5.5,5.5))
        self.assertAlmostEqual(r['rank_biserial'],4/15)
        self.assertEqual(r['n_zero'],1)
        self.assertEqual(r['n_tied_absolute_groups'],2)
        self.assertEqual(ps.matched_rank_biserial([0,-1,-1])['rank_biserial'],-1)

    def test_wilcoxon_uses_three_scenario_means(self):
        d=synthetic_rows()
        for i,gain in enumerate([1.,-1.,2.]):
            ids=(d.scenario_id=='SYN_S'+str(i)) & (d.method=='SYN_SUBJECT')
            d.loc[ids,'value']+=gain-1
        row=ps.analyze(d,synthetic_config())['results'][0]
        self.assertEqual(row['n_nonzero_scenario_differences'],3)
        # abs(1),abs(-1),abs(2) have ranks 1.5,1.5,3.
        self.assertEqual(row['wilcoxon_W'],1.5)
        self.assertAlmostEqual(row['rank_biserial'],.5)

    def test_crossed_bootstrap_resamples_training_seed_axis(self):
        r=ps.crossed_bootstrap([[0,10],[0,10]],4000,.95,np.random.default_rng(9))
        self.assertEqual(r['mean_ci'],[0,10])
        self.assertEqual(r['median_ci'],[0,10])
        # Scenario-only resampling would incorrectly return [5,5].

    def test_crossed_bootstrap_resamples_scenario_axis(self):
        r=ps.crossed_bootstrap([[0,0],[10,10]],4000,.95,np.random.default_rng(9))
        self.assertEqual(r['mean_ci'],[0,10])
        self.assertEqual(r['median_ci'],[0,10])

    def test_holm_cumulative_max_and_separate_families(self):
        def records(p):
            return [dict(wilcoxon_raw_p=x,permutation_raw_p=x,
                         mean_crossed_bootstrap_ci=[.1,1],mean_difference=.5,
                         minimum_mean_effect=0,median_crossed_bootstrap_ci=[.1,1],
                         median_difference=.5,minimum_median_effect=0) for x in p]
        a=ps.adjust_family(records([.01,.04,.03]),'SYN_FAMILY_A',3,.05)
        b=ps.adjust_family(records([.2,.8]),'SYN_FAMILY_B',2,.05)
        np.testing.assert_allclose([r['wilcoxon_holm_p'] for r in a],[.03,.06,.06])
        np.testing.assert_allclose([r['wilcoxon_holm_p'] for r in b],[.4,.8])
        self.assertEqual([r['family_size'] for r in a],[3,3,3])

    def test_missing_declared_member_does_not_shrink_family(self):
        c=synthetic_config()
        c['families'][0]['members'].append(synthetic_member('SYN_MISSING','SYN_ABSENT_CONDITION'))
        c['families'][0]['expected_size']=2
        with self.assertRaisesRegex(ValueError,'Holm family will not be reduced'):
            ps.analyze(synthetic_rows(),c)

    def test_four_group_did_is_joint_contrast(self):
        rows=[]
        for i in range(3):
            for j in range(2):
                for method,value in [('SYN_NONE',1),('SYN_ALLOC',2),('SYN_NEIGH',3),('SYN_BOTH',5)]:
                    rows.append(dict(**synthetic_member()['context'],scenario_id='SYN_S'+str(i),
                                     simulation_seed='SYN_SIM'+str(i),training_seed='SYN_T'+str(j),
                                     method=method,value=value+i+j))
        member=synthetic_member('SYN_DID')
        member.update(kind='linear_contrast',weights={'SYN_BOTH':1,'SYN_ALLOC':-1,'SYN_NEIGH':-1,'SYN_NONE':1})
        c=synthetic_config(); c['families'][0]['members']=[member]
        row=ps.analyze(pd.DataFrame(rows),c)['results'][0]
        self.assertEqual(row['mean_difference'],1)
        self.assertEqual(row['median_crossed_bootstrap_ci'],[1,1])
        # The conditional BOTH-NEIGH contrast is 2 and is not the DID.
        self.assertNotEqual(row['mean_difference'],2)

    def test_manual_correlation_pipeline_explicitly_unimplemented(self):
        c=synthetic_config(); c['families'][0]['members'][0]['kind']='manual_correlation_difference'
        with self.assertRaisesRegex(NotImplementedError,'not implemented'):
            ps.analyze(synthetic_rows(),c)

    def test_author_placeholder_rejected(self):
        c=synthetic_config(); c['families'][0]['members'][0]['requires_author_definition']=True
        with self.assertRaisesRegex(ValueError,'requires_author_definition'):
            ps.analyze(synthetic_rows(),c)

    def test_old_ambiguous_seed_schema_rejected(self):
        d=synthetic_rows().rename(columns={'simulation_seed':'seed'}).drop(columns='training_seed')
        with self.assertRaisesRegex(ValueError,'single seed'):
            ps.analyze(d,synthetic_config())

    def test_nonfinite_value_rejected(self):
        d=synthetic_rows(); d.loc[0,'value']=np.inf
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            ps.analyze(d,synthetic_config())

    def test_reproducibility(self):
        a=ps.analyze(synthetic_rows(),synthetic_config())
        b=ps.analyze(synthetic_rows(),synthetic_config())
        self.assertEqual(a,b)

    def test_config_family_size_guard(self):
        c=synthetic_config();c['families'][0]['expected_size']=3
        f=HERE/'synthetic_bad_config.yaml';f.write_text(yaml.safe_dump(c),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'expected_size'):
            ps.load_config(f)

    def test_cli_end_to_end_synthetic_only(self):
        f=HERE/'synthetic_episode_rows.csv'; synthetic_rows().to_csv(f,index=False)
        c=HERE/'synthetic_config.yaml';c.write_text(yaml.safe_dump(synthetic_config()),encoding='utf-8')
        target=HERE/'synthetic_cli_results.json'
        proc=subprocess.run([sys.executable,'-m','rc_shapleynet.statistics',str(f),
                             '--config',str(c),'--family','SYN_PAIRED','--out',str(target)],
                             capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        result=json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(result['data_kind'],'synthetic_fixture')
        self.assertEqual(result['results'][0]['family_size'],1)
        self.assertTrue(target.with_suffix('.csv').exists())
        self.assertIn('input_sha256',result['provenance'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StatisticsTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'kind':'SYNTHETIC SOFTWARE TESTS ONLY; no actual performance observations',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'all_passed':result.wasSuccessful()}
    (HERE/'synthetic_test_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
