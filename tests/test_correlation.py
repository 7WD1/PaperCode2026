import numpy as np
import pandas as pd
import pytest
from rc_shapleynet.correlation import paired_scores,crossed_difference


def fixture():
    rows=[]
    for scenario in range(4):
        for seed in range(2):
            for route in range(4):
                target=float(route+scenario*.03)
                for method,prediction in [('RC-ShapleyNet',target),('Std-Shapley',-target)]:
                    rows.append(dict(scenario_id=str(scenario),simulation_seed=str(10001+scenario),
                        training_seed=str(seed),route_id=str(route),subset='all',method=method,
                        prediction=prediction,service_loss=target))
    return pd.DataFrame(rows)


def test_resamples_whole_crossed_cells_and_retains_paired_target():
    data=paired_scores(fixture(),'RC-ShapleyNet','Std-Shapley','all')
    a=crossed_difference(data,resamples=50,seed=4)
    b=crossed_difference(data,resamples=50,seed=4)
    assert a==b
    assert a['scenarios']==4 and a['training_seeds']==2 and a['observations']==32
    assert a['difference']==pytest.approx(2)
    assert a['ci']==pytest.approx([2,2])


def test_rejects_target_mismatch_and_missing_pair_or_cell():
    frame=fixture()
    frame.loc[1,'service_loss']+=.1
    with pytest.raises(ValueError,match='same independent physical target'):
        paired_scores(frame,'RC-ShapleyNet','Std-Shapley','all')
    with pytest.raises(ValueError,match='matched observations'):
        paired_scores(fixture().iloc[1:],'RC-ShapleyNet','Std-Shapley','all')
    data=paired_scores(fixture(),'RC-ShapleyNet','Std-Shapley','all')
    with pytest.raises(ValueError,match='complete scenario-by-seed'):
        crossed_difference(data.loc[~((data.scenario_id=='0')&(data.training_seed=='0'))],50)


def test_zero_correlation_difference_is_not_significant():
    frame=fixture()
    frame['prediction']=frame.service_loss
    result=crossed_difference(paired_scores(frame,'RC-ShapleyNet','Std-Shapley','all'),50)
    assert result['difference']==0 and result['p_raw']==1
