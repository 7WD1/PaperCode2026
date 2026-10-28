"""Matched Spearman differences with scenario-by-training-seed resampling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .statistics import holm

KEYS=['scenario_id','simulation_seed','training_seed','route_id','subset']


def paired_scores(frame: pd.DataFrame, subject: str, baseline: str, subset: str) -> pd.DataFrame:
    required=set(KEYS+['method','prediction','service_loss'])
    if not required<=set(frame):
        raise ValueError('Raw correlation input needs paired route predictions and physical service-loss scores')
    data=frame.loc[frame['subset']==subset].copy()
    if data.duplicated(KEYS+['method']).any():
        raise ValueError('Duplicate attribution observation')
    left=data.loc[data.method==subject].set_index(KEYS)
    right=data.loc[data.method==baseline].set_index(KEYS)
    if left.empty or not left.index.equals(right.reindex(left.index).dropna().index) or set(left.index)!=set(right.index):
        raise ValueError('Correlation methods require the same complete matched observations')
    right=right.reindex(left.index)
    if not np.allclose(left.service_loss.to_numpy(float),right.service_loss.to_numpy(float),rtol=0,atol=1e-12):
        raise ValueError('Both attributions must use the same independent physical target')
    result=left.reset_index()[KEYS].copy()
    result['subject']=left.prediction.to_numpy(float)
    result['baseline']=right.prediction.to_numpy(float)
    result['target']=left.service_loss.to_numpy(float)
    if not np.isfinite(result[['subject','baseline','target']].to_numpy()).all():
        raise ValueError('Correlation scores must be finite')
    if result.groupby('scenario_id').simulation_seed.nunique().max()!=1:
        raise ValueError('A scenario ID must retain its shared simulation seed')
    return result


def crossed_difference(data: pd.DataFrame, resamples:int=2000, seed:int=20261028,
                       confidence:float=.95) -> dict:
    if resamples<1 or not 0<confidence<1:
        raise ValueError('Positive resampling count and valid confidence are required')
    scenarios=tuple(sorted(data.scenario_id.unique()))
    seeds=tuple(sorted(data.training_seed.unique()))
    grid={(s,z):g for (s,z),g in data.groupby(['scenario_id','training_seed'])}
    if set(grid)!={(s,z) for s in scenarios for z in seeds}:
        raise ValueError('Crossed inference requires the declared complete scenario-by-seed grid')
    def correlation(frame,column):
        if frame[column].nunique()<2 or frame.target.nunique()<2:
            return float('nan')
        return float(spearmanr(frame[column],frame.target).statistic)
    subject,baseline=correlation(data,'subject'),correlation(data,'baseline')
    if not np.isfinite([subject,baseline]).all():
        raise ValueError('Spearman correlation is undefined for constant score vectors')
    rng=np.random.Generator(np.random.PCG64(seed))
    differences=[]
    for _ in range(resamples):
        ss=rng.choice(scenarios,len(scenarios),replace=True)
        zz=rng.choice(seeds,len(seeds),replace=True)
        sample=pd.concat([grid[s,z] for s in ss for z in zz],ignore_index=True)
        delta=correlation(sample,'subject')-correlation(sample,'baseline')
        if np.isfinite(delta):
            differences.append(delta)
    if len(differences)<max(20,int(.9*resamples)):
        raise ValueError('Too many degenerate correlation resamples for a stable interval')
    a=(1-confidence)/2
    low,high=np.quantile(differences,[a,1-a])
    # Paired percentile sign-tail evidence, with finite-resample correction.
    values=np.asarray(differences)
    p=min(1.,2*min((np.count_nonzero(values<=0)+1)/(len(values)+1),
                  (np.count_nonzero(values>=0)+1)/(len(values)+1)))
    return dict(rho_subject=subject,rho_baseline=baseline,difference=subject-baseline,
                ci=[float(low),float(high)],p_raw=float(p),
                resamples_requested=resamples,resamples_valid=len(values),
                scenarios=len(scenarios),training_seeds=len(seeds),observations=len(data),
                inference='Paired crossed-cluster percentile bootstrap; route rows are repeated observations')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--subject',default='RC-ShapleyNet')
    parser.add_argument('--baseline',default='Std-Shapley')
    parser.add_argument('--subsets',nargs='+',default=['all','heterogeneous','symmetric'])
    parser.add_argument('--resamples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    frame=pd.read_csv(args.input,dtype={k:str for k in KEYS+['method']})
    rows=[]
    for subset in args.subsets:
        result=crossed_difference(paired_scores(frame,args.subject,args.baseline,subset),args.resamples,args.seed)
        rows.append(dict(subset=subset,**result))
    for result,p in zip(rows,holm([r['p_raw'] for r in rows])):
        result['p_holm']=float(p)
    payload=dict(family='E15_rho',subsets=args.subsets,results=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
