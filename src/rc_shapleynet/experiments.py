"""Explicit adapters and matched episode logging for supplied benchmark inputs."""
from __future__ import annotations
from copy import deepcopy
import csv
import importlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable,Mapping
import numpy as np
from .schema import Scenario,Plan,scenario_from_dict
from .simulation import run_episode


def load_adapter(specification:str,configuration:dict|None=None) -> Callable[[Scenario],Plan]:
    """Load a real external baseline factory, without replacing it with a stub.

    The declared module is normal user-supplied Python code. A factory receives
    configuration and returns a planner accepting immutable Scenario inputs.
    """
    if ':' not in specification:
        raise ValueError('An adapter is an explicit module:factory, not an inferred baseline name')
    module,name=specification.split(':',1)
    factory=getattr(importlib.import_module(module),name)
    planner=factory(configuration or {})
    if not callable(planner):
        raise TypeError('A baseline factory must return a planner callable')
    return planner


def learned_planner(configuration:dict) -> Callable[[Scenario],Plan]:
    from .training import load_checkpoint
    from .search import LearnedGuide,SearchConfig,search
    from .reconstruction import idle_return_route
    model,normalizer,payload=load_checkpoint(configuration['checkpoint'],device=configuration.get('device','cpu'))
    declared_seed=configuration.get('training_seed')
    if declared_seed is not None:
        fitted_seed=payload.get('training_configuration',{}).get('seed')
        if fitted_seed != declared_seed:
            raise ValueError('The declared training seed does not match the fitted checkpoint seed')
    sharing=configuration.get('sharing',{})
    guide=LearnedGuide(model,normalizer=normalizer,**sharing)
    search_parameters=dict(configuration.get('search',{}))
    if 'exploration' in search_parameters:
        alias=search_parameters.pop('exploration')
        if ('exploration_probability' in search_parameters and
                search_parameters['exploration_probability'] != alias):
            raise ValueError('Conflicting exploration and exploration_probability settings')
        search_parameters['exploration_probability']=alias
    config=SearchConfig(**search_parameters)
    def planner(scenario):
        width=model.encoder.recipient_features.embedding.num_embeddings
        if any(r.location_index >= width for r in scenario.recipients):
            raise ValueError('A location ID exceeds the checkpoint vocabulary; apply its frozen unknown-ID mapping')
        fallback=Plan(tuple(idle_return_route(scenario,c) for c in scenario.chargers))
        result=search(scenario,fallback,guide,config=config)
        if result.final_guidance is None:
            raise RuntimeError('The declared budget did not complete final attribution')
        return result.best_plan
    return planner


def canonical_metrics(metrics:Mapping[str,float]) -> dict[str,float]:
    """Log each endpoint once; conflicting compatibility aliases are errors."""
    aliases={'completion':'completion_rate','C_cov':'coverage','spatial_coverage':'coverage',
             'C_conn':'connectivity','C_alive':'powered_node_time'}
    result={}
    for name,value in metrics.items():
        name=aliases.get(name,name)
        value=float(value)
        if not math.isfinite(value):
            raise ValueError('An episode produced a nonfinite endpoint')
        if name in result and not math.isclose(result[name],value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f'Conflicting values for canonical episode metric {name}')
        result[name]=value
    return result


def _immutable_planner(planner:Callable[[Scenario],Plan],
                       decision_times:list[float]) -> Callable[[Scenario],Plan]:
    def checked(scenario):
        reference=scenario.fingerprint()
        started=perf_counter()
        plan=planner(scenario)
        elapsed=perf_counter()-started
        if scenario.fingerprint() != reference:
            raise RuntimeError('A method changed a shared planning reference')
        if not isinstance(plan,Plan):
            raise TypeError('A method adapter must return a complete Plan')
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError('The decision timer returned an invalid duration')
        decision_times.append(elapsed)
        return plan
    return checked


def run_manifest(manifest_path:str|Path,output:str|Path) -> dict:
    """Run explicitly supplied scenarios and adapters; no aggregate values are replayed.

    Every method sees the same frozen scenario, simulation seed and exogenous
    loads. One row is one whole-episode endpoint for a training-seed repeat.
    """
    path=Path(manifest_path)
    target=Path(output)
    if target.exists():
        raise FileExistsError('Use a new output path; test records are not silently overwritten')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if not manifest.get('scenarios') or not manifest.get('methods'):
        raise ValueError('A run manifest needs actual scenario files and implemented method adapters')
    training_seeds=manifest.get('training_seeds',[1,2,3,4,5])
    if (not isinstance(training_seeds,list) or not training_seeds
            or any(not isinstance(s,int) or isinstance(s,bool) or s < 0
                                  for s in training_seeds)
            or len(training_seeds)!=len(set(training_seeds))):
        raise ValueError('Declare nonempty, distinct, nonnegative integer training seeds')
    method_names=[m.get('name') for m in manifest['methods']]
    if (any(not isinstance(name,str) or not name or name != name.strip() for name in method_names)
            or len(set(method_names)) != len(method_names)):
        raise ValueError('Method names must be nonempty and distinct within a manifest')
    replan_seconds=manifest.get('replan_seconds',60)
    if (not isinstance(replan_seconds,(int,float)) or isinstance(replan_seconds,bool)
            or not math.isfinite(replan_seconds) or replan_seconds <= 0):
        raise ValueError('Replanning time must be finite and positive')
    rows=[]
    seen=set()
    for entry in manifest['scenarios']:
        source=Path(entry['input'])
        if not source.is_absolute():
            source=path.parent/source
        payload=json.loads(source.read_text(encoding='utf-8'))
        scenario=scenario_from_dict(payload.get('scenario',payload))
        interval=replan_seconds/scenario.step_seconds
        if not math.isclose(interval,round(interval),rel_tol=0,abs_tol=1e-9) or interval < 1:
            raise ValueError('Replanning time must be an integer number of physical slots')
        key=(entry['condition_id'],scenario.scenario_id,scenario.simulation_seed)
        if key in seen:
            raise ValueError('Duplicate scenario entry in the declared family')
        seen.add(key)
        fingerprint=scenario.fingerprint()
        for seed in training_seeds:
            for method in manifest['methods']:
                configuration={**deepcopy(method.get('configuration',{})),'training_seed':seed,
                               'simulation_seed':scenario.simulation_seed}
                # Checkpoints can be explicitly keyed by the fitted training seed.
                if 'checkpoints' in method:
                    if str(seed) not in method['checkpoints']:
                        raise ValueError(f'No fitted checkpoint was declared for training seed {seed}')
                    configuration['checkpoint']=method['checkpoints'][str(seed)]
                if 'checkpoint' in configuration:
                    checkpoint=Path(configuration['checkpoint'])
                    configuration['checkpoint']=str((path.parent/checkpoint).resolve()
                        if not checkpoint.is_absolute() else checkpoint)
                # Loading a model or adapter belongs outside decision timing.
                adapter=load_adapter(method['adapter'],configuration)
                decision_times=[]
                planner=_immutable_planner(adapter,decision_times)
                result=run_episode(scenario,planner,int(round(interval)))
                if scenario.fingerprint()!=fingerprint:
                    raise RuntimeError('A method changed the shared physical reference')
                endpoints=canonical_metrics(result.metrics)
                if 'objective' in endpoints and not math.isclose(endpoints['objective'],result.objective,
                                                                rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Episode objective conflicts with the recorded metric')
                endpoints['objective']=result.objective
                if not math.isfinite(result.objective):
                    raise ValueError('An episode produced a nonfinite objective')
                if not decision_times:
                    raise RuntimeError('An episode completed without a planner decision')
                latency_metrics={
                    'decision_latency_p50_seconds':float(np.percentile(decision_times,50)),
                    'decision_latency_p95_seconds':float(np.percentile(decision_times,95)),
                    'decision_count':len(decision_times)}
                if set(endpoints)&set(latency_metrics):
                    raise ValueError('Episode metrics cannot replace measured decision timing')
                endpoints.update(latency_metrics)
                for metric,value in endpoints.items():
                    rows.append(dict(experiment_id=manifest['experiment_id'],condition_id=entry['condition_id'],
                        domain=scenario.domain,profile=manifest.get('profile','query_controlled'),metric=metric,
                        scenario_id=scenario.scenario_id,simulation_seed=scenario.simulation_seed,
                        training_seed=seed,method=method['name'],value=float(value)))
    target.parent.mkdir(parents=True,exist_ok=True)
    columns=['experiment_id','condition_id','domain','profile','metric','scenario_id','simulation_seed','training_seed','method','value']
    with target.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    return {'rows':len(rows),'scenarios':len(seen),'training_seeds':training_seeds,
            'methods':[m['name'] for m in manifest['methods']]}
