"""Complete-route coalition games, signed dividends and recovery-grounded shares."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping, Iterable
import math
import numpy as np

from .schema import Plan, Scenario, Task
from .physics import Evaluation, effective_route, evaluate
from .recovery import probe

Interaction = tuple[str, ...]


def subsets(players: Iterable[str], include_empty: bool = True):
    players = tuple(sorted(players))
    for size in range(0 if include_empty else 1, len(players)+1):
        yield from combinations(players, size)


class FixedRouteGame:
    """Retain complete routes and the same q; never reoptimize a coalition.

    The empty baseline includes unavoidable travel already committed in q.
    An instance rejects a physical model lacking feasible route deletion.
    """
    def __init__(self, scenario: Scenario, plan: Plan,
                 evaluator: Callable[[Scenario, Plan], Evaluation] = evaluate):
        self.scenario, self.plan, self.evaluator = scenario, plan, evaluator
        self.players = tuple(c.id for c in scenario.chargers)
        self.reference_id = scenario.fingerprint()
        full = evaluator(scenario, plan)
        empty = evaluator(scenario, Plan(()))
        if not full.feasible or not empty.feasible:
            raise ValueError("The fixed-route game requires feasible full and empty reference plans")
        self.baseline = empty.objective
        self._cache = {frozenset(): 0.0, frozenset(self.players): full.objective-self.baseline}
        self.physical_calls = 2

    def __call__(self, members: Iterable[str]) -> float:
        if self.scenario.fingerprint() != self.reference_id:
            raise RuntimeError("The reference changed during coalition evaluation")
        key = frozenset(members)
        if key-set(self.players):
            raise ValueError("Unknown route player")
        if key not in self._cache:
            result = self.evaluator(self.scenario, self.plan.restrict(key))
            self.physical_calls += 1
            if not result.feasible:
                raise ValueError("Route deletion is infeasible under this reference; game is undefined")
            self._cache[key] = result.objective-self.baseline
        return self._cache[key]

    def enumerate(self, max_players: int = 10) -> dict[Interaction, float]:
        if len(self.players) > max_players:
            raise ValueError("Exact enumeration is restricted to explicitly small audit games")
        return {group: self(group) for group in subsets(self.players)}


def mobius_dividends(values: Mapping[Interaction, float]) -> dict[Interaction, float]:
    """Möbius inversion of a complete, normalized finite coalition game."""
    canonical = {tuple(sorted(k)): float(v) for k,v in values.items()}
    if abs(canonical.get((), math.inf)) > 1e-10:
        raise ValueError("The coalition game must have v(empty)=0")
    players = max(canonical, key=len)
    if set(canonical) != set(subsets(players)):
        raise ValueError("Möbius recovery requires every coalition")
    dividends = {}
    for group in subsets(players, include_empty=False):
        dividends[group] = sum((-1)**(len(group)-len(u))*canonical[u] for u in subsets(group))
    return dividends


def coalition_value(dividends: Mapping[Interaction, float], members: Iterable[str]) -> float:
    members = set(members)
    return float(sum(value for group,value in dividends.items() if set(group)<=members))


def normalized_shares(scores: Mapping[str, float], lambda_: float=.5,
                      tau: float=.5) -> dict[str,float]:
    if not scores or not 0<=lambda_<=1 or not math.isfinite(tau) or tau<=0:
        raise ValueError("Shares need players, lambda in [0,1] and positive temperature")
    keys = tuple(scores)
    data=np.array([scores[i] for i in keys],dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Scores must be finite")
    weights=np.exp((data-data.max())/tau)
    weights=(1-lambda_)/len(keys)+lambda_*weights/weights.sum()
    return dict(zip(keys,map(float,weights)))


def allocate(dividends: Mapping[Interaction,float],
             shares: Mapping[Interaction,Mapping[str,float]] | None=None,
             players: Iterable[str] | None=None) -> dict[str,float]:
    players = tuple(players or sorted({i for group in dividends for i in group}))
    result=dict.fromkeys(players,0.0)
    for group,value in dividends.items():
        if not group:
            raise ValueError("Empty interactions do not receive shares")
        alpha = shares[group] if shares is not None and group in shares else {i:1/len(group) for i in group}
        if set(alpha)!=set(group) or min(alpha.values()) < -1e-12 or abs(sum(alpha.values())-1)>1e-10:
            raise ValueError("Each interaction has a normalized nonnegative share vector")
        for i in group:
            result[i] += alpha[i]*value
    return result


def tasks_for_plan(scenario: Scenario, plan: Plan) -> tuple[Task,...]:
    """Explicit remote dependencies plus routes committing service to one recipient."""
    tasks=list(scenario.tasks)
    for recipient in scenario.recipients:
        members=tuple(c.id for c in scenario.chargers if any(
            s.recipient_id==recipient.id and s.energy>0
            for s in effective_route(scenario,plan,c.id).services))
        if members:
            tasks.append(Task('service:'+recipient.id,members,(recipient.id,)))
    return tuple(tasks)


def interaction_task(scenario: Scenario, plan: Plan, group: Interaction) -> Task | None:
    associated=[t for t in tasks_for_plan(scenario,plan) if set(group)<=set(t.route_ids)]
    recipients=tuple(sorted({r for task in associated for r in task.recipient_ids}))
    return Task('interaction:'+'+'.join(group),group,recipients) if recipients else None


def retained_support(scenario: Scenario, plan: Plan, order: int=2) -> tuple[Interaction,...]:
    if order<1:
        raise ValueError("Support order must be positive")
    support={(c.id,) for c in scenario.chargers}
    for task in tasks_for_plan(scenario,plan):
        for size in range(2,min(order,len(task.route_ids))+1):
            support.update(combinations(sorted(task.route_ids),size))
    return tuple(sorted(support,key=lambda t:(len(t),t)))


def urgency(scenario: Scenario, plan: Plan, task: Task, player: str) -> float:
    recipients={r.id:r for r in scenario.recipients}
    weighted=total=0.0
    for service in effective_route(scenario,plan,player).services:
        if service.recipient_id in task.recipient_ids:
            r=recipients[service.recipient_id]
            if r.arrival<=service.slot<r.departure:
                weighted+=service.energy*max(0.0,min(1.0,1-r.departure/scenario.horizon))
                total+=service.energy
    return weighted/total if total else 0.0


@dataclass(frozen=True)
class RecoveryAllocation:
    shares: Mapping[Interaction,Mapping[str,float]]
    difficulties: Mapping[Interaction,Mapping[str,float]]
    urgencies: Mapping[Interaction,Mapping[str,float]]
    allocation: Mapping[str,float]
    recovery_checker_calls: int


def physical_allocation(scenario: Scenario, plan: Plan,
                        dividends: Mapping[Interaction,float],lambda_:float=.5,
                        tau:float=.5,eta:float=.5, *,
                        evaluator: Callable[[Scenario, Plan], Evaluation] = evaluate) -> RecoveryAllocation:
    """Physical recovery shares; online callers supply their shared budget checker."""
    if eta<0:
        raise ValueError("Urgency weight must be nonnegative")
    shares, difficulties, urgencies = {}, {}, {}
    calls=0
    for group in dividends:
        task=interaction_task(scenario,plan,group)
        if task is None:
            shares[group]={i:1/len(group) for i in group}
            difficulties[group]=dict.fromkeys(group,0.0)
            urgencies[group]=dict.fromkeys(group,0.0)
        else:
            difficulties[group], urgencies[group] = {}, {}
            for i in group:
                result=probe(scenario,plan,task,i,evaluator=evaluator)
                calls+=result.checker_calls
                difficulties[group][i]=result.difficulty
                urgencies[group][i]=urgency(scenario,plan,task,i)
            scores={i:difficulties[group][i]+eta*urgencies[group][i] for i in group}
            shares[group]=normalized_shares(scores,lambda_,tau)
    return RecoveryAllocation(shares,difficulties,urgencies,
        allocate(dividends,shares,(c.id for c in scenario.chargers)),calls)


def fixed_share_contrast(p: Mapping[Interaction,float], q: Mapping[Interaction,float],
                         shares: Mapping[Interaction,Mapping[str,float]],
                         players: Iterable[str]) -> dict[str,float]:
    return allocate({t:p.get(t,0)-q.get(t,0) for t in p.keys()|q.keys()},shares,players)


def allocation_error_bound(truth: Mapping[Interaction,float],
                           learned: Mapping[Interaction,float],
                           alpha: Mapping[Interaction,Mapping[str,float]],
                           alpha_hat: Mapping[Interaction,Mapping[str,float]],
                           score_error: Mapping[Interaction,float],lambda_:float,tau:float):
    """Theorem 2's three global terms, including omitted low-order interactions."""
    if not 0<=lambda_<=1 or tau<=0:
        raise ValueError("Invalid sharing parameters")
    first=sum(abs(v-truth.get(t,0)) for t,v in learned.items())
    second=lambda_/tau*sum(abs(truth.get(t,0))*score_error[t] for t in learned)
    tail=sum(abs(v) for t,v in truth.items() if t not in learned)
    players={i for t in truth.keys()|learned.keys() for i in t}
    exact=allocate(truth,alpha,players)
    prediction=allocate(learned,alpha_hat,players)
    actual=sum(abs(prediction[i]-exact[i]) for i in players)
    return {'dividend':first,'recovery':second,'tail':tail,'bound':first+second+tail,'actual':actual}
