"""Interaction-guided, budget-controlled search with verified incumbents.

The neighborhood policy is specified here rather than attributed to an external
ALNS codebase. Positive interactions favor rebuilding their entire group and
penalize splitting it. Negative interactions favor revisiting one redundant
member. Low route allocations supply a complementary removal preference.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping
from types import MappingProxyType
import inspect
import math
import time
import numpy as np

from .schema import Plan, Scenario
from .physics import Evaluation, evaluate
from .reconstruction import destroy, greedy_reconstruct

Interaction = tuple[str, ...]


@dataclass(frozen=True)
class Guidance:
    allocation: Mapping[str, float]
    interactions: Mapping[Interaction, float]

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in self.allocation.values()):
            raise ValueError("Route allocations must be finite")
        for group, value in self.interactions.items():
            if not group or len(set(group)) != len(group) or not math.isfinite(float(value)):
                raise ValueError("Signed interactions need distinct members and finite values")
            if set(group)-self.allocation.keys():
                raise ValueError("Interaction contains an unknown allocation player")
        object.__setattr__(self, "allocation", MappingProxyType(dict(self.allocation)))
        object.__setattr__(self, "interactions", MappingProxyType(dict(self.interactions)))


class LearnedGuide:
    """One graph encoding per plan, followed by analytic fixed-context allocation.

    This adapter accepts an actual trained model. Construction or execution of
    the adapter does not imply that untrained weights reproduce paper results.
    Set lambda_=0 for the same-architecture standard-sharing comparison.
    """
    def __init__(self, model, *, lambda_: float = .5, tau: float = .5,
                 eta: float = .5, normalizer=None, graph_builder=None):
        if (not 0 <= lambda_ <= 1 or not math.isfinite(tau) or tau <= 0
                or not math.isfinite(eta) or eta < 0):
            raise ValueError("Invalid sharing configuration")
        self.model, self.normalizer = model, normalizer
        self.lambda_, self.tau, self.eta = lambda_, tau, eta
        self.graph_builder = graph_builder
        self.encoding_calls = 0

    def __call__(self, scenario: Scenario, plan: Plan, budget=None) -> Guidance:
        import torch
        from .graph import build_plan_graph
        graph = (self.graph_builder or build_plan_graph)(scenario, plan)
        if self.normalizer is not None:
            graph = self.normalizer.transform(graph)
        parameter = next(iter(self.model.parameters()), None)
        if parameter is not None:
            graph = graph.to(parameter.device)
        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                encoded = self.model.forward_plan(graph)
                self.encoding_calls += 1
                allocations = encoded.allocation(lambda_=self.lambda_, tau=self.tau, eta=self.eta)
                result = Guidance(
                    dict(zip(encoded.route_ids, map(float, allocations.detach().cpu().tolist()))),
                    {tuple(encoded.route_ids[i] for i in group): float(value)
                     for group, value in zip(encoded.support, encoded.coefficients.detach().cpu().tolist())})
        finally:
            self.model.train(training)
        return result


class BudgetExhausted(RuntimeError):
    """The next complete operation is outside the shared query/time budget."""


@dataclass(frozen=True)
class BudgetUse:
    operation: str
    phase: str
    evaluations: int
    elapsed_seconds: float
    feasible: bool | None = None


class SearchBudget:
    """Count every physical call, including repair checks, in one context.

    Since physics.evaluate performs checking and objective calculation together,
    even an infeasible call consumes one query. Cached checks do not consume a
    second physical evaluation. Time includes initialization, guidance, repair,
    selection and final attribution. Cooperative deadline checks occur before
    and after nonpreemptible Python/model calls.
    """
    def __init__(self, max_evaluations: int = 200, time_limit: float = 60.0, *,
                 final_reserve_evaluations: int = 1, final_reserve_seconds: float = .1,
                 evaluator: Callable = evaluate, clock: Callable[[], float] = time.perf_counter):
        if (max_evaluations < 1 or not math.isfinite(time_limit) or time_limit <= 0
                or not 0 <= final_reserve_evaluations < max_evaluations
                or not 0 <= final_reserve_seconds < time_limit):
            raise ValueError("Invalid search budget or final attribution reserve")
        self.max_evaluations, self.time_limit = max_evaluations, time_limit
        self.final_reserve_evaluations = final_reserve_evaluations
        self.final_reserve_seconds = final_reserve_seconds
        self.evaluator, self.clock = evaluator, clock
        self.started = clock()
        self.evaluations = 0
        self.phase = "initialization"
        self.reference_id = None
        self.records: list[BudgetUse] = []
        self.cache: dict[Plan, Evaluation] = {}

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def checkpoint(self, *, physical: bool = False):
        final = self.phase == "final_attribution"
        allowance = self.max_evaluations - (0 if final else self.final_reserve_evaluations)
        deadline = self.time_limit - (0 if final else self.final_reserve_seconds)
        if self.elapsed >= deadline or (physical and self.evaluations >= allowance):
            raise BudgetExhausted("Shared search budget exhausted")

    def evaluate(self, scenario: Scenario, plan: Plan) -> Evaluation:
        self.checkpoint()
        reference = scenario.fingerprint()
        if self.reference_id is None:
            self.reference_id = reference
        if reference != self.reference_id:
            raise ValueError("Search evaluations must retain the same frozen reference")
        if plan in self.cache:
            return self.cache[plan]
        self.checkpoint(physical=True)
        self.evaluations += 1
        result = self.evaluator(scenario, plan)
        self.cache[plan] = result
        self.records.append(BudgetUse("physical_check_and_objective", self.phase,
                                      self.evaluations, self.elapsed, result.feasible))
        self.checkpoint()
        return result

    def guidance(self, guide: Callable, scenario: Scenario, plan: Plan) -> Guidance:
        """Budget-aware guides receive this callback for every physical query.

        Two-argument guides are supported for analytic, nonphysical audit scores.
        A guide that evaluates a physical game must accept the third argument and
        pass budget.evaluate to its game/probe adapter; hidden physical calls are
        outside this API's contract.
        """
        self.checkpoint()
        if self.reference_id is not None and scenario.fingerprint() != self.reference_id:
            raise ValueError("Guidance must retain the same frozen reference")
        try:
            inspect.signature(guide).bind(scenario, plan, self)
        except TypeError:
            value = guide(scenario, plan)
        else:
            value = guide(scenario, plan, self)
        if not isinstance(value, Guidance):
            raise TypeError("A guidance provider must return Guidance")
        if set(value.allocation) != {c.id for c in scenario.chargers}:
            raise ValueError("Guidance must allocate every route player")
        self.records.append(BudgetUse("guidance", self.phase, self.evaluations, self.elapsed))
        self.checkpoint()
        return value


@dataclass(frozen=True)
class SearchConfig:
    max_evaluations: int = 200
    time_limit: float = 60.0
    exploration_probability: float = .10
    min_removed: int = 2
    max_removed: int = 4
    max_trials: int = 100
    seed: int = 1
    final_reserve_evaluations: int = 1
    final_reserve_seconds: float = .1

    def __post_init__(self):
        if not 0 <= self.exploration_probability <= 1 or not 1 <= self.min_removed <= self.max_removed:
            raise ValueError("Invalid neighborhood configuration")
        if self.max_trials < 0:
            raise ValueError("Trial limit must be nonnegative")


@dataclass(frozen=True)
class Trial:
    index: int
    removed: tuple[str, ...]
    strategy: str
    candidate_feasible: bool
    candidate_objective: float
    current_objective: float
    best_objective: float
    accepted_current: bool
    best_replaced: bool
    evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SearchResult:
    best_plan: Plan
    best_evaluation: Evaluation
    current_plan: Plan
    current_evaluation: Evaluation
    final_guidance: Guidance | None
    trace: tuple[Trial, ...]
    budget_records: tuple[BudgetUse, ...]
    evaluations: int
    elapsed_seconds: float
    stop_reason: str


def select_neighborhood(guidance: Guidance, rng: np.random.Generator, *,
                        exploration_probability: float = .1, min_removed: int = 2,
                        max_removed: int = 4) -> tuple[tuple[str, ...], str]:
    """Score bounded groups; explore independently with probability epsilon.

    Guided candidates use the 12 lowest-allocation players plus complete small
    strong interaction groups. Positive partial overlap is penalized; rebuilding
    the whole positive group receives a bonus. Revisiting one negative-interaction
    member receives the largest redundancy bonus. A cap keeps this heuristic
    usable when a scenario has many routes.
    """
    players = tuple(sorted(guidance.allocation))
    if not players:
        return (), "empty"
    lower, upper = min(min_removed, len(players)), min(max_removed, len(players))
    if rng.random() < exploration_probability:
        count = int(rng.integers(lower, upper + 1))
        selected = tuple(sorted(rng.choice(players, size=count, replace=False).tolist()))
        return selected, "independent_exploration"
    ranked = tuple(sorted(players, key=lambda i: (guidance.allocation[i], i)))
    pool = ranked[:12]
    candidates = {tuple(sorted(group)) for count in range(lower, upper + 1)
                  for group in combinations(pool, count)}
    interactions = sorted(((group, value) for group, value in guidance.interactions.items()
                           if len(group) > 1), key=lambda row: (-abs(row[1]), row[0]))
    for group, _ in interactions[:16]:
        if len(group) <= upper:
            complete = set(group)
            for i in ranked:
                if len(complete) >= lower:
                    break
                complete.add(i)
            candidates.add(tuple(sorted(complete)))
        for i in group:
            subset = {i}
            subset.update(j for j in ranked if j != i and len(subset) < lower)
            candidates.add(tuple(sorted(subset)))
    scale = max(1.0, max(abs(v) for v in guidance.allocation.values()),
                sum(abs(v) for _, v in interactions))
    def score(group):
        selected = set(group)
        value = -sum(guidance.allocation[i] for i in group)/len(group)
        for members, dividend in interactions:
            overlap = len(selected.intersection(members))
            if dividend > 0:
                value += dividend if overlap == len(members) else (-2*dividend if overlap else 0)
            elif overlap:
                value += abs(dividend)/overlap
        # The tiny cardinality tie break keeps equally useful repairs compact.
        return value/scale - 1e-8*len(group)
    winner = min(candidates, key=lambda group: (-score(group), len(group), group))
    return winner, "signed_interaction_guidance"


def improve_only(current: Evaluation, candidate: Evaluation,
                 rng: np.random.Generator) -> bool:
    return candidate.objective > current.objective


def search(scenario: Scenario, initial_plan: Plan, guide: Callable, *,
           config: SearchConfig | None = None, reconstruction: Callable = greedy_reconstruct,
           accept: Callable = improve_only, budget: SearchBudget | None = None) -> SearchResult:
    """Run the paper's verified-incumbent skeleton with the documented operator.

    Standard and RC runs differ only in their guide; their reconstruction,
    acceptance, checker, initialization and limits can be shared exactly.
    An exhausted incomplete trial cannot replace a stored incumbent.
    """
    config = config or SearchConfig()
    budget = budget or SearchBudget(config.max_evaluations, config.time_limit,
        final_reserve_evaluations=config.final_reserve_evaluations,
        final_reserve_seconds=config.final_reserve_seconds)
    rng = np.random.default_rng(config.seed)
    initial = budget.evaluate(scenario, initial_plan)
    if not initial.feasible:
        raise ValueError("Search requires a physically feasible initial fallback")
    current_plan = best_plan = initial_plan
    current_evaluation = best_evaluation = initial
    trace = []
    reason = "trial_limit"
    budget.phase = "refinement"
    for index in range(config.max_trials):
        try:
            budget.checkpoint(physical=True)
            guidance = budget.guidance(guide, scenario, current_plan)
            removed, strategy = select_neighborhood(guidance, rng,
                exploration_probability=config.exploration_probability,
                min_removed=config.min_removed, max_removed=config.max_removed)
            budget.checkpoint()
            if not removed:
                reason = "no_players"
                break
            partial = destroy(current_plan, removed)
            candidate = reconstruction(scenario, partial, removed, budget, rng)
            checked = budget.evaluate(scenario, candidate)
            replaced = checked.feasible and checked.objective > best_evaluation.objective
            accepted = checked.feasible and bool(accept(current_evaluation, checked, rng))
            budget.checkpoint()
        except BudgetExhausted:
            reason = "shared_budget"
            break
        if replaced:
            best_plan, best_evaluation = candidate, checked
        if accepted:
            current_plan, current_evaluation = candidate, checked
        trace.append(Trial(index, removed, strategy, checked.feasible, checked.objective,
                           current_evaluation.objective, best_evaluation.objective,
                           accepted, replaced, budget.evaluations, budget.elapsed))
    budget.phase = "final_attribution"
    try:
        final = budget.guidance(guide, scenario, best_plan)
    except BudgetExhausted:
        final = None
        reason += ":final_attribution_budget"
    return SearchResult(best_plan, best_evaluation, current_plan, current_evaluation,
                        final, tuple(trace), tuple(budget.records), budget.evaluations,
                        budget.elapsed, reason)
