"""Small physical search checks; these are not benchmark performance results."""
from dataclasses import replace
import math

import numpy as np
import pytest

from rc_shapleynet.schema import Arc, Charger, Location, Plan, Power, Recipient, Route, Scenario, Service, Task
from rc_shapleynet.physics import evaluate
from rc_shapleynet.reconstruction import destroy, greedy_reconstruct, idle_return_route
from rc_shapleynet.search import (BudgetExhausted, Guidance, LearnedGuide, SearchBudget,
                                 SearchConfig, search, select_neighborhood)


def scenario_fixture(count=3):
    return Scenario("EV", (Location("d", 0, 0, True), Location("x", 10, 0)),
        tuple(Charger(f"C{i}", "d", 20, 20, 1, 1, energy_per_metre=.001,
                      receiver_efficiency=1) for i in range(count)),
        (Recipient("r", "x", 1, 6, 2, 0, 2),), 8, step_seconds=3600,
        tasks=(Task("shared", tuple(f"C{i}" for i in range(count)), ("r",)),))


def analytic_guide(scenario, plan):
    ids = tuple(c.id for c in scenario.chargers)
    return Guidance(dict.fromkeys(ids, 0.0), {(i,): 0.0 for i in ids})


def full_service_plan(scenario):
    route = Route("C0", (Arc("d", "x", 0, 1), Arc("x", "x", 1, 2),
                          Arc("x", "x", 2, 3), Arc("x", "d", 3, 4))
                         + tuple(Arc("d", "d", t, t+1) for t in range(4, 8)),
                  (Service("r", 1, 1), Service("r", 2, 1)),
                  (Power(1, 1), Power(2, 1)))
    return Plan((route,))


def test_greedy_reconstruction_makes_a_real_complete_route_improvement():
    scenario = scenario_fixture()
    budget = SearchBudget(100)
    rebuilt = greedy_reconstruct(scenario, Plan(()), ("C0", "C1"), budget,
                                 np.random.default_rng(1))
    result = evaluate(scenario, rebuilt)
    assert result.feasible, result.violations
    assert result.metrics["completion"] == 1
    assert result.objective > evaluate(scenario, Plan(())).objective
    assert rebuilt.route("C0").services or rebuilt.route("C1").services
    assert any(a.origin != a.destination for r in rebuilt.routes for a in r.arcs)
    assert all(r.arcs[-1].end == scenario.horizon for r in rebuilt.routes)
    assert budget.evaluations > 1


def test_unremoved_routes_are_exactly_fixed_during_destroy_and_repair():
    scenario = scenario_fixture()
    fixed = idle_return_route(scenario, scenario.chargers[2])
    partial = Plan((fixed,))
    budget = SearchBudget(100)
    rebuilt = greedy_reconstruct(scenario, partial, ("C0", "C1"), budget,
                                 np.random.default_rng(9))
    assert rebuilt.route("C2") == fixed
    assert destroy(rebuilt, ("C0", "C1")) == partial


def test_verified_best_is_feasible_and_trace_is_monotone():
    scenario = scenario_fixture()
    result = search(scenario, Plan(()), analytic_guide,
                    config=SearchConfig(max_trials=4, max_evaluations=100))
    assert result.best_evaluation.feasible
    assert result.best_evaluation.metrics["completion"] == 1
    values = [row.best_objective for row in result.trace]
    assert values == sorted(values)
    assert all(2 <= len(row.removed) <= 3 for row in result.trace)
    assert result.final_guidance is not None
    assert result.budget_records[0].phase == "initialization"
    assert result.budget_records[-1].phase == "final_attribution"


def test_accepted_nonimproving_current_does_not_replace_verified_best():
    scenario = scenario_fixture()
    good = full_service_plan(scenario)
    proposals = iter((good, Plan(())))
    def repair(*args):
        return next(proposals)
    result = search(scenario, Plan(()), analytic_guide,
        config=SearchConfig(max_trials=2), reconstruction=repair,
        accept=lambda current, candidate, rng: True)
    assert result.current_plan == Plan(())
    assert result.best_plan == good
    assert result.best_evaluation.objective > result.current_evaluation.objective
    assert result.trace[1].accepted_current
    assert not result.trace[1].best_replaced


def test_invalid_candidate_never_enters_current_or_best():
    scenario = scenario_fixture()
    bad_route = replace(full_service_plan(scenario).route("C0"), powers=())
    invalid = Plan((bad_route,))
    result = search(scenario, Plan(()), analytic_guide,
        config=SearchConfig(max_trials=1), reconstruction=lambda *args: invalid,
        accept=lambda *args: True)
    assert result.best_plan == result.current_plan == Plan(())
    assert not result.trace[0].candidate_feasible
    assert not result.trace[0].accepted_current
    assert result.trace[0].candidate_objective == -math.inf
    assert result.evaluations == 2


def test_physical_checks_inside_repair_and_final_audit_share_one_budget():
    scenario = scenario_fixture()
    physical_calls = []
    def evaluator(q, plan):
        physical_calls.append(plan)
        return evaluate(q, plan)
    final_plan = Plan((replace(idle_return_route(scenario, scenario.chargers[0]),
                               powers=(Power(0, 0),)),))
    def budgeted_guide(q, plan, budget):
        if budget.phase == "final_attribution":
            budget.evaluate(q, final_plan)
        return analytic_guide(q, plan)
    budget = SearchBudget(8, final_reserve_evaluations=1, evaluator=evaluator)
    result = search(scenario, Plan(()), budgeted_guide,
                    config=SearchConfig(max_trials=10), budget=budget)
    assert len(physical_calls) == budget.evaluations == result.evaluations
    assert result.evaluations <= 8
    assert result.final_guidance is not None
    records = [r for r in result.budget_records if r.operation == "physical_check_and_objective"]
    assert records[0].phase == "initialization"
    assert any(r.phase == "refinement" for r in records)
    assert records[-1].phase == "final_attribution"


def test_query_reserve_is_unavailable_to_refinement_but_available_to_final():
    scenario = scenario_fixture()
    budget = SearchBudget(2, final_reserve_evaluations=1)
    budget.evaluate(scenario, Plan(()))
    different = Plan((idle_return_route(scenario, scenario.chargers[0]),))
    budget.phase = "refinement"
    with pytest.raises(BudgetExhausted):
        budget.evaluate(scenario, different)
    budget.phase = "final_attribution"
    assert budget.evaluate(scenario, different).feasible
    assert budget.evaluations == 2


def test_initialization_guidance_and_final_attribution_time_are_all_accounted():
    scenario = scenario_fixture()
    class Clock:
        value = 0.0
        def __call__(self):
            return self.value
    clock = Clock()
    def evaluator(q, plan):
        clock.value += .1
        return evaluate(q, plan)
    def guide(q, plan):
        clock.value += .05
        return analytic_guide(q, plan)
    budget = SearchBudget(20, 1, final_reserve_seconds=.2, evaluator=evaluator, clock=clock)
    result = search(scenario, Plan(()), guide, config=SearchConfig(max_trials=0), budget=budget)
    assert result.elapsed_seconds == pytest.approx(.15)
    assert result.budget_records[0].elapsed_seconds == pytest.approx(.1)
    assert result.budget_records[-1].elapsed_seconds == pytest.approx(.15)


def test_positive_group_is_not_split_and_negative_redundancy_is_revisited():
    rng = np.random.default_rng(3)
    positive = Guidance({"A": 0, "B": 0, "C": 0, "D": 0}, {("A", "B"): 5})
    chosen, strategy = select_neighborhood(positive, rng, exploration_probability=0)
    assert {"A", "B"} <= set(chosen)
    assert strategy == "signed_interaction_guidance"
    negative = Guidance({"A": 0, "B": 0, "C": 0, "D": 0}, {("A", "B"): -5})
    chosen, _ = select_neighborhood(negative, rng, exploration_probability=0)
    assert len({"A", "B"} & set(chosen)) == 1


def test_exploration_is_reproducible_and_independent_of_scores():
    first = Guidance({str(i): i for i in range(8)}, {})
    second = Guidance({str(i): -i for i in range(8)}, {})
    a = select_neighborhood(first, np.random.default_rng(7), exploration_probability=1)
    b = select_neighborhood(second, np.random.default_rng(7), exploration_probability=1)
    assert a == b
    assert a[1] == "independent_exploration"
    assert 2 <= len(a[0]) <= 4


def test_unavoidable_travel_prefix_survives_reconstruction():
    scenario = scenario_fixture(2)
    prefix = Route("C0", (Arc("d", "x", 0, 1),))
    scenario = replace(scenario, committed_prefixes=(prefix,))
    budget = SearchBudget(100)
    result = greedy_reconstruct(scenario, Plan(()), ("C0", "C1"), budget,
                                np.random.default_rng(2))
    assert result.route("C0").arcs[:1] == prefix.arcs
    assert evaluate(scenario, result).feasible


def test_wrsn_repair_delivers_load_without_overfilling_a_sensor():
    scenario = Scenario("WRSN", (Location("d", 0, 0, True),),
        (Charger("A", "d", 10, 10, 1, 1), Charger("B", "d", 10, 10, 1, 1)),
        (Recipient("r", "d", 0, 6, 0, 0, 1, load=(.05,)*6),),
        6, step_seconds=3600, required_paths=(("r",),),
        coverage_points=((0, 0, 1),))
    result = search(scenario, Plan(()), analytic_guide,
                    config=SearchConfig(max_trials=2, max_evaluations=60))
    assert result.best_evaluation.feasible
    assert result.best_evaluation.metrics["C_alive"] > 0
    assert result.best_evaluation.objective > evaluate(scenario, Plan(())).objective


def test_learned_guide_encodes_once_and_lambda_zero_is_uniform_sharing(monkeypatch):
    torch = pytest.importorskip("torch")
    from rc_shapleynet.graph import build_plan_graph
    from rc_shapleynet.learning import RCShapleyNet
    scenario = scenario_fixture()
    graph = build_plan_graph(scenario, Plan(()))
    torch.manual_seed(3)
    model = RCShapleyNet(graph.input_dims, tuple(graph.edge_index), 2,
                         hidden_dim=8, layers=1)
    original = model.forward_plan
    encoded_objects = []
    def tracked(g):
        encoded = original(g)
        encoded_objects.append(encoded)
        return encoded
    monkeypatch.setattr(model, "forward_plan", tracked)
    standard = LearnedGuide(model, lambda_=0)
    resource = LearnedGuide(model, lambda_=.5, tau=.5, eta=.5)
    uniform = standard(scenario, Plan(()))
    modified = resource(scenario, Plan(()))
    assert len(encoded_objects) == 2
    assert standard.encoding_calls == resource.encoding_calls == 1
    expected = {cid: 0.0 for cid in graph.route_ids}
    for group, value in uniform.interactions.items():
        for cid in group:
            expected[cid] += value/len(group)
    assert uniform.allocation == pytest.approx(expected)
    assert sum(uniform.allocation.values()) == pytest.approx(sum(uniform.interactions.values()))
    assert sum(modified.allocation.values()) == pytest.approx(sum(modified.interactions.values()))
    assert uniform.interactions == modified.interactions
    assert model.training  # Original mode is restored after inference.


def test_standard_and_rc_use_identical_reconstruction_and_acceptance_operators():
    scenario = scenario_fixture()
    operators = []
    def repair(q, partial, removed, budget, rng):
        operators.append((partial, removed))
        return greedy_reconstruct(q, partial, removed, budget, rng)
    def rc_guide(q, plan):
        return Guidance({c.id: 1 for c in q.chargers}, {})
    config = SearchConfig(max_trials=1, max_evaluations=60, exploration_probability=1, seed=12)
    standard = search(scenario, Plan(()), analytic_guide, config=config, reconstruction=repair)
    modified = search(scenario, Plan(()), rc_guide, config=config, reconstruction=repair)
    assert operators[0] == operators[1]
    assert standard.best_plan == modified.best_plan
    assert standard.evaluations == modified.evaluations


def test_budget_rejects_a_changed_reference_and_invalid_fallback():
    scenario = scenario_fixture()
    budget = SearchBudget(10)
    budget.evaluate(scenario, Plan(()))
    with pytest.raises(ValueError, match="frozen reference"):
        budget.evaluate(replace(scenario, scenario_id="changed"), Plan(()))
    bad = Plan((replace(full_service_plan(scenario).route("C0"), powers=()),))
    with pytest.raises(ValueError, match="feasible initial"):
        search(scenario, bad, analytic_guide)


def test_exact_small_game_guide_accounts_for_all_physical_queries():
    from rc_shapleynet.game import FixedRouteGame, allocate, mobius_dividends
    scenario = scenario_fixture(2)
    physical_calls = []
    def checker(q, plan):
        physical_calls.append(plan)
        return evaluate(q, plan)
    def exact_guide(q, plan, budget):
        game = FixedRouteGame(q, plan, evaluator=budget.evaluate)
        dividends = mobius_dividends(game.enumerate())
        return Guidance(allocate(dividends, players=game.players), dividends)
    budget = SearchBudget(60, final_reserve_evaluations=8, evaluator=checker)
    result = search(scenario, Plan(()), exact_guide,
                    config=SearchConfig(max_trials=1), budget=budget)
    assert result.best_evaluation.metrics["completion"] == 1
    assert result.final_guidance is not None
    assert result.evaluations == len(physical_calls)
    assert len(set(physical_calls)) == len(physical_calls)
    assert any(record.phase == "final_attribution" and
               record.operation == "physical_check_and_objective"
               for record in result.budget_records)


def test_frozen_guidance_and_safe_return_choice_do_not_hide_mutable_inputs():
    raw = {"A": 1.0, "B": 2.0}
    guide = Guidance(raw, {("A", "B"): 3.0})
    raw["A"] = 99
    assert guide.allocation["A"] == 1
    with pytest.raises(TypeError):
        guide.allocation["A"] = 4
    scenario = Scenario("EV", (Location("x", 0, 0), Location("slow", 1, 0, True),
                                Location("fast", 100, 0, True)),
        (Charger("A", "x", 2, 2, 1, 1, energy_per_metre=1),), (), 8,
        road_distance={"x|slow": 1, "x|fast": 100},
        road_slots={"A|x|slow": 4, "A|x|fast": 1})
    route = idle_return_route(scenario, scenario.chargers[0])
    assert route.arcs[0].destination == "slow"
    assert evaluate(scenario, Plan((route,))).feasible
