from dataclasses import replace
import pytest

from rc_shapleynet.schema import Arc, Charger, Location, Plan, Power, Recipient, Route, Scenario, Service, Task
from rc_shapleynet.recovery import probe
from rc_shapleynet.physics import evaluate


def figure2():
    scenario = Scenario("EV", (Location("r", 0, 0),),
                        (Charger("A", "r", 7, 7, 1, 1, receiver_efficiency=1),
                         Charger("B", "r", 4, 4, 1, 1, receiver_efficiency=1)),
                        (Recipient("r", "r", 0, 8, 8, 0, 8),), 8, step_seconds=3600)
    def route(cid, slots):
        return Route(cid, tuple(Arc("r", "r", t, t+1) for t in range(8)),
                     tuple(Service("r", t, 1) for t in slots), tuple(Power(t, 1) for t in slots))
    return scenario, Plan((route("A", range(4)), route("B", range(4, 8)))), Task("r", ("A", "B"), ("r",))


def test_figure2_recovery_is_zero_or_three_and_not_utility_recovery():
    scenario, plan, task = figure2()
    remove_a = probe(scenario, plan, task, "A")
    remove_b = probe(scenario, plan, task, "B")
    assert remove_a.committed_total == remove_b.committed_total == 4
    assert remove_a.restored_total == pytest.approx(0, abs=1e-7)
    assert remove_a.difficulty == pytest.approx(1, abs=1e-7)
    assert remove_b.restored_total == pytest.approx(3, abs=1e-7)
    assert remove_b.difficulty == pytest.approx(.25, abs=1e-7)
    assert remove_b.evaluation.feasible
    assert remove_b.evaluation.metrics["completion"] == 0
    assert remove_b.restored["r"] <= remove_b.committed["r"]
    assert remove_b.repaired_plan.route("A").arcs == plan.route("A").arcs
    old = {(s.recipient_id, s.slot): s.energy for s in plan.route("A").services}
    new = {(s.recipient_id, s.slot): s.energy for s in remove_b.repaired_plan.route("A").services}
    assert all(new[k] >= v for k, v in old.items())


def test_no_dwell_rerouting_or_other_route_service_is_permitted():
    scenario, plan, task = figure2()
    # A exists but is outside the task's repair set. B's original service remains
    # attributable to B; its removal cannot recruit an unrelated task participant.
    narrowed = replace(task, route_ids=("B",))
    result = probe(scenario, plan, narrowed, "B")
    assert result.restored_total == 0
    assert result.repaired_plan.route("A") == plan.route("A")


def test_zero_committed_service_has_zero_difficulty():
    scenario, plan, task = figure2()
    a = replace(plan.route("A"), services=(), powers=())
    result = probe(scenario, Plan((a, plan.route("B"))), task, "A")
    assert result.committed_total == result.restored_total == result.difficulty == 0


def test_probe_rejects_infeasible_original_plan():
    scenario, plan, task = figure2()
    a = replace(plan.route("A"), services=(Service("r", 0, 5),), powers=(Power(0, 5),))
    with pytest.raises(ValueError, match="feasible original"):
        probe(scenario, Plan((a, plan.route("B"))), task, "A")


def test_recovery_cap_is_per_recipient_not_a_transferable_total():
    scenario = Scenario("EV", (Location("x", 0, 0), Location("y", 1, 0)),
                        (Charger("A", "x", 7, 7, 3, 1, energy_per_metre=0, receiver_efficiency=1),
                         Charger("B", "y", 10, 10, 3, 1, energy_per_metre=0, receiver_efficiency=1)),
                        (Recipient("early", "x", 0, 1, 1, 0, 1),
                         Recipient("late", "y", 0, 3, 3, 0, 3)), 6, step_seconds=3600)
    a = Route("A", (Arc("x", "x", 0, 1), Arc("x", "y", 1, 2),
                    Arc("y", "y", 2, 3), Arc("y", "x", 3, 4),
                    Arc("x", "x", 4, 5), Arc("x", "x", 5, 6)),
              (Service("early", 0, 1), Service("late", 2, 3)), (Power(0, 1), Power(2, 3)))
    b = Route("B", tuple(Arc("y", "y", t, t+1) for t in range(6)))
    result = probe(scenario, Plan((a, b)), Task("both", ("A", "B"), ("early", "late")), "A")
    assert result.restored["early"] == 0
    assert result.restored["late"] == 3
    assert result.restored_total == 3
    assert result.difficulty == .25


def test_every_recovery_check_passes_through_the_injected_evaluator():
    scenario, plan, task = figure2()
    checked = []
    def counted(reference, candidate):
        checked.append((reference.fingerprint(), candidate))
        return evaluate(reference, candidate)
    result = probe(scenario, plan, task, "B", evaluator=counted)
    assert len(checked) == result.checker_calls
    assert len(checked) > 2  # Includes incremental and bisection checks.
    assert all(reference == scenario.fingerprint() for reference, _ in checked)
    assert result.difficulty == pytest.approx(.25, abs=1e-7)


def test_injected_budget_exhaustion_propagates_without_returning_partial_recovery():
    scenario, plan, task = figure2()
    class BudgetExhausted(RuntimeError):
        pass
    calls = 0
    def bounded(reference, candidate):
        nonlocal calls
        if calls == 3:
            raise BudgetExhausted("No physical calls remain")
        calls += 1
        return evaluate(reference, candidate)
    with pytest.raises(BudgetExhausted, match="No physical calls remain"):
        probe(scenario, plan, task, "B", evaluator=bounded)
    assert calls == 3
