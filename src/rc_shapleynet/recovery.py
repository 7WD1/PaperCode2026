"""A fixed, service-level local repairer over retained dwell locations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Callable

from .schema import Plan, Power, Route, Scenario, Service, Task
from .physics import Evaluation, effective_route, evaluate, service_fraction


@dataclass(frozen=True)
class RecoveryResult:
    committed: Mapping[str, float]
    restored: Mapping[str, float]
    committed_total: float
    restored_total: float
    difficulty: float
    repaired_plan: Plan
    evaluation: Evaluation
    checker_calls: int

    @property
    def total_committed(self) -> float:
        return self.committed_total

    @property
    def total_restored(self) -> float:
        return self.restored_total


def add_service(scenario: Scenario, plan: Plan, charger_id: str,
                recipient_id: str, slot: int, energy: float) -> Plan:
    """Add received energy and the minimum additional transmitter output.

    Existing arcs, services, output energy, and depot input remain committed.
    The returned candidate still needs the full checker; this helper cannot
    assert feasibility from a local power bound alone.
    """
    if energy < 0:
        raise ValueError("Recovery can only add service")
    route = effective_route(scenario, plan, charger_id)
    charger = next(c for c in scenario.chargers if c.id == charger_id)
    dwell = next((a.origin for a in route.arcs
                  if a.origin == a.destination and a.start <= slot < a.end), None)
    if dwell is None:
        raise ValueError("Recovery requires an existing waiting arc")
    services: dict[tuple[str, int], float] = {}
    for s in route.services:
        services[s.recipient_id, s.slot] = services.get((s.recipient_id, s.slot), 0.0) + s.energy
    services[recipient_id, slot] = services.get((recipient_id, slot), 0.0) + energy
    outputs, depot = {}, {}
    for p in route.powers:
        outputs[p.slot] = outputs.get(p.slot, 0.0) + p.output_energy
        depot[p.slot] = depot.get(p.slot, 0.0) + p.depot_energy
    active = {rid: value for (rid, t), value in services.items() if t == slot and value > 0}
    required = 0.0
    for rid, value in active.items():
        fraction = service_fraction(scenario, charger, dwell, rid)
        if fraction <= 0:
            raise ValueError("Recipient is incompatible with the retained dwell")
        if scenario.domain == "EV":
            required += value / fraction
        else:
            required = max(required, value / fraction)
    if scenario.domain == "WRSN":
        required = max(required, sum(active.values()))
    outputs[slot] = max(outputs.get(slot, 0.0), required)
    updated = replace(route,
                      services=tuple(Service(rid, t, value) for (rid, t), value in sorted(services.items(), key=lambda v: (v[0][1], v[0][0]))),
                      powers=tuple(Power(t, outputs.get(t, 0.0), depot.get(t, 0.0))
                                   for t in sorted(outputs.keys() | depot.keys())))
    routes = tuple(updated if r.charger_id == charger_id else r for r in plan.routes)
    if all(r.charger_id != charger_id for r in plan.routes):
        routes += (updated,)
    return Plan(routes)


def probe(scenario: Scenario, plan: Plan, task: Task, removed_route: str,
          *, tolerance: float = 1e-9, bisection_steps: int = 45,
          evaluator: Callable[[Scenario, Plan], Evaluation] = evaluate) -> RecoveryResult:
    """Recover lost task service without moving any retained route.

    The fixed order is recipient deadline, recipient ID, charger ID, then slot.
    Each positive increment is accepted only after the complete physical check.
    Bisection bounds service energy, never total utility or a completion event.
    Every check uses ``evaluator`` so online search can account for its budget.
    Evaluator exceptions, including budget exhaustion, propagate to the caller.
    """
    if removed_route not in task.route_ids:
        raise ValueError("Removed route must participate in the task")
    recipient_map = {r.id: r for r in scenario.recipients}
    charger_ids = {c.id for c in scenario.chargers}
    if any(rid not in recipient_map for rid in task.recipient_ids):
        raise ValueError("Unknown task recipient")
    if any(cid not in charger_ids for cid in task.route_ids):
        raise ValueError("Unknown task route")
    original = evaluator(scenario, plan)
    if not original.feasible:
        raise ValueError("Recovery requires a feasible original plan: " + "; ".join(original.violations))
    lost = plan.route(removed_route)
    committed = {rid: sum(s.energy for s in lost.services
                          if s.recipient_id == rid and recipient_map[rid].arrival <= s.slot < recipient_map[rid].departure)
                 for rid in task.recipient_ids}
    restored = {rid: 0.0 for rid in task.recipient_ids}
    repaired = Plan(tuple(r for r in plan.routes if r.charger_id != removed_route))
    result = evaluator(scenario, repaired)
    checks = 2
    if not result.feasible:
        raise ValueError("Route removal violates the idle-fallback closure: " + "; ".join(result.violations))
    allowed = sorted(set(task.route_ids) - {removed_route})
    ordered_recipients = sorted(task.recipient_ids, key=lambda rid: (recipient_map[rid].departure, rid))
    for rid in ordered_recipients:
        recipient = recipient_map[rid]
        for cid in allowed:
            if committed[rid] - restored[rid] <= tolerance:
                break
            route = effective_route(scenario, repaired, cid)
            slots = sorted({a.start for a in route.arcs
                            if a.origin == a.destination and a.end == a.start + 1
                            and recipient.arrival <= a.start < recipient.departure})
            for slot in slots:
                remaining = committed[rid] - restored[rid]
                if remaining <= tolerance:
                    break
                try:
                    upper_plan = add_service(scenario, repaired, cid, rid, slot, remaining)
                except ValueError:
                    continue
                upper_eval = evaluator(scenario, upper_plan)
                checks += 1
                if upper_eval.feasible:
                    increment, candidate, candidate_eval = remaining, upper_plan, upper_eval
                else:
                    low, high = 0.0, remaining
                    candidate, candidate_eval = repaired, result
                    for _ in range(bisection_steps):
                        if high - low <= tolerance:
                            break
                        midpoint = (low + high) / 2
                        trial = add_service(scenario, repaired, cid, rid, slot, midpoint)
                        trial_eval = evaluator(scenario, trial)
                        checks += 1
                        if trial_eval.feasible:
                            low, candidate, candidate_eval = midpoint, trial, trial_eval
                        else:
                            high = midpoint
                    increment = low
                if increment > tolerance:
                    repaired, result = candidate, candidate_eval
                    restored[rid] = min(committed[rid], restored[rid] + increment)
    committed_total = sum(committed.values())
    restored_total = sum(min(committed[rid], restored[rid]) for rid in committed)
    difficulty = 1 - restored_total / committed_total if committed_total > 0 else 0.0
    return RecoveryResult(committed, restored, committed_total, restored_total,
                          max(0.0, min(1.0, difficulty)), repaired, result, checks)
