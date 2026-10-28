"""Commit a physical prefix, update the reference, and replan without teleporting."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import numpy as np

from .schema import Arc, Location, Plan, Route, Scenario
from .physics import Evaluation, effective_route, evaluate


@dataclass(frozen=True)
class ExecutionStep:
    slots: int
    reference_id: str
    plan: Plan
    evaluation: Evaluation
    next_scenario: Scenario | None


@dataclass(frozen=True)
class EpisodeResult:
    steps: tuple[ExecutionStep, ...]
    battery: dict[str, np.ndarray]
    recipient_energy: dict[str, np.ndarray]
    metrics: dict[str, float]
    objective: float


def execute_interval(scenario: Scenario, plan: Plan, slots: int) -> ExecutionStep:
    """Execute only the first interval of a fully verified plan.

    In-progress travel is irrevocable. A new virtual start records its remaining
    distance and duration. Its position is an affine bookkeeping coordinate,
    not a reconstructed road trace. The next reference locks its continuation.
    """
    if not isinstance(slots, int) or not 0 < slots <= scenario.horizon:
        raise ValueError("Execution interval must contain 1..H whole slots")
    result = evaluate(scenario, plan)
    if not result.feasible:
        raise ValueError("Cannot execute an infeasible plan: " + "; ".join(result.violations))
    if slots == scenario.horizon:
        return ExecutionStep(slots, scenario.fingerprint(), plan, result, None)
    remaining_horizon = scenario.horizon - slots
    locations = {p.id: p for p in scenario.locations}
    distances: dict[str, float] = {}
    travel_slots: dict[str, int] = {}
    # Once a directed table is present, every reachable original pair is frozen.
    for origin in locations:
        for destination in locations:
            if origin == destination:
                continue
            try:
                distances[f"{origin}|{destination}"] = scenario.distance(origin, destination)
                for charger in scenario.chargers:
                    travel_slots[f"{charger.id}|{origin}|{destination}"] = scenario.travel_slots(charger, origin, destination)
            except (ValueError, KeyError):
                continue
    updated_chargers, prefixes = [], []
    for charger in scenario.chargers:
        ledger = result.ledgers["chargers"][charger.id]
        position = ledger["state_location"][slots]
        if position is None:
            arc, remaining = ledger["transit"][slots]
            fraction = (slots - arc.start) / (arc.end - arc.start)
            a, b = locations[arc.origin], locations[arc.destination]
            position = f"@transit:{charger.id}:{scenario.fingerprint()[:12]}:{slots}"
            locations[position] = Location(position, a.x + fraction * (b.x - a.x), a.y + fraction * (b.y - a.y))
            remaining_distance = scenario.distance(arc.origin, arc.destination) * remaining / (arc.end - arc.start)
            for destination in tuple(locations):
                if destination == position:
                    continue
                try:
                    tail_distance = scenario.distance(arc.destination, destination)
                    tail_slots = 0 if arc.destination == destination else scenario.travel_slots(charger, arc.destination, destination)
                except (ValueError, KeyError):
                    continue
                distances[f"{position}|{destination}"] = remaining_distance + tail_distance
                travel_slots[f"{charger.id}|{position}|{destination}"] = remaining + tail_slots
            prefixes.append(Route(charger.id, (Arc(position, arc.destination, 0, remaining),)))
        updated_chargers.append(replace(charger, initial_location=position,
                                        initial_energy=float(result.battery[charger.id][slots])))
    updated_recipients = []
    for recipient in scenario.recipients:
        received = float(np.sum(result.ledgers["recipients"][recipient.id]["received"][:slots]))
        demand = max(0.0, recipient.demand - received) if scenario.domain == "EV" else recipient.demand
        updated_recipients.append(replace(recipient, arrival=max(0, recipient.arrival-slots),
                                          departure=max(0, recipient.departure-slots), demand=demand,
                                          initial_energy=float(result.recipient_energy[recipient.id][slots]),
                                          load=recipient.load[slots:] if recipient.load else (),
                                          raw_arrival_fraction=(recipient.raw_arrival_fraction
                                              if recipient.raw_arrival_fraction is not None
                                              else recipient.arrival/scenario.horizon),
                                          declared_stay_seconds=(recipient.declared_stay_seconds
                                              if recipient.declared_stay_seconds is not None
                                              else (recipient.departure-recipient.arrival)*scenario.step_seconds),
                                          arrival_clock_seconds=(recipient.arrival_clock_seconds
                                              if recipient.arrival_clock_seconds is not None
                                              else (scenario.clock_offset_seconds
                                                    +recipient.arrival*scenario.step_seconds) % 86400)))
    deadline = (scenario.horizon if scenario.return_deadline is None else scenario.return_deadline) - slots
    safe_locations = dict(getattr(scenario, "safe_locations", {}))
    if not any(p.depot for p in scenario.locations):
        for charger in scenario.chargers:
            safe_locations.setdefault(charger.id, charger.initial_location)
    updated = replace(scenario, locations=tuple(locations.values()), chargers=tuple(updated_chargers),
                      recipients=tuple(updated_recipients), horizon=remaining_horizon,
                      road_distance=distances, road_slots=travel_slots, return_deadline=deadline,
                      committed_prefixes=tuple(prefixes), safe_locations=safe_locations,
                      clock_offset_seconds=scenario.clock_offset_seconds+slots*scenario.step_seconds)
    return ExecutionStep(slots, scenario.fingerprint(), plan, result, updated)


def shift_plan(scenario: Scenario, plan: Plan, slots: int) -> Plan:
    """Continue a verified plan under the reference returned by execution.

    This is a deterministic fallback planner, not route reoptimization.
    """
    step = execute_interval(scenario, plan, slots)
    if step.next_scenario is None:
        return Plan(())
    next_reference = step.next_scenario
    routes = []
    for charger in scenario.chargers:
        old = effective_route(scenario, plan, charger.id)
        prefix = next((r for r in next_reference.committed_prefixes if r.charger_id == charger.id), None)
        arcs = list(prefix.arcs if prefix else ())
        arcs.extend(Arc(a.origin, a.destination, a.start-slots, a.end-slots)
                    for a in old.arcs if a.start >= slots)
        services = tuple(replace(s, slot=s.slot-slots) for s in old.services if s.slot >= slots)
        powers = tuple(replace(p, slot=p.slot-slots) for p in old.powers if p.slot >= slots)
        routes.append(Route(charger.id, tuple(arcs), services, powers))
    return Plan(tuple(routes))


def run_episode(initial: Scenario, planner: Callable[[Scenario], Plan],
                replan_slots: int) -> EpisodeResult:
    """Replan on updated states and aggregate executed, rather than forecast, data."""
    if replan_slots <= 0:
        raise ValueError("Replanning interval must be positive")
    steps: list[ExecutionStep] = []
    reference: Scenario | None = initial
    while reference is not None:
        plan = planner(reference)
        step = execute_interval(reference, plan, min(replan_slots, reference.horizon))
        steps.append(step)
        reference = step.next_scenario
    battery = {c.id: np.concatenate([np.asarray([c.initial_energy])] +
                                    [s.evaluation.battery[c.id][1:s.slots+1] for s in steps])
               for c in initial.chargers}
    energy = {r.id: np.concatenate([np.asarray([r.initial_energy])] +
                                   [s.evaluation.recipient_energy[r.id][1:s.slots+1] for s in steps])
              for r in initial.recipients}
    resource = sum(float(np.sum(v["movement"][:s.slots] + v["auxiliary"][:s.slots] + v["output"][:s.slots]))
                   for s in steps for v in s.evaluation.ledgers["chargers"].values())
    cost = resource / initial.energy_reference
    metrics = {"resource_energy": resource, "resource_cost_fraction": cost}
    n, H = len(initial.recipients), initial.horizon
    if initial.domain == "EV":
        received = {r.id: np.concatenate([s.evaluation.ledgers["recipients"][r.id]["received"][:s.slots] for s in steps])
                    for r in initial.recipients}
        delivered = {r.id: min(r.demand, float(np.sum(received[r.id]))) for r in initial.recipients}
        total_demand = sum(r.demand for r in initial.recipients)
        completed = sum(delivered[r.id] >= r.demand - 1e-8 for r in initial.recipients) / n if n else 0
        waits = 0.0
        for r in initial.recipients:
            active = np.flatnonzero(received[r.id] > 1e-8)
            waits += (int(active[0]) - r.arrival if len(active) else r.departure-r.arrival) * initial.step_seconds / 60
        switches = 0
        previous = {c.id: set() for c in initial.chargers}
        for step in steps:
            for cid, ledger in step.evaluation.ledgers["chargers"].items():
                for t in range(step.slots):
                    active = ledger["active_recipients"].get(t, set())
                    if active:
                        switches += len(active-previous[cid]) if previous[cid] else 0
                        previous[cid] = active
        delivery_fraction = sum(delivered.values()) / total_demand if total_demand else 0
        wait_fraction, switch_fraction = (waits/(30*n), switches/n) if n else (0, 0)
        objective = .4*completed + .25*delivery_fraction - .15*wait_fraction - .1*cost - .1*switch_fraction
        metrics.update(completion=completed, delivery_fraction=delivery_fraction,
                       delivered_energy=sum(delivered.values()), unmet_energy=total_demand-sum(delivered.values()),
                       waiting_minutes=waits, mean_wait_minutes=waits/n if n else 0,
                       waiting_fraction=wait_fraction, switches=float(switches), switch_fraction=switch_fraction)
    else:
        alive_slots = sum(float(np.sum(s.evaluation.ledgers["recipients"][r.id]["alive_fraction"][:s.slots]))
                          for s in steps for r in initial.recipients)
        alive = alive_slots/(n*H) if n else 0
        conn = sum(float(np.sum(s.evaluation.ledgers["recipients"]["_network"]["connectivity_fraction"][:s.slots])) for s in steps)/H
        cov = sum(float(np.sum(s.evaluation.ledgers["recipients"]["_network"]["coverage_fraction"][:s.slots])) for s in steps)/H
        outage = 1-alive if n else 0
        objective = .4*alive + .3*conn - .2*outage - .1*cost
        metrics.update(C_alive=alive, C_conn=conn, C_cov=cov, powered_node_time=alive,
                       connectivity=conn, spatial_coverage=cov,
                       outage_node_hours=(n*H-alive_slots)*initial.dt_hours, outage_fraction=outage)
    return EpisodeResult(tuple(steps), battery, energy, metrics, float(objective))
