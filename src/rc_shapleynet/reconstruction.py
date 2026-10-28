"""Budgeted destroy/repair with complete, physically checked route insertions.

This is an explicit greedy insertion operator, not a copy of a published ALNS
implementation. Removed chargers are rebuilt from q's committed travel prefix.
Other routes stay byte-for-byte unchanged. Every complete trial is submitted to
the shared budget callback; this module never invokes physics.evaluate itself.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np

from .schema import Arc, Charger, Plan, Power, Recipient, Route, Scenario, Service
from .physics import Evaluation, effective_route, service_fraction


class CheckerBudget(Protocol):
    def evaluate(self, scenario: Scenario, plan: Plan) -> Evaluation: ...


def destroy(plan: Plan, route_ids: tuple[str, ...]) -> Plan:
    """Remove optional future routes; unavoidable prefixes remain in Scenario."""
    removed = set(route_ids)
    return Plan(tuple(route for route in plan.routes if route.charger_id not in removed))


def _safe_destination(scenario: Scenario, charger: Charger, location: str,
                      time_budget: int | None = None) -> str:
    choices = [p.id for p in scenario.locations if p.depot]
    if not choices:
        choices = [scenario.safe_locations.get(charger.id, charger.initial_location)]
    reachable = []
    for target in choices:
        try:
            slots = 0 if location == target else scenario.travel_slots(charger, location, target)
            energy = (scenario.distance(location, target) * charger.energy_per_metre
                      + slots * scenario.dt_hours * charger.auxiliary_kw)
            reachable.append((energy, slots, target))
        except (KeyError, ValueError):
            continue
    if not reachable:
        raise ValueError("No reachable safe return destination")
    on_time = [row for row in reachable if time_budget is None or row[1] <= time_budget]
    return min(on_time or reachable)[2]


def _travel(scenario: Scenario, charger: Charger, origin: str,
            destination: str, start: int) -> tuple[tuple[Arc, ...], int]:
    if origin == destination:
        return (), start
    end = start + scenario.travel_slots(charger, origin, destination)
    return (Arc(origin, destination, start, end),), end


def idle_return_route(scenario: Scenario, charger: Charger) -> Route:
    """A complete no-service route that preserves unavoidable initial travel."""
    prefix = next((r for r in scenario.committed_prefixes
                   if r.charger_id == charger.id), Route(charger.id))
    position = prefix.arcs[-1].destination if prefix.arcs else charger.initial_location
    clock = prefix.arcs[-1].end if prefix.arcs else 0
    destination = _safe_destination(scenario, charger, position, scenario.horizon-clock)
    tail, clock = _travel(scenario, charger, position, destination, clock)
    if clock > scenario.horizon:
        # The full checker, not this heuristic, decides feasibility.
        return Route(charger.id, prefix.arcs + tail, prefix.services, prefix.powers)
    waiting = tuple(Arc(destination, destination, t, t + 1)
                    for t in range(clock, scenario.horizon))
    return Route(charger.id, prefix.arcs + tail + waiting,
                 prefix.services, prefix.powers)


def _replace_route(plan: Plan, route: Route) -> Plan:
    existing = tuple(r for r in plan.routes if r.charger_id != route.charger_id)
    return Plan(existing + (route,))


def _build_visits(scenario: Scenario, charger: Charger, base: Plan,
                  evaluation: Evaluation, ordered: tuple[Recipient, ...],
                  power_fraction: float) -> Route:
    """Create travel/dwell/service/return arcs before the complete check.

    Local energy bounds are conservative proposals. Receiver acceptance,
    existing commitments and every intermediate safe-return constraint are
    always rechecked by the shared physical evaluator.
    """
    H, dt = scenario.horizon, scenario.dt_hours
    prefix = next((r for r in scenario.committed_prefixes
                   if r.charger_id == charger.id), Route(charger.id))
    arcs, services, powers = list(prefix.arcs), list(prefix.services), list(prefix.powers)
    position = arcs[-1].destination if arcs else charger.initial_location
    clock = arcs[-1].end if arcs else 0
    movement = sum(scenario.distance(a.origin, a.destination) * charger.energy_per_metre
                   for a in arcs if a.origin != a.destination)
    spent = sum(p.output_energy - p.depot_energy for p in powers)
    auxiliary = H * dt * charger.auxiliary_kw
    occupied = {(s.recipient_id, s.slot) for route in base.routes
                if route.charger_id != charger.id for s in route.services if s.energy > 0}
    delivered = {r.id: sum(s.energy for route in base.routes
                          if route.charger_id != charger.id for s in route.services
                          if s.recipient_id == r.id) for r in scenario.recipients}
    for r in ordered:
        if clock >= H or charger.max_power <= 0:
            break
        target = r.location_id
        try:
            travel_arcs, arrival = _travel(scenario, charger, position, target, clock)
            home = _safe_destination(scenario, charger, target,
                                     H-max(arrival, r.arrival)-1)
            return_slots = 0 if target == home else scenario.travel_slots(charger, target, home)
            travel_energy = scenario.distance(position, target) * charger.energy_per_metre
            return_energy = scenario.distance(target, home) * charger.energy_per_metre
        except (KeyError, ValueError):
            continue
        first = max(arrival, r.arrival)
        last = min(r.departure, H - return_slots)
        if first >= last:
            continue
        remaining_battery = (charger.initial_energy - auxiliary - movement
                             - travel_energy - spent - return_energy - charger.reserve_energy)
        if remaining_battery <= 1e-10:
            continue
        if scenario.domain == "EV":
            need = min(max(0.0, r.demand - delivered[r.id]),
                       max(0.0, r.capacity - r.initial_energy - delivered[r.id]))
        else:
            # Replenish future requested load rather than inventing EV demand.
            need = sum(r.load[first:last]) if r.load else max(0.0, r.capacity-r.initial_energy)
        if need <= 1e-10:
            continue
        proposed_services, proposed_powers = [], []
        proposed_spend = 0.0
        projected_energy = float(evaluation.recipient_energy[r.id][first])
        reference_received = evaluation.ledgers["recipients"][r.id]["received"]
        fraction = service_fraction(scenario, charger, target, r.id)
        if fraction <= 0:
            continue
        for slot in range(first, last):
            if (r.id, slot) in occupied:
                if scenario.domain == "WRSN":
                    load = r.load[slot] if r.load else 0.0
                    projected_energy = max(0.0, projected_energy + reference_received[slot] - load)
                continue
            receiver_room = need
            if scenario.domain == "WRSN":
                # The complete check also protects future retained deliveries.
                receiver_room = min(receiver_room,
                    max(0.0, r.capacity-projected_energy-reference_received[slot]))
            output = min(charger.max_power * dt * power_fraction,
                         remaining_battery - proposed_spend, receiver_room / fraction)
            if output <= 1e-10:
                if scenario.domain == "WRSN":
                    load = r.load[slot] if r.load else 0.0
                    projected_energy = max(0.0, projected_energy + reference_received[slot] - load)
                continue
            received = min(need, fraction * output)
            proposed_services.append(Service(r.id, slot, received))
            proposed_powers.append(Power(slot, received / fraction))
            proposed_spend += received / fraction
            need -= received
            if scenario.domain == "WRSN":
                load = r.load[slot] if r.load else 0.0
                projected_energy = max(0.0, projected_energy + reference_received[slot] + received - load)
            if need <= 1e-10 or remaining_battery-proposed_spend <= 1e-10:
                break
        if not proposed_services:
            continue
        finish = proposed_services[-1].slot + 1
        arcs.extend(travel_arcs)
        arcs.extend(Arc(target, target, t, t + 1) for t in range(arrival, finish))
        services.extend(proposed_services)
        powers.extend(proposed_powers)
        spent += proposed_spend
        movement += travel_energy
        delivered[r.id] += sum(s.energy for s in proposed_services)
        position, clock = target, finish
    home = _safe_destination(scenario, charger, position, H-clock)
    tail, clock = _travel(scenario, charger, position, home, clock)
    arcs.extend(tail)
    arcs.extend(Arc(home, home, t, t + 1) for t in range(clock, H))
    return Route(charger.id, tuple(arcs), tuple(services), tuple(powers))


def greedy_reconstruct(scenario: Scenario, partial: Plan, removed_ids: tuple[str, ...],
                       budget: CheckerBudget, rng: np.random.Generator) -> Plan:
    """Rebuild removed routes with shared, budget-counted physical evaluations.

    Deadline insertion, a bounded deadline-order perturbation, and three output
    fractions form the neighborhood. The same operator is used with standard
    and recovery-conditioned guidance. It makes no benchmark-performance claim.
    Budget exhaustion propagates so the caller keeps the last verified incumbent.
    """
    charger_map = {c.id: c for c in scenario.chargers}
    if len(set(removed_ids)) != len(removed_ids) or set(removed_ids)-charger_map.keys():
        raise ValueError("Reconstruction requires distinct known chargers")
    if set(removed_ids) & {r.charger_id for r in partial.routes}:
        raise ValueError("Destroyed routes must not remain in the partial plan")
    candidate = partial
    for cid in removed_ids:
        candidate = _replace_route(candidate, idle_return_route(scenario, charger_map[cid]))
    current = budget.evaluate(scenario, candidate)
    if not current.feasible:
        return candidate
    for cid in removed_ids:
        charger = charger_map[cid]
        deadline = tuple(sorted(scenario.recipients, key=lambda r: (r.departure, r.arrival, r.id)))
        perturbations = {r.id: float(rng.uniform(0, .2 * scenario.horizon)) for r in deadline}
        perturbed = tuple(sorted(deadline, key=lambda r: (r.departure+perturbations[r.id], r.id)))
        orders = (deadline,) if perturbed == deadline else (deadline, perturbed)
        base, base_evaluation = candidate, current
        for order in orders:
            for fraction in (1.0, .5, .25):
                route = _build_visits(scenario, charger, base, base_evaluation, order, fraction)
                trial = _replace_route(base, route)
                checked = budget.evaluate(scenario, trial)
                if checked.feasible and checked.objective > current.objective:
                    candidate, current = trial, checked
    return candidate
