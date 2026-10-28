"""Physical validation and normalized utility under a frozen planning reference.

Received energy precedes a slot's fixed sensor load. Charger input, output and
auxiliary rates are constant within a slot. Arrays include both end states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

from .schema import Arc, Charger, Plan, Route, Scenario


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    objective: float
    metrics: dict[str, float]
    battery: dict[str, np.ndarray]
    recipient_energy: dict[str, np.ndarray]
    violations: tuple[str, ...]
    ledgers: dict[str, Any]


def _idle_route(scenario: Scenario, charger: Charger) -> Route:
    prefix = next((r for r in scenario.committed_prefixes if r.charger_id == charger.id), None)
    arcs = prefix.arcs if prefix else ()
    position = arcs[-1].destination if arcs else charger.initial_location
    clock = arcs[-1].end if arcs else 0
    return Route(charger.id, arcs + tuple(Arc(position, position, t, t + 1)
                                         for t in range(clock, scenario.horizon)),
                 prefix.services if prefix else (), prefix.powers if prefix else ())


def effective_route(scenario: Scenario, plan: Plan, charger_id: str) -> Route:
    """Expand an absent route into the common travel prefix and idle waiting."""
    explicit = [r for r in plan.routes if r.charger_id == charger_id]
    if len(explicit) > 1:
        raise ValueError("A plan cannot contain duplicate charger routes")
    if explicit:
        return explicit[0]
    charger = next(c for c in scenario.chargers if c.id == charger_id)
    return _idle_route(scenario, charger)


def transfer_fraction(scenario: Scenario, distance: float) -> float:
    """Interpolate the supplied calibration; no transfer beyond its last knot."""
    curve = scenario.transfer_curve
    if not curve or any(not math.isfinite(x) or not math.isfinite(y) or x < 0
                        or not 0 <= y <= 1 for x, y in curve):
        raise ValueError("Invalid transfer calibration")
    if any(curve[j][0] >= curve[j + 1][0] for j in range(len(curve) - 1)):
        raise ValueError("Calibration distances must be strictly increasing")
    if distance > curve[-1][0]:
        return 0.0
    return float(np.interp(distance, [x for x, _ in curve], [y for _, y in curve]))


def service_fraction(scenario: Scenario, charger: Charger, location: str,
                     recipient_id: str) -> float:
    """Return the physical receiver-side limit per transmitter-side unit."""
    recipient = next(r for r in scenario.recipients if r.id == recipient_id)
    if scenario.domain == "EV":
        return charger.receiver_efficiency if location == recipient.location_id else 0.0
    locs = {v.id: v for v in scenario.locations}
    a, b = locs[location], locs[recipient.location_id]
    return transfer_fraction(scenario, math.hypot(a.x - b.x, a.y - b.y))


def _return_requirement(scenario: Scenario, charger: Charger, location: str,
                        time_budget: int | None = None
                        ) -> tuple[float, int, str | None]:
    choices = []
    safe_locations = [p for p in scenario.locations if p.depot]
    if not safe_locations:
        safe_id = getattr(scenario, "safe_locations", {}).get(charger.id, charger.initial_location)
        safe_locations = [p for p in scenario.locations if p.id == safe_id]
    for depot in safe_locations:
        try:
            distance = scenario.distance(location, depot.id)
            slots = 0 if location == depot.id else scenario.travel_slots(charger, location, depot.id)
            energy = distance * charger.energy_per_metre + slots * scenario.dt_hours * charger.auxiliary_kw
            choices.append((energy + charger.reserve_energy, slots, depot.id))
        except (KeyError, ValueError):
            continue
    within_time = [v for v in choices if time_budget is None or v[1] <= time_budget]
    return min(within_time or choices) if choices else (math.inf, 0, None)


def evaluate(scenario: Scenario, plan: Plan, strict: bool = True) -> Evaluation:
    """Validate every constraint and return a complete energy/utility ledger.

    Invalid plans return ``feasible=False`` and ``objective=-inf``. ``strict``
    selects numerical tolerance (1e-8 versus 1e-6); it never disables a constraint.
    Missing routes are explicit idle chargers with the same auxiliary load.
    """
    tol = 1e-8 if strict else 1e-6
    H, dt = scenario.horizon, scenario.dt_hours
    chargers = {c.id: c for c in scenario.chargers}
    recipients = {r.id: r for r in scenario.recipients}
    locations = {p.id: p for p in scenario.locations}
    violations: list[str] = []
    route_map: dict[str, Route] = {}
    for route in plan.routes:
        if route.charger_id not in chargers:
            violations.append(f"Unknown charger {route.charger_id}")
        elif route.charger_id in route_map:
            violations.append(f"Duplicate route for {route.charger_id}")
        else:
            route_map[route.charger_id] = route
    battery = {c.id: np.full(H + 1, np.nan) for c in scenario.chargers}
    stored = {r.id: np.zeros(H + 1) for r in scenario.recipients}
    received = {r.id: np.zeros(H) for r in scenario.recipients}
    # Allocate only active charger-recipient pairs, rather than an m*n*H cube.
    by_source: dict[tuple[str, str], np.ndarray] = {}
    source_activity: dict[tuple[str, int], set[str]] = {}
    cledger: dict[str, Any] = {}

    for c in scenario.chargers:
        cid = c.id
        supplied = route_map.get(cid)
        route = supplied if supplied is not None else _idle_route(scenario, c)
        arcs = route.arcs
        prefix = next((r for r in scenario.committed_prefixes if r.charger_id == cid), None)
        if prefix is not None:
            if (route.arcs[:len(prefix.arcs)] != prefix.arcs or
                any(a.origin == a.destination for a in prefix.arcs) or prefix.services or prefix.powers):
                violations.append(f"{cid}: route changes unavoidable in-progress travel")
        dwell: list[str | None] = [None] * H
        state_location: list[str | None] = [None] * (H + 1)
        transit: list[tuple[Arc, int] | None] = [None] * (H + 1)
        movement = np.zeros(H)
        auxiliary = np.full(H, c.auxiliary_kw * dt)
        output, depot_input = np.zeros(H), np.zeros(H)
        position, clock = c.initial_location, 0
        state_location[0] = position
        for arc in arcs:
            if not (isinstance(arc.start, int) and isinstance(arc.end, int)
                    and 0 <= arc.start < arc.end <= H):
                violations.append(f"{cid}: invalid arc interval {arc}")
                continue
            if arc.origin not in locations or arc.destination not in locations:
                violations.append(f"{cid}: unknown arc endpoint")
                continue
            if arc.start != clock or arc.origin != position:
                violations.append(f"{cid}: discontinuous flow at slot {arc.start}")
            duration = arc.end - arc.start
            if arc.origin == arc.destination:
                if duration != 1:
                    violations.append(f"{cid}: a waiting arc must occupy one slot")
                for t in range(arc.start, arc.end):
                    dwell[t] = arc.origin
                    state_location[t] = arc.origin
                    state_location[t + 1] = arc.origin
            else:
                try:
                    distance = scenario.distance(arc.origin, arc.destination)
                    expected = scenario.travel_slots(c, arc.origin, arc.destination)
                    if duration != expected:
                        violations.append(f"{cid}: travel duration {duration} differs from frozen {expected}")
                    movement[arc.start:arc.end] += distance * c.energy_per_metre / duration
                    state_location[arc.start] = arc.origin
                    for t in range(arc.start + 1, arc.end):
                        state_location[t] = None
                        transit[t] = (arc, arc.end - t)
                    state_location[arc.end] = arc.destination
                except (ValueError, KeyError) as exc:
                    violations.append(f"{cid}: {exc}")
            clock, position = arc.end, arc.destination
        if clock != H:
            violations.append(f"{cid}: route must include waiting or travel through horizon {H}")
        for power in route.powers:
            if not isinstance(power.slot, int) or not 0 <= power.slot < H:
                violations.append(f"{cid}: power outside horizon")
                continue
            if any(not math.isfinite(v) or v < -tol for v in (power.output_energy, power.depot_energy)):
                violations.append(f"{cid}: nonfinite or negative power energy")
                continue
            output[power.slot] += max(0.0, power.output_energy)
            depot_input[power.slot] += max(0.0, power.depot_energy)
        active_by_slot: dict[int, set[str]] = {}
        for service in route.services:
            rid, t, energy = service.recipient_id, service.slot, service.energy
            if rid not in recipients or not isinstance(t, int) or not 0 <= t < H:
                violations.append(f"{cid}: unknown recipient or invalid service slot")
                continue
            if not math.isfinite(energy) or energy < -tol:
                violations.append(f"{cid}: nonfinite or negative received energy")
                continue
            if energy <= tol:
                continue
            r = recipients[rid]
            if not r.arrival <= t < r.departure:
                violations.append(f"{cid}->{rid}: service outside original availability at {t}")
            if dwell[t] is None:
                violations.append(f"{cid}->{rid}: service requires a waiting arc at {t}")
            else:
                try:
                    fraction = service_fraction(scenario, c, dwell[t], rid)
                    if fraction <= 0:
                        violations.append(f"{cid}->{rid}: incompatible location at {t}")
                except (ValueError, KeyError) as exc:
                    violations.append(f"{cid}: {exc}")
            active_by_slot.setdefault(t, set()).add(rid)
            source_activity.setdefault((rid, t), set()).add(cid)
            if (cid, rid) not in by_source:
                by_source[cid, rid] = np.zeros(H)
            by_source[cid, rid][t] += energy
            received[rid][t] += energy

        for t in range(H):
            active = active_by_slot.get(t, set())
            if scenario.domain == "EV" and len(active) > c.ports:
                violations.append(f"{cid}: charger port capacity exceeded at {t}")
            if output[t] > c.max_power * dt + tol:
                violations.append(f"{cid}: transmitter power exceeded at {t}")
            if output[t] > tol and dwell[t] is None:
                violations.append(f"{cid}: transmission while travelling at {t}")
            if depot_input[t] > c.depot_power * dt + tol:
                violations.append(f"{cid}: depot input power exceeded at {t}")
            if depot_input[t] > tol:
                if dwell[t] is None or not locations[dwell[t]].depot:
                    violations.append(f"{cid}: depot input away from a depot at {t}")
                if output[t] > tol and not scenario.allow_simultaneous_depot_service:
                    violations.append(f"{cid}: simultaneous depot input and output at {t}")
            total_received = sum(by_source[cid, rid][t] for rid in active)
            if total_received > output[t] + tol:
                violations.append(f"{cid}: broadcast energy conservation violated at {t}")
            if dwell[t] is not None:
                for rid in active:
                    try:
                        fraction = service_fraction(scenario, c, dwell[t], rid)
                        if by_source[cid, rid][t] > fraction * output[t] + tol:
                            violations.append(f"{cid}->{rid}: transfer bound exceeded at {t}")
                    except (ValueError, KeyError):
                        violations.append(f"{cid}->{rid}: transfer bound could not be evaluated at {t}")
            # In EV mode each port consumes its own transmitter-side energy.
            if scenario.domain == "EV" and total_received > c.receiver_efficiency * output[t] + tol:
                violations.append(f"{cid}: EV conversion bound exceeded at {t}")
        battery[cid][0] = c.initial_energy
        for t in range(H):
            battery[cid][t + 1] = battery[cid][t] + depot_input[t] - output[t] - movement[t] - auxiliary[t]
        reserve, return_slots = np.zeros(H + 1), np.zeros(H + 1, dtype=int)
        return_destinations: list[str | None] = []
        return_deadline = scenario.horizon if scenario.return_deadline is None else scenario.return_deadline
        for t in range(H + 1):
            location = state_location[t]
            if location is not None:
                need, time, destination = _return_requirement(scenario, c, location, return_deadline-t)
            elif transit[t] is not None:
                arc, remaining = transit[t]
                need, time, destination = _return_requirement(scenario, c, arc.destination, return_deadline-t-remaining)
                need += float(np.sum(movement[t:arc.end] + auxiliary[t:arc.end]))
                time += remaining
            else:
                need, time, destination = math.inf, 0, None
            reserve[t], return_slots[t] = need, time
            return_destinations.append(destination)
            if battery[cid][t] < need - tol or battery[cid][t] > c.capacity + tol:
                violations.append(f"{cid}: battery storage or safe-return reserve violated at state {t}")
            if t + time > return_deadline:
                violations.append(f"{cid}: safe return misses deadline at state {t}")
        cledger[cid] = {"dwell": tuple(dwell), "state_location": tuple(state_location),
                        "transit": tuple(transit), "movement": movement, "auxiliary": auxiliary,
                        "output": output, "depot_input": depot_input, "return_reserve": reserve,
                        "return_slots": return_slots, "return_destinations": tuple(return_destinations),
                        "return_time_checked": True,
                        "active_recipients": active_by_slot}

    for (rid, t), sources in source_activity.items():
        if len(sources) > 1:
            violations.append(f"{rid}: multiple effective recipient inputs at {t}")
    rledger: dict[str, Any] = {}
    for r in scenario.recipients:
        stored[r.id][0] = r.initial_energy
        load = np.asarray(r.load if r.load else (0.0,) * H, dtype=float)
        executed, alive = np.zeros(H), np.zeros(H)
        for t in range(H):
            before_load = stored[r.id][t] + received[r.id][t]
            if before_load > r.capacity + tol:
                violations.append(f"{r.id}: receiver acceptance capacity exceeded at {t}")
            if scenario.domain == "EV":
                if float(np.sum(received[r.id][:t + 1])) > r.demand + tol:
                    violations.append(f"{r.id}: received energy exceeds residual requested energy at {t}")
                executed[t] = 0.0
            else:
                executed[t] = min(load[t], max(0.0, before_load))
            alive[t] = min(1.0, max(0.0, before_load) / load[t]) if load[t] > 0 else float(before_load > tol)
            stored[r.id][t + 1] = max(0.0, before_load - executed[t])
        rledger[r.id] = {"received": received[r.id], "requested_load": load,
                          "executed_load": executed, "unmet_load": load - executed,
                          "alive_fraction": alive}

    resource = sum(float(np.sum(v["movement"] + v["auxiliary"] + v["output"])) for v in cledger.values())
    n = len(recipients)
    if scenario.energy_reference <= 0:
        violations.append("Resource energy reference must be positive")
    cost_fraction = resource / scenario.energy_reference if scenario.energy_reference > 0 else math.inf
    metrics: dict[str, float] = {"resource_energy": resource, "resource_cost_fraction": cost_fraction}
    if scenario.domain == "EV":
        delivered = {rid: min(r.demand, float(np.sum(received[rid]))) for rid, r in recipients.items()}
        total_demand = sum(r.demand for r in recipients.values())
        completion = sum(delivered[rid] >= r.demand - tol for rid, r in recipients.items()) / n if n else 0.0
        wait_slots = 0.0
        for rid, r in recipients.items():
            nonzero = np.flatnonzero(received[rid] > tol)
            first = int(nonzero[0]) if len(nonzero) else r.departure
            wait_slots += max(0, first - r.arrival)
        switches = 0
        for ledger in cledger.values():
            previous: set[str] = set()
            for t in range(H):
                current = ledger["active_recipients"].get(t, set())
                if current:
                    switches += len(current - previous) if previous else 0
                    previous = current
        delivery_fraction = sum(delivered.values()) / total_demand if total_demand else 0.0
        waiting_minutes = wait_slots * scenario.step_seconds / 60
        wait_fraction = waiting_minutes / (30 * n) if n else 0.0
        switch_fraction = switches / n if n else 0.0
        objective = .4 * completion + .25 * delivery_fraction - .15 * wait_fraction - .1 * cost_fraction - .1 * switch_fraction
        metrics.update(completion=completion, delivery_fraction=delivery_fraction,
                       delivered_energy=sum(delivered.values()), unmet_energy=total_demand-sum(delivered.values()),
                       waiting_minutes=waiting_minutes, mean_wait_minutes=waiting_minutes/n if n else 0.0,
                       waiting_fraction=wait_fraction, switches=float(switches), switch_fraction=switch_fraction)
    else:
        alive = {rid: rledger[rid]["alive_fraction"] for rid in recipients}
        alive_fraction = sum(float(np.sum(v)) for v in alive.values()) / (n * H) if n else 0.0
        outage_hours = (n * H - sum(float(np.sum(v)) for v in alive.values())) * dt
        conn_slot = np.zeros(H)
        if scenario.required_paths:
            edges = {frozenset(pair) for pair in scenario.communication_edges}
            path_availability = []
            for path in scenario.required_paths:
                if not path or any(rid not in recipients for rid in path):
                    violations.append("Required path contains an unknown recipient or no nodes")
                    path_availability.append(np.zeros(H))
                elif any(frozenset((a, b)) not in edges for a, b in zip(path, path[1:])):
                    # A structurally unavailable path contributes zero; it does
                    # not invalidate other paths or the charging plan itself.
                    path_availability.append(np.zeros(H))
                else:
                    path_availability.append(np.min(np.stack([alive[rid] for rid in path]), axis=0))
            conn_slot = np.mean(np.stack(path_availability), axis=0)
        coverage_slot = np.zeros(H)
        point_weights = sum(w for _, _, w in scenario.coverage_points)
        if any(w < 0 for _, _, w in scenario.coverage_points):
            violations.append("Coverage quadrature weights must be nonnegative")
        if point_weights > 0:
            for x, y, weight in scenario.coverage_points:
                covering = [r.id for r in recipients.values()
                            if math.hypot(locations[r.location_id].x-x, locations[r.location_id].y-y) <= r.sensing_radius]
                if covering:
                    coverage_slot += weight * np.max(np.stack([alive[rid] for rid in covering]), axis=0) / point_weights
        conn, coverage = float(np.mean(conn_slot)), float(np.mean(coverage_slot))
        outage_fraction = 1 - alive_fraction if n else 0.0
        objective = .4 * alive_fraction + .3 * conn - .2 * outage_fraction - .1 * cost_fraction
        metrics.update(C_alive=alive_fraction, C_cov=coverage, C_conn=conn,
                       powered_node_time=alive_fraction, spatial_coverage=coverage, connectivity=conn,
                       outage_node_hours=outage_hours, outage_fraction=outage_fraction)
        rledger["_network"] = {"connectivity_fraction": conn_slot, "coverage_fraction": coverage_slot}
    violations = list(dict.fromkeys(violations))
    return Evaluation(not violations, float(objective) if not violations else -math.inf, metrics,
                      battery, stored, tuple(violations),
                      {"chargers": cledger, "recipients": rledger, "service_energy": by_source,
                       "reference_id": scenario.fingerprint(), "tolerance": tol})
